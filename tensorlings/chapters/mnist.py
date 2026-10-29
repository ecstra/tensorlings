"""14_mnist, Section 2's project, softmax regression on the MNIST digits, tuned on validation and measured once on test."""

from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..data import load_mnist
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
    fresh_copy,
    reject_missing_return,
    seeded,
    small_tensor,
)
from ..text import prose, source_file

MNIST = Chapter(
    folder="14_mnist",
    title="Project: MNIST digits",
    readme=source_file(
        """
        # Project: MNIST digits

        Section 2's project. You train softmax regression on MNIST, 70,000 small
        photos of handwritten digits, choose its weight decay on a validation
        set, and measure it once on the test set, as CS231n's first assignment
        does. Download the data once from the project folder.

        ```bash
        .venv/bin/python run.py data mnist
        ```

        ## Further reading

        - [LeCun and colleagues, the MNIST database](https://yann.lecun.org/exdb/mnist/)
        - [CS231n, assignment 1](https://cs231n.github.io/assignments2025/assignment1/)
        """,
    ),
)


@functools.cache
def mnist_sets() -> dict[str, Tensor]:
    """The splits the lessons quote numbers from, built once per run of the app."""
    images, labels = load_mnist("train")
    test_images, test_labels = load_mnist("test")
    x = images.reshape(len(images), -1).float() / 255
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))

    return {
        "train_x": x[order[:50000]],
        "train_y": labels[order[:50000]],
        "validation_x": x[order[50000:]],
        "validation_y": labels[order[50000:]],
        "small_x": x[order[:1000]],
        "small_y": labels[order[:1000]],
        "test_x": test_images.reshape(len(test_images), -1).float() / 255,
        "test_y": test_labels,
    }


def mnist_set(name: str) -> Tensor:
    return fresh_copy(mnist_sets()[name])


def accuracy_of(
    x: Tensor,
    y: Tensor,
    W: Tensor,
    b: Tensor,
) -> float:
    return ((x @ W + b).argmax(dim=1) == y).float().mean().item()


# ---------------------------------------------------------------------------
# mnist1
# ---------------------------------------------------------------------------

MNIST1_LESSON = prose(
    """
    This is Section 2's project. You train softmax regression on MNIST, 70,000
    small photos of handwritten digits, and measure it the honest way, as
    CS231n's first assignment does. Download the data once, about 11 MB, from
    the project folder with `.venv/bin/python run.py data mnist`.

    ```python
    from tensorlings.data import load_mnist

    images, labels = load_mnist("train")
    images.shape    # torch.Size([60000, 28, 28])
    labels[:5]      # tensor([5, 0, 4, 1, 9])
    ```

    Each image is 28 by 28 pixels, from 0 for black up to 255 for white.
    MNIST comes split already, 60,000 images for training and 10,000 for the
    test, which stay untouched until mnist3. The validation set comes out of
    the 60,000.

    Softmax regression takes one flat row of features per example, so each
    image becomes a row of 784 pixels. `.reshape` gives a tensor a new shape
    with the same entries in the same order, and -1 stands for whatever size
    makes it fit.

    ```python
    images.reshape(len(images), -1).shape    # torch.Size([60000, 784])
    ```

    The pixels get scaled to between 0 and 1, by `.float()` then dividing by
    255. Inputs in the hundreds would make the first steps of training jump.

    A tensor of `True` and `False` picks the rows where it is `True`.

    ```python
    x = torch.tensor([[1.0], [2.0], [3.0]])
    x[torch.tensor([True, False, True])]    # tensor([[1.], [3.]])
    ```

    So `images[labels == 3]` is every image of a 3.
    """,
)

MNIST1_SOURCE = source_file(
    '''
    # mnist1.py
    #
    # Section 2's project starts with the data. Download it once from the project
    # folder with
    #
    #     .venv/bin/python run.py data mnist

    import torch
    from torch import Tensor

    from tensorlings.data import load_mnist

    images, labels = load_mnist("train")       # shapes (60000, 28, 28) and (60000,)

    # .reshape gives the same entries a new shape, and -1 stands for whatever fits.
    rows = images.reshape(len(images), -1)      # shape (60000, 784)

    # A tensor of True and False picks the rows where it is True.
    threes = images[labels == 3]                # every image of a 3


    # TODO: Return the images as rows of 784 pixels each, as floats from 0 to 1.
    def flatten_and_scale(images: Tensor) -> Tensor:
        ...


    # TODO: Finish the middle line, so this shuffles the examples with generator,
    #   and the last validation_size shuffled positions go to validation. It
    #   returns the training rows and labels, then the validation rows and
    #   labels.
    def train_validation_split(
        x: Tensor,
        y: Tensor,
        validation_size: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        order = torch.randperm(len(x), generator=generator)
        train, validation = ...

        return x[train], y[train], x[validation], y[validation]


    # TODO: Return the average of the rows of x whose label is digit, one number
    #   per pixel. It shows what that digit looks like on average.
    def average_image(
        x: Tensor,
        labels: Tensor,
        digit: int,
    ) -> Tensor:
        ...
    ''',
)

MNIST1_SOLUTION = source_file(
    '''
    # mnist1.py, a solution for comparison

    import torch
    from torch import Tensor

    from tensorlings.data import load_mnist

    images, labels = load_mnist("train")

    rows = images.reshape(len(images), -1)

    threes = images[labels == 3]


    def flatten_and_scale(images: Tensor) -> Tensor:
        return images.reshape(len(images), -1).float() / 255


    def train_validation_split(
        x: Tensor,
        y: Tensor,
        validation_size: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        order = torch.randperm(len(x), generator=generator)
        train, validation = order[:-validation_size], order[-validation_size:]

        return x[train], y[train], x[validation], y[validation]


    def average_image(
        x: Tensor,
        labels: Tensor,
        digit: int,
    ) -> Tensor:
        return x[labels == digit].mean(dim=0)
    ''',
)


def small_images(
    seed: int,
    count: int,
) -> Tensor:
    return torch.randint(0, 256, (count, 28, 28), generator=torch.Generator().manual_seed(seed), dtype=torch.uint8)


def reference_split(
    x: Tensor,
    y: Tensor,
    validation_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    order = torch.randperm(len(x), generator=generator)
    train, validation = order[:-validation_size], order[-validation_size:]

    return x[train], y[train], x[validation], y[validation]


def check_mnist1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="flatten_and_scale",
        description="gives rows of 784 floats from 0 to 1",
        inputs=(
            (Described(small_images(154, 3), "3 images"),),
            (Described(small_images(155, 5), "5 images"),),
        ),
        reference=lambda images: images.reshape(len(images), -1).float() / 255,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="train_validation_split",
        description="shuffles and puts the last positions in validation",
        inputs=(
            (Described(small_tensor(156, 10, 2), "x with 10 rows"), Described(torch.arange(10), "y"), 3, seeded(0)),
            (Described(small_tensor(157, 12, 3), "x with 12 rows"), Described(torch.arange(12) % 4, "y"), 2, seeded(5)),
        ),
        reference=reference_split,
        expect=expect_parts("the training rows", "the training labels", "the validation rows", "the validation labels"),
    )
    check_on_inputs(
        checker=checker,
        function="average_image",
        description="gives the mean of the rows with that label",
        inputs=(
            (small_tensor(158, 6, 4), torch.tensor([0, 1, 0, 2, 1, 0]), 0),
            (small_tensor(159, 6, 4), torch.tensor([0, 1, 0, 2, 1, 0]), 2),
        ),
        reference=lambda x, labels, digit: x[labels == digit].mean(dim=0),
        expect=expect_tensor,
    )


MNIST1 = Exercise(
    name="mnist1",
    chapter=MNIST,
    title="Project: the data",
    lesson=MNIST1_LESSON,
    source=MNIST1_SOURCE,
    solution=MNIST1_SOLUTION,
    hints={
        "flatten_and_scale": "Reshape to len(images) rows, as the line that sets rows does, then .float() and "
        "divide by 255.",
        "train_validation_split": "Slice order twice. Everything but the last validation_size positions is "
        "order[:-validation_size], and the last ones are order[-validation_size:].",
        "average_image": "labels == digit picks that digit's rows of x. Take their mean down the rows, dim=0.",
    },
    introduces=(),
    check=check_mnist1,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# mnist2
# ---------------------------------------------------------------------------

MNIST2_LESSON = prose(
    """
    Training is descent2's loop with the pieces of Section 2, softmax
    regression for the model, cross entropy for the loss and minibatches for
    the steps. The weights have shape `(784, 10)`, one column per digit, and the
    bias has one number per digit.

    The weights start at zero. Every class then gets the same score, so the
    loss starts at `log(10)`, about 2.303, as likelihood2 said. A first loss
    far from that means a bug before training has even begun. Karpathy's
    recipe for training lists this check among its first.

    With minibatches of 128, an epoch of 50,000 examples is 391 steps. Two
    epochs with a learning rate of 0.5 take about a second here, and reach
    about 91 percent on the validation set.

    That is close to the best a linear model does on MNIST, about 92 percent.
    Each pixel gets one weight per digit. So the model learns which pixels tend
    to be ink for each digit, but nothing about strokes or shapes. The models
    of Section 3 pass 98 percent.

    Measuring accuracy needs no gradients, so it can run inside
    `with torch.no_grad():`. That saves the work of recording.
    """,
)

MNIST2_SOURCE = source_file(
    '''
    # mnist2.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Finish the line that sets loss, the cross entropy of softmax
    #   regression on the minibatch's rows, so this trains W and b and returns
    #   them.
    def train(
        x: Tensor,
        y: Tensor,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        W = torch.zeros(x.shape[1], 10, requires_grad=True)
        b = torch.zeros(10, requires_grad=True)

        for _ in range(epochs):
            order = torch.randperm(len(x), generator=generator)

            for start in range(0, len(x), batch_size):
                positions = order[start:start + batch_size]
                loss = ...
                loss.backward()

                with torch.no_grad():
                    W -= learning_rate * W.grad
                    b -= learning_rate * b.grad

                W.grad = None
                b.grad = None

        return W.detach(), b.detach()


    # TODO: Return the fraction of rows of x whose predicted class equals its
    #   label in y.
    def accuracy(
        x: Tensor,
        y: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> float:
        ...
    ''',
)

MNIST2_SOLUTION = source_file(
    '''
    # mnist2.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    # The first loss, with W and b at zero, should be log(10), about 2.303.
    def train(
        x: Tensor,
        y: Tensor,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        W = torch.zeros(x.shape[1], 10, requires_grad=True)
        b = torch.zeros(10, requires_grad=True)

        for _ in range(epochs):
            order = torch.randperm(len(x), generator=generator)

            for start in range(0, len(x), batch_size):
                positions = order[start:start + batch_size]
                loss = F.cross_entropy(x[positions] @ W + b, y[positions])
                loss.backward()

                with torch.no_grad():
                    W -= learning_rate * W.grad
                    b -= learning_rate * b.grad

                W.grad = None
                b.grad = None

        return W.detach(), b.detach()


    def accuracy(
        x: Tensor,
        y: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> float:
        with torch.no_grad():
            return ((x @ W + b).argmax(dim=1) == y).float().mean().item()
    ''',
)


def check_trained_on_mnist(found: object) -> None:
    call = "train on the 50,000 training images, for 2 epochs of minibatches of 128 with learning rate 0.5"
    trained = found(mnist_set("train_x"), mnist_set("train_y"), 2, 128, 0.5, torch.Generator().manual_seed(1))
    reject_missing_return(
        call=call,
        actual=trained,
    )
    W, b = trained
    accuracy = accuracy_of(mnist_set("validation_x"), mnist_set("validation_y"), W, b)

    if accuracy < 0.9:
        raise Mismatch(f"After {call}, the validation accuracy is {accuracy:.3f}, but it should be at least 0.9.")


def check_mnist2(checker: Checker) -> None:
    checker.test(
        function="train",
        description="trains softmax regression past 90 percent on the validation set",
        run=check_trained_on_mnist,
    )
    check_on_inputs(
        checker=checker,
        function="accuracy",
        description="gives the fraction of rows predicted right",
        inputs=(
            (small_tensor(161, 6, 4), torch.tensor([0, 2, 1, 1, 0, 2]), small_tensor(162, 4, 3), small_tensor(163, 3)),
            (small_tensor(164, 5, 4), torch.tensor([1, 1, 0, 2, 2]), small_tensor(165, 4, 3), small_tensor(166, 3)),
        ),
        reference=accuracy_of,
        expect=expect_close,
    )


MNIST2 = Exercise(
    name="mnist2",
    chapter=MNIST,
    title="Project: training",
    lesson=MNIST2_LESSON,
    source=MNIST2_SOURCE,
    solution=MNIST2_SOLUTION,
    hints={
        "train": "The scores of the minibatch are x[positions] @ W + b, and its labels are y[positions]. Pass both "
        "to F.cross_entropy.",
        "accuracy": "Predict with argmax of the scores along dim=1, compare with y, then the mean of the matches as "
        "floats, as softmax3 does.",
    },
    introduces=(),
    check=check_mnist2,
    time_limit=120,
)


# ---------------------------------------------------------------------------
# mnist3
# ---------------------------------------------------------------------------

MNIST3_LESSON = prose(
    """
    With 50,000 examples and only 7,850 parameters, softmax regression barely
    overfits, so weight decay has little to fix. Train on 1,000 examples
    instead, for 100 epochs, and it does overfit. Here is that run with each
    strength.

    | Strength | Training accuracy | Validation accuracy |
    |---|---|---|
    | 0 | 1.000 | 0.860 |
    | 0.001 | 0.999 | 0.864 |
    | 0.01 | 0.931 | 0.860 |
    | 0.1 | 0.724 | 0.694 |

    Every strength fits the training set almost perfectly, while validation
    stays near 86 percent. Decay closes that gap only a little here, and too
    much of it hurts both.

    Pick the strength on the validation set, then measure on the test set,
    once.

    A **confusion matrix** shows which mistakes a model makes. It counts, for
    every true class and every predicted class, how many examples had that
    pair. The diagonal holds the right answers, and the largest count off the
    diagonal is the mistake made most. For the model of mnist2 that is a 9
    read as a 4, 41 times in the test set.

    `.fill_diagonal_(0)` sets a square matrix's diagonal to 0 in place, so
    clone the matrix first. `.argmax()` with no `dim` gives a position in the
    matrix read row after row, as if it were flat. Python's `divmod(a, b)`
    gives `a // b` and `a % b`, and with `b` the number of columns, that is the
    row and the column.

    ```python
    torch.tensor([[1, 9], [5, 2]]).argmax()    # tensor(1), row 0, column 1
    divmod(1, 2)                               # (0, 1)
    ```
    """,
)

MNIST3_SOURCE = source_file(
    '''
    # mnist3.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    # Trains softmax regression with weight decay as mnist2 does, for 100 epochs
    # of minibatches of 100, with the same seed every time so runs compare fairly.
    def train_with_decay(
        x: Tensor,
        y: Tensor,
        strength: float,
    ) -> tuple[Tensor, Tensor]:
        generator = torch.Generator().manual_seed(1)
        W = torch.zeros(x.shape[1], 10, requires_grad=True)
        b = torch.zeros(10, requires_grad=True)

        for _ in range(100):
            order = torch.randperm(len(x), generator=generator)

            for start in range(0, len(x), 100):
                positions = order[start:start + 100]
                loss = F.cross_entropy(x[positions] @ W + b, y[positions]) + strength * (W ** 2).sum()
                loss.backward()

                with torch.no_grad():
                    W -= 0.5 * W.grad
                    b -= 0.5 * b.grad

                W.grad = None
                b.grad = None

        return W.detach(), b.detach()


    def accuracy(
        x: Tensor,
        y: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> float:
        return ((x @ W + b).argmax(dim=1) == y).float().mean().item()


    # TODO: Finish the line in the loop, so this returns the strength from
    #   strengths whose model, trained on the training set, has the highest
    #   accuracy on the validation set.
    def pick_strength(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        strengths: list[float],
    ) -> float:
        scores = []

        for strength in strengths:
            W, b = train_with_decay(train_x, train_y, strength)
            scores.append(...)

        return strengths[scores.index(max(scores))]


    # TODO: Finish the first and last lines, so this trains with the strength
    #   pick_strength chooses and returns its accuracy on the test set.
    def test_accuracy(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        test_x: Tensor,
        test_y: Tensor,
        strengths: list[float],
    ) -> float:
        strength = ...
        W, b = train_with_decay(train_x, train_y, strength)

        return ...


    # TODO: Finish the line in the loop, so counts holds, for every true class
    #   and predicted class, how many examples had that pair. The row is the
    #   true class and the column the predicted one.
    def confusion_matrix(
        predicted: Tensor,
        labels: Tensor,
        classes: int,
    ) -> Tensor:
        counts = torch.zeros(classes, classes, dtype=torch.long)

        for true_class, guess in zip(labels.tolist(), predicted.tolist()):
            ...

        return counts


    # TODO: Finish the last line, so this returns the true class and the
    #   predicted class of the largest count off the diagonal of matrix, the
    #   mistake made most.
    def most_confused(matrix: Tensor) -> tuple[int, int]:
        mistakes = matrix.clone()
        mistakes.fill_diagonal_(0)

        return ...
    ''',
)

MNIST3_SOLUTION = source_file(
    '''
    # mnist3.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    def train_with_decay(
        x: Tensor,
        y: Tensor,
        strength: float,
    ) -> tuple[Tensor, Tensor]:
        generator = torch.Generator().manual_seed(1)
        W = torch.zeros(x.shape[1], 10, requires_grad=True)
        b = torch.zeros(10, requires_grad=True)

        for _ in range(100):
            order = torch.randperm(len(x), generator=generator)

            for start in range(0, len(x), 100):
                positions = order[start:start + 100]
                loss = F.cross_entropy(x[positions] @ W + b, y[positions]) + strength * (W ** 2).sum()
                loss.backward()

                with torch.no_grad():
                    W -= 0.5 * W.grad
                    b -= 0.5 * b.grad

                W.grad = None
                b.grad = None

        return W.detach(), b.detach()


    def accuracy(
        x: Tensor,
        y: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> float:
        return ((x @ W + b).argmax(dim=1) == y).float().mean().item()


    def pick_strength(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        strengths: list[float],
    ) -> float:
        scores = []

        for strength in strengths:
            W, b = train_with_decay(train_x, train_y, strength)
            scores.append(accuracy(validation_x, validation_y, W, b))

        return strengths[scores.index(max(scores))]


    def test_accuracy(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        test_x: Tensor,
        test_y: Tensor,
        strengths: list[float],
    ) -> float:
        strength = pick_strength(train_x, train_y, validation_x, validation_y, strengths)
        W, b = train_with_decay(train_x, train_y, strength)

        return accuracy(test_x, test_y, W, b)


    def confusion_matrix(
        predicted: Tensor,
        labels: Tensor,
        classes: int,
    ) -> Tensor:
        counts = torch.zeros(classes, classes, dtype=torch.long)

        for true_class, guess in zip(labels.tolist(), predicted.tolist()):
            counts[true_class, guess] += 1

        return counts


    # The flattened position of the largest count, split back into its row and column.
    def most_confused(matrix: Tensor) -> tuple[int, int]:
        mistakes = matrix.clone()
        mistakes.fill_diagonal_(0)

        return divmod(mistakes.argmax().item(), matrix.shape[1])
    ''',
)


def train_small(
    x: Tensor,
    y: Tensor,
    strength: float,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(1)
    W = torch.zeros(x.shape[1], 10, requires_grad=True)
    b = torch.zeros(10, requires_grad=True)

    for _ in range(100):
        order = torch.randperm(len(x), generator=generator)

        for start in range(0, len(x), 100):
            positions = order[start:start + 100]
            (F.cross_entropy(x[positions] @ W + b, y[positions]) + strength * (W ** 2).sum()).backward()

            with torch.no_grad():
                W -= 0.5 * W.grad
                b -= 0.5 * b.grad

            W.grad = None
            b.grad = None

    return W.detach(), b.detach()


STRENGTHS = [0.0, 0.001, 0.1]


@functools.cache
def best_small_strength() -> float:
    scores = [
        accuracy_of(mnist_set("validation_x"), mnist_set("validation_y"), *train_small(mnist_set("small_x"), mnist_set("small_y"), strength))
        for strength in STRENGTHS
    ]

    return STRENGTHS[scores.index(max(scores))]


def check_pick_strength(found: object) -> None:
    call = "pick_strength on 1,000 training images and the validation set, with strengths 0, 0.001 and 0.1"
    picked = found(mnist_set("small_x"), mnist_set("small_y"), mnist_set("validation_x"), mnist_set("validation_y"), STRENGTHS)
    expect_close(
        call=call,
        actual=picked,
        expected=best_small_strength(),
    )


def check_test_accuracy(found: object) -> None:
    call = "test_accuracy on 1,000 training images, picking among strengths 0, 0.001 and 0.1"
    measured = found(
        mnist_set("small_x"),
        mnist_set("small_y"),
        mnist_set("validation_x"),
        mnist_set("validation_y"),
        mnist_set("test_x"),
        mnist_set("test_y"),
        STRENGTHS,
    )
    W, b = train_small(mnist_set("small_x"), mnist_set("small_y"), best_small_strength())
    expect_close(
        call=call,
        actual=measured,
        expected=accuracy_of(mnist_set("test_x"), mnist_set("test_y"), W, b),
    )


def reference_confusion(
    predicted: Tensor,
    labels: Tensor,
    classes: int,
) -> Tensor:
    return torch.bincount(labels * classes + predicted, minlength=classes * classes).reshape(classes, classes)


def reference_most_confused(matrix: Tensor) -> tuple[float, float]:
    mistakes = matrix.clone()
    mistakes.fill_diagonal_(0)
    row, column = divmod(mistakes.argmax().item(), matrix.shape[1])

    return float(row), float(column)


def check_mnist3(checker: Checker) -> None:
    checker.test(
        function="pick_strength",
        description="picks the strength with the best validation accuracy",
        run=check_pick_strength,
    )
    checker.test(
        function="test_accuracy",
        description="measures the picked model once on the test set",
        run=check_test_accuracy,
    )
    check_on_inputs(
        checker=checker,
        function="confusion_matrix",
        description="counts every pair of true and predicted class",
        inputs=(
            (torch.tensor([0, 1, 1, 2, 2, 0]), torch.tensor([0, 1, 2, 2, 1, 1]), 3),
            (torch.tensor([3, 3, 0, 1]), torch.tensor([3, 2, 0, 1]), 4),
        ),
        reference=reference_confusion,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="most_confused",
        description="gives the true and predicted class of the largest mistake",
        inputs=(
            (torch.tensor([[50, 2, 1], [9, 40, 3], [0, 4, 30]]),),
            (torch.tensor([[10, 0, 0, 1], [0, 10, 0, 0], [0, 0, 10, 7], [0, 2, 0, 10]]),),
        ),
        reference=reference_most_confused,
        expect=expect_parts("the true class", "the predicted class"),
    )


MNIST3 = Exercise(
    name="mnist3",
    chapter=MNIST,
    title="Project: tuning and the test",
    lesson=MNIST3_LESSON,
    source=MNIST3_SOURCE,
    solution=MNIST3_SOLUTION,
    hints={
        "pick_strength": "Append the accuracy of W and b on the validation set.",
        "test_accuracy": "The strength comes from pick_strength. The last line is accuracy on test_x and test_y.",
        "confusion_matrix": "Add 1 to the entry at row true_class and column guess.",
        "most_confused": "The position of the largest entry of mistakes, from .argmax().item(), splits into a row "
        "and a column with divmod by the number of columns, matrix.shape[1].",
    },
    introduces=(),
    check=check_mnist3,
    time_limit=120,
)
