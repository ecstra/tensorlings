"""14_exoplanets, Section 2's project, softmax regression that vets NASA's Kepler signals, planet candidate or false
positive, tuned on validation and measured once on test."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..data import KEPLER_COLUMNS, load_kepler
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
    reject_missing_return,
    seeded,
    small_tensor,
)
from ..text import prose, source_file

EXOPLANETS = Chapter(
    folder="14_exoplanets",
    title="Project: exoplanet hunter",
    readme=source_file(
        """
        # Project: exoplanet hunter

        Section 2's project. NASA's Kepler telescope watched stars for four
        years, looking for the small dip in light when a planet passes in front
        of its star. Many of the dips it flagged came from something else. You
        train softmax regression to tell a planet candidate from a false
        positive with 18 measurements of each signal, choose its weight decay on
        a validation set, and measure it once on the test set. Download the data
        once from the project folder.

        ```bash
        .venv/bin/python run.py data kepler
        ```

        ## Further reading

        - [NASA Exoplanet Archive, the Kepler tables of objects of interest](https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html)
        - [Thompson and colleagues, the DR25 catalog and its Robovetter](https://arxiv.org/abs/1710.06758)
        - [CS231n, assignment 1](https://cs231n.github.io/assignments2025/assignment1/)
        """,
    ),
)

# The test and validation sets each hold this many signals, and the early survey of planets3 this many.
HELD = 1000
SMALL = 100
STRENGTHS = [0.0, 0.003, 0.03, 0.1, 0.3]


def reference_products(x: Tensor) -> Tensor:
    first, second = torch.triu_indices(x.shape[1], x.shape[1])

    return torch.cat([x, x[:, first] * x[:, second]], dim=1)


@functools.cache
def kepler_sets(seed: int) -> dict[str, Tensor]:
    """Every set the checks and the lessons use, for one shuffle of the complete signals."""
    features, labels = load_kepler()
    complete = ~features.isnan().any(dim=1)
    x, y = features[complete], labels[complete]
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    train, validation, test = order[: -2 * HELD], order[-2 * HELD : -HELD], order[-HELD:]
    mean, spread = x[train].mean(dim=0), x[train].std(dim=0)
    small = train[:SMALL]
    small_products = reference_products(x[small])
    products_mean, products_spread = small_products.mean(dim=0), small_products.std(dim=0)

    def scaled(rows: Tensor) -> Tensor:
        return (x[rows] - mean) / spread

    def scaled_products(rows: Tensor) -> Tensor:
        return (reference_products(x[rows]) - products_mean) / products_spread

    return {
        "raw_x": x[train],
        "train_x": scaled(train),
        "train_y": y[train],
        "validation_x": scaled(validation),
        "validation_y": y[validation],
        "test_x": scaled(test),
        "test_y": y[test],
        "raw_validation_x": x[validation],
        "raw_test_x": x[test],
        "small_x": scaled_products(small),
        "small_y": y[small],
        "small_validation_x": scaled_products(validation),
        "small_test_x": scaled_products(test),
    }


def kepler_set(name: str) -> Tensor:
    return kepler_sets(0)[name].clone()


def accuracy_of(
    x: Tensor,
    y: Tensor,
    W: Tensor,
    b: Tensor,
) -> float:
    return ((x @ W + b).argmax(dim=1) == y).float().mean().item()


def loss_of(
    x: Tensor,
    y: Tensor,
    W: Tensor,
    b: Tensor,
) -> float:
    return F.cross_entropy(x @ W + b, y).item()


# ---------------------------------------------------------------------------
# planets1
# ---------------------------------------------------------------------------

PLANETS1_LESSON = prose(
    """
    This is Section 2's project. NASA's Kepler telescope watched stars for
    four years, looking for the dip in light when a planet crosses its star.
    Its DR25 table holds 8,054 such signals, each vetted by a program, the
    Robovetter. About half came out as planet candidates, and half as false
    positives. Download the table once, about 1.2 MB, from the project folder
    with `.venv/bin/python run.py data kepler`.

    ```python
    from tensorlings.data import KEPLER_COLUMNS, load_kepler

    features, labels = load_kepler()
    features.shape       # torch.Size([8054, 18])
    labels[:5]           # tensor([0, 0, 0, 0, 0]), 0 a planet candidate, 1 a false positive
    KEPLER_COLUMNS[:3]   # ('koi_period', 'koi_duration', 'koi_depth')
    ```

    Each row measures one signal. Some describe the dip, such as its period
    in days and its depth in parts per million. Some describe the star. Two
    are tests a vetter runs, whether odd and even dips differ in depth, and how
    far the dip's light sits from the star's position.

    The table leaves a measurement empty when it has none, and those become
    `nan`, not a number. 225 rows hold one, and the model can't use them.

    There is no test set given, so the split makes all three. The last 1,000
    shuffled rows go to test, the 1,000 before them to validation.

    The measurements sit on wildly different scales. Depth runs past 900,000,
    while nine in ten impact parameters stay under 1.2. So every column gets
    standardized, as descent4 did, with the training set's mean and spread.
    Validation and test use the training numbers too, since a model in use
    never sees its new data ahead of time.
    """,
)

PLANETS1_SOURCE = source_file(
    '''
    # planets1.py
    #
    # Section 2's project starts with the data. Download it once from the project
    # folder with
    #
    #     .venv/bin/python run.py data kepler

    import math

    import torch
    from torch import Tensor

    from tensorlings.data import KEPLER_COLUMNS, load_kepler

    features, labels = load_kepler()                # shapes (8054, 18) and (8054,)

    # .isnan() marks every nan, and .any(dim=1) asks whether each row has one.
    gaps = torch.tensor([[1.0, math.nan], [2.0, 3.0]]).isnan().any(dim=1)    # tensor([ True, False])

    # ~ turns every True into False and every False into True.
    kept = ~gaps                                    # tensor([False,  True])


    # TODO: Return the rows of features and their labels, keeping only the rows
    #   that hold no nan.
    def complete_rows(
        features: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ...


    # A negative slice counts from the end.
    middle = torch.arange(6)[-4:-2]                 # tensor([2, 3])


    # TODO: Finish the middle line, so this shuffles the rows with generator, the
    #   last held shuffled positions go to test, the held positions before them
    #   to validation, and the rest to training.
    def three_way_split(
        x: Tensor,
        y: Tensor,
        held: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        order = torch.randperm(len(x), generator=generator)
        train, validation, test = ...

        return x[train], y[train], x[validation], y[validation], x[test], y[test]


    spread = torch.tensor([[1.0], [3.0]]).std(dim=0)    # tensor([1.4142])


    # TODO: Return x standardized with the mean and spread of each column of
    #   train_x, so every set gets the training set's scale.
    def standardize(
        train_x: Tensor,
        x: Tensor,
    ) -> Tensor:
        ...
    ''',
)

PLANETS1_SOLUTION = source_file(
    '''
    # planets1.py, a solution for comparison

    import math

    import torch
    from torch import Tensor

    from tensorlings.data import KEPLER_COLUMNS, load_kepler

    features, labels = load_kepler()

    gaps = torch.tensor([[1.0, math.nan], [2.0, 3.0]]).isnan().any(dim=1)

    kept = ~gaps


    def complete_rows(
        features: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        complete = ~features.isnan().any(dim=1)

        return features[complete], labels[complete]


    middle = torch.arange(6)[-4:-2]


    def three_way_split(
        x: Tensor,
        y: Tensor,
        held: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        order = torch.randperm(len(x), generator=generator)
        train, validation, test = order[: -2 * held], order[-2 * held : -held], order[-held:]

        return x[train], y[train], x[validation], y[validation], x[test], y[test]


    spread = torch.tensor([[1.0], [3.0]]).std(dim=0)


    def standardize(
        train_x: Tensor,
        x: Tensor,
    ) -> Tensor:
        return (x - train_x.mean(dim=0)) / train_x.std(dim=0)
    ''',
)


def with_gaps(
    seed: int,
    rows: int,
    columns: int,
    gaps: list[tuple[int, int]],
) -> Tensor:
    x = small_tensor(seed, rows, columns)

    for row, column in gaps:
        x[row, column] = float("nan")

    return x


def reference_complete(
    features: Tensor,
    labels: Tensor,
) -> tuple[Tensor, Tensor]:
    complete = ~features.isnan().any(dim=1)

    return features[complete], labels[complete]


def expect_complete(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    expect_parts("the rows", "their labels")(
        call=call,
        actual=actual,
        expected=expected,
    )

    if actual[1].dtype != expected[1].dtype:
        raise Mismatch(f"The labels from {call} are {actual[1].dtype}, but they should stay {expected[1].dtype}, whole numbers.")


def reference_three_way(
    x: Tensor,
    y: Tensor,
    held: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    order = torch.randperm(len(x), generator=generator)
    train, validation, test = order[: -2 * held], order[-2 * held : -held], order[-held:]

    return x[train], y[train], x[validation], y[validation], x[test], y[test]


def check_planets1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="complete_rows",
        description="keeps the rows with no nan, and their labels",
        inputs=(
            (Described(with_gaps(140, 5, 3, [(1, 2)]), "5 rows, row 1 with a nan"), torch.tensor([0, 1, 1, 0, 1])),
            (
                Described(with_gaps(141, 6, 4, [(0, 0), (3, 1), (3, 3)]), "6 rows, rows 0 and 3 with nans"),
                torch.tensor([1, 0, 0, 1, 1, 0]),
            ),
        ),
        reference=reference_complete,
        expect=expect_complete,
    )
    check_on_inputs(
        checker=checker,
        function="three_way_split",
        description="shuffles, then the last positions go to test and the ones before them to validation",
        inputs=(
            (Described(small_tensor(142, 10, 2), "x with 10 rows"), Described(torch.arange(10), "y"), 2, seeded(0)),
            (Described(small_tensor(143, 13, 3), "x with 13 rows"), Described(torch.arange(13) % 2, "y"), 3, seeded(4)),
        ),
        reference=reference_three_way,
        expect=expect_parts(
            "the training rows",
            "the training labels",
            "the validation rows",
            "the validation labels",
            "the test rows",
            "the test labels",
        ),
    )
    check_on_inputs(
        checker=checker,
        function="standardize",
        description="uses the training rows' mean and spread",
        inputs=(
            (small_tensor(144, 6, 3), small_tensor(145, 4, 3)),
            (small_tensor(146, 5, 2) * 100, small_tensor(147, 3, 2)),
        ),
        reference=lambda train_x, x: (x - train_x.mean(dim=0)) / train_x.std(dim=0),
        expect=expect_tensor,
    )


PLANETS1 = Exercise(
    name="planets1",
    chapter=EXOPLANETS,
    title="Project: the data",
    lesson=PLANETS1_LESSON,
    source=PLANETS1_SOURCE,
    solution=PLANETS1_SOLUTION,
    hints={
        "complete_rows": "Mark the rows that hold a nan as the line that sets gaps does, and flip the marks with ~. "
        "Then use them to pick the rows of both features and labels.",
        "three_way_split": "Slice order three times. Training ends 2 * held positions from the end, validation is the "
        "held positions after that, and test is order[-held:].",
        "standardize": "Take the mean and the .std of train_x down its rows, dim=0. Subtract the mean from x and "
        "divide by the spread.",
    },
    introduces=(),
    check=check_planets1,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# planets2
# ---------------------------------------------------------------------------

PLANETS2_LESSON = prose(
    """
    Training is descent2's loop with the pieces of Section 2, softmax
    regression for the model, cross entropy for the loss and minibatches for
    the steps. The weights have shape `(18, 2)`, one column per class.

    The weights start at zero, so both classes get the same score and the
    first loss is `log(2)`, about 0.693. Karpathy's recipe for training lists
    this check among its first.

    With minibatches of 64, ten epochs over the 5,829 training signals take
    about a second, and reach 88.6% on validation. Over five shuffles of the
    table the model lands between 86.8% and 89.0%, where guessing gets 50%.

    Since every column was standardized, the weights compare. The push of a
    feature toward false positive is `W[:, 1] - W[:, 0]`, and the largest
    pushes say what the model looks at.

    | Feature | Push |
    |---|---|
    | `koi_depth`, the dip's depth | 2.62 |
    | `koi_ror`, the planet's radius over the star's | 2.24 |
    | `koi_dikco_msky`, how far the dip's light sits from the star | 2.22 |
    | `koi_max_sngle_ev`, the strongest single dip | 1.36 |
    | `koi_prad`, the planet's radius | 1.02 |

    Each pushes toward false positive as it grows, and a vetter reads them the
    same way. A planet blocks at most a few percent of its star's light. A far
    deeper dip, with a radius ratio near 1, comes from a second star. A dip
    whose light sits away from the star comes from a neighbour.

    Measuring accuracy needs no gradients, so it runs inside
    `with torch.no_grad():`.
    """,
)

PLANETS2_SOURCE = source_file(
    '''
    # planets2.py

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
        W = torch.zeros(x.shape[1], 2, requires_grad=True)
        b = torch.zeros(2, requires_grad=True)

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
    #   label in y, as a plain number.
    def accuracy(
        x: Tensor,
        y: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> float:
        ...


    # .argsort(descending=True) gives the positions from the largest entry down.
    order = torch.tensor([0.5, -2.0, 1.0]).abs().argsort(descending=True)    # tensor([1, 2, 0])


    # TODO: Return the names from columns of the count features with the
    #   largest push, largest first. A feature's push is the size of
    #   W[:, 1] - W[:, 0], its weight toward false positive less its weight
    #   toward planet candidate.
    def strongest(
        W: Tensor,
        columns: tuple[str, ...],
        count: int,
    ) -> list[str]:
        ...
    ''',
)

PLANETS2_SOLUTION = source_file(
    '''
    # planets2.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    # The first loss, with W and b at zero, should be log(2), about 0.693.
    def train(
        x: Tensor,
        y: Tensor,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        W = torch.zeros(x.shape[1], 2, requires_grad=True)
        b = torch.zeros(2, requires_grad=True)

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


    order = torch.tensor([0.5, -2.0, 1.0]).abs().argsort(descending=True)


    def strongest(
        W: Tensor,
        columns: tuple[str, ...],
        count: int,
    ) -> list[str]:
        pushes = (W[:, 1] - W[:, 0]).abs()

        return [columns[position] for position in pushes.argsort(descending=True)[:count].tolist()]
    ''',
)


def reference_train(
    x: Tensor,
    y: Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    W = torch.zeros(x.shape[1], 2, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)

    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)

        for start in range(0, len(x), batch_size):
            positions = order[start : start + batch_size]
            F.cross_entropy(x[positions] @ W + b, y[positions]).backward()

            with torch.no_grad():
                W -= learning_rate * W.grad
                b -= learning_rate * b.grad

            W.grad = None
            b.grad = None

    return W.detach(), b.detach()


def reference_strongest(
    W: Tensor,
    columns: tuple[str, ...],
    count: int,
) -> list[str]:
    pushes = (W[:, 1] - W[:, 0]).abs()

    return [columns[position] for position in pushes.argsort(descending=True)[:count].tolist()]


def check_trained_on_kepler(found: object) -> None:
    call = "train on the 5,829 standardized training signals, for 10 epochs of minibatches of 64 with learning rate 0.5"
    trained = found(kepler_set("train_x"), kepler_set("train_y"), 10, 64, 0.5, torch.Generator().manual_seed(1))
    reject_missing_return(
        call=call,
        actual=trained,
    )
    W, b = trained

    if tuple(W.shape) != (18, 2):
        raise Mismatch(f"After {call}, W has shape {tuple(W.shape)}, but it should have shape (18, 2).")

    accuracy = accuracy_of(kepler_set("validation_x"), kepler_set("validation_y"), W, b)

    if accuracy < 0.85:
        raise Mismatch(f"After {call}, the validation accuracy is {accuracy:.3f}, but it should be at least 0.85.")


def check_planets2(checker: Checker) -> None:
    checker.test(
        function="train",
        description="trains softmax regression past 85 percent on the validation set",
        run=check_trained_on_kepler,
    )
    check_on_inputs(
        checker=checker,
        function="accuracy",
        description="gives the fraction of rows predicted right",
        inputs=(
            (small_tensor(171, 6, 4), torch.tensor([0, 1, 1, 0, 0, 1]), small_tensor(172, 4, 2), small_tensor(173, 2)),
            (small_tensor(174, 7, 3), torch.tensor([1, 1, 0, 0, 1, 0, 1]), small_tensor(175, 3, 2), small_tensor(176, 2)),
        ),
        reference=accuracy_of,
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="strongest",
        description="names the features with the largest push, largest first",
        inputs=(
            (torch.tensor([[0.5, -1.0], [2.0, 2.5], [-1.0, 3.0], [0.0, 0.2]]), ("a", "b", "c", "d"), 2),
            (torch.tensor([[3.0, -1.0], [0.0, 0.5], [1.0, -0.5], [2.0, 0.0], [0.0, -6.0]]), ("p", "q", "r", "s", "t"), 3),
        ),
        reference=reference_strongest,
        expect=expect_value,
    )


PLANETS2 = Exercise(
    name="planets2",
    chapter=EXOPLANETS,
    title="Project: training",
    lesson=PLANETS2_LESSON,
    source=PLANETS2_SOURCE,
    solution=PLANETS2_SOLUTION,
    hints={
        "train": "The scores of the minibatch are x[positions] @ W + b, and its labels are y[positions]. Pass both "
        "to F.cross_entropy.",
        "accuracy": "Predict with argmax of the scores along dim=1, compare with y, then the mean of the matches as "
        "floats, as softmax3 does. .item() makes it a plain number.",
        "strongest": "Take the size of the difference of W's two columns with .abs(), then the first count positions "
        "of its argsort from the largest down. Look each position up in columns.",
    },
    introduces=(),
    check=check_planets2,
    time_limit=120,
)


# ---------------------------------------------------------------------------
# planets3
# ---------------------------------------------------------------------------

PLANETS3_LESSON = prose(
    """
    Early in a survey only a few signals have been vetted, say 100. Give the
    model the product of every pair of columns too, as overfitting1 added
    powers. That makes 189 features, more than the 100 examples, and the
    model overfits.

    Here it is on 100 signals with each strength of weight decay, averaged over
    five shuffles of the table.

    | Strength | Training accuracy | Validation accuracy | Validation loss |
    |---|---|---|---|
    | 0 | 0.986 | 0.838 | 3.50 |
    | 0.003 | 0.984 | 0.846 | 2.12 |
    | 0.03 | 0.944 | 0.862 | 0.52 |
    | 0.1 | 0.938 | 0.860 | 0.46 |
    | 0.3 | 0.912 | 0.853 | 0.44 |

    With no decay the model gets nearly every training signal right. It also
    grows sure of its wrong answers on new ones, so the validation loss blows
    up. Decay keeps the weights small. From 0.03 on, the validation loss
    falls from 3.50 to about 0.5, and accuracy rises from 83.8% to 86.2%.

    The loss shows overfitting best here, so `pick_strength` chooses the
    strength with the lowest validation loss. Then the test set gets used,
    once.

    A **confusion matrix** counts, for every true class and every predicted
    class, how many examples had that pair. The diagonal holds the right
    answers. On the first shuffle the picked strength is 0.3, and the test
    set gives 83.6%. Of 501 false positives, 137 pass as planets, while only
    27 of 499 planet candidates get thrown out.

    ```python
    counts = torch.zeros(2, 2, dtype=torch.long)
    counts[1, 0] += 1    # one false positive taken for a planet
    ```
    """,
)

PLANETS3_SOURCE = source_file(
    '''
    # planets3.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    # Every pair of positions of 3 columns, a column with itself included.
    pairs = torch.triu_indices(3, 3)    # tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]])


    # TODO: Return x with the product of every pair of its columns added after
    #   them, in the order torch.triu_indices gives the pairs.
    def products(x: Tensor) -> Tensor:
        ...


    # Trains softmax regression with weight decay as planets2 does, for 300 epochs
    # of minibatches of 20, with the same seed every time so runs compare fairly.
    def train_with_decay(
        x: Tensor,
        y: Tensor,
        strength: float,
    ) -> tuple[Tensor, Tensor]:
        generator = torch.Generator().manual_seed(1)
        W = torch.zeros(x.shape[1], 2, requires_grad=True)
        b = torch.zeros(2, requires_grad=True)

        for _ in range(300):
            order = torch.randperm(len(x), generator=generator)

            for start in range(0, len(x), 20):
                positions = order[start:start + 20]
                loss = F.cross_entropy(x[positions] @ W + b, y[positions]) + strength * (W ** 2).sum()
                loss.backward()

                with torch.no_grad():
                    W -= 0.1 * W.grad
                    b -= 0.1 * b.grad

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


    # TODO: Return the cross entropy of W and b on the rows of x and their labels
    #   y, as a plain number, without recording for autograd.
    def validation_loss(
        x: Tensor,
        y: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> float:
        ...


    # TODO: Finish the line in the loop, so this returns the strength from
    #   strengths whose model, trained on the training set, has the lowest loss
    #   on the validation set.
    def pick_strength(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        strengths: list[float],
    ) -> float:
        losses = []

        for strength in strengths:
            W, b = train_with_decay(train_x, train_y, strength)
            losses.append(...)

        return strengths[losses.index(min(losses))]


    # TODO: Finish the first and last lines, so this trains with the strength
    #   pick_strength chooses and returns its accuracy and its loss on the test
    #   set, in that order.
    def test_scores(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        test_x: Tensor,
        test_y: Tensor,
        strengths: list[float],
    ) -> tuple[float, float]:
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
    ''',
)

PLANETS3_SOLUTION = source_file(
    '''
    # planets3.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    pairs = torch.triu_indices(3, 3)


    def products(x: Tensor) -> Tensor:
        first, second = torch.triu_indices(x.shape[1], x.shape[1])

        return torch.cat([x, x[:, first] * x[:, second]], dim=1)


    def train_with_decay(
        x: Tensor,
        y: Tensor,
        strength: float,
    ) -> tuple[Tensor, Tensor]:
        generator = torch.Generator().manual_seed(1)
        W = torch.zeros(x.shape[1], 2, requires_grad=True)
        b = torch.zeros(2, requires_grad=True)

        for _ in range(300):
            order = torch.randperm(len(x), generator=generator)

            for start in range(0, len(x), 20):
                positions = order[start:start + 20]
                loss = F.cross_entropy(x[positions] @ W + b, y[positions]) + strength * (W ** 2).sum()
                loss.backward()

                with torch.no_grad():
                    W -= 0.1 * W.grad
                    b -= 0.1 * b.grad

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


    def validation_loss(
        x: Tensor,
        y: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> float:
        with torch.no_grad():
            return F.cross_entropy(x @ W + b, y).item()


    def pick_strength(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        strengths: list[float],
    ) -> float:
        losses = []

        for strength in strengths:
            W, b = train_with_decay(train_x, train_y, strength)
            losses.append(validation_loss(validation_x, validation_y, W, b))

        return strengths[losses.index(min(losses))]


    def test_scores(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        test_x: Tensor,
        test_y: Tensor,
        strengths: list[float],
    ) -> tuple[float, float]:
        strength = pick_strength(train_x, train_y, validation_x, validation_y, strengths)
        W, b = train_with_decay(train_x, train_y, strength)

        return accuracy(test_x, test_y, W, b), validation_loss(test_x, test_y, W, b)


    def confusion_matrix(
        predicted: Tensor,
        labels: Tensor,
        classes: int,
    ) -> Tensor:
        counts = torch.zeros(classes, classes, dtype=torch.long)

        for true_class, guess in zip(labels.tolist(), predicted.tolist()):
            counts[true_class, guess] += 1

        return counts
    ''',
)


def train_small(
    x: Tensor,
    y: Tensor,
    strength: float,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(1)
    W = torch.zeros(x.shape[1], 2, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)

    for _ in range(300):
        order = torch.randperm(len(x), generator=generator)

        for start in range(0, len(x), 20):
            positions = order[start : start + 20]
            (F.cross_entropy(x[positions] @ W + b, y[positions]) + strength * (W**2).sum()).backward()

            with torch.no_grad():
                W -= 0.1 * W.grad
                b -= 0.1 * b.grad

            W.grad = None
            b.grad = None

    return W.detach(), b.detach()


@functools.cache
def small_models(seed: int) -> dict[float, tuple[Tensor, Tensor]]:
    sets = kepler_sets(seed)

    return {strength: train_small(sets["small_x"], sets["small_y"], strength) for strength in STRENGTHS}


def best_small_strength(seed: int) -> float:
    sets = kepler_sets(seed)
    losses = [loss_of(sets["small_validation_x"], sets["validation_y"], *small_models(seed)[strength]) for strength in STRENGTHS]

    return STRENGTHS[losses.index(min(losses))]


class RecordingFunctional:
    """Stands in for the file's F during a call, noting whether autograd was recording at each cross entropy."""

    def __init__(self) -> None:
        self.recording: list[bool] = []

    def cross_entropy(
        self,
        *arguments: object,
        **keywords: object,
    ) -> Tensor:
        self.recording.append(torch.is_grad_enabled())

        return F.cross_entropy(*arguments, **keywords)

    def __getattr__(
        self,
        name: str,
    ) -> object:
        return getattr(F, name)


def check_validation_loss_recording(found: object) -> None:
    call = "validation_loss of W and b that need gradients"
    W = torch.tensor([[1.0, -1.0], [0.5, 2.0]], requires_grad=True)
    b = torch.tensor([0.0, 1.0], requires_grad=True)
    stand_in = RecordingFunctional()
    original = found.__globals__.get("F")
    found.__globals__["F"] = stand_in

    try:
        loss = found(torch.tensor([[1.0, 2.0], [-1.0, 0.0]]), torch.tensor([0, 1]), W, b)
    finally:
        found.__globals__["F"] = original

    reject_missing_return(
        call=call,
        actual=loss,
    )

    if any(stand_in.recording) or (isinstance(loss, torch.Tensor) and loss.requires_grad):
        raise Mismatch(f"{call} recorded for autograd. Take the loss inside torch.no_grad().")


def check_pick_strength(found: object) -> None:
    call = "pick_strength on 100 training signals and the validation set, with strengths 0, 0.003, 0.03, 0.1 and 0.3"
    picked = found(
        kepler_set("small_x"),
        kepler_set("small_y"),
        kepler_set("small_validation_x"),
        kepler_set("validation_y"),
        STRENGTHS,
    )
    expect_close(
        call=call,
        actual=picked,
        expected=best_small_strength(0),
    )


def check_test_scores(found: object) -> None:
    call = "test_scores on 100 training signals, picking among strengths 0, 0.003, 0.03, 0.1 and 0.3"
    measured = found(
        kepler_set("small_x"),
        kepler_set("small_y"),
        kepler_set("small_validation_x"),
        kepler_set("validation_y"),
        kepler_set("small_test_x"),
        kepler_set("test_y"),
        STRENGTHS,
    )
    W, b = small_models(0)[best_small_strength(0)]
    expect_parts("the test accuracy", "the test loss")(
        call=call,
        actual=measured,
        expected=(
            accuracy_of(kepler_set("small_test_x"), kepler_set("test_y"), W, b),
            loss_of(kepler_set("small_test_x"), kepler_set("test_y"), W, b),
        ),
    )


def reference_confusion(
    predicted: Tensor,
    labels: Tensor,
    classes: int,
) -> Tensor:
    return torch.bincount(labels * classes + predicted, minlength=classes * classes).reshape(classes, classes)


def check_planets3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="products",
        description="adds the product of every pair of columns after them",
        inputs=(
            (small_tensor(181, 4, 2),),
            (small_tensor(182, 3, 4),),
        ),
        reference=reference_products,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="validation_loss",
        description="gives the cross entropy as a plain number",
        inputs=(
            (small_tensor(183, 5, 3), torch.tensor([0, 1, 1, 0, 1]), small_tensor(184, 3, 2), small_tensor(185, 2)),
            (small_tensor(186, 4, 2), torch.tensor([1, 0, 0, 1]), small_tensor(187, 2, 2), small_tensor(188, 2)),
        ),
        reference=loss_of,
        expect=expect_close,
    )
    checker.test(
        function="validation_loss",
        description="takes the loss without recording for autograd",
        run=check_validation_loss_recording,
    )
    checker.test(
        function="pick_strength",
        description="picks the strength with the lowest validation loss",
        run=check_pick_strength,
    )
    checker.test(
        function="test_scores",
        description="measures the picked model once on the test set",
        run=check_test_scores,
    )
    check_on_inputs(
        checker=checker,
        function="confusion_matrix",
        description="counts every pair of true and predicted class",
        inputs=(
            (torch.tensor([0, 1, 1, 0, 0, 1]), torch.tensor([0, 1, 0, 0, 1, 1]), 2),
            (torch.tensor([2, 2, 0, 1]), torch.tensor([2, 1, 0, 1]), 3),
        ),
        reference=reference_confusion,
        expect=expect_tensor,
    )


PLANETS3 = Exercise(
    name="planets3",
    chapter=EXOPLANETS,
    title="Project: tuning and the test",
    lesson=PLANETS3_LESSON,
    source=PLANETS3_SOURCE,
    solution=PLANETS3_SOLUTION,
    hints={
        "products": "torch.triu_indices of the number of columns twice gives two rows of positions. Pick the "
        "columns at each row with x[:, ...], multiply them, and join them after x with torch.cat along dim=1.",
        "validation_loss": "Inside torch.no_grad(), pass the scores x @ W + b and y to F.cross_entropy, then .item().",
        "pick_strength": "Append the validation loss of W and b on the validation set.",
        "test_scores": "The strength comes from pick_strength. The last line gives accuracy and validation_loss, "
        "both on test_x and test_y.",
        "confusion_matrix": "Add 1 to the entry at row true_class and column guess.",
    },
    introduces=(),
    check=check_planets3,
    time_limit=120,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_exoplanets(seeds: range = range(5)) -> dict[str, object]:
    """Every number in the three lessons. Threads are pinned, so the runs come out the same each time."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    results: dict[str, object] = {}

    try:
        features, labels = load_kepler()
        results["rows with a nan"] = int(features.isnan().any(dim=1).sum())
        results["training signals"] = len(kepler_sets(0)["train_x"])

        accuracies = []

        for seed in seeds:
            sets = kepler_sets(seed)
            W, b = reference_train(sets["train_x"], sets["train_y"], 10, 64, 0.5, torch.Generator().manual_seed(1))
            accuracies.append(accuracy_of(sets["validation_x"], sets["validation_y"], W, b))

            if seed == 0:
                pushes = W[:, 1] - W[:, 0]
                order = pushes.abs().argsort(descending=True)[:5].tolist()
                results["strongest pushes"] = [(KEPLER_COLUMNS[position], pushes[position].item()) for position in order]
                predicted = (sets["test_x"] @ W + b).argmax(dim=1)
                results["planets2 test confusion"] = reference_confusion(predicted, sets["test_y"], 2).tolist()

        results["planets2 validation accuracies"] = accuracies

        table = {strength: ([], [], []) for strength in STRENGTHS}

        for seed in seeds:
            sets = kepler_sets(seed)

            for strength, (W, b) in small_models(seed).items():
                table[strength][0].append(accuracy_of(sets["small_x"], sets["small_y"], W, b))
                table[strength][1].append(accuracy_of(sets["small_validation_x"], sets["validation_y"], W, b))
                table[strength][2].append(loss_of(sets["small_validation_x"], sets["validation_y"], W, b))

        results["planets3 table"] = {
            strength: tuple(sum(values) / len(values) for values in columns) for strength, columns in table.items()
        }
        results["planets3 validation losses by seed"] = {strength: columns[2] for strength, columns in table.items()}

        sets = kepler_sets(0)
        strength = best_small_strength(0)
        W, b = small_models(0)[strength]
        predicted = (sets["small_test_x"] @ W + b).argmax(dim=1)
        results["planets3 picked strength"] = strength
        results["planets3 test accuracy"] = accuracy_of(sets["small_test_x"], sets["test_y"], W, b)
        results["planets3 test loss"] = loss_of(sets["small_test_x"], sets["test_y"], W, b)
        results["planets3 test confusion"] = reference_confusion(predicted, sets["test_y"], 2).tolist()
    finally:
        torch.set_num_threads(threads)

    return results
