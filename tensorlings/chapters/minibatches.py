"""10_minibatches, training on small random groups of examples, epochs, and PyTorch's DataLoader."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_parts,
    expect_tensor,
    expect_value,
    reject_missing_return,
    seeded,
    shape_text,
    small_tensor,
)
from ..text import prose, source_file
from .gradient_descent import linear_data, named
from .tensors import pinned_threads

MINIBATCHES = Chapter(
    folder="10_minibatches",
    title="Minibatches",
    readme=source_file(
        """
        # Minibatches

        Real datasets are too big to read in full for every step. Each step uses
        a small random group of examples instead, a minibatch. This chapter
        picks minibatches, shows why their noisy gradients still work, runs
        training in epochs, and ends with PyTorch's DataLoader, which does the
        bookkeeping.

        ## Further reading

        - [Dive into Deep Learning, minibatch stochastic gradient descent](https://d2l.ai/chapter_optimization/minibatch-sgd.html)
        - [PyTorch, datasets and dataloaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
        """,
    ),
)


def mse_gradients_on(
    x: Tensor,
    y: Tensor,
    w: Tensor,
    b: Tensor,
    positions: Tensor,
) -> tuple[Tensor, Tensor]:
    w = w.clone().requires_grad_()
    b = b.clone().requires_grad_()
    ((x[positions] @ w + b - y[positions]) ** 2).mean().backward()

    return w.grad, b.grad


def expect_tensor_list(
    call: str,
    actual: object,
    expected: list[Tensor],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, list) or len(actual) != len(expected):
        sizes = ", ".join(str(len(piece)) for piece in expected)
        raise Mismatch(f"{call} should be a list of {len(expected)} tensors of positions, of sizes {sizes}.")

    for position, (got, wanted) in enumerate(zip(actual, expected)):
        expect_tensor(
            call=f"entry {position} of {call}",
            actual=got,
            expected=wanted,
        )


# ---------------------------------------------------------------------------
# minibatch1
# ---------------------------------------------------------------------------

MINIBATCH1_LESSON = prose(
    """
    So far every step used every example. With a million examples, each step
    would read all of them to move the parameters a little. So each step uses
    a small random group of examples instead, often 32 to 512 of them. A group
    of examples the model takes in together is a **batch**, and a batch drawn
    at random from the data is a **minibatch**.

    A minibatch's gradient is a noisy estimate of the gradient over every
    example, and on average it points the same way. Here is the gradient of
    `w` for a linear model on 10 examples, with minibatches of 3.

    ```text
    every example              [48.00, -7.15, 49.40]
    one minibatch of 3         [36.50, 11.33, 43.00]
    mean of 200 minibatches    [47.70, -6.28, 47.55]
    ```

    One minibatch is far off, but the mean of many comes back to the full
    gradient. Over 10 runs, one minibatch missed some entry by 10 to 47, and
    the mean of 200 by 0.2 to 3.6.

    So each step is noisy, but it reads far fewer examples. For the same
    number of examples read, many cheap noisy steps beat a few exact ones.
    Here is a linear model with 10 features fit to 10,000 examples. Full batch
    means every step uses every example. Each way ran at its best learning
    rate. Each entry is how far the loss ends above the least squares best,
    the median of 10 runs.

    | Examples read | Full batch | Minibatches of 32 |
    |---|---|---|
    | 10,000 | 2.23 after 1 step | 0.043 after 313 steps |
    | 50,000 | 0.23 after 5 steps | 0.0066 after 1,565 steps |
    | 200,000 | 0.022 after 20 steps | 0.0011 after 6,260 steps |

    The minibatches ended lower in all 10 runs.

    A tensor of positions picks those rows, in that order.

    ```python
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    x[torch.tensor([3, 0])]    # tensor([[4.], [1.]])
    ```

    `torch.randperm(n)` gives the numbers 0 to `n - 1` in a random order, so
    its first `k` entries are `k` different random positions. Pass it a
    generator, as in `torch.randperm(5, generator=g)`, and the order comes out
    the same every run. A generator made with
    `torch.Generator().manual_seed(0)` always draws the same random numbers,
    which is how the checks repeat your draws.

    `.clone()` makes a copy of a tensor. A function can change the copy
    without touching the tensor it was given.
    """,
)

MINIBATCH1_SOURCE = source_file(
    '''
    # minibatch1.py

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

    # A tensor of positions picks those rows, in that order.
    picked = x[torch.tensor([3, 0])]                       # tensor([[4.], [1.]])

    # torch.randperm(n) gives 0 to n - 1 in a random order, the same order every
    # run for a generator with the same seed.
    order = torch.randperm(5, generator=torch.Generator().manual_seed(0))


    # TODO: Return the rows of x at the first size positions of a randperm made
    #   with generator, and the entries of y at the same positions.
    def random_minibatch(
        x: Tensor,
        y: Tensor,
        size: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Finish the line that sets loss, the mean squared error of x @ w + b
    #   against y on the rows at positions only. The lines after it give its
    #   gradients by autograd. The first two lines work on copies, so the w and
    #   b passed in keep no gradient from one call to the next.
    def minibatch_gradients(
        x: Tensor,
        y: Tensor,
        w: Tensor,
        b: Tensor,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        w = w.clone().requires_grad_()
        b = b.clone().requires_grad_()
        loss = ...
        loss.backward()

        return w.grad, b.grad


    # TODO: Finish the line in the loop, so this returns the mean of grad_w over
    #   the random minibatches of the given size, one per draw. It comes out
    #   close to the gradient over every example.
    def mean_minibatch_gradient(
        x: Tensor,
        y: Tensor,
        w: Tensor,
        b: Tensor,
        size: int,
        draws: int,
        generator: torch.Generator,
    ) -> Tensor:
        total = torch.zeros_like(w)

        for _ in range(draws):
            positions = torch.randperm(len(x), generator=generator)[:size]
            grad_w, grad_b = minibatch_gradients(x, y, w, b, positions)
            total = ...

        return total / draws
    ''',
)

MINIBATCH1_SOLUTION = source_file(
    '''
    # minibatch1.py, a solution for comparison

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

    picked = x[torch.tensor([3, 0])]

    order = torch.randperm(5, generator=torch.Generator().manual_seed(0))


    def random_minibatch(
        x: Tensor,
        y: Tensor,
        size: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        positions = torch.randperm(len(x), generator=generator)[:size]

        return x[positions], y[positions]


    def minibatch_gradients(
        x: Tensor,
        y: Tensor,
        w: Tensor,
        b: Tensor,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        w = w.clone().requires_grad_()
        b = b.clone().requires_grad_()
        loss = ((x[positions] @ w + b - y[positions]) ** 2).mean()
        loss.backward()

        return w.grad, b.grad


    def mean_minibatch_gradient(
        x: Tensor,
        y: Tensor,
        w: Tensor,
        b: Tensor,
        size: int,
        draws: int,
        generator: torch.Generator,
    ) -> Tensor:
        total = torch.zeros_like(w)

        for _ in range(draws):
            positions = torch.randperm(len(x), generator=generator)[:size]
            grad_w, grad_b = minibatch_gradients(x, y, w, b, positions)
            total = total + grad_w

        return total / draws
    ''',
)


def reference_minibatch(
    x: Tensor,
    y: Tensor,
    size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    positions = torch.randperm(len(x), generator=generator)[:size]

    return x[positions], y[positions]


def reference_mean_gradient(
    x: Tensor,
    y: Tensor,
    w: Tensor,
    b: Tensor,
    size: int,
    draws: int,
    generator: torch.Generator,
) -> Tensor:
    total = torch.zeros_like(w)

    for _ in range(draws):
        positions = torch.randperm(len(x), generator=generator)[:size]
        total = total + mse_gradients_on(x, y, w, b, positions)[0]

    return total / draws


def check_minibatch1(checker: Checker) -> None:
    x_six, y_six = linear_data(82, 6, 2)
    x_ten, y_ten = linear_data(83, 10, 3)

    check_on_inputs(
        checker=checker,
        function="random_minibatch",
        description="picks size random rows of x and the matching entries of y",
        inputs=(
            (*named(x_six, y_six), 3, seeded(0)),
            (*named(x_ten, y_ten), 4, seeded(7)),
        ),
        reference=reference_minibatch,
        expect=expect_parts("the rows of x", "the entries of y"),
    )
    check_on_inputs(
        checker=checker,
        function="minibatch_gradients",
        description="gives the gradients of the loss on the chosen rows",
        inputs=(
            (*named(x_six, y_six), small_tensor(84, 2), torch.tensor(0.5), torch.tensor([4, 1, 2])),
            (*named(x_ten, y_ten), small_tensor(85, 3), torch.tensor(-1.0), torch.tensor([0, 9, 3, 5])),
        ),
        reference=mse_gradients_on,
        expect=expect_parts("grad_w", "grad_b"),
    )
    check_on_inputs(
        checker=checker,
        function="mean_minibatch_gradient",
        description="gives the mean of grad_w over many random minibatches",
        inputs=(
            (*named(x_ten, y_ten), small_tensor(86, 3), torch.tensor(0.0), 3, 20, seeded(1)),
        ),
        reference=reference_mean_gradient,
        expect=expect_tensor,
    )


MINIBATCH1 = Exercise(
    name="minibatch1",
    chapter=MINIBATCHES,
    title="A minibatch's gradient",
    lesson=MINIBATCH1_LESSON,
    source=MINIBATCH1_SOURCE,
    solution=MINIBATCH1_SOLUTION,
    hints={
        "random_minibatch": "Take torch.randperm(len(x), generator=generator) and keep its first size entries with "
        "a slice. Those positions pick the rows of x and the entries of y.",
        "minibatch_gradients": "The mean squared error as in descent2, with x[positions] in place of x and "
        "y[positions] in place of y.",
        "mean_minibatch_gradient": "Add this draw's grad_w to total.",
    },
    introduces=("batch", "minibatch"),
    check=check_minibatch1,
)


# ---------------------------------------------------------------------------
# minibatch2
# ---------------------------------------------------------------------------

MINIBATCH2_LESSON = prose(
    """
    Drawing every minibatch at random can pick one example many times and
    another never. The usual way is to shuffle the examples once, then cut the
    shuffled order into pieces, so every example is used exactly once. One full
    pass over the data like this is an **epoch**, and training runs for many
    epochs, with a new shuffle each time.

    ```python
    order = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    pieces = [order[start:start + 4] for start in range(0, 10, 4)]
    # three pieces of positions, of sizes 4, 4 and 2
    ```

    `range(0, 10, 4)` counts 0, 4 and 8, the start of each piece. The last
    piece is smaller when the count of examples doesn't divide evenly. Some
    training leaves that piece out, so every step sees the same number of
    examples.

    The shuffle matters. Data often arrives sorted, by date or by kind. Cut
    without a shuffle, each minibatch would hold one kind of example, and
    every step would pull the parameters toward that kind alone.

    Here is a linear model trained for 5 epochs in minibatches of 32, on 2,000
    examples sorted by their target. Each entry is how far the loss ends above
    the least squares best, the median of 10 runs.

    | Learning rate | Shuffled every epoch | Kept sorted |
    |---|---|---|
    | 0.01 | 0.0005 | 0.022 |
    | 0.03 | 0.0005 | 0.12 |
    | 0.1 | 0.0023 | 0.80 |

    Shuffling ended lower in all 30 pairs of runs.

    Training on minibatches is the loop from descent2 inside a loop over the
    pieces of each epoch. Each step uses the rows of its piece only.
    """,
)

MINIBATCH2_SOURCE = source_file(
    '''
    # minibatch2.py

    import torch
    from torch import Tensor

    # One epoch over 10 examples in minibatches of 4. The order is shuffled once,
    # then cut into pieces, so every example is used once.
    order = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    pieces = [order[start:start + 4] for start in range(0, 10, 4)]    # sizes 4, 4 and 2


    # TODO: Finish the last line, so this returns the positions of every
    #   minibatch of one epoch over n examples, the shuffled order cut into
    #   pieces of size. The last piece may be smaller.
    def epoch_batches(
        n: int,
        size: int,
        generator: torch.Generator,
    ) -> list[Tensor]:
        order = torch.randperm(n, generator=generator)

        return ...


    # TODO: Finish the last line the same way, but leave out the last piece when
    #   it is smaller than size, so every minibatch has size positions.
    def full_batches(
        n: int,
        size: int,
        generator: torch.Generator,
    ) -> list[Tensor]:
        order = torch.randperm(n, generator=generator)

        return ...


    # TODO: Finish the two lines that set xb and yb, the rows of the minibatch, so
    #   this trains for the given epochs, one epoch_batches pass each, and
    #   returns the final w and b.
    def train_epochs(
        x: Tensor,
        y: Tensor,
        epochs: int,
        size: int,
        learning_rate: float,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        for _ in range(epochs):
            for positions in epoch_batches(len(x), size, generator):
                xb = ...
                yb = ...
                loss = ((xb @ w + b - yb) ** 2).mean()
                loss.backward()

                with torch.no_grad():
                    w -= learning_rate * w.grad
                    b -= learning_rate * b.grad

                w.grad = None
                b.grad = None

        return w.detach(), b.detach()
    ''',
)

MINIBATCH2_SOLUTION = source_file(
    '''
    # minibatch2.py, a solution for comparison

    import torch
    from torch import Tensor

    order = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    pieces = [order[start:start + 4] for start in range(0, 10, 4)]


    def epoch_batches(
        n: int,
        size: int,
        generator: torch.Generator,
    ) -> list[Tensor]:
        order = torch.randperm(n, generator=generator)

        return [order[start:start + size] for start in range(0, n, size)]


    # Starts up to n - size leave room for a whole piece.
    def full_batches(
        n: int,
        size: int,
        generator: torch.Generator,
    ) -> list[Tensor]:
        order = torch.randperm(n, generator=generator)

        return [order[start:start + size] for start in range(0, n - size + 1, size)]


    def train_epochs(
        x: Tensor,
        y: Tensor,
        epochs: int,
        size: int,
        learning_rate: float,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        for _ in range(epochs):
            for positions in epoch_batches(len(x), size, generator):
                xb = x[positions]
                yb = y[positions]
                loss = ((xb @ w + b - yb) ** 2).mean()
                loss.backward()

                with torch.no_grad():
                    w -= learning_rate * w.grad
                    b -= learning_rate * b.grad

                w.grad = None
                b.grad = None

        return w.detach(), b.detach()
    ''',
)


def reference_epoch_batches(
    n: int,
    size: int,
    generator: torch.Generator,
    drop_last: bool = False,
) -> list[Tensor]:
    order = torch.randperm(n, generator=generator)
    pieces = [order[start:start + size] for start in range(0, n, size)]

    return [piece for piece in pieces if len(piece) == size] if drop_last else pieces


def reference_train_epochs(
    x: Tensor,
    y: Tensor,
    epochs: int,
    size: int,
    learning_rate: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    for _ in range(epochs):
        for positions in reference_epoch_batches(len(x), size, generator):
            loss = ((x[positions] @ w + b - y[positions]) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

    return w.detach(), b.detach()


def check_minibatch2(checker: Checker) -> None:
    sizes = (
        (10, 4, seeded(0)),
        (9, 3, seeded(5)),
        (7, 5, seeded(2)),
    )
    check_on_inputs(
        checker=checker,
        function="epoch_batches",
        description="shuffles once and cuts the order into pieces",
        inputs=sizes,
        reference=reference_epoch_batches,
        expect=expect_tensor_list,
    )
    check_on_inputs(
        checker=checker,
        function="full_batches",
        description="leaves out a last piece that is too small",
        inputs=sizes,
        reference=lambda n, size, generator: reference_epoch_batches(n, size, generator, drop_last=True),
        expect=expect_tensor_list,
    )
    check_on_inputs(
        checker=checker,
        function="train_epochs",
        description="trains on the rows of each minibatch, epoch after epoch",
        inputs=(
            (*named(*linear_data(87, 10, 2)), 5, 4, 0.05, seeded(3)),
            (*named(*linear_data(88, 12, 3)), 4, 5, 0.02, seeded(4)),
        ),
        reference=reference_train_epochs,
        expect=expect_parts("w", "b"),
    )


MINIBATCH2 = Exercise(
    name="minibatch2",
    chapter=MINIBATCHES,
    title="Epochs",
    lesson=MINIBATCH2_LESSON,
    source=MINIBATCH2_SOURCE,
    solution=MINIBATCH2_SOLUTION,
    hints={
        "epoch_batches": "Cut order the way the line that sets pieces does, with n and size in place of 10 and 4.",
        "full_batches": "Cut the same way, but stop the starts early enough that every piece is whole. The last "
        "start that leaves room for a whole piece is n - size.",
        "train_epochs": "xb is the rows of x at positions, and yb the entries of y at positions, picked as in "
        "minibatch1.",
    },
    introduces=("epoch",),
    check=check_minibatch2,
)


# ---------------------------------------------------------------------------
# minibatch3
# ---------------------------------------------------------------------------

MINIBATCH3_LESSON = prose(
    """
    PyTorch does the shuffling and cutting for you. A `TensorDataset` pairs
    row `i` of `x` with entry `i` of `y`. A `DataLoader` shuffles the pairs and
    hands them out in minibatches, a fresh shuffle every time you loop over it.

    ```python
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )

    for xb, yb in loader:    # one epoch
        ...
    ```

    Each loop over the loader is one epoch, and `xb` and `yb` are the rows of
    one minibatch, already picked. `drop_last=True` leaves out a last
    minibatch that is smaller than `batch_size`. `len(loader)` gives the number
    of minibatches in an epoch.

    The size of a minibatch trades noise for speed. Bigger minibatches give
    steadier gradients, but each step costs more. Smaller ones give more steps
    per epoch with noisier gradients. Sizes from 32 to 512 are common, and the
    learning rate usually has to change when the size does.

    The number of steps a run takes, epochs times minibatches per epoch, is
    worth knowing. Later chapters change the learning rate over a run by that
    count.
    """,
)

MINIBATCH3_SOURCE = source_file(
    '''
    # minibatch3.py

    import math

    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.arange(10.0).unsqueeze(1)
    y = torch.arange(10.0)

    # A TensorDataset pairs row i of x with entry i of y. A DataLoader shuffles the
    # pairs and hands them out in minibatches, one epoch per loop over it.
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    sizes = [len(yb) for xb, yb in loader]                 # [4, 4, 2]


    # TODO: Return a DataLoader over x and y with minibatches of size, shuffled
    #   with generator, that leaves out a last minibatch smaller than size.
    def make_loader(
        x: Tensor,
        y: Tensor,
        size: int,
        generator: torch.Generator,
    ) -> DataLoader:
        ...


    # TODO: Finish the line that starts the inner loop, so this trains like
    #   train_epochs in minibatch2, with each epoch's minibatches from loader.
    def train_with_loader(
        x: Tensor,
        y: Tensor,
        epochs: int,
        size: int,
        learning_rate: float,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        loader = DataLoader(TensorDataset(x, y), batch_size=size, shuffle=True, generator=generator)
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        for _ in range(epochs):
            for xb, yb in ...:
                loss = ((xb @ w + b - yb) ** 2).mean()
                loss.backward()

                with torch.no_grad():
                    w -= learning_rate * w.grad
                    b -= learning_rate * b.grad

                w.grad = None
                b.grad = None

        return w.detach(), b.detach()


    # TODO: Return how many steps a run takes over n examples, for the given
    #   epochs and minibatch size. drop_last says whether a last, smaller
    #   minibatch is left out. math.ceil rounds up and // rounds down.
    def count_steps(
        n: int,
        size: int,
        epochs: int,
        drop_last: bool,
    ) -> int:
        ...
    ''',
)

MINIBATCH3_SOLUTION = source_file(
    '''
    # minibatch3.py, a solution for comparison

    import math

    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.arange(10.0).unsqueeze(1)
    y = torch.arange(10.0)

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    sizes = [len(yb) for xb, yb in loader]


    def make_loader(
        x: Tensor,
        y: Tensor,
        size: int,
        generator: torch.Generator,
    ) -> DataLoader:
        return DataLoader(
            TensorDataset(x, y),
            batch_size=size,
            shuffle=True,
            drop_last=True,
            generator=generator,
        )


    def train_with_loader(
        x: Tensor,
        y: Tensor,
        epochs: int,
        size: int,
        learning_rate: float,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        loader = DataLoader(TensorDataset(x, y), batch_size=size, shuffle=True, generator=generator)
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        for _ in range(epochs):
            for xb, yb in loader:
                loss = ((xb @ w + b - yb) ** 2).mean()
                loss.backward()

                with torch.no_grad():
                    w -= learning_rate * w.grad
                    b -= learning_rate * b.grad

                w.grad = None
                b.grad = None

        return w.detach(), b.detach()


    # A smaller last minibatch is still a step, unless drop_last leaves it out.
    def count_steps(
        n: int,
        size: int,
        epochs: int,
        drop_last: bool,
    ) -> int:
        per_epoch = n // size if drop_last else math.ceil(n / size)

        return epochs * per_epoch
    ''',
)


def expect_same_loader(
    call: str,
    actual: object,
    expected: DataLoader,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, DataLoader):
        raise Mismatch(f"{call} is a {type(actual).__name__}. It should be a DataLoader.")

    got = list(actual)
    wanted = list(expected)

    if [len(yb) for _, yb in got] != [len(yb) for _, yb in wanted]:
        sizes = [len(yb) for _, yb in wanted]
        raise Mismatch(f"One epoch of {call} gives minibatches of sizes {[len(yb) for _, yb in got]}, but it should give {sizes}.")

    for (xb, yb), (wanted_x, wanted_y) in zip(got, wanted):
        if not torch.equal(yb, wanted_y) or not torch.equal(xb, wanted_x):
            raise Mismatch(f"One epoch of {call} has the right sizes but a different order. Shuffle with the generator passed in.")


def reference_train_with_loader(
    x: Tensor,
    y: Tensor,
    epochs: int,
    size: int,
    learning_rate: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    loader = DataLoader(TensorDataset(x, y), batch_size=size, shuffle=True, generator=generator)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    for _ in range(epochs):
        for xb, yb in loader:
            loss = ((xb @ w + b - yb) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

    return w.detach(), b.detach()


def check_minibatch3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="make_loader",
        description="shuffles with the generator and leaves out a short last minibatch",
        inputs=(
            (*named(*linear_data(89, 10, 2)), 4, seeded(0)),
            (*named(*linear_data(90, 9, 3)), 3, seeded(6)),
        ),
        reference=lambda x, y, size, generator: DataLoader(
            TensorDataset(x, y),
            batch_size=size,
            shuffle=True,
            drop_last=True,
            generator=generator,
        ),
        expect=expect_same_loader,
    )
    check_on_inputs(
        checker=checker,
        function="train_with_loader",
        description="trains on the loader's minibatches",
        inputs=(
            (*named(*linear_data(91, 10, 2)), 5, 4, 0.05, seeded(3)),
        ),
        reference=reference_train_with_loader,
        expect=expect_parts("w", "b"),
    )
    check_on_inputs(
        checker=checker,
        function="count_steps",
        description="counts the steps of a run",
        inputs=(
            (10, 4, 3, False),
            (10, 4, 3, True),
            (60000, 128, 5, False),
            (60000, 128, 5, True),
        ),
        reference=lambda n, size, epochs, drop_last: epochs * (n // size if drop_last else math.ceil(n / size)),
        expect=expect_value,
    )


MINIBATCH3 = Exercise(
    name="minibatch3",
    chapter=MINIBATCHES,
    title="PyTorch's DataLoader",
    lesson=MINIBATCH3_LESSON,
    source=MINIBATCH3_SOURCE,
    solution=MINIBATCH3_SOLUTION,
    hints={
        "make_loader": "Build it the way the line that sets loader does, with batch_size=size, generator=generator "
        "and drop_last=True.",
        "train_with_loader": "Loop over loader itself. Each turn gives xb and yb.",
        "count_steps": "One epoch has n / size minibatches, rounded up when the last smaller one counts and down "
        "when drop_last leaves it out. Multiply by epochs.",
    },
    introduces=(),
    check=check_minibatch3,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def related_features(
    seed: int,
    examples: int,
    features: int,
) -> tuple[Tensor, Tensor]:
    """Standardized features made by mixing independent ones, so they move together as real features do, and targets
    a linear function of them plus noise of spread 0.5. Features that never move together would let full batch
    descent land on the best fit in about one step, which real data doesn't allow."""
    generator = torch.Generator().manual_seed(seed)
    independent = torch.randn(examples, features, generator=generator)
    mixed = independent @ torch.randn(features, features, generator=generator)
    x = (mixed - mixed.mean(dim=0)) / mixed.std(dim=0)
    true_w = torch.randn(features, generator=generator)

    return x, x @ true_w + 1.0 + 0.5 * torch.randn(examples, generator=generator)


def full_loss(
    x: Tensor,
    y: Tensor,
    w: Tensor,
    b: Tensor,
) -> float:
    with torch.no_grad():
        return ((x @ w + b - y) ** 2).mean().item()


def best_fit_loss(
    x: Tensor,
    y: Tensor,
) -> float:
    """The least squares loss, solved in doubles so the gaps above it stay exact down to 1e-4."""
    with_ones = torch.cat([x, torch.ones(len(x), 1)], dim=1).double()
    solution = torch.linalg.lstsq(with_ones, y.double().unsqueeze(1)).solution.squeeze(1).float()

    return full_loss(x, y, solution[:-1], solution[-1])


def epoch_losses(
    x: Tensor,
    y: Tensor,
    epochs: int,
    size: int,
    learning_rate: float,
    order_of_epoch: Callable[[], Tensor],
) -> list[float]:
    """Training from zeros in minibatches of size, cut from order_of_epoch() every epoch. The loss on every example
    after each epoch."""
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    losses: list[float] = []

    for _ in range(epochs):
        order = order_of_epoch()

        for start in range(0, len(x), size):
            positions = order[start:start + size]
            loss = ((x[positions] @ w + b - y[positions]) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

        losses.append(full_loss(x, y, w.detach(), b.detach()))

    return losses


def shuffles(
    seed: int,
    n: int,
) -> Callable[[], Tensor]:
    """A fresh shuffle of n positions on every call, from a generator of its own, so each run draws the same orders."""
    generator = torch.Generator().manual_seed(seed)

    return lambda: torch.randperm(n, generator=generator)


def measure_gradient_estimates(seeds: range = range(10)) -> dict[str, object]:
    """minibatch1's first comparison: the gradient of w on minibatch1's checked data, 10 examples and 3 features,
    with w at small_tensor(86, 3), b at 0 and minibatches of 3. The gradient over every example, and per generator
    seed the first minibatch's gradient and the mean over 200 minibatches, the first of them included, each with
    its largest entry gap from the full gradient."""
    x, y = linear_data(83, 10, 3)
    w = small_tensor(86, 3)
    b = torch.tensor(0.0)
    full = mse_gradients_on(x, y, w, b, torch.arange(len(x)))[0]
    results: dict[str, object] = {"full": full.tolist(), "one": [], "mean of 200": [], "one, gap": [], "mean of 200, gap": []}

    with pinned_threads():
        for seed in seeds:
            first = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[:3]
            one = mse_gradients_on(x, y, w, b, first)[0]
            mean = reference_mean_gradient(x, y, w, b, 3, 200, torch.Generator().manual_seed(seed))
            results["one"].append(one.tolist())
            results["mean of 200"].append(mean.tolist())
            results["one, gap"].append((one - full).abs().max().item())
            results["mean of 200, gap"].append((mean - full).abs().max().item())

    return results


# Each way's rate is the one of these with the lowest loss after 20 epochs, so neither is held back by a bad rate.
COMPARED_RATES: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def measure_full_against_minibatch(seeds: range = range(10)) -> dict[str, list[tuple[float, list[float]]]]:
    """minibatch1's table: related_features with 10,000 examples and 10 features. Full batch descent, every example
    in every step, against minibatches of 32 shuffled every epoch. Per seed and way, the rate it ran at and its loss
    above the least squares loss after epochs 1, 2, 5 and 20. An epoch is one step for full batch and 313 for
    minibatches of 32."""
    results: dict[str, list[tuple[float, list[float]]]] = {}

    with pinned_threads():
        for seed in seeds:
            x, y = related_features(seed, 10_000, 10)
            best = best_fit_loss(x, y)

            for label, size in (("full batch", len(x)), ("minibatches of 32", 32)):
                runs = {rate: epoch_losses(x, y, 20, size, rate, shuffles(seed, len(x))) for rate in COMPARED_RATES}
                finite = {rate: losses for rate, losses in runs.items() if all(math.isfinite(loss) for loss in losses)}
                rate = min(finite, key=lambda rate: finite[rate][-1])
                above = [finite[rate][epochs - 1] - best for epochs in (1, 2, 5, 20)]
                results.setdefault(label, []).append((rate, above))

    return results


def measure_shuffling(seeds: range = range(10)) -> dict[tuple[float, str], list[float]]:
    """minibatch2's table: related_features with 2,000 examples and 2 features, trained for 5 epochs in minibatches
    of 32. Shuffled every epoch against kept sorted by target. Per rate and order, the loss above least squares per
    seed."""
    results: dict[tuple[float, str], list[float]] = {}

    with pinned_threads():
        for seed in seeds:
            x, y = related_features(seed, 2_000, 2)
            best = best_fit_loss(x, y)
            by_target = torch.argsort(y)

            for rate in (0.01, 0.03, 0.1):
                shuffled = epoch_losses(x, y, 5, 32, rate, shuffles(seed, len(x)))
                in_order = epoch_losses(x, y, 5, 32, rate, lambda: by_target)
                results.setdefault((rate, "shuffled"), []).append(shuffled[-1] - best)
                results.setdefault((rate, "sorted"), []).append(in_order[-1] - best)

    return results
