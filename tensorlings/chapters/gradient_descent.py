"""09_gradient_descent, walking downhill on the loss, by hand and with autograd, and what makes it work."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    reject_missing_return,
    shape_text,
    small_tensor,
)
from ..text import prose, source_file
from .tensors import pinned_threads

GRADIENT_DESCENT = Chapter(
    folder="09_gradient_descent",
    title="Gradient descent",
    readme=source_file(
        """
        # Gradient descent

        Most models have no formula for their best parameters. Gradient descent
        finds good ones by stepping against the gradient of the loss, over and
        over. This chapter takes the steps by hand, then with autograd, and
        shows the two things that make it work or fail, the learning rate and
        the scale of the features.

        ## Further reading

        - [Dive into Deep Learning, gradient descent](https://d2l.ai/chapter_optimization/gd.html)
        - [Dive into Deep Learning, linear regression from scratch](https://d2l.ai/chapter_linear-regression/linear-regression-scratch.html)
        - [Goh, why momentum really works, for pictures of descent on a loss](https://distill.pub/2017/momentum/)
        """,
    ),
)


def linear_data(
    seed: int,
    examples: int,
    features: int,
) -> tuple[Tensor, Tensor]:
    """Small whole number features and targets near a linear function of them, the same every run."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(-3, 4, (examples, features), generator=generator).float()
    true_w = torch.randint(-2, 3, (features,), generator=generator).float()
    noise = torch.randint(-1, 2, (examples,), generator=generator).float() / 4

    return x, x @ true_w + 1.0 + noise


def named(
    x: Tensor,
    y: Tensor,
) -> tuple[Described, Described]:
    """Training data that a failing call shows by its shape, since the numbers themselves would bury the message."""
    return Described(x, f"x of shape {shape_text(tuple(x.shape))}"), Described(y, "y")


def run_descent(
    x: Tensor,
    y: Tensor,
    steps: int,
    learning_rate: float,
) -> tuple[Tensor, Tensor]:
    """Gradient descent on the mean squared error with the gradients worked out by hand."""
    w = torch.zeros(x.shape[1])
    b = torch.tensor(0.0)

    for _ in range(steps):
        errors = x @ w + b - y
        grad_errors = 2 * errors / errors.numel()
        w = w - learning_rate * (x.T @ grad_errors)
        b = b - learning_rate * grad_errors.sum()

    return w, b


def autograd_descent(
    x: Tensor,
    y: Tensor,
    steps: int,
    learning_rate: float,
    loss_of: object = lambda predictions, y: ((predictions - y) ** 2).mean(),
) -> tuple[Tensor, Tensor, list[float]]:
    """Gradient descent with autograd, giving the final parameters and the loss before each step."""
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    history: list[float] = []

    for _ in range(steps):
        loss = loss_of(x @ w + b, y)
        history.append(loss.item())
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        w.grad = None
        b.grad = None

    return w.detach(), b.detach(), history


# ---------------------------------------------------------------------------
# descent1
# ---------------------------------------------------------------------------

DESCENT1_LESSON = prose(
    """
    Least squares only works for a few simple models. Most models have no
    formula for their best parameters, so training finds them another way. Start anywhere, then walk downhill on the loss, a small step
    at a time.

    The gradient of the loss points the way each parameter would raise the
    loss fastest, as derivatives3 showed. So each step moves the parameters
    against it.

    ```python
    w = w - learning_rate * grad_w
    ```

    The **learning rate** sets the size of each step. Take the loss
    `(3 * w - 6) ** 2`, which is lowest at `w = 2` and has derivative
    `6 * (3 * w - 6)`.

    ```python
    w = 0.0
    for step in range(5):
        grad = 6 * (3 * w - 6)
        w = w - 0.05 * grad
    # w goes 1.8, 1.98, 1.998, 1.9998, 1.99998
    ```

    Each step closes most of the gap to 2. Walking downhill on the loss like
    this is **gradient descent**.

    For a linear model with the mean squared error, Section 1's backprop by
    hand gives the gradients. The loss is the mean of `errors ** 2`, where
    `errors = x @ w + b - y`. Its gradient with respect to `errors` is
    `2 * errors / n` for `n` examples. From there the rules of the backprop
    chapter give `grad_w` and `grad_b`. Remember that `b` was copied into
    every prediction.
    """,
)

DESCENT1_SOURCE = source_file(
    '''
    # descent1.py

    import torch
    from torch import Tensor

    # Gradient descent on the loss (3 * w - 6) ** 2, whose derivative is
    # 6 * (3 * w - 6). w moves toward 2, the bottom of the loss.
    w = 0.0

    for _ in range(5):
        grad = 6 * (3 * w - 6)
        w = w - 0.05 * grad                          # 1.8, 1.98, 1.998, ...


    # TODO: Return the weights after one step of gradient descent.
    def step(
        w: Tensor,
        grad_w: Tensor,
        learning_rate: float,
    ) -> Tensor:
        ...


    # TODO: Finish the last two lines, so this returns grad_w and grad_b, the
    #   gradients of the mean squared error of x @ w + b against y. grad_errors,
    #   the gradient of the errors, is filled in from the lesson. x @ w is a
    #   matrix times a flat tensor, as in matvec_grad_v of backprop3, and b is
    #   copied into every prediction, as row is in backprop2.
    def mse_gradients(
        x: Tensor,
        w: Tensor,
        b: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        errors = x @ w + b - y
        grad_errors = 2 * errors / errors.numel()
        grad_w = ...
        grad_b = ...

        return grad_w, grad_b


    # TODO: Finish the two lines in the loop, so this runs gradient descent from
    #   w at zeros and b at 0 for the given number of steps, and returns the
    #   final w and b.
    def descend(
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> tuple[Tensor, Tensor]:
        w = torch.zeros(x.shape[1])
        b = torch.tensor(0.0)

        for _ in range(steps):
            grad_w, grad_b = mse_gradients(x, w, b, y)
            w = ...
            b = ...

        return w, b
    ''',
)

DESCENT1_SOLUTION = source_file(
    '''
    # descent1.py, a solution for comparison

    import torch
    from torch import Tensor

    w = 0.0

    for _ in range(5):
        grad = 6 * (3 * w - 6)
        w = w - 0.05 * grad


    def step(
        w: Tensor,
        grad_w: Tensor,
        learning_rate: float,
    ) -> Tensor:
        return w - learning_rate * grad_w


    # The mean divides by the number of examples. b was added to every
    # prediction, so its gradient adds grad_errors up.
    def mse_gradients(
        x: Tensor,
        w: Tensor,
        b: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        errors = x @ w + b - y
        grad_errors = 2 * errors / errors.numel()
        grad_w = x.T @ grad_errors
        grad_b = grad_errors.sum()

        return grad_w, grad_b


    def descend(
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> tuple[Tensor, Tensor]:
        w = torch.zeros(x.shape[1])
        b = torch.tensor(0.0)

        for _ in range(steps):
            grad_w, grad_b = mse_gradients(x, w, b, y)
            w = step(w, grad_w, learning_rate)
            b = step(b, grad_b, learning_rate)

        return w, b
    ''',
)


def autograd_mse_gradients(
    x: Tensor,
    w: Tensor,
    b: Tensor,
    y: Tensor,
) -> tuple[Tensor, Tensor]:
    w = w.clone().requires_grad_()
    b = b.clone().requires_grad_()
    ((x @ w + b - y) ** 2).mean().backward()

    return w.grad, b.grad


def check_descent1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="step",
        description="moves the weights against the gradient",
        inputs=(
            (small_tensor(60, 3), small_tensor(61, 3), 0.1),
            (small_tensor(62, 2, 2), small_tensor(63, 2, 2), 0.5),
        ),
        reference=lambda w, grad_w, learning_rate: w - learning_rate * grad_w,
        expect=expect_tensor,
    )
    x_two, y_two = linear_data(64, 6, 2)
    x_three, y_three = linear_data(66, 5, 3)
    check_on_inputs(
        checker=checker,
        function="mse_gradients",
        description="gives the gradients of the mean squared error",
        inputs=(
            (named(x_two, y_two)[0], small_tensor(65, 2), torch.tensor(0.5), named(x_two, y_two)[1]),
            (named(x_three, y_three)[0], small_tensor(67, 3), torch.tensor(-1.0), named(x_three, y_three)[1]),
        ),
        reference=autograd_mse_gradients,
        expect=expect_parts("grad_w", "grad_b"),
    )
    check_on_inputs(
        checker=checker,
        function="descend",
        description="runs gradient descent from zeros",
        inputs=(
            (*named(*linear_data(68, 8, 2)), 50, 0.05),
            (*named(*linear_data(69, 10, 3)), 80, 0.02),
        ),
        reference=run_descent,
        expect=expect_parts("w", "b"),
    )


DESCENT1 = Exercise(
    name="descent1",
    chapter=GRADIENT_DESCENT,
    title="One step downhill",
    lesson=DESCENT1_LESSON,
    source=DESCENT1_SOURCE,
    solution=DESCENT1_SOLUTION,
    hints={
        "step": "Take the learning rate times the gradient away from w, as the line in the loop above does.",
        "mse_gradients": "w plays the part of v in matvec_grad_v of backprop3, with grad_errors as the gradient "
        "handed down. b was copied into every prediction, so its gradient adds grad_errors up.",
        "descend": "Call step for w with grad_w, and for b with grad_b.",
    },
    introduces=("learning rate", "gradient descent"),
    check=check_descent1,
)


# ---------------------------------------------------------------------------
# descent2
# ---------------------------------------------------------------------------

DESCENT2_LESSON = prose(
    """
    Working out gradients by hand stops being practical once a model has more
    than a few steps. Autograd does it instead. Give the parameters
    `requires_grad=True`, compute the loss, call `.backward()`, and every
    parameter's `.grad` holds its gradient.

    ```python
    w = torch.zeros(2, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    for _ in range(100):
        loss = ((x @ w + b - y) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            w -= 0.1 * w.grad
            b -= 0.1 * b.grad

        w.grad = None
        b.grad = None
    ```

    Three lines are new.

    - `with torch.no_grad():` stops autograd from recording what runs inside it. The update is not part of the loss, so it must not join the graph.
    - `-=` changes `w` in place, so it stays the tensor that asked for gradients. `w = w - ...` would make a new tensor.
    - `w.grad = None` clears the gradient. Gradients add up across `.backward()` calls, as autograd4 showed, so without it each step would add every gradient before it.

    Watching the loss tells you whether training works. It should fall fast at
    first, then level off. `loss.item()` turns the loss into a plain number to
    keep or print.

    The loops here end with `return w.detach(), b.detach()`. `.detach()`
    gives the same numbers as a tensor that autograd no longer records. So
    whatever the caller does with them stays out of the training's graph.
    """,
)

DESCENT2_SOURCE = source_file(
    '''
    # descent2.py

    import torch
    from torch import Tensor


    # TODO: This should run gradient descent with autograd and return the final
    #   w and b, but they come out wrong, since each step's gradient piles onto
    #   the ones before. Read the lesson, then fix it.
    def train(
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> tuple[Tensor, Tensor]:
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        for _ in range(steps):
            loss = ((x @ w + b - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

        return w.detach(), b.detach()


    # TODO: Finish the line that sets loss, so this trains on the mean absolute
    #   error from linreg2 instead.
    def train_absolute(
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> tuple[Tensor, Tensor]:
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        for _ in range(steps):
            loss = ...
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

        return w.detach(), b.detach()


    # TODO: Finish the line that fills history, so this returns the loss at
    #   every step, before that step's update, as plain numbers.
    def loss_history(
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> list[float]:
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        history = []

        for _ in range(steps):
            loss = ((x @ w + b - y) ** 2).mean()
            history.append(...)
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

        return history
    ''',
)

DESCENT2_SOLUTION = source_file(
    '''
    # descent2.py, a solution for comparison

    import torch
    from torch import Tensor


    # Without clearing, each .backward() adds onto every gradient before it.
    def train(
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> tuple[Tensor, Tensor]:
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        for _ in range(steps):
            loss = ((x @ w + b - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

        return w.detach(), b.detach()


    def train_absolute(
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> tuple[Tensor, Tensor]:
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        for _ in range(steps):
            loss = (x @ w + b - y).abs().mean()
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

        return w.detach(), b.detach()


    def loss_history(
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> list[float]:
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        history = []

        for _ in range(steps):
            loss = ((x @ w + b - y) ** 2).mean()
            history.append(loss.item())
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

        return history
    ''',
)


def expect_history(
    call: str,
    actual: object,
    expected: list[float],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, list) or len(actual) != len(expected):
        raise Mismatch(f"{call} should be a list of {len(expected)} losses, one per step.")

    for position, (got, wanted) in enumerate(zip(actual, expected)):
        if got is Ellipsis:
            raise Mismatch(f"{call} still holds a ... where a TODO goes, so it is not written yet.")

        if not isinstance(got, float):
            raise Mismatch(f"Entry {position} of {call} is a {type(got).__name__}. Keep each loss as a plain number with .item().")

        if abs(got - wanted) > 1e-4 * max(1.0, abs(wanted)):
            raise Mismatch(f"Entry {position} of {call} is {got:.6g}, but the loss at that step is {wanted:.6g}.")


def check_descent2(checker: Checker) -> None:
    training_data = (
        (*named(*linear_data(70, 8, 2)), 40, 0.05),
        (*named(*linear_data(71, 10, 3)), 60, 0.02),
    )
    check_on_inputs(
        checker=checker,
        function="train",
        description="trains with autograd, clearing the gradients every step",
        inputs=training_data,
        reference=lambda x, y, steps, learning_rate: autograd_descent(x, y, steps, learning_rate)[:2],
        expect=expect_parts("w", "b"),
    )
    check_on_inputs(
        checker=checker,
        function="train_absolute",
        description="trains on the mean absolute error",
        inputs=training_data,
        reference=lambda x, y, steps, learning_rate: autograd_descent(
            x,
            y,
            steps,
            learning_rate,
            loss_of=lambda predictions, y: (predictions - y).abs().mean(),
        )[:2],
        expect=expect_parts("w", "b"),
    )
    check_on_inputs(
        checker=checker,
        function="loss_history",
        description="gives the loss before every step, as plain numbers",
        inputs=training_data,
        reference=lambda x, y, steps, learning_rate: autograd_descent(x, y, steps, learning_rate)[2],
        expect=expect_history,
    )


DESCENT2 = Exercise(
    name="descent2",
    chapter=GRADIENT_DESCENT,
    title="Training with autograd",
    lesson=DESCENT2_LESSON,
    source=DESCENT2_SOURCE,
    solution=DESCENT2_SOLUTION,
    hints={
        "train": "The gradients pile up from one step to the next. Clear w.grad and b.grad at the end of each step.",
        "train_absolute": "The mean absolute error of x @ w + b against y, written as in linreg2.",
        "loss_history": "loss is a tensor holding one number. Append loss.item(), the plain number.",
    },
    introduces=(),
    check=check_descent2,
)


# ---------------------------------------------------------------------------
# descent3
# ---------------------------------------------------------------------------

DESCENT3_LESSON = prose(
    """
    The learning rate decides whether training works at all. Take the loss
    `w ** 2`, with gradient `2 * w`, starting at `w = 1`. Each step multiplies
    `w` by `1 - 2 * learning_rate`. After 10 steps the loss is

    | Learning rate | Loss after 10 steps | What happens |
    |---|---|---|
    | 0.01 | 0.668 | crawls toward the bottom |
    | 0.1 | 0.0115 | gets there |
    | 0.5 | 0 | lands on the bottom in one step |
    | 0.9 | 0.0115 | jumps across the bottom and back, and still gets there |
    | 1.1 | 38.3 | jumps farther each time, and blows up |

    Too small wastes steps. Too big overshoots, and past a limit every step
    lands farther from the bottom than the last. The loss then grows without
    end, until it is `inf`, infinity, or `nan`, short for not a number. A loss like that
    is the first sign of a learning rate that is too big.

    `math.isfinite(x)` is True for a plain number and False for `inf` and
    `nan`. `math.inf` is infinity as a plain number, above every finite loss,
    so it makes a good starting best loss. Comparing a `nan` with anything
    gives False, so test for it before comparing losses.

    There is no rule that gives the right rate for a new model. The usual way
    is to try a few rates ten times apart, like 0.001, 0.01 and 0.1, and keep
    the one with the lowest loss. Another is to start high and halve the rate
    until training stops blowing up.
    """,
)

DESCENT3_SOURCE = source_file(
    '''
    # descent3.py

    import math

    import torch
    from torch import Tensor


    # The training loop from descent2, giving the loss after the last step.
    def final_loss(
        x: Tensor,
        y: Tensor,
        learning_rate: float,
        steps: int,
    ) -> float:
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        for _ in range(steps):
            loss = ((x @ w + b - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

        return ((x @ w + b - y) ** 2).mean().item()


    # TODO: Finish the if line, so this returns the learning rate from rates whose
    #   final_loss after the given steps is lowest, skipping any rate whose loss
    #   is not finite.
    def best_learning_rate(
        x: Tensor,
        y: Tensor,
        rates: list[float],
        steps: int,
    ) -> float:
        best_rate = None
        best_loss = math.inf

        for rate in rates:
            loss = final_loss(x, y, rate, steps)

            if ...:
                best_rate = rate
                best_loss = loss

        return best_rate


    # TODO: Finish blows_up, the test the while loop runs. It returns True when
    #   final_loss at that rate is not finite, or is above start_loss, the loss
    #   with w and b at zero before any step.
    def halve_until_stable(
        x: Tensor,
        y: Tensor,
        rate: float,
        steps: int,
    ) -> float:
        start_loss = (y ** 2).mean().item()

        def blows_up(rate: float) -> bool:
            ...

        while blows_up(rate):
            rate = rate / 2

        return rate
    ''',
)

DESCENT3_SOLUTION = source_file(
    '''
    # descent3.py, a solution for comparison

    import math

    import torch
    from torch import Tensor


    def final_loss(
        x: Tensor,
        y: Tensor,
        learning_rate: float,
        steps: int,
    ) -> float:
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        for _ in range(steps):
            loss = ((x @ w + b - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

        return ((x @ w + b - y) ** 2).mean().item()


    def best_learning_rate(
        x: Tensor,
        y: Tensor,
        rates: list[float],
        steps: int,
    ) -> float:
        best_rate = None
        best_loss = math.inf

        for rate in rates:
            loss = final_loss(x, y, rate, steps)

            if math.isfinite(loss) and loss < best_loss:
                best_rate = rate
                best_loss = loss

        return best_rate


    def halve_until_stable(
        x: Tensor,
        y: Tensor,
        rate: float,
        steps: int,
    ) -> float:
        start_loss = (y ** 2).mean().item()

        # A nan fails every comparison, so loss > start_loss alone would call a nan stable.
        def blows_up(rate: float) -> bool:
            loss = final_loss(x, y, rate, steps)

            return not math.isfinite(loss) or loss > start_loss

        while blows_up(rate):
            rate = rate / 2

        return rate
    ''',
)


def reference_final_loss(
    x: Tensor,
    y: Tensor,
    learning_rate: float,
    steps: int,
) -> float:
    w, b, _ = autograd_descent(x, y, steps, learning_rate)

    return ((x @ w + b - y) ** 2).mean().item()


def reference_best_rate(
    x: Tensor,
    y: Tensor,
    rates: list[float],
    steps: int,
) -> float:
    finite = [(reference_final_loss(x, y, rate, steps), rate) for rate in rates]

    return min((loss, rate) for loss, rate in finite if math.isfinite(loss))[1]


def reference_halving(
    x: Tensor,
    y: Tensor,
    rate: float,
    steps: int,
) -> float:
    start_loss = (y ** 2).mean().item()

    while not math.isfinite(reference_final_loss(x, y, rate, steps)) or reference_final_loss(x, y, rate, steps) > start_loss:
        rate = rate / 2

    return rate


def check_descent3(checker: Checker) -> None:
    # The first rate of each list gives a nan loss, and min with a key keeps a nan that comes first.
    # Halving from 10.0 starts at a nan, which loss > start_loss alone would call stable.
    check_on_inputs(
        checker=checker,
        function="best_learning_rate",
        description="gives the rate with the lowest finite loss",
        inputs=(
            (*named(*linear_data(72, 10, 2)), [10.0, 0.001, 0.01, 0.1, 1.0], 30),
            (*named(*linear_data(73, 12, 3)), [3.0, 0.3, 0.03, 0.003], 30),
        ),
        reference=reference_best_rate,
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="halve_until_stable",
        description="halves the rate until training stops blowing up",
        inputs=(
            (*named(*linear_data(72, 10, 2)), 10.0, 30),
            (*named(*linear_data(74, 10, 2)), 8.0, 20),
            (*named(*linear_data(75, 12, 3)), 0.05, 20),
        ),
        reference=reference_halving,
        expect=expect_close,
    )


DESCENT3 = Exercise(
    name="descent3",
    chapter=GRADIENT_DESCENT,
    title="The learning rate",
    lesson=DESCENT3_LESSON,
    source=DESCENT3_SOURCE,
    solution=DESCENT3_SOLUTION,
    hints={
        "best_learning_rate": "Keep a rate when math.isfinite says its loss is finite and the loss is below "
        "best_loss.",
        "halve_until_stable": "In blows_up, get final_loss(x, y, rate, steps). Return True when math.isfinite "
        "says it isn't finite, or when it is above start_loss.",
    },
    introduces=(),
    check=check_descent3,
)


# ---------------------------------------------------------------------------
# descent4
# ---------------------------------------------------------------------------

DESCENT4_LESSON = prose(
    """
    Features often live on very different scales. Here one feature is a
    fraction and the other is in the thousands.

    ```python
    x = torch.tensor([[0.2, 1500.0],
                      [0.9, 2600.0],
                      [0.5, 1100.0],
                      [0.7, 2900.0]])
    ```

    The gradient of each weight is scaled by its feature, so the second
    weight's gradient is about a thousand times the first's. A learning rate
    small enough to keep the second weight from blowing up leaves the first
    one barely moving. No single rate suits both.

    The fix is to put every feature on the same scale before training.
    Standardizing takes each column's mean away, then divides by the column's
    standard deviation, its typical distance from the mean. Every column then
    has mean 0 and spread 1.

    ```python
    x.mean(dim=0)    # about 0.575 and 2025, one mean per column
    x.std(dim=0)     # about 0.299 and 862, one spread per column
    ```

    Here is training on 20 examples like these for 100 steps, over 10 runs.
    The targets are `3 * fraction + thousands / 1000 + 0.5`, and the rates
    tried were half a power of ten apart.

    | Features | Learning rate | Loss after 100 steps |
    |---|---|---|
    | raw | 3e-7 and up | blows up |
    | raw | 1e-7, the best | 0.83 to 1.70 |
    | standardized | 0.1 | below 3e-9 |

    On the raw features the first weight ends near 0.00001, where it should
    be 3. The loss, 1.35 at the median, is about what predicting the mean
    target for every example gives, 1.24. Standardized, the first weight ends
    at 3.

    Keep the mean and the spread you computed on the training data, and use
    those same numbers on any new example. Don't standardize a new example
    with its own numbers, and don't fold new examples into the mean and the
    spread either.
    """,
)

DESCENT4_SOURCE = source_file(
    '''
    # descent4.py

    import torch
    from torch import Tensor

    x = torch.tensor([[0.2, 1500.0],
                      [0.9, 2600.0],
                      [0.5, 1100.0],
                      [0.7, 2900.0]])

    # One mean and one spread per column, so dim=0, down the rows.
    means = x.mean(dim=0)                        # about 0.575 and 2025
    spreads = x.std(dim=0)                       # about 0.299 and 862


    # TODO: Return the mean and the spread of every column of x, as the two lines
    #   above work them out.
    def column_stats(x: Tensor) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return x standardized with the given mean and spread, each column's
    #   mean taken away, then divided by its spread.
    def standardize_with(
        x: Tensor,
        mean: Tensor,
        spread: Tensor,
    ) -> Tensor:
        ...


    # The training loop from descent2, giving the loss after the last step.
    def final_loss(
        x: Tensor,
        y: Tensor,
        learning_rate: float,
        steps: int,
    ) -> float:
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        for _ in range(steps):
            loss = ((x @ w + b - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

        return ((x @ w + b - y) ** 2).mean().item()


    # TODO: This trains on x as it is, and the loss blows up. x here is the
    #   training data, so its own column stats are the right ones. Make it train
    #   on x standardized with them, keeping the rate and the steps.
    def fit_loss(
        x: Tensor,
        y: Tensor,
    ) -> float:
        return final_loss(x, y, 0.1, 100)
    ''',
)

DESCENT4_SOLUTION = source_file(
    '''
    # descent4.py, a solution for comparison

    import torch
    from torch import Tensor

    x = torch.tensor([[0.2, 1500.0],
                      [0.9, 2600.0],
                      [0.5, 1100.0],
                      [0.7, 2900.0]])

    means = x.mean(dim=0)
    spreads = x.std(dim=0)


    def column_stats(x: Tensor) -> tuple[Tensor, Tensor]:
        return x.mean(dim=0), x.std(dim=0)


    # The mean and the spread have shape (columns,), which lines up with every row.
    def standardize_with(
        x: Tensor,
        mean: Tensor,
        spread: Tensor,
    ) -> Tensor:
        return (x - mean) / spread


    def final_loss(
        x: Tensor,
        y: Tensor,
        learning_rate: float,
        steps: int,
    ) -> float:
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        for _ in range(steps):
            loss = ((x @ w + b - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

        return ((x @ w + b - y) ** 2).mean().item()


    def fit_loss(
        x: Tensor,
        y: Tensor,
    ) -> float:
        mean, spread = column_stats(x)

        return final_loss(standardize_with(x, mean, spread), y, 0.1, 100)
    ''',
)


def scaled_data(
    seed: int,
    examples: int,
) -> tuple[Tensor, Tensor]:
    """One feature between 0 and 1 and one in the thousands, with targets linear in both."""
    generator = torch.Generator().manual_seed(seed)
    small = torch.randint(0, 11, (examples,), generator=generator).float() / 10
    large = torch.randint(10, 31, (examples,), generator=generator).float() * 100
    x = torch.stack([small, large], dim=1)

    return x, 3 * small + large / 1000 + 0.5


def check_descent4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="column_stats",
        description="gives the mean and the spread of every column",
        inputs=(
            (named(*scaled_data(76, 6))[0],),
            (small_tensor(77, 5, 3),),
        ),
        reference=lambda x: (x.mean(dim=0), x.std(dim=0)),
        expect=expect_parts("the means", "the spreads"),
    )
    check_on_inputs(
        checker=checker,
        function="standardize_with",
        description="takes the mean away and divides by the spread, column by column",
        inputs=(
            (named(*scaled_data(78, 4))[0], torch.tensor([0.5, 2000.0]), torch.tensor([0.25, 800.0])),
            (small_tensor(79, 3, 3), torch.tensor([1.0, -1.0, 0.0]), torch.tensor([2.0, 0.5, 1.0])),
        ),
        reference=lambda x, mean, spread: (x - mean) / spread,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="fit_loss",
        description="trains on standardized features to a low loss",
        inputs=(
            named(*scaled_data(80, 12)),
            named(*scaled_data(81, 20)),
        ),
        reference=lambda x, y: reference_final_loss((x - x.mean(dim=0)) / x.std(dim=0), y, 0.1, 100),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-4,
        ),
    )


DESCENT4 = Exercise(
    name="descent4",
    chapter=GRADIENT_DESCENT,
    title="Features on different scales",
    lesson=DESCENT4_LESSON,
    source=DESCENT4_SOURCE,
    solution=DESCENT4_SOLUTION,
    hints={
        "column_stats": "Return the two lines that set means and spreads, with this function's x, as a pair.",
        "standardize_with": "Take mean away from x, then divide by spread. Both have one number per column, which "
        "broadcasting lines up with every row.",
        "fit_loss": "Get the mean and the spread from column_stats(x), standardize x with them, and pass the "
        "standardized x to final_loss in place of x.",
    },
    introduces=(),
    check=check_descent4,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

# Rates half a power of ten apart, from 1e-10 up to 1.
RAW_RATES: tuple[float, ...] = tuple(10.0 ** (power / 2) for power in range(-20, 1))


def measure_standardizing(seeds: range = range(10)) -> dict[str, list[float]]:
    """descent4's table: scaled_data with 20 examples and 100 steps from zeros, one entry per seed. On the raw
    features, the smallest rate of RAW_RATES whose loss is not finite or ends above the starting loss, the best rate
    below it, that rate's loss and the first weight it ends with. The loss of predicting the mean of y for every
    example. On standardized features at a rate of 0.1, the loss and the first weight, put back on the raw feature's
    scale. The targets are 3 * the fraction + the thousands / 1000 + 0.5, so the first weight should end at 3."""
    results: dict[str, list[float]] = {}

    with pinned_threads():
        for seed in seeds:
            x, y = scaled_data(seed, 20)
            start_loss = (y ** 2).mean().item()
            raw_losses = {rate: reference_final_loss(x, y, rate, 100) for rate in RAW_RATES}
            blowing_up = min(rate for rate, loss in raw_losses.items() if not math.isfinite(loss) or loss > start_loss)
            stable = {rate: loss for rate, loss in raw_losses.items() if rate < blowing_up}
            best_rate = min(stable, key=stable.get)
            raw_w, _, _ = autograd_descent(x, y, 100, best_rate)

            mean, spread = x.mean(dim=0), x.std(dim=0)
            standardized_w, _, _ = autograd_descent((x - mean) / spread, y, 100, 0.1)

            for key, value in (
                ("raw, first rate that blows up", blowing_up),
                ("raw, best rate", best_rate),
                ("raw, best loss", stable[best_rate]),
                ("raw, first weight", raw_w[0].item()),
                ("predicting the mean", ((y - y.mean()) ** 2).mean().item()),
                ("standardized, loss", reference_final_loss((x - mean) / spread, y, 0.1, 100)),
                ("standardized, first weight", (standardized_w[0] / spread[0]).item()),
            ):
                results.setdefault(key, []).append(value)

    return results
