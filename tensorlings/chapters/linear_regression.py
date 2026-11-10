"""08_linear_regression, a model with parameters, the loss that measures it, and the exact best fit."""

from __future__ import annotations

import torch
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import Checker, check_on_inputs, expect_close, expect_parts, expect_tensor, expect_value, small_tensor
from ..text import prose, source_file

LINEAR_REGRESSION = Chapter(
    folder="08_linear_regression",
    title="Linear regression",
    readme=source_file(
        """
        # Linear regression

        Section 2 is about how a model learns. This chapter starts with the
        simplest model, a weighted sum of the features plus a bias. It also
        introduces the loss, a number that measures how wrong the model is. For
        this model, the best parameters can be computed exactly. That gives the
        next chapters an answer to check their training against.

        ## Further reading

        - [Dive into Deep Learning, linear regression](https://d2l.ai/chapter_linear-regression/linear-regression.html)
        - [CS229 notes, linear regression and least squares](https://cs229.stanford.edu/main_notes.pdf)
        """,
    ),
)


# ---------------------------------------------------------------------------
# linreg1
# ---------------------------------------------------------------------------

LINREG1_LESSON = prose(
    """
    Section 2 is about how a model learns. It starts with the simplest model
    there is.

    The data is a tensor with one row per example. The numbers in a row are
    that example's **features**. Each example also has a target, the number
    the model should output for it. The number the model actually outputs is
    its **prediction**.

    A linear model multiplies each feature by its own weight, adds up the
    results, and then adds one more number, the **bias**.

    ```python
    x = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])    # 3 examples, 2 features each
    w = torch.tensor([0.5, 1.0])     # one weight per feature
    b = 2.0
    x @ w + b                         # tensor([ 4.5000,  7.5000, 10.5000])
    ```

    `x @ w` computes the weighted sum of every row at once, as in matmul1.
    Then `+ b` adds the bias to each of those sums. The examples are the rows
    of `x`, so `x` goes on the left of `@`.

    The weights and the bias are the model's **parameters**, the numbers that
    training changes. The bias lets a prediction be something other than 0
    when every feature is 0. Predicting a number this way is called linear
    **regression**. A regression model is any model whose output is a number.
    Models that choose among a set of options come in a later chapter.

    A model can also predict several numbers per example. Then the weights
    form a matrix of shape `(features, outputs)`, with one column per output.
    The bias then holds one number per output. A model's size is its number of
    parameters, which counts every weight and every bias.
    """,
)

LINREG1_SOURCE = source_file(
    '''
    # linreg1.py

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])            # 3 examples, 2 features each
    w = torch.tensor([0.5, 1.0])             # one weight per feature
    b = 2.0                                  # the bias

    # One prediction per example, which is the example's weighted sum plus the bias.
    predictions = x @ w + b                  # tensor([ 4.5000,  7.5000, 10.5000])


    # TODO: Return the prediction for every example in x, for any weights w and
    #   bias b. x has shape (examples, features), w has shape (features,), and
    #   the result has shape (examples,).
    #   Example: predict(x, w, b) with the x, w and b above returns
    #   tensor([ 4.5000,  7.5000, 10.5000]).
    def predict(
        x: Tensor,
        w: Tensor,
        b: float,
    ) -> Tensor:
        ...


    # TODO: Return the predictions of a model with several outputs, as a tensor
    #   of shape (examples, outputs). x has shape (examples, features), W has
    #   shape (features, outputs), and biases has shape (outputs,).
    def predict_outputs(
        x: Tensor,
        W: Tensor,
        biases: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return how many parameters a linear model has with this many
    #   features and outputs, counting every weight and every bias. The model
    #   above has 2 weights and 1 bias, so count_parameters(2, 1) returns 3.
    def count_parameters(
        features: int,
        outputs: int,
    ) -> int:
        ...


    # TODO: This crashes when the checker calls it. Read the error, then fix it
    #   so it returns one prediction per example.
    def predict_rows(
        x: Tensor,
        w: Tensor,
        b: float,
    ) -> Tensor:
        return w @ x + b
    ''',
)

LINREG1_SOLUTION = source_file(
    '''
    # linreg1.py, a solution for comparison

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])
    w = torch.tensor([0.5, 1.0])
    b = 2.0

    predictions = x @ w + b


    def predict(
        x: Tensor,
        w: Tensor,
        b: float,
    ) -> Tensor:
        return x @ w + b


    # (examples, features) @ (features, outputs) gives (examples, outputs). Then
    # broadcasting adds each bias to its own column.
    def predict_outputs(
        x: Tensor,
        W: Tensor,
        biases: Tensor,
    ) -> Tensor:
        return x @ W + biases


    def count_parameters(
        features: int,
        outputs: int,
    ) -> int:
        return features * outputs + outputs


    # The examples are the rows of x, so x goes on the left of @.
    def predict_rows(
        x: Tensor,
        w: Tensor,
        b: float,
    ) -> Tensor:
        return x @ w + b
    ''',
)


def check_linreg1(checker: Checker) -> None:
    for function in ("predict", "predict_rows"):
        check_on_inputs(
            checker=checker,
            function=function,
            description="gives one prediction per example",
            inputs=(
                (small_tensor(1, 4, 3), small_tensor(2, 3), 0.5),
                (small_tensor(3, 2, 2), small_tensor(4, 2), -1.0),
            ),
            reference=lambda x, w, b: x @ w + b,
            expect=expect_tensor,
        )

    check_on_inputs(
        checker=checker,
        function="predict_outputs",
        description="gives one prediction per example and output",
        inputs=(
            (small_tensor(5, 4, 3), small_tensor(6, 3, 2), small_tensor(7, 2)),
            (small_tensor(8, 3, 3), small_tensor(9, 3, 3), small_tensor(10, 3)),
        ),
        reference=lambda x, W, biases: x @ W + biases,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="count_parameters",
        description="counts every weight and every bias",
        inputs=(
            (2, 1),
            (784, 10),
            (3, 5),
        ),
        reference=lambda features, outputs: features * outputs + outputs,
        expect=expect_value,
    )


LINREG1 = Exercise(
    name="linreg1",
    chapter=LINEAR_REGRESSION,
    title="A model with parameters",
    lesson=LINREG1_LESSON,
    source=LINREG1_SOURCE,
    solution=LINREG1_SOLUTION,
    hints={
        "predict": "Copy the pattern of the line that sets predictions, using this function's x, w and b.",
        "predict_outputs": "Do the same as in predict, with a matrix of weights. x @ W has one column per output, "
        "and broadcasting adds each bias to its own column.",
        "count_parameters": "There is one weight for each pair of a feature and an output, and one bias for each "
        "output.",
        "predict_rows": "The error says the two shapes cannot be multiplied. Each row of x is an example, so x has "
        "to be on the left of @.",
    },
    introduces=("feature", "prediction", "parameter", "bias", "regression"),
    check=check_linreg1,
)


# ---------------------------------------------------------------------------
# linreg2
# ---------------------------------------------------------------------------

LINREG2_LESSON = prose(
    """
    Training needs one number that says how wrong the model is. That number is
    the **loss**. A lower loss is better, and training searches for parameters
    that give a low loss.

    For regression, the usual loss is the **mean squared error**. Take each
    prediction's error, `prediction - target`, square it, and then take the
    mean of those squares.

    ```python
    predictions = torch.tensor([4.5, 7.5, 10.5])
    targets = torch.tensor([5.0, 7.0, 12.0])
    errors = predictions - targets    # tensor([-0.5000,  0.5000, -1.5000])
    (errors ** 2).mean()              # tensor(0.9167)
    errors.abs()                      # tensor([0.5000, 0.5000, 1.5000]), the sign dropped
    ```

    Squaring makes every error positive, so errors in opposite directions add
    up instead of canceling out. Squaring also makes a big error cost far more
    than a small one. An error of 3 adds 9 to the sum, and an error of 1 only
    adds 1.

    The mean absolute error is the mean of the absolute values of the errors.
    It counts each error by its size alone. So a single target that is far
    from its prediction changes it less than it changes the squared error.

    PyTorch has both losses built in, as `F.mse_loss` and `F.l1_loss`. Here
    `F` is `torch.nn.functional`, which code usually imports with
    `import torch.nn.functional as F`.

    The data stays fixed while a model trains, so the loss depends only on the
    parameters. Changing the weights changes the predictions, which changes
    the loss.

    Be careful with shapes. Say the predictions have shape `(3, 1)` and the
    targets have shape `(3,)`. Subtracting them broadcasts to shape `(3, 3)`,
    which compares every prediction with every target. Taking the mean of that
    runs without an error, but the number is meaningless. `.squeeze(1)`
    removes dimension 1 when its size is 1, which undoes `.unsqueeze(1)`. So
    it turns shape `(3, 1)` into `(3,)`.
    """,
)

LINREG2_SOURCE = source_file(
    '''
    # linreg2.py

    import torch
    from torch import Tensor

    predictions = torch.tensor([4.5, 7.5, 10.5])
    targets = torch.tensor([5.0, 7.0, 12.0])

    # The mean squared error. Square each error, then take the mean.
    errors = predictions - targets                 # tensor([-0.5000,  0.5000, -1.5000])
    mse = (errors ** 2).mean()                     # tensor(0.9167)


    # TODO: Return the mean squared error between predictions and targets,
    #   which have the same shape.
    #   Example: with the predictions and targets above, it returns
    #   tensor(0.9167).
    def mean_squared_error(
        predictions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the mean absolute error, the mean of the absolute values of
    #   the errors predictions - targets.
    #   Example: with the predictions and targets above, it returns
    #   tensor(0.8333).
    def mean_absolute_error(
        predictions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        ...


    # TODO: Here w has shape (features, 1), so the predictions come out with
    #   shape (examples, 1). targets has shape (examples,). The function runs
    #   without an error, but the loss it returns is wrong. Compare it with the
    #   expected loss in the Result tab, then fix the line that sets
    #   predictions.
    def column_model_loss(
        x: Tensor,
        w: Tensor,
        b: float,
        targets: Tensor,
    ) -> Tensor:
        predictions = x @ w + b

        return ((predictions - targets) ** 2).mean()
    ''',
)

LINREG2_SOLUTION = source_file(
    '''
    # linreg2.py, a solution for comparison

    import torch
    from torch import Tensor

    predictions = torch.tensor([4.5, 7.5, 10.5])
    targets = torch.tensor([5.0, 7.0, 12.0])

    errors = predictions - targets
    mse = (errors ** 2).mean()


    def mean_squared_error(
        predictions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        return ((predictions - targets) ** 2).mean()


    def mean_absolute_error(
        predictions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        return (predictions - targets).abs().mean()


    # Shape (examples, 1) minus shape (examples,) broadcasts to (examples, examples).
    def column_model_loss(
        x: Tensor,
        w: Tensor,
        b: float,
        targets: Tensor,
    ) -> Tensor:
        predictions = (x @ w + b).squeeze(1)

        return ((predictions - targets) ** 2).mean()
    ''',
)


def check_linreg2(checker: Checker) -> None:
    for function, description, reference in (
        ("mean_squared_error", "gives the mean of the squared errors", lambda p, t: ((p - t) ** 2).mean()),
        ("mean_absolute_error", "gives the mean of the absolute errors", lambda p, t: (p - t).abs().mean()),
    ):
        check_on_inputs(
            checker=checker,
            function=function,
            description=description,
            inputs=(
                (small_tensor(11, 4), small_tensor(12, 4)),
                (small_tensor(13, 6), small_tensor(14, 6)),
            ),
            reference=reference,
            expect=expect_close,
        )

    check_on_inputs(
        checker=checker,
        function="column_model_loss",
        description="gives the mean squared error of a model whose weights have shape (features, 1)",
        inputs=(
            (small_tensor(15, 4, 2), small_tensor(16, 2, 1), 0.5, small_tensor(17, 4)),
            (small_tensor(18, 3, 3), small_tensor(19, 3, 1), -1.0, small_tensor(20, 3)),
        ),
        reference=lambda x, w, b, targets: (((x @ w + b).squeeze(1) - targets) ** 2).mean(),
        expect=expect_close,
    )


LINREG2 = Exercise(
    name="linreg2",
    chapter=LINEAR_REGRESSION,
    title="The loss",
    lesson=LINREG2_LESSON,
    source=LINREG2_SOURCE,
    solution=LINREG2_SOLUTION,
    hints={
        "mean_squared_error": "Copy the pattern of the two lines that set errors and mse, using this function's "
        "predictions and targets.",
        "mean_absolute_error": "Compute the errors as in mean_squared_error. Then call .abs() on them instead of "
        "squaring, and take the mean.",
        "column_model_loss": "The predictions have shape (examples, 1) and the targets have shape (examples,). "
        "Subtracting them broadcasts to shape (examples, examples). Remove the size 1 dimension of the "
        "predictions with .squeeze(1).",
    },
    introduces=("loss",),
    check=check_linreg2,
)


# ---------------------------------------------------------------------------
# linreg3
# ---------------------------------------------------------------------------

LINREG3_LESSON = prose(
    """
    For a linear model with the squared error loss, the best parameters can be
    computed directly, without any searching. This method is called **least
    squares**. `torch.linalg.lstsq(A, y).solution` returns one weight per
    column of `A`. Those weights give `A @ weights` the lowest possible
    squared error against `y`.

    `lstsq` has no separate bias. The usual trick is to add a column of ones
    to the data. A weight on a feature that is always 1 adds the same number
    to every prediction, which is exactly what a bias does. So if you add a
    column of ones on the right of `x`, the last weight that `lstsq` returns
    is the bias.

    `torch.cat` joins tensors along a dimension. Along dimension 1, it puts
    their columns side by side. `torch.ones(4, 1)` makes a tensor of ones with
    shape `(4, 1)`. `x.shape[0]` is the number of rows of `x`.

    ```python
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([3.1, 4.9, 7.2, 8.8])
    with_ones = torch.cat([x, torch.ones(4, 1)], dim=1)    # shape (4, 2)
    torch.linalg.lstsq(with_ones, y).solution             # tensor([1.9400, 1.1500])
    ```

    The best line here is `1.94 * x + 1.15`.

    Only a few simple models have an exact formula like this. The models in
    later sections have none. So the next chapter finds parameters by changing
    them step by step in the direction that lowers the loss, like walking
    downhill. On a linear model, least squares gives the answer that this walk
    should reach. So least squares can check whether the walk works.
    """,
)

LINREG3_SOURCE = source_file(
    '''
    # linreg3.py

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([3.1, 4.9, 7.2, 8.8])

    # torch.cat along dimension 1 puts columns side by side, so this adds a
    # column of ones on the right of x.
    with_ones = torch.cat([x, torch.ones(4, 1)], dim=1)    # shape (4, 2)

    # The weights with the lowest squared error. The last weight, the one on the
    # column of ones, is the bias.
    best = torch.linalg.lstsq(with_ones, y).solution        # tensor([1.9400, 1.1500])


    # TODO: Return x with a column of ones added on the right. x can have any
    #   number of rows and columns. For x of shape (rows, columns), the result
    #   has shape (rows, columns + 1).
    #   Example: add_ones_column(x) with the x above returns the same tensor as
    #   with_ones.
    def add_ones_column(x: Tensor) -> Tensor:
        ...


    # TODO: Finish the middle line, so this returns the weights and the bias
    #   that give the lowest mean squared error on x and y, found by least
    #   squares. x has shape (examples, features) and y has shape (examples,).
    #   Example: best_fit(x, y) with the x and y above returns
    #   (tensor([1.9400]), tensor(1.1500)).
    def best_fit(
        x: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        with_ones = add_ones_column(x)
        solution = ...

        return solution[:-1], solution[-1]


    # TODO: Finish the last two lines, so this returns the mean squared error of
    #   the best fit on x and y. Make the predictions the way linreg1 does, and
    #   compute the loss the way linreg2 does.
    #   Example: lowest_loss(x, y) with the x and y above returns about
    #   tensor(0.0205).
    def lowest_loss(
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        w, b = best_fit(x, y)
        predictions = ...

        return ...
    ''',
)

LINREG3_SOLUTION = source_file(
    '''
    # linreg3.py, a solution for comparison

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([3.1, 4.9, 7.2, 8.8])

    with_ones = torch.cat([x, torch.ones(4, 1)], dim=1)

    best = torch.linalg.lstsq(with_ones, y).solution


    def add_ones_column(x: Tensor) -> Tensor:
        return torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)


    def best_fit(
        x: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        with_ones = add_ones_column(x)
        solution = torch.linalg.lstsq(with_ones, y).solution

        return solution[:-1], solution[-1]


    def lowest_loss(
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        w, b = best_fit(x, y)
        predictions = x @ w + b

        return ((predictions - y) ** 2).mean()
    ''',
)


def least_squares(
    x: Tensor,
    y: Tensor,
) -> tuple[Tensor, Tensor]:
    solution = torch.linalg.lstsq(torch.cat([x, torch.ones(x.shape[0], 1)], dim=1), y).solution

    return solution[:-1], solution[-1]


FIT_DATA: tuple[tuple[Tensor, Tensor], ...] = (
    (small_tensor(21, 6, 2), small_tensor(22, 6)),
    (small_tensor(23, 8, 3), small_tensor(24, 8)),
)


def check_linreg3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="add_ones_column",
        description="adds a column of ones on the right",
        inputs=(
            (small_tensor(25, 3, 2),),
            (small_tensor(26, 5, 1),),
        ),
        reference=lambda x: torch.cat([x, torch.ones(x.shape[0], 1)], dim=1),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="best_fit",
        description="gives the weights and bias with the lowest squared error",
        inputs=FIT_DATA,
        reference=least_squares,
        expect=expect_parts("the weights", "the bias"),
    )
    check_on_inputs(
        checker=checker,
        function="lowest_loss",
        description="gives the mean squared error of the best fit",
        inputs=FIT_DATA,
        reference=lambda x, y: ((x @ least_squares(x, y)[0] + least_squares(x, y)[1] - y) ** 2).mean(),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )


LINREG3 = Exercise(
    name="linreg3",
    chapter=LINEAR_REGRESSION,
    title="The best fit, exactly",
    lesson=LINREG3_LESSON,
    source=LINREG3_SOURCE,
    solution=LINREG3_SOLUTION,
    hints={
        "add_ones_column": "Copy the pattern of the line that sets with_ones, using x.shape[0] rows of ones "
        "instead of 4.",
        "best_fit": "Pass with_ones and y to torch.linalg.lstsq, and take .solution from the result.",
        "lowest_loss": "Get w and b from best_fit. Predict with x @ w + b, then take the mean squared error "
        "against y.",
    },
    introduces=("least squares",),
    check=check_linreg3,
)
