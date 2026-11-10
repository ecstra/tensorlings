"""11_softmax_regression, scores to probabilities, keeping softmax from overflowing, and a model that picks a class."""

from __future__ import annotations

import torch
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import Checker, check_on_inputs, expect_close, expect_tensor, small_tensor
from ..text import prose, source_file

SOFTMAX_REGRESSION = Chapter(
    folder="11_softmax_regression",
    title="Softmax regression",
    readme=source_file(
        """
        # Softmax regression

        Many tasks ask a model to choose one answer from a fixed set of options.
        This chapter turns a model's scores into probabilities with softmax. It
        shows how to keep softmax from overflowing when the scores are big. Then
        it builds softmax regression, a linear model that predicts which option,
        or class, an example belongs to.

        ## Further reading

        - [Dive into Deep Learning, softmax regression](https://d2l.ai/chapter_linear-classification/softmax-regression.html)
        - [CS231n, linear classification](https://cs231n.github.io/linear-classify/)
        """,
    ),
)


# ---------------------------------------------------------------------------
# softmax1
# ---------------------------------------------------------------------------

SOFTMAX1_LESSON = prose(
    """
    Regression predicts a number. Often the answer is instead one choice from a
    fixed set of options, like which of the ten digits an image shows. Each
    option is called a **class**. This kind of class has nothing to do with a
    Python class.

    The model outputs one score per class. A raw score like this, which can be
    any number, is called a **logit**. To turn the scores into probabilities,
    first take `exp` of each score, which always gives a positive number. Then
    divide each result by the sum of all of them, so that they add up to 1.

    ```python
    scores = torch.tensor([2.0, 1.0, 0.1])
    exps = torch.exp(scores)    # tensor([7.3891, 2.7183, 1.1052])
    exps / exps.sum()           # tensor([0.6590, 0.2424, 0.0986])
    ```

    This function is called **softmax**. Its output is a **probability
    distribution**, which means one number per class, each between 0 and 1,
    all adding up to 1. A higher score always gets a higher probability, so
    the largest score gets the largest probability.

    Adding the same number `c` to every score does not change the softmax.
    Each `exp(s + c)` equals `exp(s) * exp(c)`, so every exp gets multiplied by
    the same factor `exp(c)`. Dividing by the sum cancels that factor.

    For a batch of examples, the scores form a matrix with one row per example
    and one column per class. Softmax applies to each row separately, so each
    row must be divided by its own sum. Summing with `keepdim=True`, as in
    reductions1, gives the sums a shape that broadcasting lines up with the
    rows.
    """,
)

SOFTMAX1_SOURCE = source_file(
    '''
    # softmax1.py

    import torch
    from torch import Tensor

    scores = torch.tensor([2.0, 1.0, 0.1])

    # exp makes every score positive, and dividing by the sum makes them add up to 1.
    exps = torch.exp(scores)                          # tensor([7.3891, 2.7183, 1.1052])
    probabilities = exps / exps.sum()                 # tensor([0.6590, 0.2424, 0.0986])


    # TODO: Return the softmax of each row of scores, so every row of the result
    #   adds up to 1. scores is a matrix of shape (examples, classes), and the
    #   result has the same shape.
    #   Example: softmax_rows(torch.tensor([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]]))
    #   returns tensor([[0.6590, 0.2424, 0.0986], [0.3333, 0.3333, 0.3333]]).
    def softmax_rows(scores: Tensor) -> Tensor:
        ...


    # TODO: Return the most likely class for each row of scores, as a tensor with
    #   one class number per row. The most likely class is the one with the
    #   largest probability.
    #   Example: most_likely(torch.tensor([[2.0, 1.0, 0.1], [0.0, 3.0, 1.0]]))
    #   returns tensor([0, 1]).
    def most_likely(scores: Tensor) -> Tensor:
        ...
    ''',
)

SOFTMAX1_SOLUTION = source_file(
    '''
    # softmax1.py, a solution for comparison

    import torch
    from torch import Tensor

    scores = torch.tensor([2.0, 1.0, 0.1])

    exps = torch.exp(scores)
    probabilities = exps / exps.sum()


    def softmax_rows(scores: Tensor) -> Tensor:
        exps = torch.exp(scores)

        return exps / exps.sum(dim=1, keepdim=True)


    # The largest score has the largest probability, so softmax isn't needed here.
    def most_likely(scores: Tensor) -> Tensor:
        return scores.argmax(dim=1)
    ''',
)

SCORE_ROWS: tuple[tuple[Tensor], ...] = (
    (small_tensor(92, 2, 3),),
    (small_tensor(93, 4, 4),),
)


def check_softmax1(checker: Checker) -> None:
    # The square scores are where adding up the columns instead of the rows still runs.
    check_on_inputs(
        checker=checker,
        function="softmax_rows",
        description="gives each row its own softmax",
        inputs=SCORE_ROWS,
        reference=lambda scores: torch.softmax(scores, dim=1),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="most_likely",
        description="gives the class with the largest score in each row",
        inputs=(
            (torch.tensor([[2.0, 1.0, 0.1], [0.0, 3.0, 1.0]]),),
            (torch.tensor([[1.0, 4.0, 2.0, 0.0], [5.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [2.0, 2.5, 1.0, 0.0]]),),
        ),
        reference=lambda scores: scores.argmax(dim=1),
        expect=expect_tensor,
    )


SOFTMAX1 = Exercise(
    name="softmax1",
    chapter=SOFTMAX_REGRESSION,
    title="From scores to probabilities",
    lesson=SOFTMAX1_LESSON,
    source=SOFTMAX1_SOURCE,
    solution=SOFTMAX1_SOLUTION,
    hints={
        "softmax_rows": "Copy the pattern of the two lines that set exps and probabilities, but divide each row by "
        "its own sum. Sum along dim=1 with keepdim=True, so the sums have shape (rows, 1) and line up with the rows.",
        "most_likely": "The largest score gets the largest probability, so argmax of the scores along each row "
        "gives the class.",
    },
    introduces=("logit", "softmax", "probability distribution"),
    check=check_softmax1,
)


# ---------------------------------------------------------------------------
# softmax2
# ---------------------------------------------------------------------------

SOFTMAX2_LESSON = prose(
    """
    `exp` grows very fast. Unless you ask for another type, tensors hold
    float32 numbers, which keep about 7 significant digits, as micrograd3
    said. The largest float32 number is about `3.4e38`. `exp(89)` is already
    bigger than that, so it comes out as `inf`. A result too big for its type
    like this is called an overflow. Then `inf / inf` gives `nan`.

    ```python
    scores = torch.tensor([1000.0, 999.0])
    torch.exp(scores)                                # tensor([inf, inf])
    torch.exp(scores) / torch.exp(scores).sum()      # tensor([nan, nan])
    ```

    Scores that big do happen, for example when the features are not
    standardized or the learning rate is too high. The fix uses a fact from
    softmax1. Subtracting the same number from every score does not change the
    softmax. So subtract the largest score from every score first. Then the
    largest score becomes 0, every exp is at most 1, and nothing overflows.

    ```python
    shifted = scores - scores.max()                  # tensor([ 0., -1.])
    torch.exp(shifted) / torch.exp(shifted).sum()    # tensor([0.7311, 0.2689])
    ```

    For a matrix, subtract each row's own largest score from that row.
    `scores.amax(dim=1, keepdim=True)` gives the largest score of each row,
    with shape `(rows, 1)`, so broadcasting lines it up with the rows.

    PyTorch's own `torch.softmax(scores, dim=1)` subtracts the largest score
    for you. Every library does this, because a single `nan` spreads into
    every later step of training.
    """,
)

SOFTMAX2_SOURCE = source_file(
    '''
    # softmax2.py

    import torch
    from torch import Tensor

    scores = torch.tensor([1000.0, 999.0])

    # Subtracting the largest score first keeps every exp at most 1.
    shifted = scores - scores.max()                       # tensor([ 0., -1.])
    probabilities = torch.exp(shifted) / torch.exp(shifted).sum()    # tensor([0.7311, 0.2689])


    # TODO: This returns nan for rows with big scores. Fix it by subtracting each
    #   row's largest score from that row before the exp, as the lesson shows.
    def softmax_rows(scores: Tensor) -> Tensor:
        exps = torch.exp(scores)

        return exps / exps.sum(dim=1, keepdim=True)


    # TODO: Return how confident the model is for each row of scores. That is the
    #   probability it gives to its most likely class, so the result has one
    #   number per row. torch.softmax(scores, dim=1) gives the probabilities.
    #   Example: confidence(torch.tensor([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]]))
    #   returns tensor([0.6590, 0.3333]).
    def confidence(scores: Tensor) -> Tensor:
        ...
    ''',
)

SOFTMAX2_SOLUTION = source_file(
    '''
    # softmax2.py, a solution for comparison

    import torch
    from torch import Tensor

    scores = torch.tensor([1000.0, 999.0])

    shifted = scores - scores.max()
    probabilities = torch.exp(shifted) / torch.exp(shifted).sum()


    # keepdim=True keeps each row's largest score in shape (rows, 1), so it lines up with its row.
    def softmax_rows(scores: Tensor) -> Tensor:
        exps = torch.exp(scores - scores.amax(dim=1, keepdim=True))

        return exps / exps.sum(dim=1, keepdim=True)


    def confidence(scores: Tensor) -> Tensor:
        return torch.softmax(scores, dim=1).amax(dim=1)
    ''',
)


def check_softmax2(checker: Checker) -> None:
    # Each row has its own scale, so taking one largest from the whole matrix still overflows. The last rows spread
    # past 88, so taking away a row's smallest score, its mean or its first entry overflows too.
    check_on_inputs(
        checker=checker,
        function="softmax_rows",
        description="gives each row its softmax, even for big scores",
        inputs=(
            (torch.tensor([[1000.0, 999.0, 998.0], [1.0, 2.0, 3.0]]),),
            (torch.tensor([[-500.0, -501.0], [800.0, 795.0]]),),
            (small_tensor(96, 3, 3),),
            (torch.tensor([[1000.0, 0.0], [0.0, 1000.0]]),),
        ),
        reference=lambda scores: torch.softmax(scores, dim=1),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="confidence",
        description="gives the probability of each row's most likely class",
        inputs=(
            (torch.tensor([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]]),),
            (small_tensor(97, 4, 3),),
        ),
        reference=lambda scores: torch.softmax(scores, dim=1).amax(dim=1),
        expect=expect_tensor,
    )


SOFTMAX2 = Exercise(
    name="softmax2",
    chapter=SOFTMAX_REGRESSION,
    title="Softmax without overflow",
    lesson=SOFTMAX2_LESSON,
    source=SOFTMAX2_SOURCE,
    solution=SOFTMAX2_SOLUTION,
    hints={
        "softmax_rows": "Subtract each row's largest score before the exp. scores.amax(dim=1, keepdim=True) gives "
        "those largest scores in a shape that lines up with the rows.",
        "confidence": "Compute the probabilities, then take the largest one in each row with .amax(dim=1).",
    },
    introduces=(),
    check=check_softmax2,
)


# ---------------------------------------------------------------------------
# softmax3
# ---------------------------------------------------------------------------

SOFTMAX3_LESSON = prose(
    """
    Softmax regression is linear regression with one output per class,
    followed by softmax. The weight matrix `W` has shape `(features, classes)`,
    and the bias `b` has one number per class. So `x @ W + b` gives one row of
    class scores per example.

    ```python
    x = torch.tensor([[1.0, 2.0],
                      [3.0, -1.0]])                # 2 examples, 2 features
    W = torch.tensor([[1.0, -1.0, 0.0],
                      [0.5, 0.0, 1.0]])            # 2 features, 3 classes
    b = torch.tensor([0.0, 0.5, -0.5])
    scores = x @ W + b    # tensor([[ 2.0, -0.5,  1.5], [ 2.5, -2.5, -1.5]])
    ```

    Each example's correct class is called its **label**, a whole number from
    0 up to the number of classes minus 1. The model's predicted class is the
    one with the largest score. **Accuracy** is the fraction of examples whose
    predicted class equals the label.

    `==` between two tensors compares them entry by entry, and gives a tensor
    of `True` and `False` values. `.mean()` raises an error on `True` and
    `False` values. `.float()` turns them into 1.0 and 0.0, and the mean of
    those is the fraction that were `True`.

    ```python
    predicted = torch.tensor([2, 0, 1, 1])
    (predicted == 1).float().mean()    # tensor(0.5000), half were predicted as class 1
    ```

    People report accuracy, but training can't use it as the loss. A small
    change in the weights rarely changes any predicted class, so accuracy
    usually stays exactly the same. That makes its gradient 0 almost
    everywhere, which gives gradient descent no direction to follow. The next
    chapter builds the loss that training uses instead.
    """,
)

SOFTMAX3_SOURCE = source_file(
    '''
    # softmax3.py

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0, 2.0],
                      [3.0, -1.0]])                    # 2 examples, 2 features
    W = torch.tensor([[1.0, -1.0, 0.0],
                      [0.5, 0.0, 1.0]])                # 2 features, 3 classes
    b = torch.tensor([0.0, 0.5, -0.5])

    # One row of class scores per example.
    scores = x @ W + b                                 # tensor([[2.0, -0.5, 1.5], [2.5, -2.5, -1.5]])


    # TODO: Return the probability of every class for every example, as a tensor
    #   of shape (examples, classes). x has shape (examples, features), W has
    #   shape (features, classes) and b has shape (classes,).
    #   torch.softmax(scores, dim=1) gives the softmax of each row of scores.
    #   Example: class_probabilities(x, W, b), with the x, W and b above,
    #   returns tensor([[0.5922, 0.0486, 0.3592], [0.9756, 0.0066, 0.0179]]).
    def class_probabilities(
        x: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the predicted class of every example, which is the class with
    #   the largest score. The result has one class number per example.
    #   Example: predict_classes(x, W, b), with the x, W and b above, returns
    #   tensor([0, 0]).
    def predict_classes(
        x: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> Tensor:
        ...


    # TODO: This should return the fraction of examples whose predicted class
    #   equals their label, as a Python float, but it crashes. Read the error,
    #   then fix it.
    #   Example: accuracy(torch.tensor([2, 0, 1, 1]), torch.tensor([2, 1, 1, 0]))
    #   should return 0.5.
    def accuracy(
        predicted: Tensor,
        labels: Tensor,
    ) -> float:
        return (predicted == labels).mean().item()
    ''',
)

SOFTMAX3_SOLUTION = source_file(
    '''
    # softmax3.py, a solution for comparison

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0, 2.0],
                      [3.0, -1.0]])
    W = torch.tensor([[1.0, -1.0, 0.0],
                      [0.5, 0.0, 1.0]])
    b = torch.tensor([0.0, 0.5, -0.5])

    scores = x @ W + b


    def class_probabilities(
        x: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> Tensor:
        return torch.softmax(x @ W + b, dim=1)


    def predict_classes(
        x: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> Tensor:
        return (x @ W + b).argmax(dim=1)


    # True and False need to become 1.0 and 0.0 before their mean.
    def accuracy(
        predicted: Tensor,
        labels: Tensor,
    ) -> float:
        return (predicted == labels).float().mean().item()
    ''',
)

MODELS: tuple[tuple[Tensor, Tensor, Tensor], ...] = (
    (small_tensor(98, 4, 2), small_tensor(99, 2, 3), small_tensor(100, 3)),
    (small_tensor(101, 3, 3), small_tensor(102, 3, 3), small_tensor(103, 3)),
)


def check_softmax3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="class_probabilities",
        description="gives every class's probability for every example",
        inputs=MODELS,
        reference=lambda x, W, b: torch.softmax(x @ W + b, dim=1),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="predict_classes",
        description="gives the class with the largest score for every example",
        inputs=MODELS,
        reference=lambda x, W, b: (x @ W + b).argmax(dim=1),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="accuracy",
        description="gives the fraction of predictions that match their labels",
        inputs=(
            (torch.tensor([2, 0, 1, 1]), torch.tensor([2, 1, 1, 0])),
            (torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 0, 4])),
        ),
        reference=lambda predicted, labels: (predicted == labels).float().mean().item(),
        expect=expect_close,
    )


SOFTMAX3 = Exercise(
    name="softmax3",
    chapter=SOFTMAX_REGRESSION,
    title="Softmax regression",
    lesson=SOFTMAX3_LESSON,
    source=SOFTMAX3_SOURCE,
    solution=SOFTMAX3_SOLUTION,
    hints={
        "class_probabilities": "Compute the scores the way the line that sets scores does, then pass them to "
        "torch.softmax with dim=1.",
        "predict_classes": "The largest score has the largest probability, so take argmax of the scores along each "
        "row, as most_likely does in softmax1.",
        "accuracy": "The error says .mean() can't work on True and False. Turn them into 1.0 and 0.0 with .float() "
        "before the mean.",
    },
    introduces=("label",),
    check=check_softmax3,
)
