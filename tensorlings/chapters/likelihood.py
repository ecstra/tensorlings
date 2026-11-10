"""12_likelihood, losses as probability, the likelihood, cross entropy, squared error as a likelihood, and its gradient."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import Checker, Described, check_on_inputs, expect_close, expect_parts, expect_tensor, small_tensor
from ..text import prose, source_file
from .tensor_backprop import autograd_backward
from .tensors import pinned_threads

LIKELIHOOD = Chapter(
    folder="12_likelihood",
    title="Losses as probability",
    readme=source_file(
        """
        # Losses as probability

        Where do loss functions come from? This chapter shows that the usual
        losses all come from one idea. A good model should give a high
        probability to the correct answers in the data. Applied to classes, this
        idea gives cross entropy. Applied to numbers, it gives the squared error.
        The chapter ends by working out the gradient of cross entropy by hand.

        ## Further reading

        - [Dive into Deep Learning, the softmax loss](https://d2l.ai/chapter_linear-classification/softmax-regression.html#loss-function)
        - [Karpathy, building makemore, the negative log likelihood](https://www.youtube.com/watch?v=PaCmpygFfXo)
        - [Bishop, Pattern Recognition and Machine Learning, section 1.2.5](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
        """,
    ),
)


def pick_labels(
    probabilities: Tensor,
    labels: Tensor,
) -> Tensor:
    return probabilities[torch.arange(len(labels)), labels]


def probability_rows(
    seed: int,
    rows: int,
    classes: int,
) -> Tensor:
    return torch.softmax(small_tensor(seed, rows, classes), dim=1)


def label_tensor(
    seed: int,
    rows: int,
    classes: int,
) -> Tensor:
    return torch.randint(0, classes, (rows,), generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# likelihood1
# ---------------------------------------------------------------------------

LIKELIHOOD1_LESSON = prose(
    """
    A softmax model gives each example a probability for every class,
    including its correct label. Say it gives probabilities 0.7, 0.9 and 0.2
    to the correct labels of three examples. Their product, 0.126, is the
    probability that the model gives to all three labels together. That
    product is called the **likelihood** of the labels. A better model gives
    the labels a higher likelihood.

    Training could try to increase the likelihood directly. The problem is
    that a product of thousands of probabilities is so small that a float
    rounds it to 0.

    ```python
    torch.full((1000,), 0.5).prod()            # tensor(0.)
    torch.log(torch.full((1000,), 0.5)).sum()  # tensor(-693.1473)
    ```

    `torch.full((1000,), 0.5)` makes a tensor of 1000 entries that are all
    0.5, and `.prod()` multiplies all the entries together. 0.5 to the power
    1000 is far smaller than the smallest positive float32, the number type a
    tensor holds. So the product comes out as 0, which is called underflow.
    When every likelihood is 0, you can't tell a good model from a bad one.

    Taking the log fixes this. The log of a product is the sum of the logs,
    `log(a * b) = log(a) + log(b)`. The log of a division is the difference of
    the logs, so `math.log(12 / 3)` equals `math.log(12) - math.log(3)`. The
    log of a probability is negative. A sum of thousands of them is a large
    negative number, which a float stores without trouble.

    The log increases when its input increases. So the model with the higher
    likelihood also has the higher log likelihood, which is the sum of the
    logs. Training lowers a loss, so the loss flips the sign. The loss is the
    negative log likelihood divided by the number of examples.

    To pick each row's probability for its label, index with two tensors. The
    first holds the row position of each example, and the second holds the
    column position. `torch.arange(n)` gives the row positions 0 to `n - 1`.

    ```python
    probabilities = torch.tensor([[0.7, 0.2, 0.1],
                                  [0.05, 0.9, 0.05],
                                  [0.5, 0.3, 0.2]])
    labels = torch.tensor([0, 1, 2])
    probabilities[torch.arange(3), labels]     # tensor([0.7000, 0.9000, 0.2000])
    ```
    """,
)

LIKELIHOOD1_SOURCE = source_file(
    '''
    # likelihood1.py

    import torch
    from torch import Tensor

    probabilities = torch.tensor([[0.7, 0.2, 0.1],
                                  [0.05, 0.9, 0.05],
                                  [0.5, 0.3, 0.2]])
    labels = torch.tensor([0, 1, 2])

    # Indexing with a tensor of row positions and a tensor of column positions
    # picks one entry per row, here each row's probability for its label.
    picked = probabilities[torch.arange(3), labels]    # tensor([0.7, 0.9, 0.2])


    # TODO: Return each row's probability for its label, for any number of rows.
    #   probabilities has shape (examples, classes), labels has shape
    #   (examples,), and the result has shape (examples,).
    #   Example: label_probabilities(probabilities, labels), with the tensors
    #   above, returns tensor([0.7000, 0.9000, 0.2000]).
    def label_probabilities(
        probabilities: Tensor,
        labels: Tensor,
    ) -> Tensor:
        ...


    # TODO: This should return the log likelihood of the labels, the log of the
    #   product of each label's probability. For a few hundred examples it
    #   returns -inf instead. Read the lesson, then fix the last line.
    def log_likelihood(
        probabilities: Tensor,
        labels: Tensor,
    ) -> Tensor:
        picked = probabilities[torch.arange(len(labels)), labels]

        return torch.log(picked.prod())


    # TODO: Return the negative log likelihood loss. Take the log of each label's
    #   probability, put a minus sign in front, and average over the examples.
    #   Example: negative_log_likelihood(probabilities, labels), with the
    #   tensors above, returns about tensor(0.6905).
    def negative_log_likelihood(
        probabilities: Tensor,
        labels: Tensor,
    ) -> Tensor:
        ...
    ''',
)

LIKELIHOOD1_SOLUTION = source_file(
    '''
    # likelihood1.py, a solution for comparison

    import torch
    from torch import Tensor

    probabilities = torch.tensor([[0.7, 0.2, 0.1],
                                  [0.05, 0.9, 0.05],
                                  [0.5, 0.3, 0.2]])
    labels = torch.tensor([0, 1, 2])

    picked = probabilities[torch.arange(3), labels]


    def label_probabilities(
        probabilities: Tensor,
        labels: Tensor,
    ) -> Tensor:
        return probabilities[torch.arange(len(labels)), labels]


    # The log of a product is the sum of the logs, and the sum doesn't underflow.
    def log_likelihood(
        probabilities: Tensor,
        labels: Tensor,
    ) -> Tensor:
        picked = probabilities[torch.arange(len(labels)), labels]

        return torch.log(picked).sum()


    def negative_log_likelihood(
        probabilities: Tensor,
        labels: Tensor,
    ) -> Tensor:
        return -torch.log(label_probabilities(probabilities, labels)).mean()
    ''',
)

MANY_ROWS = (
    Described(probability_rows(104, 400, 5), "probabilities with 400 rows"),
    Described(label_tensor(105, 400, 5), "labels"),
)


def check_likelihood1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="label_probabilities",
        description="picks each row's probability for its label",
        inputs=(
            (probability_rows(106, 4, 3), label_tensor(107, 4, 3)),
            (probability_rows(108, 5, 5), label_tensor(111, 5, 5)),
        ),
        reference=pick_labels,
        expect=expect_tensor,
    )
    # 400 rows is enough for the product to reach 0.
    check_on_inputs(
        checker=checker,
        function="log_likelihood",
        description="adds up the logs, so many examples don't give -inf",
        inputs=(
            (probability_rows(110, 4, 3), label_tensor(111, 4, 3)),
            MANY_ROWS,
        ),
        reference=lambda probabilities, labels: torch.log(pick_labels(probabilities, labels)).sum(),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-3,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="negative_log_likelihood",
        description="gives the mean of minus the log of each label's probability",
        inputs=(
            (probability_rows(112, 4, 3), label_tensor(113, 4, 3)),
            MANY_ROWS,
        ),
        reference=lambda probabilities, labels: -torch.log(pick_labels(probabilities, labels)).mean(),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )


LIKELIHOOD1 = Exercise(
    name="likelihood1",
    chapter=LIKELIHOOD,
    title="The likelihood of the labels",
    lesson=LIKELIHOOD1_LESSON,
    source=LIKELIHOOD1_SOURCE,
    solution=LIKELIHOOD1_SOLUTION,
    hints={
        "label_probabilities": "Copy the pattern of the line that sets picked, using torch.arange(len(labels)) "
        "instead of torch.arange(3).",
        "log_likelihood": "The product of hundreds of probabilities is 0, and log(0) is -inf. Take the log of each "
        "probability first, then add the logs up.",
        "negative_log_likelihood": "Take the log of each label's probability, then the mean, then flip the sign.",
    },
    introduces=("likelihood",),
    check=check_likelihood1,
)


# ---------------------------------------------------------------------------
# likelihood2
# ---------------------------------------------------------------------------

LIKELIHOOD2_LESSON = prose(
    """
    The negative log likelihood of softmax probabilities, averaged over the
    examples, is called **cross entropy**. It is the standard loss for models
    that pick a class.

    Computing softmax first and then taking the log loses precision. A
    probability smaller than the smallest float32 rounds to 0, and its log is
    `-inf`. The true log is an ordinary number, like -120 in this example.

    ```python
    scores = torch.tensor([[60.0, 0.0, -60.0]])
    torch.log(torch.softmax(scores, dim=1))    # tensor([[  0., -60., -inf]]), the last should be -120
    ```

    So work with logs from the start. Softmax is `exp(score) / total`, where
    `total` is the sum of the exps of all the scores in the row. By the rule
    for the log of a division, the log of softmax is `score - log(total)`.
    `torch.logsumexp` computes `log(total)`, the log of a sum of exps. It
    subtracts the largest score before taking the exps, as softmax2 did, so it
    never overflows.

    ```python
    torch.logsumexp(scores, dim=1, keepdim=True)    # tensor([[60.]])
    ```

    PyTorch's `F.cross_entropy(logits, labels)` takes the raw scores and the
    labels, and does all of these steps. This is why models output logits and
    leave the softmax to the loss function.

    A model that has learned nothing gives every class the same score, so
    every class gets probability `1 / classes`. With 10 classes, its cross
    entropy is 2.303. A newly created model should start with a loss near that
    number. A starting loss far above it means the model is confidently wrong
    before it has learned anything.
    """,
)

LIKELIHOOD2_SOURCE = source_file(
    '''
    # likelihood2.py

    import math

    import torch
    from torch import Tensor

    scores = torch.tensor([[60.0, 0.0, -60.0]])

    # The log of the sum of the exps in each row. It subtracts the largest score
    # first, so nothing overflows.
    log_total = torch.logsumexp(scores, dim=1, keepdim=True)     # tensor([[60.]])


    # TODO: Return the log of the softmax of each row of scores. Compute it as
    #   each score minus its row's log_total, so very small probabilities get
    #   their correct logs instead of -inf. The result has the same shape as
    #   scores, (examples, classes).
    #   Example: log_softmax_rows(torch.tensor([[60.0, 0.0, -60.0]])) returns
    #   tensor([[   0.,  -60., -120.]]).
    def log_softmax_rows(scores: Tensor) -> Tensor:
        ...


    # TODO: Finish the two lines, so this returns the cross entropy of logits
    #   against labels. That is the mean over the rows of minus each row's log
    #   probability for its label. The result should match PyTorch's
    #   F.cross_entropy.
    #   Example: for logits = torch.tensor([[2.0, 1.0, 0.1], [0.0, 3.0, 1.0]])
    #   and labels = torch.tensor([0, 2]), cross_entropy returns about
    #   tensor(1.2934).
    def cross_entropy(
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        log_probabilities = log_softmax_rows(logits)
        picked = ...

        return ...


    # TODO: Return the cross entropy of a model that gives every class the same
    #   score, for the given number of classes, as a Python float.
    #   Example: starting_loss(10) returns about 2.303.
    def starting_loss(classes: int) -> float:
        ...
    ''',
)

LIKELIHOOD2_SOLUTION = source_file(
    '''
    # likelihood2.py, a solution for comparison

    import math

    import torch
    from torch import Tensor

    scores = torch.tensor([[60.0, 0.0, -60.0]])

    log_total = torch.logsumexp(scores, dim=1, keepdim=True)


    def log_softmax_rows(scores: Tensor) -> Tensor:
        return scores - torch.logsumexp(scores, dim=1, keepdim=True)


    def cross_entropy(
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        log_probabilities = log_softmax_rows(logits)
        picked = log_probabilities[torch.arange(len(labels)), labels]

        return -picked.mean()


    # Every class gets 1 / classes, and minus its log is log(classes).
    def starting_loss(classes: int) -> float:
        return math.log(classes)
    ''',
)


def check_likelihood2(checker: Checker) -> None:
    # Scores far apart give probabilities that round to 0, where softmax then log gives -inf.
    check_on_inputs(
        checker=checker,
        function="log_softmax_rows",
        description="gives the log of each row's softmax, even for scores far apart",
        inputs=(
            (torch.tensor([[60.0, 0.0, -60.0], [1.0, 2.0, 3.0]]),),
            (small_tensor(114, 3, 4),),
            (torch.tensor([[1000.0, 0.0], [-1000.0, 0.0]]),),
        ),
        reference=lambda scores: torch.log_softmax(scores, dim=1),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="cross_entropy",
        description="matches F.cross_entropy",
        inputs=(
            (small_tensor(115, 4, 3), label_tensor(116, 4, 3)),
            (small_tensor(117, 6, 5) * 20, label_tensor(118, 6, 5)),
        ),
        reference=lambda logits, labels: F.cross_entropy(logits, labels),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-4,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="starting_loss",
        description="gives the cross entropy of equal scores",
        inputs=(
            (10,),
            (2,),
            (27,),
        ),
        reference=lambda classes: math.log(classes),
        expect=expect_close,
    )


LIKELIHOOD2 = Exercise(
    name="likelihood2",
    chapter=LIKELIHOOD,
    title="Cross entropy from logits",
    lesson=LIKELIHOOD2_LESSON,
    source=LIKELIHOOD2_SOURCE,
    solution=LIKELIHOOD2_SOLUTION,
    hints={
        "log_softmax_rows": "Compute torch.logsumexp of the scores along dim=1 with keepdim=True, as the line that "
        "sets log_total does. Then subtract it from the scores.",
        "cross_entropy": "Pick each row's log probability for its label, the way likelihood1 picks probabilities. "
        "The loss is minus their mean.",
        "starting_loss": "Every class gets probability 1 / classes, so the loss is minus the log of that. "
        "math.log gives the log of a plain number.",
    },
    introduces=("cross entropy",),
    check=check_likelihood2,
)


# ---------------------------------------------------------------------------
# likelihood3
# ---------------------------------------------------------------------------

LIKELIHOOD3_LESSON = prose(
    """
    The squared error also comes from a likelihood. Suppose each target is the
    prediction plus random noise. Suppose also that the noise follows a
    Gaussian distribution, the bell curve, with standard deviation `sigma`. A
    target here can be any real number, so it gets a probability density
    instead of a probability. A density is higher where targets are more
    likely. The density of seeing the target `y` is

    ```text
    exp(-(y - prediction) ** 2 / (2 * sigma ** 2)) / (sigma * sqrt(2 * pi))
    ```

    Using the rules for the log of a product and of a division, and
    `log(exp(u)) = u`, its negative log is

    ```text
    (y - prediction) ** 2 / (2 * sigma ** 2) + log(sigma * sqrt(2 * pi))
    ```

    `math.pi` is the number pi, and `math.sqrt` takes the square root of a
    plain number. For one target of 3.0, a prediction of 2.5 and `sigma` of 1,
    the negative log is 1.0439.

    If `sigma` is fixed, the second term is the same for every model. The
    first term is the squared error divided by the constant `2 * sigma ** 2`.
    So the model with the lowest squared error has the highest likelihood
    under Gaussian noise. The mean squared error is the negative log
    likelihood with the constants dropped. So choosing the squared error as
    the loss amounts to assuming that the errors are Gaussian.

    Other kinds of noise give other losses. A Laplace distribution with scale
    `b` has density `exp(-abs(y - prediction) / b) / (2 * b)`. Its negative
    log is the absolute error divided by `b`, plus `log(2 * b)`. A Laplace
    distribution gives large errors more probability than a Gaussian does. So
    a target far from the others, called an outlier, raises the absolute error
    much less than it raises the squared error.

    The table below fits a line by gradient descent to 20 points whose true
    slope is 2, once with each loss. The noise is Gaussian with standard
    deviation 0.3. In the second row, one target is also raised by 30. Each
    entry is the distance between the fitted slope and 2, the median over 10
    runs.

    | Targets | Squared error | Absolute error |
    |---|---|---|
    | Gaussian noise only | 0.032 | 0.045 |
    | one target raised by 30 | 1.36 | 0.083 |

    With Gaussian noise only, the squared error gives the closer slope in 7 of
    10 runs, as its likelihood says it should. With the outlier, the squared
    error's slope moves to about 3.4. The absolute error's slope barely
    changes.
    """,
)

LIKELIHOOD3_SOURCE = source_file(
    '''
    # likelihood3.py

    import math

    import torch
    from torch import Tensor

    # The Gaussian negative log likelihood for one target of 3.0, a prediction
    # of 2.5 and sigma of 1.
    one_target = (3.0 - 2.5) ** 2 / (2 * 1.0 ** 2) + math.log(1.0 * math.sqrt(2 * math.pi))    # 1.0439


    # TODO: Return the Gaussian negative log likelihood from the lesson, for
    #   noise with standard deviation sigma, averaged over the examples.
    #   predictions and targets are 1D tensors of the same length.
    #   Example: gaussian_nll(torch.tensor([2.5]), torch.tensor([3.0]), 1.0)
    #   returns about tensor(1.0439).
    def gaussian_nll(
        predictions: Tensor,
        targets: Tensor,
        sigma: float,
    ) -> Tensor:
        ...


    # TODO: Return the mean Gaussian negative log likelihood when all you know is
    #   mse, the mean squared error of the predictions, and sigma.
    #   Example: nll_from_mse(0.25, 1.0) returns about 1.0439.
    def nll_from_mse(
        mse: float,
        sigma: float,
    ) -> float:
        ...


    # TODO: Return the mean over the examples of the Laplace negative log
    #   likelihood from the lesson, for Laplace noise with scale b.
    #   Example: laplace_nll(torch.tensor([2.5]), torch.tensor([3.0]), 1.0)
    #   returns about tensor(1.1931).
    def laplace_nll(
        predictions: Tensor,
        targets: Tensor,
        b: float,
    ) -> Tensor:
        ...
    ''',
)

LIKELIHOOD3_SOLUTION = source_file(
    '''
    # likelihood3.py, a solution for comparison

    import math

    import torch
    from torch import Tensor

    one_target = (3.0 - 2.5) ** 2 / (2 * 1.0 ** 2) + math.log(1.0 * math.sqrt(2 * math.pi))


    def gaussian_nll(
        predictions: Tensor,
        targets: Tensor,
        sigma: float,
    ) -> Tensor:
        squared = (targets - predictions) ** 2 / (2 * sigma ** 2)

        return (squared + math.log(sigma * math.sqrt(2 * math.pi))).mean()


    def nll_from_mse(
        mse: float,
        sigma: float,
    ) -> float:
        return mse / (2 * sigma ** 2) + math.log(sigma * math.sqrt(2 * math.pi))


    def laplace_nll(
        predictions: Tensor,
        targets: Tensor,
        b: float,
    ) -> Tensor:
        return ((targets - predictions).abs() / b + math.log(2 * b)).mean()
    ''',
)

NOISE_PAIRS = (
    (small_tensor(119, 5), small_tensor(120, 5)),
    (small_tensor(121, 8), small_tensor(122, 8)),
)


def check_likelihood3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="gaussian_nll",
        description="gives the mean Gaussian negative log likelihood",
        inputs=(
            (*NOISE_PAIRS[0], 1.0),
            (*NOISE_PAIRS[1], 0.5),
        ),
        reference=lambda predictions, targets, sigma: -torch.distributions.Normal(predictions, sigma).log_prob(targets).mean(),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-4,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="nll_from_mse",
        description="gives the Gaussian negative log likelihood from the squared error",
        inputs=(
            (2.5, 1.0),
            (0.3, 0.2),
            (4.0, 3.0),
        ),
        reference=lambda mse, sigma: mse / (2 * sigma ** 2) + math.log(sigma * math.sqrt(2 * math.pi)),
        expect=expect_close,
    )
    # At b of 1 and 2, math.log(2) * b happens to equal math.log(2 * b), so b of 0.5 tells them apart.
    check_on_inputs(
        checker=checker,
        function="laplace_nll",
        description="gives the mean Laplace negative log likelihood",
        inputs=(
            (*NOISE_PAIRS[0], 1.0),
            (*NOISE_PAIRS[1], 2.0),
            (*NOISE_PAIRS[0], 0.5),
        ),
        reference=lambda predictions, targets, b: -torch.distributions.Laplace(predictions, b).log_prob(targets).mean(),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-4,
        ),
    )


LIKELIHOOD3 = Exercise(
    name="likelihood3",
    chapter=LIKELIHOOD,
    title="Squared error is a likelihood",
    lesson=LIKELIHOOD3_LESSON,
    source=LIKELIHOOD3_SOURCE,
    solution=LIKELIHOOD3_SOLUTION,
    hints={
        "gaussian_nll": "Write the negative log from the lesson for every example, using math.log, math.sqrt and "
        "math.pi for the constant term. Then take the mean.",
        "nll_from_mse": "The first term of the negative log, averaged over the examples, is the mean squared error "
        "divided by 2 * sigma ** 2. Then add the constant term.",
        "laplace_nll": "Divide the absolute error of each example by b, add math.log(2 * b), then take the mean.",
    },
    introduces=(),
    check=check_likelihood3,
)


# ---------------------------------------------------------------------------
# likelihood4
# ---------------------------------------------------------------------------

LIKELIHOOD4_LESSON = prose(
    """
    The gradient of cross entropy with respect to the logits has a simple
    formula. It is worth working it out by hand once.

    Take one example with logits `z` and label `y`. Its loss is
    `-z[y] + logsumexp(z)`, because the log of softmax is the score minus
    `logsumexp`. Write `total` for the sum of `exp(z[k])` over all classes
    `k`, so `logsumexp(z)` is `log(total)`. The derivative of `log(t)` is
    `1 / t`. So by the chain rule, the partial derivative of `log(total)` with
    respect to `z[j]` is `1 / total` times the partial of `total`. The partial
    of `total` with respect to `z[j]` is `exp(z[j])`. That makes
    `exp(z[j]) / total`, which is `softmax(z)[j]`. The partial of `-z[y]` is
    -1 when `j == y` and 0 for every other `j`. So the gradient is the softmax
    probabilities, with 1 subtracted at the label.

    ```text
    gradient = softmax(z) - one_hot(y)
    ```

    A **one hot** row has a 1 at one position and 0 everywhere else. Here the
    1 is at the label's position. For the mean loss over `n` examples, divide
    the gradient by `n`.

    Training changes the logits in the direction of the negative gradient. So
    the correct class's logit is raised in proportion to `1 - p`, where `p` is
    the probability the model gave it. Every other class's logit is lowered in
    proportion to the probability the model gave that class.

    You can also assign to the entries that two position tensors pick.

    ```python
    grid = torch.zeros(2, 3)
    grid[torch.arange(2), torch.tensor([2, 0])] = 1.0
    # tensor([[0., 0., 1.], [1., 0., 0.]])
    ```

    PyTorch has `F.one_hot(labels, classes)`, which returns a tensor of whole
    numbers. `.float()` turns them into floats.
    """,
)

LIKELIHOOD4_SOURCE = source_file(
    '''
    # likelihood4.py

    import torch
    from torch import Tensor

    # Assigning through a tensor of row positions and a tensor of column
    # positions sets one entry per row.
    grid = torch.zeros(2, 3)
    grid[torch.arange(2), torch.tensor([2, 0])] = 1.0       # tensor([[0., 0., 1.], [1., 0., 0.]])


    # TODO: Finish the second line, so this returns a one hot row for each label,
    #   with the 1 at the label's position. The result has shape
    #   (len(labels), classes).
    #   Example: one_hot(torch.tensor([2, 0, 1]), 3) returns
    #   tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]).
    def one_hot(
        labels: Tensor,
        classes: int,
    ) -> Tensor:
        encoded = torch.zeros(len(labels), classes)
        ...

        return encoded


    # TODO: Finish the last line, so this returns the gradient of the mean cross
    #   entropy of logits against labels, with respect to logits. Use the
    #   formula from the lesson instead of autograd. The result has the same
    #   shape as logits, (examples, classes).
    #   Example: cross_entropy_grad(torch.tensor([[0.0, 0.0]]), torch.tensor([1]))
    #   returns tensor([[ 0.5000, -0.5000]]).
    def cross_entropy_grad(
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        probabilities = torch.softmax(logits, dim=1)

        return ...


    # TODO: Finish the last two lines, so this returns the gradients of the mean
    #   cross entropy of softmax regression with respect to W and b. The first
    #   line computes grad_logits, the gradient of the loss with respect to the
    #   logits x @ W + b. x has shape (examples, features), W has shape
    #   (features, classes) and b has shape (classes,).
    def softmax_regression_grads(
        x: Tensor,
        W: Tensor,
        b: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        grad_logits = cross_entropy_grad(x @ W + b, labels)
        grad_W = ...
        grad_b = ...

        return grad_W, grad_b
    ''',
)

LIKELIHOOD4_SOLUTION = source_file(
    '''
    # likelihood4.py, a solution for comparison

    import torch
    from torch import Tensor

    grid = torch.zeros(2, 3)
    grid[torch.arange(2), torch.tensor([2, 0])] = 1.0


    def one_hot(
        labels: Tensor,
        classes: int,
    ) -> Tensor:
        encoded = torch.zeros(len(labels), classes)
        encoded[torch.arange(len(labels)), labels] = 1.0

        return encoded


    def cross_entropy_grad(
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        probabilities = torch.softmax(logits, dim=1)

        return (probabilities - one_hot(labels, logits.shape[1])) / len(labels)


    # W is on the right of x @ W. b is added to every row, so its gradient sums grad_logits over the rows.
    def softmax_regression_grads(
        x: Tensor,
        W: Tensor,
        b: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        grad_logits = cross_entropy_grad(x @ W + b, labels)
        grad_W = x.T @ grad_logits
        grad_b = grad_logits.sum(dim=0)

        return grad_W, grad_b
    ''',
)


def check_likelihood4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="one_hot",
        description="puts a 1 at each label's position",
        inputs=(
            (torch.tensor([2, 0, 1]), 3),
            (torch.tensor([4, 4, 0, 2]), 5),
        ),
        reference=lambda labels, classes: F.one_hot(labels, classes).float(),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="cross_entropy_grad",
        description="gives softmax minus one hot, over the number of examples",
        inputs=(
            (small_tensor(123, 4, 3), label_tensor(124, 4, 3)),
            (small_tensor(125, 5, 5), label_tensor(126, 5, 5)),
        ),
        reference=lambda logits, labels: autograd_backward(
            lambda logits: F.cross_entropy(logits, labels),
            (logits,),
            torch.tensor(1.0),
        )[0],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="softmax_regression_grads",
        description="gives the gradients of W and b",
        inputs=(
            (small_tensor(127, 4, 2), small_tensor(128, 2, 3) / 2, small_tensor(129, 3) / 2, label_tensor(130, 4, 3)),
            (small_tensor(131, 3, 3), small_tensor(132, 3, 3) / 2, small_tensor(133, 3) / 2, label_tensor(134, 3, 3)),
        ),
        reference=lambda x, W, b, labels: autograd_backward(
            lambda W, b: F.cross_entropy(x @ W + b, labels),
            (W, b),
            torch.tensor(1.0),
        ),
        expect=expect_parts("grad_W", "grad_b"),
    )


LIKELIHOOD4 = Exercise(
    name="likelihood4",
    chapter=LIKELIHOOD,
    title="The gradient of cross entropy",
    lesson=LIKELIHOOD4_LESSON,
    source=LIKELIHOOD4_SOURCE,
    solution=LIKELIHOOD4_SOLUTION,
    hints={
        "one_hot": "Copy the pattern of the line that sets one entry per row of grid, using torch.arange(len(labels)) "
        "for the rows and labels for the columns.",
        "cross_entropy_grad": "Start from the softmax probabilities, subtract one_hot(labels, logits.shape[1]), and "
        "divide by the number of examples.",
        "softmax_regression_grads": "W is on the right of x @ W, like b in matmul_grad_b of backprop3. b is added to "
        "every row, so its gradient is grad_logits summed over the rows.",
    },
    introduces=("one hot",),
    check=check_likelihood4,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def fit_line(
    x: Tensor,
    y: Tensor,
    loss_of: Callable[[Tensor, Tensor], Tensor],
) -> tuple[float, float]:
    """Gradient descent from zeros, 2000 steps at each of the rates 0.05, 0.01 and 0.002, giving the slope and the
    bias. The falling rate lets the absolute error settle, since its gradient keeps its size right up to the best
    fit."""
    w = torch.zeros(1, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    for rate in (0.05, 0.01, 0.002):
        for _ in range(2000):
            loss = loss_of(x @ w + b, y)
            loss.backward()

            with torch.no_grad():
                w -= rate * w.grad
                b -= rate * b.grad

            w.grad = None
            b.grad = None

    return w.item(), b.item()


def measure_wild_target(seeds: range = range(10)) -> dict[str, list[tuple[float, float]]]:
    """likelihood3's table: 20 points at x = 0, 0.25 and on up to 4.75, with targets 2 * x + 1 plus Gaussian noise
    of spread 0.3. A line fit on the squared error and on the absolute error, on those targets and again with target
    17 raised by 30. Per seed, how far the slope and the bias land from 2 and 1."""
    squared = lambda predictions, targets: ((predictions - targets) ** 2).mean()
    absolute = lambda predictions, targets: (predictions - targets).abs().mean()
    results: dict[str, list[tuple[float, float]]] = {}

    with pinned_threads():
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)
            x = torch.arange(20.0).unsqueeze(1) / 4
            y = 2 * x.squeeze(1) + 1 + 0.3 * torch.randn(20, generator=generator)
            wild = y.clone()
            wild[17] += 30.0

            for label, targets, loss_of in (
                ("squared, Gaussian noise", y, squared),
                ("absolute, Gaussian noise", y, absolute),
                ("squared, one wild target", wild, squared),
                ("absolute, one wild target", wild, absolute),
            ):
                slope, bias = fit_line(x, targets, loss_of)
                results.setdefault(label, []).append((abs(slope - 2), abs(bias - 1)))

    return results
