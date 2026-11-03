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

        Where do losses come from? This chapter shows that the usual ones are the
        same idea, the probability a model gives the data it should have
        predicted. That gives cross entropy for classes, and the squared error
        for numbers, and ends with the gradient of cross entropy worked out by
        hand.

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
    A softmax model gives each example a probability for its correct label.
    Say it gives 0.7, 0.9 and 0.2 to three examples' labels. Their product,
    0.126, is the probability the model gives to all three labels at once.
    That product is the **likelihood** of the labels, and a better model gives
    a higher one.

    Training could push the likelihood up, but a product of thousands of
    probabilities shrinks to 0 in a float.

    ```python
    torch.full((1000,), 0.5).prod()            # tensor(0.)
    torch.log(torch.full((1000,), 0.5)).sum()  # tensor(-693.1473)
    ```

    `torch.full((1000,), 0.5)` is 1000 entries of 0.5, and `.prod()`
    multiplies all entries. 0.5 to the power 1000 is far below the smallest
    float32, the kind of number a tensor holds, so it comes out as 0. A 0
    can't tell a good model from a bad one.

    The log fixes it. The log of a product is the sum of the logs, and the log
    of a division is the difference, so `math.log(12 / 3)` equals
    `math.log(12) - math.log(3)`. A sum of thousands of negative numbers is a
    big negative number, nothing worse.

    The log grows when its input grows, so the model with the higher
    likelihood has the higher log likelihood too. Training lowers a loss, so
    it flips the sign. The negative log likelihood, divided by the number of
    examples, is the loss.

    To pick each row's probability for its label, index with two tensors, one
    row position and one column position per example. `torch.arange(n)` gives
    0 to `n - 1`.

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

    # One row position and one column position per example picks each row's
    # probability for its label.
    picked = probabilities[torch.arange(3), labels]    # tensor([0.7, 0.9, 0.2])


    # TODO: Return each row's probability for its label, for any number of rows.
    def label_probabilities(
        probabilities: Tensor,
        labels: Tensor,
    ) -> Tensor:
        ...


    # TODO: This should return the log likelihood of the labels, but for a few
    #   hundred examples it gives -inf. Read the lesson, then fix the last line.
    def log_likelihood(
        probabilities: Tensor,
        labels: Tensor,
    ) -> Tensor:
        picked = probabilities[torch.arange(len(labels)), labels]

        return torch.log(picked.prod())


    # TODO: Return the loss, minus the log of each label's probability, averaged
    #   over the examples.
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
        "label_probabilities": "Follow the line that sets picked, with torch.arange(len(labels)) in place of "
        "torch.arange(3).",
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
    The negative log likelihood of softmax probabilities is called **cross
    entropy**. It is the standard loss for picking a class.

    Taking softmax and then the log throws precision away. A probability far
    below the smallest float rounds to 0, and its log is `-inf`, even though
    the true log is an ordinary number.

    ```python
    scores = torch.tensor([[60.0, 0.0, -60.0]])
    torch.log(torch.softmax(scores, dim=1))    # tensor([[  0., -60., -inf]]), the last should be -120
    ```

    So work in logs from the start. Softmax is `exp(score) / total`. By the
    rule for the log of a division, its log is each score minus the log of the
    row's total of exps. `torch.logsumexp`
    computes that log of a sum of exps with the largest score taken out first,
    as softmax2 did, so it never overflows.

    ```python
    torch.logsumexp(scores, dim=1, keepdim=True)    # tensor([[60.]])
    ```

    PyTorch's `F.cross_entropy(logits, labels)` takes the raw scores and the
    labels and does all of it. That is why models output logits and leave the
    softmax to the loss.

    A model that knows nothing gives every class the same score, so every
    class gets probability `1 / classes`. With 10 classes its cross entropy
    comes out at 2.303. A new model should start near that number. A starting
    loss far above it means the model is confidently wrong before it has
    learned anything.
    """,
)

LIKELIHOOD2_SOURCE = source_file(
    '''
    # likelihood2.py

    import math

    import torch
    from torch import Tensor

    scores = torch.tensor([[60.0, 0.0, -60.0]])

    # The log of a sum of exps, with the largest taken out first so nothing overflows.
    log_total = torch.logsumexp(scores, dim=1, keepdim=True)     # tensor([[60.]])


    # TODO: Return the log of the softmax of each row of scores, each score
    #   minus its row's log_total, so very small probabilities keep their logs.
    def log_softmax_rows(scores: Tensor) -> Tensor:
        ...


    # TODO: Finish the two lines, so this returns the cross entropy of logits
    #   against labels, the mean of minus each row's log probability for its
    #   label. It should match PyTorch's own cross entropy.
    def cross_entropy(
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        log_probabilities = log_softmax_rows(logits)
        picked = ...

        return ...


    # TODO: Return the cross entropy of a model that gives every class the same
    #   score, for this many classes.
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
        "log_softmax_rows": "Take torch.logsumexp of the scores along dim=1 with keepdim=True, as the line that "
        "sets log_total does. Then take it away from the scores.",
        "cross_entropy": "Pick each row's log probability for its label as likelihood1 picks probabilities. The "
        "loss is minus their mean.",
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
    Squared error is a likelihood too. Suppose each target is the prediction
    plus random noise, and the noise follows a Gaussian, the bell curve, with
    spread `sigma`. A target here can be any number, so it gets a probability
    density in place of a probability. A density is higher where targets are
    more likely. The density of seeing target `y` is

    ```text
    exp(-(y - prediction) ** 2 / (2 * sigma ** 2)) / (sigma * sqrt(2 * pi))
    ```

    Its negative log, by the rules for the log of a product and a division, is

    ```text
    (y - prediction) ** 2 / (2 * sigma ** 2) + log(sigma * sqrt(2 * pi))
    ```

    `math.pi` and `math.sqrt` give pi and square roots of plain numbers. For
    one target of 3.0, a prediction of 2.5 and `sigma` of 1, that is 1.0439.

    With `sigma` fixed, the second part is the same for every model, and the
    first is the squared error divided by a constant. So the model with the
    lowest squared error has the highest likelihood under Gaussian noise. The
    mean squared error is the negative log likelihood with the constants
    dropped. Picking squared error says the misses are Gaussian.

    Other noise gives other losses. A Laplace distribution has density
    `exp(-abs(y - prediction) / b) / (2 * b)` for a spread `b`. Its negative
    log is the absolute error divided by `b`, plus `log(2 * b)`. A Laplace
    gives far misses more probability than a Gaussian does. So one wild target
    costs the absolute error less than it costs the squared error.

    Here is a line fit by gradient descent to 20 points with a true slope of
    2, on each loss. The noise is Gaussian with spread 0.3, and then one target
    is raised by 30. Each entry is how far the fitted slope lands from 2, the
    median of 10 runs.

    | Targets | Squared error | Absolute error |
    |---|---|---|
    | Gaussian noise only | 0.032 | 0.045 |
    | one target raised by 30 | 1.36 | 0.083 |

    On Gaussian noise the squared error lands closer, in 7 of 10 runs, as its
    likelihood says it should. One wild target drags its slope to about 3.4,
    while the absolute error barely moves.
    """,
)

LIKELIHOOD3_SOURCE = source_file(
    '''
    # likelihood3.py

    import math

    import torch
    from torch import Tensor

    # The Gaussian negative log for one target of 3.0, a prediction of 2.5 and
    # sigma of 1.
    one_target = (3.0 - 2.5) ** 2 / (2 * 1.0 ** 2) + math.log(1.0 * math.sqrt(2 * math.pi))    # 1.0439


    # TODO: Return the Gaussian negative log likelihood from the lesson, for
    #   noise with spread sigma, averaged over the examples.
    def gaussian_nll(
        predictions: Tensor,
        targets: Tensor,
        sigma: float,
    ) -> Tensor:
        ...


    # TODO: Return the mean Gaussian negative log likelihood, given only the mean
    #   squared error of the predictions and sigma.
    def nll_from_mse(
        mse: float,
        sigma: float,
    ) -> float:
        ...


    # TODO: Return the mean over the examples of the Laplace negative log
    #   likelihood from the lesson, for noise with spread b.
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
        "gaussian_nll": "Write the negative log from the lesson for every example, with math.log, math.sqrt and "
        "math.pi for the constant. Then take the mean.",
        "nll_from_mse": "The first part of the negative log, averaged over the examples, is the mean squared error "
        "divided by 2 * sigma ** 2. Add the constant part.",
        "laplace_nll": "The absolute error of each example divided by b, plus math.log(2 * b), then the mean.",
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
    form, worth working out by hand once.

    For one example with logits `z` and label `y`, the loss is
    `-z[y] + logsumexp(z)`. `logsumexp(z)` is the log of a total. By the chain
    rule, its partial with respect to `z[j]` is `1 / total` times the total's
    own partial, which is `exp(z[j])`. That makes `exp(z[j]) / total`, which
    is `softmax(z)[j]`. The partial of `-z[y]` is -1 for `j == y` and 0
    for every other `j`. So the gradient is the softmax probabilities, with 1
    taken away at the label.

    ```text
    gradient = softmax(z) - one_hot(y)
    ```

    A **one hot** row has a 1 at one position and 0 everywhere else. Here the
    1 sits at the label. For the mean loss over `n` examples, divide by
    `n`.

    So the correct class gets pushed up by the probability it is missing, and
    every other class gets pushed down by the probability it took.

    Two position tensors also set entries, the way they pick them.

    ```python
    grid = torch.zeros(2, 3)
    grid[torch.arange(2), torch.tensor([2, 0])] = 1.0
    # tensor([[0., 0., 1.], [1., 0., 0.]])
    ```

    PyTorch has `F.one_hot(labels, classes)`, which gives whole numbers, and
    `.float()` turns them into floats.
    """,
)

LIKELIHOOD4_SOURCE = source_file(
    '''
    # likelihood4.py

    import torch
    from torch import Tensor

    # Two position tensors set one entry per row, the way they pick them.
    grid = torch.zeros(2, 3)
    grid[torch.arange(2), torch.tensor([2, 0])] = 1.0       # tensor([[0., 0., 1.], [1., 0., 0.]])


    # TODO: Finish the second line, so this returns one one hot row per label,
    #   with the 1 at the label's position.
    def one_hot(
        labels: Tensor,
        classes: int,
    ) -> Tensor:
        encoded = torch.zeros(len(labels), classes)
        ...

        return encoded


    # TODO: Finish the last line, so this returns the gradient of the mean cross
    #   entropy of logits against labels, with respect to the logits, worked out
    #   by hand.
    def cross_entropy_grad(
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        probabilities = torch.softmax(logits, dim=1)

        return ...


    # TODO: Finish the last two lines, so this returns the gradients of the mean
    #   cross entropy of softmax regression with respect to W and b. The first
    #   line gives the gradient of the logits x @ W + b.
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


    # W is on the right of x @ W, and b was copied into every row.
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
        "one_hot": "Follow the line that sets one entry per row of grid, with torch.arange(len(labels)) for the "
        "rows and labels for the columns.",
        "cross_entropy_grad": "Take the softmax of each row of logits, take away one_hot(labels, logits.shape[1]), "
        "and divide by the number of examples.",
        "softmax_regression_grads": "W is on the right of x @ W, like b in backprop3. b was copied into every row, "
        "so its gradient adds grad_logits up over the rows.",
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
