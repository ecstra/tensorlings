"""16_activations, the step between two Linears that lets models bend, which one to use, and why."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
    reject_missing_return,
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from .tensors import pinned_threads

ACTIVATIONS = Chapter(
    folder="16_activations",
    title="Activations",
    readme=source_file(
        """
        # Activations

        Two Linear layers in a row can always be collapsed into one Linear
        layer. An activation between them prevents that. An activation is a
        simple function applied to every entry on its own, and it lets a model
        fit curved shapes. This chapter builds ReLU, and shows a task that a
        Linear model cannot learn but a model with ReLU can. Then it covers
        sigmoid, tanh and GELU, and which job each one suits. It ends with
        operations like `argmax` that pass no gradient back, and the smooth
        versions of them that sigmoid and softmax give.

        ## Further reading

        - [Dive into Deep Learning, multilayer perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)
        - [Hendrycks and Gimpel, Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415)
        - [Glorot, Bordes and Bengio, deep sparse rectifier networks](https://proceedings.mlr.press/v15/glorot11a.html)
        """,
    ),
)

XOR_X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_Y = torch.tensor([0.0, 1.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# relu1
# ---------------------------------------------------------------------------

RELU1_LESSON = prose(
    """
    linear3 showed that two Linear layers in a row collapse into one Linear
    layer. The fix is a step between them that is not a weighted sum, a
    function applied to every entry on its own. Such a step is called an
    **activation**.

    The simplest activation is ReLU, short for rectified linear unit. It
    keeps positive numbers and turns negative numbers into 0. As a formula,
    `relu(x) = max(x, 0)`.

    ```python
    x = torch.tensor([-2.0, -0.5, 0.0, 1.5])
    torch.relu(x)    # tensor([0.0000, 0.0000, 0.0000, 1.5000])
    ```

    The derivative of ReLU is 1 where the input is positive and 0 where it is
    negative. At exactly 0 the derivative is taken to be 0. So in
    backpropagation, ReLU passes the gradient back unchanged where its input
    was positive, and sets it to 0 everywhere else.

    Put ReLU between two Linear layers, `Linear, ReLU, Linear`, and the two
    layers can no longer be collapsed into one. The outputs of the first
    layer after the ReLU are called the hidden features, because nothing
    outside the model sees them. Each hidden neuron outputs 0 for the inputs
    where its weighted sum is negative. For the other inputs it outputs its
    weighted sum unchanged.

    So, as a function of the input, the model's output is made of straight
    pieces that meet at corners. With enough hidden neurons, those pieces can
    follow any smooth curve over a limited range.

    `.float()` turns a tensor of `True` and `False` into 1.0 and 0.0. So
    `(x < 0).float()` gives 1.0 for each entry below 0, and 0.0 for the rest.
    """,
)

RELU1_SOURCE = source_file(
    '''
    # relu1.py

    import torch
    from torch import Tensor

    x = torch.tensor([-2.0, -0.5, 0.0, 1.5])

    # ReLU keeps the positive entries and turns the rest into 0.
    activated = torch.relu(x)                          # tensor([0.0, 0.0, 0.0, 1.5])


    # TODO: Return the derivative of ReLU at every entry of x, which is 1.0 where
    #   the entry is greater than 0 and 0.0 everywhere else.
    #   Example: relu_derivative(torch.tensor([-2.0, 0.0, 1.5])) returns
    #   tensor([0., 0., 1.]).
    def relu_derivative(x: Tensor) -> Tensor:
        ...


    # TODO: Return the hidden features. Pass x through a Linear layer with
    #   weights W1 and bias b1, then apply ReLU to the result.
    def hidden_features(
        x: Tensor,
        W1: Tensor,
        b1: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish the last line, so this passes x through Linear, ReLU, Linear.
    #   hidden already holds the output of the first Linear and the ReLU. The
    #   second Linear layer has weights W2 and bias b2.
    def two_layer_net(
        x: Tensor,
        W1: Tensor,
        b1: Tensor,
        W2: Tensor,
        b2: Tensor,
    ) -> Tensor:
        hidden = hidden_features(x, W1, b1)

        return ...
    ''',
)

RELU1_SOLUTION = source_file(
    '''
    # relu1.py, a solution for comparison

    import torch
    from torch import Tensor

    x = torch.tensor([-2.0, -0.5, 0.0, 1.5])

    activated = torch.relu(x)


    def relu_derivative(x: Tensor) -> Tensor:
        return (x > 0).float()


    def hidden_features(
        x: Tensor,
        W1: Tensor,
        b1: Tensor,
    ) -> Tensor:
        return torch.relu(x @ W1.T + b1)


    def two_layer_net(
        x: Tensor,
        W1: Tensor,
        b1: Tensor,
        W2: Tensor,
        b2: Tensor,
    ) -> Tensor:
        hidden = hidden_features(x, W1, b1)

        return hidden @ W2.T + b2
    ''',
)

NETS: tuple[tuple[Tensor, ...], ...] = (
    (small_tensor(200, 3, 2), small_tensor(201, 4, 2), small_tensor(202, 4), small_tensor(203, 1, 4), small_tensor(204, 1)),
    (small_tensor(205, 2, 3), small_tensor(206, 3, 3), small_tensor(207, 3), small_tensor(208, 2, 3), small_tensor(209, 2)),
)


def check_relu1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="relu_derivative",
        description="gives 1 above 0 and 0 everywhere else",
        inputs=(
            (torch.tensor([-2.0, -0.5, 0.0, 1.5]),),
            (small_tensor(210, 2, 4),),
        ),
        reference=lambda x: (x > 0).float(),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="hidden_features",
        description="gives ReLU of the first Linear layer",
        inputs=tuple(net[:3] for net in NETS),
        reference=lambda x, W1, b1: torch.relu(F.linear(x, W1, b1)),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="two_layer_net",
        description="gives Linear, ReLU, Linear",
        inputs=NETS,
        reference=lambda x, W1, b1, W2, b2: F.linear(torch.relu(F.linear(x, W1, b1)), W2, b2),
        expect=expect_tensor,
    )


RELU1 = Exercise(
    name="relu1",
    chapter=ACTIVATIONS,
    title="ReLU",
    lesson=RELU1_LESSON,
    source=RELU1_SOURCE,
    solution=RELU1_SOLUTION,
    hints={
        "relu_derivative": "Compare x with 0, then turn True and False into 1.0 and 0.0 with .float().",
        "hidden_features": "The Linear layer is x @ W1.T + b1, as in linear1. Pass that to torch.relu.",
        "two_layer_net": "The second layer is a Linear layer applied to hidden, with W2 and b2.",
    },
    introduces=("activation",),
    check=check_relu1,
)


# ---------------------------------------------------------------------------
# relu2
# ---------------------------------------------------------------------------

RELU2_LESSON = prose(
    """
    Here is a task that one Linear layer cannot learn. There are four points,
    each with two features. The target is 1 when exactly one of the two
    features is 1, and 0 otherwise.

    ```python
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    ```

    This is called XOR, short for exclusive or. Starting from `(0, 0)`,
    raising either feature from 0 to 1 raises the target to 1. But raising
    both features gives a target of 0 again. A weighted sum cannot do that.
    In a weighted sum, raising a feature always changes the output by the
    same amount, whatever the other feature is.

    Trained on these points, a Linear model ends up predicting 0.5 for every
    point. That gives a mean squared error of 0.25, which is the best any
    Linear model can do here. Two Linear layers in a row, with 8 neurons
    between them, also end at an error of 0.25. That happened from all 10
    random starting weights tried, as linear3 says it must.

    Now put ReLU between the two Linear layers, with 8 hidden neurons.
    Gradient descent then brings the error down to almost 0. Some hidden
    neurons output a positive number for one point and 0 for another. The
    second layer adds up these hidden features with its weights to give the
    right answer at every point.

    The hidden layer's weights start random. They are drawn with a
    generator, so every run gets the same random numbers. If the weights all
    started equal, every hidden neuron would compute the same thing, and they
    would stay copies of each other.

    This is why deep learning stacks layers with activations between them.
    More neurons can fit more complicated shapes. The next chapter of
    Section 3 builds the model that results, called the multilayer
    perceptron.
    """,
)

RELU2_SOURCE = source_file(
    '''
    # relu2.py

    import math

    import torch
    from torch import Tensor


    # TODO: Finish the line that sets predictions to the output of one Linear
    #   layer on x, with weights W and bias b. Then this trains the layer by
    #   gradient descent and returns the final mean squared error.
    def fit_linear(
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> float:
        W = torch.zeros(1, x.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)

        for _ in range(steps):
            predictions = ...
            loss = ((predictions.squeeze(1) - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                for parameter in (W, b):
                    parameter -= learning_rate * parameter.grad
                    parameter.grad = None

        return loss.item()


    # TODO: Finish the line that sets predictions to the output of Linear, ReLU,
    #   Linear on x. Then this trains the two layer network and returns the
    #   final mean squared error. W2 is divided by sqrt(hidden) so the output
    #   starts small. The initialization chapter explains why.
    def fit_relu_net(
        x: Tensor,
        y: Tensor,
        hidden: int,
        steps: int,
        learning_rate: float,
        generator: torch.Generator,
    ) -> float:
        W1 = torch.randn(hidden, x.shape[1], generator=generator).requires_grad_()
        b1 = torch.zeros(hidden, requires_grad=True)
        W2 = (torch.randn(1, hidden, generator=generator) / math.sqrt(hidden)).requires_grad_()
        b2 = torch.zeros(1, requires_grad=True)

        for _ in range(steps):
            predictions = ...
            loss = ((predictions.squeeze(1) - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                for parameter in (W1, b1, W2, b2):
                    parameter -= learning_rate * parameter.grad
                    parameter.grad = None

        return loss.item()
    ''',
)

RELU2_SOLUTION = source_file(
    '''
    # relu2.py, a solution for comparison

    import math

    import torch
    from torch import Tensor


    def fit_linear(
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> float:
        W = torch.zeros(1, x.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)

        for _ in range(steps):
            predictions = x @ W.T + b
            loss = ((predictions.squeeze(1) - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                for parameter in (W, b):
                    parameter -= learning_rate * parameter.grad
                    parameter.grad = None

        return loss.item()


    def fit_relu_net(
        x: Tensor,
        y: Tensor,
        hidden: int,
        steps: int,
        learning_rate: float,
        generator: torch.Generator,
    ) -> float:
        W1 = torch.randn(hidden, x.shape[1], generator=generator).requires_grad_()
        b1 = torch.zeros(hidden, requires_grad=True)
        W2 = (torch.randn(1, hidden, generator=generator) / math.sqrt(hidden)).requires_grad_()
        b2 = torch.zeros(1, requires_grad=True)

        for _ in range(steps):
            predictions = torch.relu(x @ W1.T + b1) @ W2.T + b2
            loss = ((predictions.squeeze(1) - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                for parameter in (W1, b1, W2, b2):
                    parameter -= learning_rate * parameter.grad
                    parameter.grad = None

        return loss.item()
    ''',
)

XOR_DATA = (Described(XOR_X, "the four XOR points"), Described(XOR_Y, "their targets"))

# AND is 1 only when both features are, which one Linear layer gets close to, so a typed in 0.25 fails here.
AND_DATA = (Described(XOR_X, "the four points"), Described(torch.tensor([0.0, 0.0, 0.0, 1.0]), "AND targets"))


def train_with(
    parameters: list[Tensor],
    forward: object,
    x: Tensor,
    y: Tensor,
    steps: int,
    learning_rate: float,
) -> float:
    for _ in range(steps):
        loss = ((forward(x).squeeze(1) - y) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            for parameter in parameters:
                parameter -= learning_rate * parameter.grad
                parameter.grad = None

    return loss.item()


def reference_fit_linear(
    x: Tensor,
    y: Tensor,
    steps: int,
    learning_rate: float,
) -> float:
    W = torch.zeros(1, x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    return train_with([W, b], lambda x: x @ W.T + b, x, y, steps, learning_rate)


def reference_fit_relu_net(
    x: Tensor,
    y: Tensor,
    hidden: int,
    steps: int,
    learning_rate: float,
    generator: torch.Generator,
) -> float:
    W1 = torch.randn(hidden, x.shape[1], generator=generator).requires_grad_()
    b1 = torch.zeros(hidden, requires_grad=True)
    W2 = (torch.randn(1, hidden, generator=generator) / math.sqrt(hidden)).requires_grad_()
    b2 = torch.zeros(1, requires_grad=True)

    return train_with([W1, b1, W2, b2], lambda x: torch.relu(x @ W1.T + b1) @ W2.T + b2, x, y, steps, learning_rate)


def check_relu2(checker: Checker) -> None:
    close = lambda call, actual, expected: expect_close(
        call=call,
        actual=actual,
        expected=expected,
        tolerance=1e-4,
    )
    check_on_inputs(
        checker=checker,
        function="fit_linear",
        description="trains one Linear layer, which gets stuck at 0.25 on XOR",
        inputs=(
            (*XOR_DATA, 2000, 0.1),
            (*AND_DATA, 500, 0.1),
        ),
        reference=reference_fit_linear,
        expect=close,
    )
    check_on_inputs(
        checker=checker,
        function="fit_relu_net",
        description="trains Linear, ReLU, Linear down to almost 0 on XOR",
        inputs=(
            (*XOR_DATA, 8, 2000, 0.1, seeded(0)),
            (*AND_DATA, 4, 300, 0.1, seeded(3)),
        ),
        reference=reference_fit_relu_net,
        expect=close,
    )


RELU2 = Exercise(
    name="relu2",
    chapter=ACTIVATIONS,
    title="Why an activation",
    lesson=RELU2_LESSON,
    source=RELU2_SOURCE,
    solution=RELU2_SOLUTION,
    hints={
        "fit_linear": "One Linear layer is x @ W.T + b, as in linear1.",
        "fit_relu_net": "Apply the first layer, x @ W1.T + b1, then torch.relu, then the second layer with W2 and b2, "
        "as two_layer_net does in relu1.",
    },
    introduces=(),
    check=check_relu2,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# activations1
# ---------------------------------------------------------------------------

ACTIVATIONS1_LESSON = prose(
    """
    Two older activations squeeze any number into a fixed range. Sigmoid
    gives a number between 0 and 1, and tanh gives a number between -1 and 1.

    ```text
    sigmoid(x) = 1 / (1 + exp(-x))
    tanh(x)    = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    ```

    ```python
    x = torch.tensor([-4.0, 0.0, 4.0])
    torch.sigmoid(x)    # tensor([0.0180, 0.5000, 0.9820])
    torch.tanh(x)       # tensor([-0.9993,  0.0000,  0.9993])
    ```

    Sigmoid equals softmax over two classes with scores `x` and 0, so its
    output can be read as a probability. The derivatives of both activations
    can be written in terms of their own outputs. With `s = sigmoid(x)`, the
    derivative of sigmoid is `s * (1 - s)`, which is at most 0.25. With
    `t = tanh(x)`, the derivative of tanh is `1 - (t ** 2)`, which is at most 1.

    Both curves become flat for inputs far from 0. At `x = 4`, sigmoid's
    derivative is already 0.018. So a neuron whose input is far from 0 passes
    back almost no gradient, and it stops learning. This is called
    saturation. In a stack of sigmoid layers, backpropagation multiplies many
    of these small derivatives together. So the gradient that reaches the
    first layers is almost 0. ReLU's derivative is 1 for any positive input.

    The table below compares models of six layers with 64 neurons each, with
    one of the three activations between the layers. The task is to classify
    200 points that lie on two spirals wound around each other. Each model
    was trained for 3000 steps of gradient descent from PyTorch's starting
    weights. The numbers are averages over 5 seeds, and the gradient column
    is measured at the first step.

    | Between the layers | First layer's gradient divided by the last layer's | Final loss | Training accuracy |
    |---|---|---|---|
    | sigmoid | 0.00008 | 0.692 | 54% |
    | tanh | 0.11 | 0.220 | 90% |
    | ReLU | 0.065 | 0.018 | 99% |

    The loss of the sigmoid model stays at `log(2)`, the loss of guessing a
    probability of 0.5 for both classes. Tanh learns slowly, and ReLU fits
    the spirals. This is the main reason ReLU replaced sigmoid and tanh
    between layers.

    Sigmoid and tanh are still used for other jobs. A sigmoid gives the
    probability of a yes or no answer. It can also be a gate, a number
    between 0 and 1 that multiplies a signal. A gate scales the signal
    anywhere from fully off to fully on. A tanh gives an output that must stay between -1 and 1.
    """,
)

ACTIVATIONS1_SOURCE = source_file(
    '''
    # activations1.py

    import torch
    from torch import Tensor


    # TODO: Return sigmoid of every entry of x, computed with the formula from
    #   the lesson, without calling torch.sigmoid.
    #   Example: sigmoid(torch.tensor([-4.0, 0.0, 4.0])) returns
    #   tensor([0.0180, 0.5000, 0.9820]).
    def sigmoid(x: Tensor) -> Tensor:
        ...


    # The derivative of sigmoid, written in terms of sigmoid's own output s.
    def sigmoid_derivative(x: Tensor) -> Tensor:
        s = torch.sigmoid(x)

        return s * (1 - s)


    # TODO: Return the derivative of tanh at every entry of x. Write it in terms
    #   of tanh's own output, the way sigmoid_derivative does for sigmoid.
    #   Example: tanh_derivative(torch.tensor([0.0])) returns tensor([1.]).
    def tanh_derivative(x: Tensor) -> Tensor:
        ...


    # TODO: Return the fraction of entries of x where sigmoid is saturated, which
    #   here means its derivative is below 0.01, as a float.
    #   Example: saturated_fraction(torch.tensor([-6.0, 0.0, 2.0, 8.0])) returns
    #   0.5.
    def saturated_fraction(x: Tensor) -> float:
        ...
    ''',
)

ACTIVATIONS1_SOLUTION = source_file(
    '''
    # activations1.py, a solution for comparison

    import torch
    from torch import Tensor


    def sigmoid(x: Tensor) -> Tensor:
        return 1 / (1 + torch.exp(-x))


    def sigmoid_derivative(x: Tensor) -> Tensor:
        s = torch.sigmoid(x)

        return s * (1 - s)


    def tanh_derivative(x: Tensor) -> Tensor:
        t = torch.tanh(x)

        return 1 - t ** 2


    def saturated_fraction(x: Tensor) -> float:
        return (sigmoid_derivative(x) < 0.01).float().mean().item()
    ''',
)


class Without:
    """Stands in for a module the file imported, during one call, so a function its TODO rules out fails by name."""

    def __init__(
        self,
        module: object,
        banned: str,
        message: str,
    ) -> None:
        self.module = module
        self.banned = banned
        self.message = message

    def __getattr__(
        self,
        name: str,
    ) -> object:
        if name == self.banned:
            raise Mismatch(self.message)

        return getattr(self.module, name)


def expect_without(
    found: object,
    imported_as: str,
    banned: str,
    message: str,
) -> None:
    """Calls found with the module the file imported as imported_as standing in, so a call to banned fails."""
    original = found.__globals__.get(imported_as)
    found.__globals__[imported_as] = Without(original, banned, message)

    try:
        actual = found(torch.tensor([-2.0, 0.0, 1.5]))
    finally:
        found.__globals__[imported_as] = original

    reject_missing_return(
        call=f"{found.__name__}(tensor([-2.0, 0.0, 1.5]))",
        actual=actual,
    )


def check_activations1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="sigmoid",
        description="squeezes every entry into the range from 0 to 1",
        inputs=(
            (torch.tensor([-4.0, 0.0, 4.0]),),
            (small_tensor(211, 2, 3),),
        ),
        reference=torch.sigmoid,
        expect=expect_tensor,
    )
    checker.test(
        function="sigmoid",
        description="works from the formula, without torch.sigmoid",
        run=lambda found: expect_without(
            found=found,
            imported_as="torch",
            banned="sigmoid",
            message="sigmoid calls torch.sigmoid, which its TODO rules out. Write the formula with torch.exp.",
        ),
    )
    check_on_inputs(
        checker=checker,
        function="tanh_derivative",
        description="gives 1 minus tanh squared",
        inputs=(
            (torch.tensor([-4.0, 0.0, 4.0]),),
            (small_tensor(212, 2, 3),),
        ),
        reference=lambda x: 1 - torch.tanh(x) ** 2,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="saturated_fraction",
        description="gives the fraction of entries where sigmoid's derivative is below 0.01",
        inputs=(
            (torch.tensor([-6.0, -1.0, 0.0, 2.0, 5.0, 8.0]),),
            (torch.linspace(-10.0, 10.0, 21),),
        ),
        reference=lambda x: ((torch.sigmoid(x) * (1 - torch.sigmoid(x))) < 0.01).float().mean().item(),
        expect=expect_close,
    )


ACTIVATIONS1 = Exercise(
    name="activations1",
    chapter=ACTIVATIONS,
    title="Sigmoid and tanh",
    lesson=ACTIVATIONS1_LESSON,
    source=ACTIVATIONS1_SOURCE,
    solution=ACTIVATIONS1_SOLUTION,
    hints={
        "sigmoid": "Write the formula with torch.exp. It works on every entry at once.",
        "tanh_derivative": "Set t = torch.tanh(x), then return the derivative from the lesson, written in terms of t.",
        "saturated_fraction": "Compare sigmoid_derivative(x) with 0.01, then take the mean of the matches as "
        "floats, as accuracy does in softmax3. Turn it into a plain number with .item().",
    },
    introduces=(),
    check=check_activations1,
)


# ---------------------------------------------------------------------------
# gelu1
# ---------------------------------------------------------------------------

GELU1_LESSON = prose(
    """
    Language models like GPT-2 and BERT use an activation called GELU, from a
    2016 paper by Hendrycks and Gimpel. ReLU passes an input through
    unchanged when it is positive, and outputs 0 when it is not. GELU instead
    multiplies the input by a number between 0 and 1. That number is the
    probability that a random number from the bell curve is smaller than the
    input.

    ```text
    GELU(x) = x * Phi(x),   Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    ```

    `Phi(x)` is the probability that a number drawn from the bell curve with
    mean 0 and standard deviation 1 is below `x`. This bell curve is called
    the standard normal distribution. `erf` is the error function, which
    `torch.erf` computes.

    For a large positive `x`, `Phi(x)` is almost 1, so GELU gives almost `x`,
    like ReLU. For a large negative `x`, GELU gives almost 0. Near 0 the GELU
    curve bends smoothly and dips a little below 0, where ReLU has a sharp
    corner.

    ```python
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    F.gelu(x)    # tensor([-0.1587,  0.0000,  0.8413,  1.9545])
    ```

    The paper also gives a faster approximation that uses tanh, and GPT-2
    used it.

    ```text
    0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * (x ** 3))))
    ```

    `F.gelu(x, approximate="tanh")` computes this approximation.

    A close relative is SiLU, `x * sigmoid(x)`, which multiplies the input by
    its sigmoid instead of by `Phi(x)`. The same paper named it, and
    `F.silu(x)` computes it. Llama and most current language models use
    SiLU, inside a gated layer that Section 7 builds.

    Why not use ReLU? A ReLU neuron whose input is below 0 on every example
    always outputs 0 and gets no gradient. So its weights never change
    again, and it stays at 0 for good. GELU and SiLU still pass a small
    gradient back for inputs below 0. Here is the XOR network from relu2
    with 4 hidden neurons, trained from 100 different random starting
    weights.

    | Between the layers | Runs stuck above an error of 0.01 | Mean final error |
    |---|---|---|
    | ReLU | 37 of 100 | 0.061 |
    | GELU | 0 of 100 | 0.000 |
    | SiLU | 0 of 100 | 0.000 |

    Every stuck ReLU run has a hidden neuron that outputs 0 on all four
    points.
    """,
)

GELU1_SOURCE = source_file(
    '''
    # gelu1.py

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Return GELU of every entry of x, computed with the paper's formula
    #   and torch.erf, without calling F.gelu. It should match F.gelu(x).
    #   Example: gelu(torch.tensor([-1.0, 1.0])) returns
    #   tensor([-0.1587,  0.8413]).
    def gelu(x: Tensor) -> Tensor:
        ...


    # TODO: Return the tanh approximation of GELU from the lesson, for every
    #   entry of x. It should match F.gelu(x, approximate="tanh").
    def gelu_tanh(x: Tensor) -> Tensor:
        ...


    # TODO: Return SiLU of every entry of x, which is x times sigmoid of x. It
    #   should match F.silu(x).
    def silu(x: Tensor) -> Tensor:
        ...
    ''',
)

GELU1_SOLUTION = source_file(
    '''
    # gelu1.py, a solution for comparison

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    def gelu(x: Tensor) -> Tensor:
        return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))


    def gelu_tanh(x: Tensor) -> Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


    def silu(x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)
    ''',
)

GELU_INPUTS: tuple[tuple[Tensor], ...] = (
    (torch.tensor([-1.0, 0.0, 1.0, 2.0]),),
    (torch.linspace(-4.0, 4.0, 9),),
)


def check_gelu1(checker: Checker) -> None:
    for function, description, reference in (
        ("gelu", "matches F.gelu", F.gelu),
        ("gelu_tanh", "matches F.gelu with the tanh approximation", lambda x: F.gelu(x, approximate="tanh")),
        ("silu", "matches F.silu", F.silu),
    ):
        check_on_inputs(
            checker=checker,
            function=function,
            description=description,
            inputs=GELU_INPUTS,
            reference=reference,
            expect=expect_tensor,
        )

    checker.test(
        function="gelu",
        description="works from the paper's formula, without F.gelu",
        run=lambda found: expect_without(
            found=found,
            imported_as="F",
            banned="gelu",
            message="gelu calls F.gelu, which its TODO rules out. Write x times Phi(x) with torch.erf.",
        ),
    )


GELU1 = Exercise(
    name="gelu1",
    chapter=ACTIVATIONS,
    title="GELU from its paper",
    lesson=GELU1_LESSON,
    source=GELU1_SOURCE,
    solution=GELU1_SOLUTION,
    hints={
        "gelu": "Phi(x) is 0.5 * (1 + torch.erf(x / math.sqrt(2))). Multiply x by it.",
        "gelu_tanh": "Write the formula from the lesson as it is, using torch.tanh, math.sqrt and math.pi.",
        "silu": "Multiply x by torch.sigmoid(x).",
    },
    introduces=(),
    check=check_gelu1,
)


# ---------------------------------------------------------------------------
# activations2
# ---------------------------------------------------------------------------

ACTIVATIONS2_LESSON = prose(
    """
    An activation does a different job depending on where it sits. Between
    layers, its job is to let the model fit curved shapes. ReLU is the usual
    default there, and GELU and SiLU are common in language models. At the
    output, its job is to put the answer in the form the task needs.

    | The model predicts | Output activation | Because it gives |
    |---|---|---|
    | any number, like a temperature | none | any number |
    | the probability of yes | sigmoid | a number between 0 and 1 |
    | a value held between -1 and 1 | tanh | a number between -1 and 1 |
    | one of several classes | softmax over each row | a probability for each class |

    For classes, a model usually outputs the logits and leaves the softmax to
    the loss, because `F.cross_entropy` applies the softmax itself. The same
    goes for a yes or no answer. `F.binary_cross_entropy_with_logits` takes
    the raw score and applies the sigmoid itself.

    Choosing the wrong output activation does not raise an error. ReLU at the
    output can never give a negative number, so a model predicting
    temperatures could never predict -5. A sigmoid can never give more than
    1, so a model that counts things could never predict more than 1. The
    model trains and the loss stops going down, with nothing to say why.

    Here is Linear, ReLU, Linear with 32 hidden neurons, fitted to 200 points
    by gradient descent. The errors are averages over 5 seeds.

    | Target | Output activation | Final error |
    |---|---|---|
    | temperatures from -5 to 5 | none | 0.050 |
    | temperatures from -5 to 5 | ReLU | 13.0 |
    | counts from 0 to 4 | none | 0.000 |
    | counts from 0 to 4 | sigmoid | 2.25 |

    With ReLU at the output, the temperature model does no better than always
    guessing the mean temperature, which has an error of 13.0. The sigmoid
    model never predicts a count above 1.

    So choose the output activation based on what the answer has to look
    like. This rule comes up whenever you design a model, including in
    Section 14.
    """,
)

ACTIVATIONS2_SOURCE = source_file(
    '''
    # activations2.py

    import torch
    from torch import Tensor


    # TODO: Apply the output activation that the job needs to scores, and return
    #   the result. kind is "number" for any number, "probability" for the
    #   probability of yes, "bounded" for a value that must stay between -1 and
    #   1, and "choice" for a probability for each class. For "choice", scores
    #   has one row per example and one column per class. Return the
    #   probabilities themselves, even though a real model would leave the
    #   softmax to the loss.
    def output_activation(
        scores: Tensor,
        kind: str,
    ) -> Tensor:
        ...


    # TODO: This model should give the probability of yes for each example, but
    #   its last line can give numbers above 1. Fix the last line.
    def yes_probability(
        x: Tensor,
        W1: Tensor,
        b1: Tensor,
        W2: Tensor,
        b2: Tensor,
    ) -> Tensor:
        hidden = torch.relu(x @ W1.T + b1)

        return torch.relu(hidden @ W2.T + b2)
    ''',
)

ACTIVATIONS2_SOLUTION = source_file(
    '''
    # activations2.py, a solution for comparison

    import torch
    from torch import Tensor


    def output_activation(
        scores: Tensor,
        kind: str,
    ) -> Tensor:
        if kind == "probability":
            return torch.sigmoid(scores)

        if kind == "bounded":
            return torch.tanh(scores)

        if kind == "choice":
            return torch.softmax(scores, dim=1)

        return scores


    # ReLU is right between the layers, but the answer has to be between 0 and 1.
    def yes_probability(
        x: Tensor,
        W1: Tensor,
        b1: Tensor,
        W2: Tensor,
        b2: Tensor,
    ) -> Tensor:
        hidden = torch.relu(x @ W1.T + b1)

        return torch.sigmoid(hidden @ W2.T + b2)
    ''',
)

OUTPUT_SCORES = small_tensor(213, 2, 3)


def reference_output(
    scores: Tensor,
    kind: str,
) -> Tensor:
    return {
        "number": scores,
        "probability": torch.sigmoid(scores),
        "bounded": torch.tanh(scores),
        "choice": torch.softmax(scores, dim=1),
    }[kind]


def check_activations2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="output_activation",
        description="gives the output activation each job needs",
        inputs=tuple((OUTPUT_SCORES, kind) for kind in ("number", "probability", "bounded", "choice")),
        reference=reference_output,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="yes_probability",
        description="gives a probability between 0 and 1",
        inputs=NETS,
        reference=lambda x, W1, b1, W2, b2: torch.sigmoid(F.linear(torch.relu(F.linear(x, W1, b1)), W2, b2)),
        expect=expect_tensor,
    )


ACTIVATIONS2 = Exercise(
    name="activations2",
    chapter=ACTIVATIONS,
    title="Which activation, where",
    lesson=ACTIVATIONS2_LESSON,
    source=ACTIVATIONS2_SOURCE,
    solution=ACTIVATIONS2_SOLUTION,
    hints={
        "output_activation": "Use one if for each kind, returning torch.sigmoid, torch.tanh or torch.softmax with "
        "dim=1. For a number, return the scores unchanged.",
        "yes_probability": "The ReLU on the hidden layer is correct. The last step needs the activation that gives a "
        "number between 0 and 1.",
    },
    introduces=(),
    check=check_activations2,
)


# ---------------------------------------------------------------------------
# soft1
# ---------------------------------------------------------------------------

SOFT1_LESSON = prose(
    """
    A model learns only through gradients. Some common operations in code
    pass no gradient back, and then nothing before them can learn.

    ```python
    w = torch.tensor(0.3, requires_grad=True)
    w > 0                     # tensor(True), with no grad_fn
    torch.sigmoid(w / 0.1)    # tensor(0.9526, grad_fn=<SigmoidBackward0>)
    ```

    A comparison gives True or False. If you change `w` by a small amount,
    the answer stays the same. So its derivative is 0, and autograd does not
    record the comparison at all. The same is true for an `if` on a tensor,
    for `argmax`, and for indexing with an index the model computed. With
    indexing, the row that was picked still gets a gradient and can learn.
    But the scores that chose that row get no gradient.

    Each of these hard steps has a soft version, which changes smoothly as
    its input changes.

    | Hard step | Soft version |
    |---|---|
    | `x > 0`, a jump from 0 to 1 | `torch.sigmoid(x / t)` |
    | `values[scores.argmax()]`, one row | `torch.softmax(scores / t, dim=0) @ values`, a weighted sum of rows |

    The number `t` is called the **temperature**. A small temperature makes
    the soft version almost the same as the hard step. A large temperature
    makes the change more gradual.

    `fit_scores` below learns which row of `values` to pick. In this example
    the rows are `[1, 0]`, `[0, 1]` and `[1, 1]`, and the target is `[0, 1]`,
    which is row 1. The scores start at `[1, 0, 0.5]`, which favours row 0.
    The table shows the result of 100 steps of gradient descent with a
    learning rate of 0.5, at three temperatures.

    | Temperature | Final softmax weights on the rows | Loss |
    |---|---|---|
    | 1 | 0.025, 0.939, 0.036 | 0.004 |
    | 0.1 | 0.000, 0.000, 1.000 | 1.0 |
    | 0.01 | 1.000, 0.000, 0.000 | 2.0 |

    At temperature 0.01, the softmax puts all its weight on row 0. Its
    gradient rounds to 0, so the scores never change. That is the same
    problem the hard step has. At 0.1, row 1 starts with a weight of 0.00005,
    so its score gets almost no gradient. Row 2 is closer to the target than
    row 0, so the weight moves over to row 2 and stays there. At temperature
    1, the weight ends up on row 1, the right answer.

    So a soft step gives up being exact in exchange for having a gradient,
    and the temperature controls that balance. Gates in Section 6 and lookups in Section 7 are built from these
    two soft steps.
    """,
)

SOFT1_SOURCE = source_file(
    '''
    # soft1.py

    import torch
    from torch import Tensor

    # The soft version of x > 0 for x = -1 and x = 1, at temperature 0.5.
    torch.sigmoid(torch.tensor([-1.0, 1.0]) / 0.5)    # tensor([0.1192, 0.8808])


    # TODO: Return the soft version of x > 0 from the lesson's table, for every
    #   entry of x. Each result is a number between 0 and 1, and it gets closer
    #   to the hard step as temperature gets smaller.
    def soft_step(
        x: Tensor,
        temperature: float,
    ) -> Tensor:
        ...


    # TODO: Return the soft version of values[scores.argmax()] from the lesson's
    #   table, at the given temperature. scores has shape (n,) and values has
    #   shape (n, d). The result is a weighted sum of the rows of values, with
    #   shape (d,).
    def soft_pick(
        scores: Tensor,
        values: Tensor,
        temperature: float,
    ) -> Tensor:
        ...


    # TODO: Finish the two lines in the loop. The first sets loss to the sum of
    #   the squared differences between soft_pick's result and target. The
    #   second takes one gradient descent step on scores with learning rate lr.
    def fit_scores(
        scores: Tensor,
        values: Tensor,
        target: Tensor,
        temperature: float,
        steps: int,
        lr: float,
    ) -> Tensor:
        scores = scores.clone().requires_grad_()

        for step in range(steps):
            loss = ...
            loss.backward()

            with torch.no_grad():
                ...
                scores.grad = None

        return scores.detach()
    ''',
)

SOFT1_SOLUTION = source_file(
    '''
    # soft1.py, a solution for comparison

    import torch
    from torch import Tensor


    def soft_step(
        x: Tensor,
        temperature: float,
    ) -> Tensor:
        return torch.sigmoid(x / temperature)


    def soft_pick(
        scores: Tensor,
        values: Tensor,
        temperature: float,
    ) -> Tensor:
        return torch.softmax(scores / temperature, dim=0) @ values


    def fit_scores(
        scores: Tensor,
        values: Tensor,
        target: Tensor,
        temperature: float,
        steps: int,
        lr: float,
    ) -> Tensor:
        scores = scores.clone().requires_grad_()

        for step in range(steps):
            loss = ((soft_pick(scores, values, temperature) - target) ** 2).sum()
            loss.backward()

            with torch.no_grad():
                scores -= lr * scores.grad
                scores.grad = None

        return scores.detach()
    ''',
)


def reference_fit_scores(
    scores: Tensor,
    values: Tensor,
    target: Tensor,
    temperature: float,
    steps: int,
    lr: float,
) -> Tensor:
    scores = scores.clone().requires_grad_()

    for _ in range(steps):
        loss = ((torch.softmax(scores / temperature, dim=0) @ values - target) ** 2).sum()
        loss.backward()

        with torch.no_grad():
            scores -= lr * scores.grad
            scores.grad.zero_()

    return scores.detach()


def check_soft1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="soft_step",
        description="gives sigmoid of x divided by the temperature",
        inputs=(
            (torch.tensor([-0.3, 0.0, 0.3]), 0.1),
            (small_tensor(214, 2, 3), 2.0),
        ),
        reference=lambda x, temperature: torch.sigmoid(x / temperature),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="soft_pick",
        description="weights the rows by softmax of the scores divided by the temperature",
        inputs=(
            (torch.tensor([0.5, 2.0, 1.0]), small_tensor(215, 3, 2), 1.0),
            (small_tensor(216, 4), small_tensor(217, 4, 3), 0.25),
        ),
        reference=lambda scores, values, temperature: torch.softmax(scores / temperature, dim=0) @ values,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="fit_scores",
        description="learns the scores by gradient descent through soft_pick",
        inputs=(
            (torch.tensor([1.0, 0.0, 0.5]), small_tensor(218, 3, 2), torch.tensor([0.0, 1.0]), 1.0, 20, 0.5),
            (torch.zeros(4), small_tensor(219, 4, 3), small_tensor(220, 3), 0.5, 15, 0.2),
        ),
        reference=reference_fit_scores,
        expect=expect_tensor,
    )


SOFT1 = Exercise(
    name="soft1",
    chapter=ACTIVATIONS,
    title="Steps that pass no gradient",
    lesson=SOFT1_LESSON,
    source=SOFT1_SOURCE,
    solution=SOFT1_SOLUTION,
    hints={
        "soft_step": "Divide x by the temperature, then take torch.sigmoid.",
        "soft_pick": "torch.softmax of the scores divided by the temperature, with dim=0, gives one weight per row. "
        "Then @ values adds up the rows, each multiplied by its weight.",
        "fit_scores": "The loss is the .sum() of the squared differences between soft_pick's result and target. The "
        "step subtracts lr * scores.grad from scores with -=, as the loop in relu2 does.",
    },
    introduces=(),
    check=check_soft1,
)


# ---------------------------------------------------------------------------
# soft2
# ---------------------------------------------------------------------------

SOFT2_LESSON = prose(
    """
    A Python dict finds the key that equals the query, and returns that key's
    value. Checking whether two things are equal is a hard step, so no
    gradient could teach a model what to look up. The soft pick from soft1
    fixes that. It needs a score for how well the query matches each key.

    When the keys and the query are rows of numbers, their dot product can be
    that score. The dot product is large when both rows have big entries in
    the same places. It is near 0 when they have no big entries in common,
    and negative when their signs disagree.

    ```python
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    keys @ torch.tensor([3.0, -1.0])    # tensor([ 3., -1.,  2.])
    ```

    A **soft lookup** turns those scores into weights with softmax, and
    returns the weighted sum of the values. Say the three keys above have the
    values 10, 20 and 30. Then the weights are 0.721, 0.013 and 0.265, and
    the lookup gives 15.4. A hard lookup would give 10, the value of the best
    matching key.

    A query with bigger entries spreads the scores further apart. That puts
    more of the softmax weight on the best match, the same effect a lower
    temperature has.

    | Query | Weights | Soft lookup |
    |---|---|---|
    | `[3, -1]` | 0.721, 0.013, 0.265 | 15.4 |
    | `[6, -2]` | 0.881, 0.000, 0.119 | 12.4 |
    | `[12, -4]` | 0.982, 0.000, 0.018 | 10.4 |

    So a model controls how sharp its lookups are through the size of the
    numbers in its queries. It learns that size through gradients, like any
    other part of the model.

    To do many lookups at once, put the queries in the rows of a matrix.
    `queries @ keys.T` gives one row of scores per query. A softmax over each
    row gives the weights. Then `@ values` computes the weighted sum of the
    values for every query at once. Section 7 builds its main block on
    exactly this calculation.
    """,
)

SOFT2_SOURCE = source_file(
    '''
    # soft2.py

    import torch
    from torch import Tensor

    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    scores = keys @ torch.tensor([3.0, -1.0])    # tensor([ 3., -1.,  2.])


    # TODO: Return the row of values for the key that matches query best, where
    #   the match score is the dot product of query with each key. query has
    #   shape (d,), keys has shape (n, d) and values has shape (n, v).
    def hard_lookup(
        query: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the soft lookup of query. Weight the rows of values by the
    #   softmax of the dot products of query with each key, and add them up.
    #   The result has shape (v,).
    #   Example: soft_lookup(torch.tensor([3.0, -1.0]), keys,
    #   torch.tensor([[10.0], [20.0], [30.0]])) returns tensor([15.4399]).
    def soft_lookup(
        query: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> Tensor:
        ...


    # TODO: queries has shape (m, d), with one query per row. Return the soft
    #   lookup of every query at once, without a loop. Row i of the result is
    #   the soft lookup of row i of queries, so the result has shape (m, v).
    def soft_lookups(
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> Tensor:
        ...
    ''',
)

SOFT2_SOLUTION = source_file(
    '''
    # soft2.py, a solution for comparison

    import torch
    from torch import Tensor

    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    scores = keys @ torch.tensor([3.0, -1.0])


    def hard_lookup(
        query: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> Tensor:
        return values[(keys @ query).argmax()]


    def soft_lookup(
        query: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> Tensor:
        return torch.softmax(keys @ query, dim=0) @ values


    def soft_lookups(
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> Tensor:
        return torch.softmax(queries @ keys.T, dim=1) @ values
    ''',
)


def check_soft2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="hard_lookup",
        description="gives the value of the best matching key",
        inputs=(
            (torch.tensor([-1.0, 3.0]), torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), small_tensor(221, 3, 2)),
            (small_tensor(222, 4), small_tensor(223, 5, 4), small_tensor(224, 5, 3)),
        ),
        reference=lambda query, keys, values: values[(keys @ query).argmax()],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="soft_lookup",
        description="weights the values by softmax of the dot products",
        inputs=(
            (torch.tensor([-1.0, 3.0]), torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), small_tensor(225, 3, 2)),
            (small_tensor(226, 4) / 2, small_tensor(227, 5, 4) / 2, small_tensor(228, 5, 3)),
        ),
        reference=lambda query, keys, values: torch.softmax(keys @ query, dim=0) @ values,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="soft_lookups",
        description="does a soft lookup for every row of queries at once",
        inputs=(
            (small_tensor(229, 2, 3) / 2, small_tensor(230, 4, 3) / 2, small_tensor(231, 4, 2)),
            (small_tensor(232, 3, 4) / 2, small_tensor(233, 5, 4) / 2, small_tensor(234, 5, 3)),
        ),
        reference=lambda queries, keys, values: torch.softmax(queries @ keys.T, dim=1) @ values,
        expect=expect_tensor,
    )


SOFT2 = Exercise(
    name="soft2",
    chapter=ACTIVATIONS,
    title="A lookup that learns",
    lesson=SOFT2_LESSON,
    source=SOFT2_SOURCE,
    solution=SOFT2_SOLUTION,
    hints={
        "hard_lookup": "keys @ query gives one score per key. Index values with the argmax of the scores.",
        "soft_lookup": "torch.softmax of keys @ query with dim=0 gives the weights, then @ values gives the weighted "
        "sum of the values.",
        "soft_lookups": "queries @ keys.T has one row of scores per query. Take the softmax over each row with "
        "dim=1, then @ values.",
    },
    introduces=(),
    check=check_soft2,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def xor_net(
    activation: Callable[[Tensor], Tensor],
    hidden: int,
    seed: int,
    steps: int = 2000,
    learning_rate: float = 0.1,
) -> tuple[float, int]:
    """relu2's Linear, activation, Linear on XOR, from relu2's start and with its training. Gives the final error
    and how many hidden neurons ended at or below 0 on all four points, where ReLU passes no gradient."""
    generator = torch.Generator().manual_seed(seed)
    W1 = torch.randn(hidden, 2, generator=generator).requires_grad_()
    b1 = torch.zeros(hidden, requires_grad=True)
    W2 = (torch.randn(1, hidden, generator=generator) / math.sqrt(hidden)).requires_grad_()
    b2 = torch.zeros(1, requires_grad=True)
    error = train_with([W1, b1, W2, b2], lambda x: activation(x @ W1.T + b1) @ W2.T + b2, XOR_X, XOR_Y, steps, learning_rate)

    with torch.no_grad():
        off = int(((XOR_X @ W1.T + b1) <= 0).all(dim=0).sum())

    return error, off


def measure_relu2(seeds: range = range(10)) -> dict[str, object]:
    """relu2's numbers. One Linear on XOR, Linear, ReLU, Linear with 8 neurons, and two Linears with 8 neurons
    between them and no activation, over the seeds."""
    with pinned_threads():
        return {
            "one Linear": reference_fit_linear(XOR_X, XOR_Y, 2000, 0.1),
            "Linear, ReLU, Linear, seed 0": reference_fit_relu_net(XOR_X, XOR_Y, 8, 2000, 0.1, torch.Generator().manual_seed(0)),
            "two Linears by seed": [xor_net(lambda x: x, 8, seed)[0] for seed in seeds],
        }


def spirals(
    seed: int,
    count: int = 200,
) -> tuple[Tensor, Tensor]:
    """Points on two spirals wound round each other, one class per spiral, with a little noise."""
    generator = torch.Generator().manual_seed(seed)
    turn = torch.rand(count, generator=generator) * 3 * math.pi
    labels = torch.randint(0, 2, (count,), generator=generator)
    radius = turn / (3 * math.pi)
    angle = turn + labels * math.pi
    points = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)

    return points + 0.02 * torch.randn(count, 2, generator=generator), labels


def deep_spiral_run(
    activation: Callable[[Tensor], Tensor],
    seed: int,
    depth: int = 6,
    width: int = 64,
    steps: int = 3000,
    learning_rate: float = 0.1,
) -> tuple[float, float, float]:
    """depth Linears of width neurons with the activation after each, then a Linear to the two classes, with
    PyTorch's starting weights and plain gradient descent on the spirals. Gives the first layer's gradient over the
    last layer's at the first step, then the final loss and the training accuracy."""
    torch.manual_seed(seed)
    layers = nn.ModuleList([nn.Linear(2, width)] + [nn.Linear(width, width) for _ in range(depth - 1)] + [nn.Linear(width, 2)])
    x, y = spirals(100 + seed)

    def scores(points: Tensor) -> Tensor:
        for layer in layers[:-1]:
            points = activation(layer(points))

        return layers[-1](points)

    for step in range(steps):
        F.cross_entropy(scores(x), y).backward()

        if step == 0:
            ratio = (layers[0].weight.grad.norm() / layers[-1].weight.grad.norm()).item()

        with torch.no_grad():
            for parameter in layers.parameters():
                parameter -= learning_rate * parameter.grad
                parameter.grad = None

    with torch.no_grad():
        final = scores(x)

    return ratio, F.cross_entropy(final, y).item(), (final.argmax(dim=1) == y).float().mean().item()


def measure_activations1(seeds: range = range(5)) -> dict[str, tuple[float, float, float]]:
    """activations1's table. For sigmoid, tanh and ReLU between six layers on the spirals, averaged over the seeds,
    the first layer's gradient over the last's at the start, the final loss and the training accuracy."""
    results: dict[str, tuple[float, float, float]] = {}

    with pinned_threads():
        for name, activation in (("sigmoid", torch.sigmoid), ("tanh", torch.tanh), ("ReLU", torch.relu)):
            runs = [deep_spiral_run(activation, seed) for seed in seeds]
            results[name] = tuple(sum(run[column] for run in runs) / len(runs) for column in range(3))

    return results


def measure_gelu1(
    hidden: int = 4,
    seeds: range = range(100),
) -> dict[str, tuple[int, float, int]]:
    """gelu1's table. relu2's XOR net with 4 hidden neurons and each activation, over 100 starts. For each, the runs
    stuck above an error of 0.01, the mean final error, and the stuck runs with a neuron off on all four points."""
    results: dict[str, tuple[int, float, int]] = {}

    with pinned_threads():
        for name, activation in (("ReLU", torch.relu), ("GELU", F.gelu), ("SiLU", F.silu)):
            runs = [xor_net(activation, hidden, seed) for seed in seeds]
            stuck = [run for run in runs if run[0] > 0.01]
            results[name] = (len(stuck), sum(run[0] for run in runs) / len(runs), sum(run[1] > 0 for run in stuck))

    return results


def output_run(
    target: Callable[[Tensor], Tensor],
    output: Callable[[Tensor], Tensor],
    seed: int,
    hidden: int = 32,
    steps: int = 4000,
    learning_rate: float = 0.01,
) -> tuple[float, float]:
    """Linear, ReLU, Linear with the given output activation, fitted to 200 points of target by gradient descent on
    the mean squared error. Gives the final error and the targets' variance, the error of always guessing their mean."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(200, 1, generator=generator) * 2 - 1
    y = target(x).squeeze(1)
    torch.manual_seed(seed)
    first, second = nn.Linear(1, hidden), nn.Linear(hidden, 1)
    parameters = [*first.parameters(), *second.parameters()]
    error = train_with(parameters, lambda x: output(second(torch.relu(first(x)))), x, y, steps, learning_rate)

    return error, y.var(unbiased=False).item()


def measure_activations2(seeds: range = range(5)) -> dict[str, tuple[float, float]]:
    """activations2's table. Temperatures from -5 to 5 and counts from 0 to 4, each with the right output activation
    and a wrong one, averaged over the seeds. Each entry is the final error and the error of guessing the mean."""
    targets = {
        "temperature": lambda x: 5 * torch.sin(3 * x),
        "count": lambda x: 2 + 2 * x,
    }
    cases = (
        ("temperature", "none", lambda scores: scores),
        ("temperature", "ReLU", torch.relu),
        ("count", "none", lambda scores: scores),
        ("count", "sigmoid", torch.sigmoid),
    )
    results: dict[str, tuple[float, float]] = {}

    with pinned_threads():
        for target, name, output in cases:
            runs = [output_run(targets[target], output, seed) for seed in seeds]
            results[f"{target}, {name}"] = (sum(run[0] for run in runs) / len(runs), sum(run[1] for run in runs) / len(runs))

    return results
