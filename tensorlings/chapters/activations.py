"""16_activations, the step between two Linears that lets models bend, which one to use, and why."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import Checker, Described, check_on_inputs, expect_close, expect_tensor, expect_value, seeded, small_tensor
from ..text import prose, source_file

ACTIVATIONS = Chapter(
    folder="16_activations",
    title="Activations",
    readme=source_file(
        """
        # Activations

        Two Linear layers in a row are one Linear layer. An activation between
        them, a simple function applied to every entry, stops that and lets a
        model bend to fit curved shapes. This chapter builds ReLU, shows a task
        a Linear model can't learn and a model with ReLU can, then covers
        sigmoid, tanh and GELU, and which job each one fits. It ends with
        steps like `argmax` that pass no gradient, and the soft versions that
        sigmoid and softmax give them.

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
    linear3 showed that two Linear layers in a row multiply out into one. The
    fix is a step between them that isn't a weighted sum, applied to every
    entry on its own. Such a step is an **activation**.

    The simplest is ReLU, short for rectified linear unit. It keeps positive
    numbers and turns negative ones into 0.

    ```python
    x = torch.tensor([-2.0, -0.5, 0.0, 1.5])
    torch.relu(x)    # tensor([0.0000, 0.0000, 0.0000, 1.5000])
    ```

    Its derivative is 1 where the input is positive and 0 where it is
    negative. At exactly 0 the derivative is taken as 0. So in backprop a
    ReLU passes the gradient on where its input was positive and blocks it
    everywhere else.

    Put ReLU between two Linears, `Linear, ReLU, Linear`, and the product no
    longer multiplies out. The outputs of the middle layer are called the
    hidden features, since nothing outside the model sees them. Each hidden
    neuron is off, giving 0, for some inputs and passes its weighted sum for
    others. The output is then made of straight pieces that meet at corners.
    With enough neurons those pieces can follow any smooth curve over a
    limited range.

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


    # TODO: Return the derivative of ReLU at every entry of x, 1.0 where the
    #   entry is above 0 and 0.0 everywhere else.
    def relu_derivative(x: Tensor) -> Tensor:
        ...


    # TODO: Return the hidden features, x through a Linear layer with weights W1
    #   and bias b1, then through ReLU.
    def hidden_features(
        x: Tensor,
        W1: Tensor,
        b1: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish the last line, so this passes x through Linear, ReLU, Linear,
    #   with the second layer's weights W2 and bias b2.
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
        "two_layer_net": "The second layer is a Linear on hidden, with W2 and b2.",
    },
    introduces=("activation",),
    check=check_relu1,
)


# ---------------------------------------------------------------------------
# relu2
# ---------------------------------------------------------------------------

RELU2_LESSON = prose(
    """
    Here is a task one Linear layer can't learn. Four points, each with two
    features, and a target that is 1 when exactly one feature is 1.

    ```python
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    ```

    This is called XOR. Raising either feature from 0 to 1 raises the target,
    but raising both lowers it again. A weighted sum can't do that, since
    raising a feature always moves the output the same way, whatever the other
    feature is. Trained on these points, a
    Linear model ends up giving 0.5 for every point. That is a mean squared
    error of 0.25, the best a straight answer can do.

    With 8 hidden neurons and ReLU between two Linears, gradient descent drives
    the error to almost 0. Some neurons switch on for one point and off for
    another. The second layer adds up the pieces into the right answer at
    every point.

    The hidden layer's weights start random, with a generator so every run
    matches. If they all started equal, every hidden neuron would compute the
    same thing and stay copies of each other.

    This is the whole reason deep learning stacks layers with activations
    between them. More neurons can fit more complicated shapes. Section 3's
    next chapter builds the model this makes, the multilayer perceptron.
    """,
)

RELU2_SOURCE = source_file(
    '''
    # relu2.py

    import math

    import torch
    from torch import Tensor


    # TODO: Finish the line that sets predictions, one Linear layer on x, so this
    #   trains it by gradient descent and returns the final mean squared error.
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


    # TODO: Finish the line that sets predictions, Linear, ReLU, Linear on x, so
    #   this trains the two layer net and returns the final mean squared error.
    #   W2 is scaled down so the output starts small. The initialization chapter
    #   shows why.
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
        "fit_relu_net": "The first layer is x @ W1.T + b1, through torch.relu, then the second layer with W2 and b2, "
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
    Two older activations squash any number into a fixed range. Sigmoid gives
    a number between 0 and 1, and tanh one between -1 and 1.

    ```text
    sigmoid(x) = 1 / (1 + exp(-x))
    tanh(x)    = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    ```

    ```python
    x = torch.tensor([-4.0, 0.0, 4.0])
    torch.sigmoid(x)    # tensor([0.0180, 0.5000, 0.9820])
    torch.tanh(x)       # tensor([-0.9993,  0.0000,  0.9993])
    ```

    Sigmoid is softmax over two classes, with scores `x` and 0, so its output
    reads as a probability. Their derivatives come out in terms of their own
    outputs. With `s = sigmoid(x)` the derivative is `s * (1 - s)`, at most
    0.25. With `t = tanh(x)` it is `1 - t ** 2`, at most 1.

    Both flatten out for big inputs. At `x = 4`, sigmoid's derivative is
    already 0.018, so a neuron pushed far to one side passes back almost no
    gradient and stops learning. This is saturation. A stack of sigmoid layers
    multiplies many of those small derivatives together, and the gradient
    reaching the first layers all but vanishes. ReLU's derivative is 1 for any
    positive input, which is the main reason it replaced them in hidden
    layers.

    They still have jobs. A sigmoid gives a yes or no probability, or a gate
    that scales a signal between off and on. A tanh gives a value held between
    -1 and 1.
    """,
)

ACTIVATIONS1_SOURCE = source_file(
    '''
    # activations1.py

    import torch
    from torch import Tensor


    # TODO: Return sigmoid of every entry of x, from its formula in the lesson,
    #   without torch.sigmoid.
    def sigmoid(x: Tensor) -> Tensor:
        ...


    # sigmoid's derivative comes out in terms of its own output.
    def sigmoid_derivative(x: Tensor) -> Tensor:
        s = torch.sigmoid(x)

        return s * (1 - s)


    # TODO: Return the derivative of tanh at every entry of x, the way
    #   sigmoid_derivative does for sigmoid.
    def tanh_derivative(x: Tensor) -> Tensor:
        ...


    # TODO: Return the fraction of entries of x where sigmoid is saturated, where
    #   its derivative is below 0.01.
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


def check_activations1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="sigmoid",
        description="squashes every entry between 0 and 1",
        inputs=(
            (torch.tensor([-4.0, 0.0, 4.0]),),
            (small_tensor(211, 2, 3),),
        ),
        reference=torch.sigmoid,
        expect=expect_tensor,
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
        "tanh_derivative": "Take t = torch.tanh(x), then the derivative from the lesson in terms of t.",
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
    Language models like GPT-2 and BERT use GELU, from Hendrycks and Gimpel's
    2016 paper. ReLU lets an input through when it is positive and blocks it
    when it isn't. GELU lets it through in part, weighted by how likely a
    random number from the bell curve is to be smaller than it.

    ```text
    GELU(x) = x * Phi(x),   Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    ```

    `Phi` is the chance that a number drawn from the bell curve with mean 0
    and spread 1 is below `x`. `erf` is the error function, which `torch.erf` computes. For a
    big positive `x`, `Phi` is almost 1 and GELU is almost `x`, like ReLU. For
    a big negative `x` it is almost 0. Near 0 it bends smoothly, with a small
    dip below 0, where ReLU has a sharp corner.

    ```python
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    F.gelu(x)    # tensor([-0.1587,  0.0000,  0.8413,  1.9545])
    ```

    The paper also gives a faster approximation with tanh, which GPT-2 used.

    ```text
    0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3)))
    ```

    `F.gelu(x, approximate="tanh")` computes that one.

    A close cousin is SiLU, `x * sigmoid(x)`, which lets the input through
    weighted by a sigmoid instead. Llama and most current language models use
    it, inside a gated layer that Section 7 builds.
    """,
)

GELU1_SOURCE = source_file(
    '''
    # gelu1.py

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Return GELU of every entry of x, from the paper's formula with
    #   torch.erf. It should match F.gelu(x).
    def gelu(x: Tensor) -> Tensor:
        ...


    # TODO: Return the tanh approximation of GELU from the lesson. It should
    #   match F.gelu(x, approximate="tanh").
    def gelu_tanh(x: Tensor) -> Tensor:
        ...


    # TODO: Return SiLU of every entry of x, x times sigmoid of x. It should match
    #   F.silu(x).
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


GELU1 = Exercise(
    name="gelu1",
    chapter=ACTIVATIONS,
    title="GELU from its paper",
    lesson=GELU1_LESSON,
    source=GELU1_SOURCE,
    solution=GELU1_SOLUTION,
    hints={
        "gelu": "Phi(x) is 0.5 * (1 + torch.erf(x / math.sqrt(2))). Multiply x by it.",
        "gelu_tanh": "Write the formula as it stands, with torch.tanh, math.sqrt and math.pi.",
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
    Each activation has a job. Between layers its job is to bend, and ReLU is
    the plain default there, with GELU and SiLU common in language models. At the
    output, its job is to put the answer in the form the task needs.

    | The model predicts | Output activation | Because it gives |
    |---|---|---|
    | any number, like a temperature | none | any number |
    | the probability of yes | sigmoid | a number between 0 and 1 |
    | a value held between -1 and 1 | tanh | a number between -1 and 1 |
    | one of several classes | softmax over each row | a probability for each class |

    For classes, a model usually outputs the logits and leaves the softmax to
    the loss, as `F.cross_entropy` does. The same goes for a yes or no answer,
    where `F.binary_cross_entropy_with_logits` takes the raw score and does
    the sigmoid itself.

    The wrong choice fails without an error. ReLU at the output can never give
    a negative number, so a model predicting temperatures could never say -5.
    A sigmoid can never give more than 1, so a model counting things would top
    out at 1. The model trains, the loss stops falling, and nothing says why.

    So pick the output activation from what the answer has to look like. This
    rule comes back whenever you design a model, in Section 13 too.
    """,
)

ACTIVATIONS2_SOURCE = source_file(
    '''
    # activations2.py

    import torch
    from torch import Tensor


    # TODO: Return scores through the output activation the job needs. kind is
    #   "number" for any number, "probability" for the probability of yes,
    #   "bounded" for a value held between -1 and 1, and "choice" for a
    #   probability for each class, one row of classes per example. Here, return
    #   the probabilities themselves.
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
        "output_activation": "One if for each kind, returning torch.sigmoid, torch.tanh or torch.softmax with "
        "dim=1, and the scores as they are for a number.",
        "yes_probability": "The hidden ReLU is right. The last step needs the activation that gives a number "
        "between 0 and 1.",
    },
    introduces=(),
    check=check_activations2,
)


# ---------------------------------------------------------------------------
# soft1
# ---------------------------------------------------------------------------

SOFT1_LESSON = prose(
    """
    A model learns only through gradients. Some everyday steps of code pass
    none back, and then nothing before them can learn.

    ```python
    w = torch.tensor(0.3, requires_grad=True)
    w > 0                     # tensor(True), with no grad_fn
    torch.sigmoid(w / 0.1)    # tensor(0.9526, grad_fn=<SigmoidBackward0>)
    ```

    A comparison gives True or False. Nudge `w` a little and the answer stays
    the same, so its derivative is 0 and autograd doesn't record it at all.
    The same goes for an `if` on a tensor, for `argmax`, and for indexing with
    an index the model computed. The row that was picked can still learn, but
    the scores that chose it get no gradient.

    Each hard step has a soft version that changes smoothly.

    | Hard step | Soft version |
    |---|---|
    | `x > 0`, a jump from 0 to 1 | `torch.sigmoid(x / t)` |
    | `values[scores.argmax()]`, one row | `torch.softmax(scores / t, dim=0) @ values`, a weighted sum of rows |

    The number `t` is the **temperature**. A small one makes the soft version
    nearly hard, and a large one blurs it.

    `fit_scores` below learns which row of `values` to pick. It runs gradient
    descent on scores that start out favouring row 0, while the target is row
    1. Here are 100 steps at three temperatures.

    | Temperature | Weights on the rows at the end | Loss |
    |---|---|---|
    | 1 | 0.025, 0.939, 0.036 | 0.004 |
    | 0.1 | 0.000, 0.000, 1.000 | 1.0 |
    | 0.01 | 1.000, 0.000, 0.000 | 2.0 |

    At 0.01 the softmax sits entirely on row 0. Its gradient rounds to 0, so
    the scores never move, the hard step's problem back again. At 0.1 the
    weights slide onto row 2 and stick there. At 1 they find row 1.

    So a soft step trades exactness for a gradient, and the temperature sets
    the trade. Gates in Section 6 and lookups in Section 7 are built from these
    two soft steps.
    """,
)

SOFT1_SOURCE = source_file(
    '''
    # soft1.py

    import torch
    from torch import Tensor

    torch.sigmoid(torch.tensor([-1.0, 1.0]) / 0.5)    # tensor([0.1192, 0.8808])


    # TODO: Return a soft version of x > 0, a number between 0 and 1 for each
    #   entry of x, closer to the hard step the smaller temperature is.
    def soft_step(
        x: Tensor,
        temperature: float,
    ) -> Tensor:
        ...


    # TODO: Return a soft version of values[scores.argmax()], with scores of
    #   shape (n,) and values of shape (n, d). The result has shape (d,).
    def soft_pick(
        scores: Tensor,
        values: Tensor,
        temperature: float,
    ) -> Tensor:
        ...


    # TODO: Finish the loop. Each step takes the loss, the sum of the squared
    #   differences between soft_pick's result and target, then one gradient
    #   descent step on scores with learning rate lr.
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
            ...

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
                scores.grad.zero_()

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
        description="gives sigmoid of x over the temperature",
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
        description="weights the rows by softmax of the scores over the temperature",
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
        "soft_pick": "torch.softmax of the scores over the temperature with dim=0 gives one weight per row. "
        "Then @ values sums the rows with those weights.",
        "fit_scores": "Take the loss with soft_pick and target and call loss.backward(). Then inside "
        "torch.no_grad(), take lr * scores.grad from scores and zero the gradient, as relu2's loop does.",
    },
    introduces=(),
    check=check_soft1,
)


# ---------------------------------------------------------------------------
# soft2
# ---------------------------------------------------------------------------

SOFT2_LESSON = prose(
    """
    A Python dict finds the key equal to the query and gives back its value.
    "Equal" is a hard step, so no gradient could teach a model what to look
    up. soft1's soft pick fixes that, once a score says how well the query
    matches each key.

    With the keys and the query as rows of numbers, the dot product is that
    score. It is large when both have big entries in the same places, near 0
    when they share nothing, and negative when their signs disagree.

    ```python
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    keys @ torch.tensor([3.0, -1.0])    # tensor([ 3., -1.,  2.])
    ```

    A **soft lookup** turns those scores into weights with softmax and returns
    the weighted sum of the values. With values 10, 20 and 30 for the three
    keys, the weights are 0.721, 0.013 and 0.265, and the lookup gives 15.4. A
    hard lookup would give 10, the value of the best match.

    A query with bigger entries spreads the scores further apart, which
    sharpens the softmax the way a lower temperature does.

    | Query | Weights | Soft lookup |
    |---|---|---|
    | `[3, -1]` | 0.721, 0.013, 0.265 | 15.4 |
    | `[6, -2]` | 0.881, 0.000, 0.119 | 12.4 |
    | `[12, -4]` | 0.982, 0.000, 0.018 | 10.4 |

    So a model sets how sharp its lookups are through the size of its numbers,
    and learns that like anything else.

    Many queries go at once as the rows of a matrix. `queries @ keys.T` gives
    one row of scores per query, a softmax over each row gives the weights,
    and `@ values` sums the values for every query together. Section 7 builds
    its main block on exactly this.
    """,
)

SOFT2_SOURCE = source_file(
    '''
    # soft2.py

    import torch
    from torch import Tensor

    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    scores = keys @ torch.tensor([3.0, -1.0])    # tensor([ 3., -1.,  2.])


    # TODO: Return the value of the key that matches query best, by the dot
    #   product. query has shape (d,), keys (n, d) and values (n, v).
    def hard_lookup(
        query: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the soft lookup of query, the values weighted by softmax of
    #   how well query matches each key. The result has shape (v,).
    def soft_lookup(
        query: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the soft lookup of every row of queries, of shape (m, d), as
    #   one row each, in one go without a loop. The result has shape (m, v).
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
        "soft_lookup": "torch.softmax of keys @ query with dim=0 gives the weights, then @ values sums the values.",
        "soft_lookups": "queries @ keys.T has one row of scores per query. Take the softmax over each row with "
        "dim=1, then @ values.",
    },
    introduces=(),
    check=check_soft2,
)
