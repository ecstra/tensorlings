"""17_mlp, Linear layers with activations between them, built as a module, trained, and sized."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import Checker, Described, Mismatch, check_on_inputs, expect_close, expect_tensor, expect_value, small_tensor
from ..text import prose, source_file
from .linear_layers import expect_module

MLP = Chapter(
    folder="17_mlp",
    title="Multilayer perceptrons",
    readme=source_file(
        """
        # Multilayer perceptrons

        A multilayer perceptron, or MLP, is a stack of linear layers with an
        activation between each pair of layers. It is the simplest model that
        can learn curves and not only straight lines, and nearly every bigger
        model has MLPs inside it. In this chapter you build an MLP with
        `nn.Sequential`, then build it again as your own module. Then you train
        it and see how its size affects what it can fit.

        ## Further reading

        - [Dive into Deep Learning, multilayer perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)
        - [3Blue1Brown, but what is a neural network](https://www.3blue1brown.com/lessons/neural-networks)
        """,
    ),
)

CURVE_X = torch.linspace(-3.0, 3.0, 64).unsqueeze(1)
CURVE_Y = torch.sin(CURVE_X).squeeze(1)


def noisy_curve(
    seed: int,
    count: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(count, 1, generator=generator) * 6 - 3

    return x, torch.sin(x).squeeze(1) + 0.3 * torch.randn(count, generator=generator)


def seeded_mlp(
    seed: int,
    sizes: tuple[int, ...],
) -> nn.Sequential:
    torch.manual_seed(seed)
    layers: list[nn.Module] = []

    for index, (inputs, outputs) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(inputs, outputs))

        if index < len(sizes) - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def train_reference(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    steps: int,
    learning_rate: float,
) -> float:
    for _ in range(steps):
        loss = ((model(x).squeeze(1) - y) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= learning_rate * parameter.grad
                parameter.grad = None

    return loss.item()


def layer_shapes(model: nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    return [(type(layer).__name__, tuple(layer.weight.shape) if hasattr(layer, "weight") else ()) for layer in model]


# ---------------------------------------------------------------------------
# mlp1
# ---------------------------------------------------------------------------

MLP1_LESSON = prose(
    """
    A multilayer perceptron, or MLP, is a stack of linear layers with an
    activation between each pair of layers. It is the simplest model that can
    learn curves and not only straight lines. Nearly every bigger model has
    MLPs inside it.

    `nn.Sequential` takes modules and runs them in order. When you call it, it
    passes the input to the first module, passes that output to the second
    module, and so on. `nn.ReLU()` is the ReLU activation wrapped as a module,
    so it can go in the chain.

    ```python
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    model(torch.zeros(5, 2)).shape    # torch.Size([5, 1])
    model[0]                          # Linear(in_features=2, out_features=8, bias=True)
    ```

    The first layer's input size must equal the number of input features. The
    last layer's output size must equal the number of outputs you want. Each
    Linear and ReLU pair before the last Linear is a hidden layer, and you
    choose its size. The number of neurons in a hidden layer is its width. The
    number of hidden layers is the depth of the MLP.

    `nn.Sequential(*layers)` builds the chain from a Python list named
    `layers`. This helps when the depth is a number passed in, because you can
    build the list in a loop. The `*` unpacks the list into separate
    arguments.

    A Linear layer from `n` inputs to `h` outputs holds `n * h + h`
    parameters. That is `n * h` weights and `h` biases. Every Linear layer in
    the MLP counts the same way, with its own input and output sizes.
    """,
)

MLP1_SOURCE = source_file(
    '''
    # mlp1.py

    import torch
    import torch.nn as nn

    # A Linear layer, a ReLU and a second Linear layer, run in order by
    # nn.Sequential. It takes 2 inputs, has 8 hidden neurons, and gives 1 output.
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


    # TODO: Return an MLP with one hidden layer, as an nn.Sequential of a Linear
    #   layer, a ReLU and a second Linear layer. It takes `inputs` features, has
    #   `hidden` neurons in its hidden layer, and gives `outputs` outputs.
    #   Example: make_mlp(2, 8, 1) holds the same layers as model above.
    def make_mlp(
        inputs: int,
        hidden: int,
        outputs: int,
    ) -> nn.Sequential:
        ...


    # TODO: Finish the line in the loop, so this returns an MLP with `depth`
    #   hidden layers. Each hidden layer is a Linear layer with `hidden` outputs,
    #   followed by a ReLU. After them, a last Linear layer gives `outputs`
    #   outputs. Example: make_deep_mlp(2, 8, 1, 3) holds Linear, ReLU, Linear,
    #   ReLU, Linear, ReLU, Linear.
    def make_deep_mlp(
        inputs: int,
        hidden: int,
        outputs: int,
        depth: int,
    ) -> nn.Sequential:
        layers = []
        size = inputs

        for _ in range(depth):
            layers.extend(...)
            size = hidden

        layers.append(nn.Linear(size, outputs))

        return nn.Sequential(*layers)


    # TODO: Return the number of parameters in make_mlp(inputs, hidden, outputs),
    #   which is the weights and biases of both Linear layers. Compute it from
    #   the three sizes, without building the model.
    #   Example: mlp_parameters(2, 8, 1) returns 33.
    def mlp_parameters(
        inputs: int,
        hidden: int,
        outputs: int,
    ) -> int:
        ...
    ''',
)

MLP1_SOLUTION = source_file(
    '''
    # mlp1.py, a solution for comparison

    import torch
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


    def make_mlp(
        inputs: int,
        hidden: int,
        outputs: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, outputs),
        )


    def make_deep_mlp(
        inputs: int,
        hidden: int,
        outputs: int,
        depth: int,
    ) -> nn.Sequential:
        layers = []
        size = inputs

        for _ in range(depth):
            layers.extend([nn.Linear(size, hidden), nn.ReLU()])
            size = hidden

        layers.append(nn.Linear(size, outputs))

        return nn.Sequential(*layers)


    # The hidden layer's weights and biases, then the output layer's.
    def mlp_parameters(
        inputs: int,
        hidden: int,
        outputs: int,
    ) -> int:
        return inputs * hidden + hidden + hidden * outputs + outputs
    ''',
)


def expect_layers(
    call: str,
    actual: object,
    expected: nn.Sequential,
) -> None:
    model = expect_module(call, actual)

    if not isinstance(model, nn.Sequential):
        raise Mismatch(f"{call} is a {type(model).__name__}. It should be an nn.Sequential.")

    got = layer_shapes(model)
    wanted = layer_shapes(expected)

    if got != wanted:
        names = ", ".join(name for name, _ in wanted)
        raise Mismatch(f"{call} should hold {names}, with the Linear weights shaped {[shape for _, shape in wanted if shape]}, but it holds {', '.join(name for name, _ in got)} shaped {[shape for _, shape in got if shape]}.")


def check_mlp1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="make_mlp",
        description="gives Linear, ReLU, Linear with the right sizes",
        inputs=(
            (2, 8, 1),
            (784, 100, 10),
        ),
        reference=lambda inputs, hidden, outputs: seeded_mlp(0, (inputs, hidden, outputs)),
        expect=expect_layers,
    )
    check_on_inputs(
        checker=checker,
        function="make_deep_mlp",
        description="gives depth hidden layers, then the output layer",
        inputs=(
            (2, 8, 1, 3),
            (5, 4, 2, 1),
            (3, 6, 3, 0),
        ),
        reference=lambda inputs, hidden, outputs, depth: seeded_mlp(0, (inputs, *[hidden] * depth, outputs)),
        expect=expect_layers,
    )
    check_on_inputs(
        checker=checker,
        function="mlp_parameters",
        description="counts the parameters without building the model",
        inputs=(
            (2, 8, 1),
            (784, 100, 10),
        ),
        reference=lambda inputs, hidden, outputs: sum(parameter.numel() for parameter in seeded_mlp(0, (inputs, hidden, outputs)).parameters()),
        expect=expect_value,
    )
    checker.test(
        function="mlp_parameters",
        description="works the count out without building the model",
        run=check_counts_without_building,
    )


class NoLayers:
    """Stands in for torch.nn while mlp_parameters runs, so a count that builds the model fails."""

    def __getattr__(self, name: str) -> object:
        raise Mismatch(f"mlp_parameters used nn.{name}, so it built the model. Work the count out from inputs, hidden and outputs alone.")


def check_counts_without_building(found: object) -> None:
    namespace = found.__globals__
    had_nn = "nn" in namespace
    original = namespace.get("nn")
    namespace["nn"] = NoLayers()

    try:
        found(784, 100, 10)
    finally:
        if had_nn:
            namespace["nn"] = original
        else:
            del namespace["nn"]


MLP1 = Exercise(
    name="mlp1",
    chapter=MLP,
    title="Stacking an MLP",
    lesson=MLP1_LESSON,
    source=MLP1_SOURCE,
    solution=MLP1_SOLUTION,
    hints={
        "make_mlp": "Copy the pattern of the line that sets model, using inputs, hidden and outputs instead of 2, 8 "
        "and 1.",
        "make_deep_mlp": "Each hidden layer adds two modules to the list, a Linear layer from size to hidden and then "
        "a ReLU. layers.extend takes a list and appends each of its items.",
        "mlp_parameters": "The hidden layer holds inputs * hidden weights and hidden biases. The output layer counts "
        "the same way, with hidden in place of inputs and outputs in place of hidden.",
    },
    introduces=(),
    check=check_mlp1,
)


# ---------------------------------------------------------------------------
# mlp2
# ---------------------------------------------------------------------------

MLP2_LESSON = prose(
    """
    `nn.Sequential` works for a model whose layers run one after another in a
    single chain. Most models need more control than that. So they are written
    as a subclass of `nn.Module` with their own `forward` method. This exercise
    writes the MLP that way. It takes a list of sizes, like `[2, 8, 8, 1]`.
    The first number is the number of inputs and the last is the number of
    outputs. The numbers in between are the widths of the hidden layers.

    The MLP keeps its layers in a list, and a plain Python list causes a
    silent bug. A module finds the parameters of the modules stored directly in
    its attributes, or in containers it knows about. It does not look inside a
    plain list. So the Linear layers in a plain list are missing from
    `model.parameters()`. A training loop over `model.parameters()` then
    trains nothing, and no error warns you.

    ```python
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(2, 2)]

    len(list(Broken().parameters()))    # 0
    ```

    `nn.ModuleList` is a list that a module does know about. You can loop over
    it and index it like a list. The parameters of every module inside it are
    included in `model.parameters()`.

    In `forward`, an activation goes after every layer except the last one.
    The last layer produces the model's output, so it gets no ReLU.
    activations2 showed that ReLU at the output can break a model. `enumerate`
    gives each layer's position as you loop, so you can tell when you have
    reached the last layer.
    """,
)

MLP2_SOURCE = source_file(
    '''
    # mlp2.py

    import torch
    import torch.nn as nn
    from torch import Tensor


    # TODO: Fix two things. First, self.layers is a plain list, so the model has
    #   no parameters. Change that line so the model finds the layers. Second,
    #   finish the loop in forward, so ReLU is applied after every layer except
    #   the last. Example: MLP([2, 8, 1]) has 33 parameters.
    class MLP(nn.Module):
        def __init__(self, sizes: list[int]) -> None:
            super().__init__()
            self.layers = [nn.Linear(inputs, outputs) for inputs, outputs in zip(sizes[:-1], sizes[1:])]

        def forward(self, x: Tensor) -> Tensor:
            for position, layer in enumerate(self.layers):
                x = layer(x)

                if position < len(self.layers) - 1:
                    x = ...

            return x
    ''',
)

MLP2_SOLUTION = source_file(
    '''
    # mlp2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor


    # nn.ModuleList registers each layer, so model.parameters() includes their
    # parameters.
    class MLP(nn.Module):
        def __init__(self, sizes: list[int]) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(inputs, outputs) for inputs, outputs in zip(sizes[:-1], sizes[1:])])

        def forward(self, x: Tensor) -> Tensor:
            for position, layer in enumerate(self.layers):
                x = layer(x)

                if position < len(self.layers) - 1:
                    x = torch.relu(x)

            return x
    ''',
)


def check_mlp_class(found: type) -> None:
    for sizes in ([2, 8, 1], [3, 5, 5, 2]):
        call = f"MLP({sizes})"
        model = expect_module(call, found(sizes))
        reference = seeded_mlp(0, tuple(sizes))
        count = sum(parameter.numel() for parameter in model.parameters())
        wanted = sum(parameter.numel() for parameter in reference.parameters())

        if count != wanted:
            raise Mismatch(f"{call} has {count} parameters, but it should have {wanted}. Its layers have to be in an nn.ModuleList.")

        linears = [layer for layer in reference if isinstance(layer, nn.Linear)]

        with torch.no_grad():
            for mine, theirs in zip(model.layers, linears):
                mine.weight.copy_(theirs.weight)
                mine.bias.copy_(theirs.bias)

        x = small_tensor(214 + len(sizes), 4, sizes[0])
        expect_tensor(
            call=f"{call}, with the weights of an nn.Sequential copied in, on x of shape {tuple(x.shape)}",
            actual=model(x),
            expected=reference(x),
        )


def check_mlp2(checker: Checker) -> None:
    checker.test(
        function="MLP",
        description="registers its layers and puts ReLU between them",
        run=check_mlp_class,
    )


MLP2 = Exercise(
    name="mlp2",
    chapter=MLP,
    title="An MLP as a module",
    lesson=MLP2_LESSON,
    source=MLP2_SOURCE,
    solution=MLP2_SOLUTION,
    hints={
        "MLP": "Wrap the list in nn.ModuleList(...). In forward, the missing line passes x through torch.relu.",
    },
    introduces=(),
    check=check_mlp2,
)


# ---------------------------------------------------------------------------
# mlp3
# ---------------------------------------------------------------------------

MLP3_LESSON = prose(
    """
    You train an MLP with the same loop as in Section 2. The difference is
    that the parameters belong to the model, instead of being a `w` and `b`
    you made by hand. In each step the model makes predictions, and the loss
    compares them with the targets. Then `.backward()` computes the gradient
    of every parameter, and a gradient step updates all the parameters.

    ```python
    for _ in range(steps):
        loss = ((model(x).squeeze(1) - y) ** 2).mean()
        loss.backward()
        gradient_step(model, learning_rate)
    ```

    `gradient_step` is the function from modules1. It subtracts the learning
    rate times the gradient from every parameter of the model, then clears
    the gradient. `.squeeze(1)` turns the model's output from shape
    `(examples, 1)` into shape `(examples,)`, which is the shape of the
    targets. linreg2 showed what goes wrong when those shapes differ.

    In this exercise an MLP with one hidden layer of 32 neurons learns
    `sin(x)` for `x` between -3 and 3, from 64 points. After 1000 steps with a
    learning rate of 0.1, the mean squared error falls to about 0.0004. A
    single Linear layer trained on the same points can only fit a straight
    line, so its error stays above 0.1.

    The same loop trains any model built from modules, however many layers it
    has. Only the model, the loss and the data change from task to task.
    """,
)

MLP3_SOURCE = source_file(
    '''
    # mlp3.py

    import torch
    import torch.nn as nn
    from torch import Tensor


    # The gradient step from modules1.
    def gradient_step(
        model: nn.Module,
        learning_rate: float,
    ) -> None:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= learning_rate * parameter.grad
                parameter.grad = None


    # TODO: Finish the line that sets loss, so this trains the model and returns
    #   the final loss. loss is the mean squared error between the model's
    #   predictions on x and the targets y. x has shape (examples, features), y
    #   has shape (examples,), and the model's output has shape (examples, 1).
    def train(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> float:
        for _ in range(steps):
            loss = ...
            loss.backward()
            gradient_step(model, learning_rate)

        return loss.item()


    # TODO: Return the mean squared error of the model's predictions on x against
    #   y, as a plain Python number. Autograd must not record anything while
    #   this runs.
    def evaluate(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        ...
    ''',
)

MLP3_SOLUTION = source_file(
    '''
    # mlp3.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor


    def gradient_step(
        model: nn.Module,
        learning_rate: float,
    ) -> None:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= learning_rate * parameter.grad
                parameter.grad = None


    def train(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> float:
        for _ in range(steps):
            loss = ((model(x).squeeze(1) - y) ** 2).mean()
            loss.backward()
            gradient_step(model, learning_rate)

        return loss.item()


    def evaluate(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        with torch.no_grad():
            return ((model(x).squeeze(1) - y) ** 2).mean().item()
    ''',
)


def check_mlp_training(found: object) -> None:
    call = "train on sin(x) with an MLP of width 32, for 1000 steps with learning rate 0.1"
    final = found(seeded_mlp(0, (1, 32, 1)), CURVE_X.clone(), CURVE_Y.clone(), 1000, 0.1)
    expected = train_reference(seeded_mlp(0, (1, 32, 1)), CURVE_X, CURVE_Y, 1000, 0.1)

    # A loss stuck far above the reference is the shape trap of linreg2, (examples, 1) against (examples,).
    if isinstance(final, (int, float)) and final > 100 * expected:
        raise Mismatch(
            f"{call} ends at a loss of {final:.4g}, where it should reach about {expected:.4g}. Check that the "
            "predictions have shape (examples,) before subtracting y, as linreg2 warned.",
        )

    expect_close(
        call=call,
        actual=final,
        expected=expected,
        tolerance=1e-5,
    )


def check_mlp3(checker: Checker) -> None:
    checker.test(
        function="train",
        description="trains the MLP down to a small loss",
        run=check_mlp_training,
    )
    check_on_inputs(
        checker=checker,
        function="evaluate",
        description="gives the mean squared error of the model",
        inputs=(
            (Described(seeded_mlp(1, (1, 8, 1)), "an MLP of width 8"), Described(CURVE_X, "x"), Described(CURVE_Y, "y")),
            (Described(seeded_mlp(2, (2, 4, 1)), "an MLP with 2 inputs"), small_tensor(220, 5, 2), small_tensor(221, 5)),
        ),
        reference=lambda model, x, y: ((model(x).squeeze(1) - y) ** 2).mean().item(),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )
    checker.test(
        function="evaluate",
        description="runs the model without recording for autograd",
        run=check_evaluate_records_nothing,
    )


class GradWatcher(nn.Module):
    """An MLP that notes whether autograd was recording each time it ran."""

    def __init__(self, sizes: tuple[int, ...] = (1, 8, 1)) -> None:
        super().__init__()
        self.inner = seeded_mlp(3, sizes)
        self.recording: list[bool] = []

    def forward(self, x: Tensor) -> Tensor:
        self.recording.append(torch.is_grad_enabled())

        return self.inner(x)


def check_evaluate_records_nothing(found: object) -> None:
    watcher = GradWatcher()
    found(watcher, CURVE_X.clone(), CURVE_Y.clone())

    if any(watcher.recording):
        raise Mismatch("evaluate ran the model while autograd was recording. Put the work inside a with torch.no_grad() block.")


MLP3 = Exercise(
    name="mlp3",
    chapter=MLP,
    title="Training an MLP",
    lesson=MLP3_LESSON,
    source=MLP3_SOURCE,
    solution=MLP3_SOLUTION,
    hints={
        "train": "Call the model on x, .squeeze(1) the result, and take the mean squared error against y.",
        "evaluate": "Compute the same mean squared error inside a with torch.no_grad() block, and return it as a "
        "plain number with .item().",
    },
    introduces=(),
    check=check_mlp3,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# mlp4
# ---------------------------------------------------------------------------

MLP4_LESSON = prose(
    """
    How wide should a hidden layer be? To find out, MLPs with six different
    widths were each trained on 20 noisy points of `sin(x)`. Each trained for
    10,000 steps with a learning rate of 0.005. Then each was measured on a
    test set of 1000 new points. Each number in the table is an average over
    10 seeds, and each seed has its own points and its own starting weights.
    Even a model that predicted `sin(x)` exactly would have a mean squared
    error of 0.09 on new points, because of the noise. So no model can do
    better than 0.09.

    The last column uses early stopping, from Section 2. Every 100 steps the
    model was measured on 200 validation points. The column shows the test
    error at the check where the validation error was lowest.

    | Width | Training error | Test error at the end | Test error, stopped early |
    |---|---|---|---|
    | 1 | 0.238 | 0.305 | 0.294 |
    | 4 | 0.106 | 0.230 | 0.197 |
    | 8 | 0.067 | 0.177 | 0.173 |
    | 32 | 0.057 | 0.155 | 0.152 |
    | 128 | 0.050 | 0.154 | 0.147 |
    | 512 | 0.041 | 0.163 | 0.138 |

    Width 1 is too narrow to bend into a curve, so it underfits. Both its
    training error and its test error are high. Trained to the end, width 512
    overfits a little. Its test error was worse than width 32's on 9 of 10
    seeds. With early stopping, more width kept helping, and width 512 beat
    width 32 on 8 of 10 seeds.

    In 2014, Neyshabur and colleagues found that test error kept falling as
    they added hidden units, even after the training error reached 0. They
    argued that the training method itself stops a big network from fitting
    the noise as much as its size would allow. So in practice people make a
    model a little bigger than it needs to be. Then they limit its overfitting
    with early stopping and weight decay. The size of the model is a
    hyperparameter, chosen by comparing errors on the validation set.

    A deeper MLP can make more bends in its curve with the same number of
    parameters. In the next table, two MLPs fitted 200 points of `sin(3x)`
    with no noise. Each trained for 20,000 steps at a learning rate of 0.01,
    and each error is an average over 10 seeds.

    | Hidden layers | Parameters | Training error |
    |---|---|---|
    | one of 48 | 145 | 0.042 |
    | two of 11 | 166 | 0.025 |

    The MLP with two hidden layers ended with a lower error on 9 of 10 seeds.
    A plain stack with many more layers soon stops training at all. Careful
    initialization (init3), normalization and residual connections are what
    make extra depth keep helping.
    """,
)

MLP4_SOURCE = source_file(
    '''
    # mlp4.py

    import torch
    import torch.nn as nn
    from torch import Tensor


    # The training loop of mlp3.
    def train(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> float:
        for _ in range(steps):
            loss = ((model(x).squeeze(1) - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= learning_rate * parameter.grad
                    parameter.grad = None

        return loss.item()


    # TODO: Finish the line that sets model to an MLP with 1 input, one hidden
    #   layer of `width` neurons, and 1 output. The line comes after
    #   torch.manual_seed(0), so every run starts from the same weights. train_x
    #   and validation_x have shape (points, 1). The function returns the
    #   training error and the validation error after training.
    def errors_for_width(
        width: int,
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        steps: int,
    ) -> tuple[float, float]:
        torch.manual_seed(0)
        model = ...
        train_error = train(model, train_x, train_y, steps, 0.005)

        with torch.no_grad():
            validation_error = ((model(validation_x).squeeze(1) - validation_y) ** 2).mean().item()

        return train_error, validation_error


    # TODO: Return the width in widths whose MLP has the lowest validation error,
    #   as measured by errors_for_width.
    def best_width(
        widths: list[int],
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        steps: int,
    ) -> int:
        ...
    ''',
)

MLP4_SOLUTION = source_file(
    '''
    # mlp4.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor


    def train(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> float:
        for _ in range(steps):
            loss = ((model(x).squeeze(1) - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= learning_rate * parameter.grad
                    parameter.grad = None

        return loss.item()


    def errors_for_width(
        width: int,
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        steps: int,
    ) -> tuple[float, float]:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, 1))
        train_error = train(model, train_x, train_y, steps, 0.005)

        with torch.no_grad():
            validation_error = ((model(validation_x).squeeze(1) - validation_y) ** 2).mean().item()

        return train_error, validation_error


    def best_width(
        widths: list[int],
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        steps: int,
    ) -> int:
        errors = [errors_for_width(width, train_x, train_y, validation_x, validation_y, steps)[1] for width in widths]

        return widths[errors.index(min(errors))]
    ''',
)


def reference_width_errors(
    width: int,
    train_x: Tensor,
    train_y: Tensor,
    validation_x: Tensor,
    validation_y: Tensor,
    steps: int,
) -> tuple[float, float]:
    model = seeded_mlp(0, (1, width, 1))
    train_error = train_reference(model, train_x, train_y, steps, 0.005)

    with torch.no_grad():
        validation_error = ((model(validation_x).squeeze(1) - validation_y) ** 2).mean().item()

    return train_error, validation_error


def named_noisy(
    seed: int,
    count: int,
    name: str,
) -> tuple[Described, Described]:
    x, y = noisy_curve(seed, count)

    return Described(x, f"{name}_x with {count} points"), Described(y, f"{name}_y")


def check_mlp4(checker: Checker) -> None:
    from ..engine import expect_parts

    data = (*named_noisy(0, 20, "train"), *named_noisy(1, 200, "validation"))
    check_on_inputs(
        checker=checker,
        function="errors_for_width",
        description="trains an MLP of that width and measures both errors",
        inputs=(
            (4, *data, 500),
            (32, *data, 500),
        ),
        reference=reference_width_errors,
        expect=expect_parts("the training error", "the validation error"),
    )
    check_on_inputs(
        checker=checker,
        function="best_width",
        description="gives the width with the lowest validation error",
        inputs=(
            ([1, 4, 32], *data, 1000),
            # At 5000 steps width 512 has the lowest training error and width 32 the lowest validation error.
            ([32, 512], *data, 5000),
        ),
        reference=lambda widths, tx, ty, vx, vy, steps: min(widths, key=lambda width: reference_width_errors(width, tx, ty, vx, vy, steps)[1]),
        expect=expect_value,
    )


MLP4 = Exercise(
    name="mlp4",
    chapter=MLP,
    title="Width and depth",
    lesson=MLP4_LESSON,
    source=MLP4_SOURCE,
    solution=MLP4_SOLUTION,
    hints={
        "errors_for_width": "Build an nn.Sequential of a Linear layer from 1 input to width outputs, a ReLU, and a "
        "Linear layer from width inputs to 1 output.",
        "best_width": "Call errors_for_width for each width and keep the one whose second number, the validation "
        "error, is lowest.",
    },
    introduces=(),
    check=check_mlp4,
    time_limit=60,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def mean_squared_error(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> float:
    with torch.no_grad():
        return ((model(x).squeeze(1) - y) ** 2).mean().item()


def measure_width(
    width: int,
    seeds: range = range(10),
    steps: int = 10000,
    learning_rate: float = 0.005,
    check_every: int = 100,
) -> list[dict[str, float]]:
    """mlp4's width table for one width, one row per seed. Seed s trains on 20 points from noisy_curve(s), with the
    model drawn after torch.manual_seed(s). Every check_every steps it measures 200 validation points from
    noisy_curve(1000 + s) and 1000 test points from noisy_curve(2000 + s). A row holds the training error and the test
    error at the end, and the test error at the check with the lowest validation error, which is early stopping."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rows: list[dict[str, float]] = []

    try:
        for seed in seeds:
            train_x, train_y = noisy_curve(seed, 20)
            validation_x, validation_y = noisy_curve(1000 + seed, 200)
            test_x, test_y = noisy_curve(2000 + seed, 1000)
            model = seeded_mlp(seed, (1, width, 1))
            lowest_validation = float("inf")
            stopped_test = float("nan")
            stopped_step = 0

            for step in range(1, steps + 1):
                train_reference(model, train_x, train_y, 1, learning_rate)

                if step % check_every:
                    continue

                validation_error = mean_squared_error(model, validation_x, validation_y)

                if validation_error < lowest_validation:
                    lowest_validation = validation_error
                    stopped_test = mean_squared_error(model, test_x, test_y)
                    stopped_step = step

            rows.append(
                {
                    "train": mean_squared_error(model, train_x, train_y),
                    "test": mean_squared_error(model, test_x, test_y),
                    "stopped_test": stopped_test,
                    "stopped_step": stopped_step,
                },
            )
    finally:
        torch.set_num_threads(threads)

    return rows


def measure_widths(widths: tuple[int, ...] = (1, 4, 8, 32, 128, 512)) -> dict[int, list[dict[str, float]]]:
    """Every row of mlp4's width table. It takes a few minutes, so widths can also be spread over processes with
    measure_width."""
    return {width: measure_width(width) for width in widths}


def measure_depth(
    sizes: tuple[int, ...],
    seeds: range = range(10),
    steps: int = 20000,
    learning_rate: float = 0.01,
) -> list[float]:
    """mlp4's depth table for one model, the training error on each seed. The model learns 200 evenly spaced points of
    sin(3x) between -3 and 3, with no noise, after torch.manual_seed(s)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    x = torch.linspace(-3.0, 3.0, 200).unsqueeze(1)
    y = torch.sin(3 * x).squeeze(1)
    errors: list[float] = []

    try:
        for seed in seeds:
            model = seeded_mlp(seed, sizes)
            train_reference(model, x, y, steps, learning_rate)
            errors.append(mean_squared_error(model, x, y))
    finally:
        torch.set_num_threads(threads)

    return errors


def measure_depths() -> dict[tuple[int, ...], list[float]]:
    """Both rows of mlp4's depth table, one hidden layer of 48 against two of 11, at about the same parameter count."""
    return {sizes: measure_depth(sizes) for sizes in ((1, 48, 1), (1, 11, 11, 1))}
