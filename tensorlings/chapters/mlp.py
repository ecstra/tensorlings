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

        Linear layers with activations between them make the multilayer
        perceptron, the MLP, the plainest model that can learn curved shapes and
        a part of nearly every bigger model. This chapter stacks one, builds it
        as a module, trains it, and shows how its size decides what it fits.

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
    Linear layers with an activation between each pair make a multilayer
    perceptron, an MLP. It is the plainest model that can learn curved shapes,
    and a part of nearly every bigger model.

    `nn.Sequential` chains modules, so calling it passes the input through
    each in turn. `nn.ReLU()` is ReLU as a module.

    ```python
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    model(torch.zeros(5, 2)).shape    # torch.Size([5, 1])
    model[0]                          # Linear(in_features=2, out_features=8, bias=True)
    ```

    The first layer's input size is the number of features, and the last
    layer's output size is the number of outputs. The sizes in between are
    yours to choose. The number of neurons in a hidden layer is its width, and
    the number of hidden layers is the depth.

    `nn.Sequential(*layers)` builds one from a list, which helps when the depth
    is a number you pass in. The `*` spreads the list out into separate
    arguments.

    A hidden layer of width `h` after `n` inputs holds `n * h + h`
    parameters, and every layer after it counts the same way.
    """,
)

MLP1_SOURCE = source_file(
    '''
    # mlp1.py

    import torch
    import torch.nn as nn

    # Linear, ReLU, Linear, chained by nn.Sequential.
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


    # TODO: Return an MLP with one hidden layer, Linear, ReLU, Linear, for the
    #   given numbers of inputs, hidden neurons and outputs.
    def make_mlp(
        inputs: int,
        hidden: int,
        outputs: int,
    ) -> nn.Sequential:
        ...


    # TODO: Finish the line in the loop. This returns an MLP with `depth` hidden
    #   layers, each `hidden` wide and each a Linear then a ReLU. A last Linear
    #   then gives the outputs.
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


    # TODO: Return how many parameters make_mlp(inputs, hidden, outputs) holds,
    #   worked out without building it.
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


MLP1 = Exercise(
    name="mlp1",
    chapter=MLP,
    title="Stacking an MLP",
    lesson=MLP1_LESSON,
    source=MLP1_SOURCE,
    solution=MLP1_SOLUTION,
    hints={
        "make_mlp": "Follow the line that sets model, with inputs, hidden and outputs in place of 2, 8 and 1.",
        "make_deep_mlp": "Each hidden layer adds two modules to the list, a Linear from size to hidden, then a "
        "ReLU. .extend takes a list of them.",
        "mlp_parameters": "The hidden layer holds inputs * hidden weights and hidden biases. The output layer counts "
        "the same way from hidden to outputs.",
    },
    introduces=(),
    check=check_mlp1,
)


# ---------------------------------------------------------------------------
# mlp2
# ---------------------------------------------------------------------------

MLP2_LESSON = prose(
    """
    `nn.Sequential` fits a model that is one chain. Most models need more
    control, so they are written as a module with a `forward` of their own.
    Here is the MLP that way. It takes a list of sizes, the inputs, each
    hidden width, then the outputs, like `[2, 8, 8, 1]`.

    The layers live in a list, and a plain Python list is a trap. A module only
    finds parameters stored in attributes it knows about, so the Linear layers
    of a plain list are invisible to `model.parameters()`. A training loop over
    `model.parameters()` then trains nothing, and nothing warns you.

    ```python
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(2, 2)]

    len(list(Broken().parameters()))    # 0
    ```

    `nn.ModuleList` is a list a module does know about. It works like a list.
    You can loop over it and index it, and its modules' parameters belong to
    the model.

    In `forward`, the activation goes between the layers and not after the
    last one, since the last layer gives the answer, as activations2 said.
    `enumerate` gives each layer's position as you loop, which tells the last
    one apart.
    """,
)

MLP2_SOURCE = source_file(
    '''
    # mlp2.py

    import torch
    import torch.nn as nn
    from torch import Tensor


    # TODO: Two things to finish. The layers are stored in a plain list, so the
    #   model has no parameters. Fix that line. Then finish the loop in forward,
    #   so every layer but the last is followed by ReLU.
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


    # nn.ModuleList registers each layer, so their parameters belong to the model.
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
    Training an MLP is the loop of Section 2, with the model's parameters in
    place of a hand made `w` and `b`. The model gives the predictions, the
    loss compares them with the targets, `.backward()` fills every
    parameter's gradient, and a step moves them all.

    ```python
    for _ in range(steps):
        loss = ((model(x).squeeze(1) - y) ** 2).mean()
        loss.backward()
        gradient_step(model, learning_rate)
    ```

    `gradient_step` is the one from modules1, which steps every parameter of
    the model against its gradient and clears it. `.squeeze(1)` turns the
    model's output of shape `(examples, 1)` into `(examples,)`, to line up with
    the targets, as linreg2 warned.

    Here an MLP with one hidden layer of 32 neurons learns `sin(x)` between -3
    and 3 from 64 points. In 1000 steps with a learning rate of 0.1 the mean
    squared error falls to about 0.0004. One Linear layer on the same points
    can only fit a straight line, and stays above 0.1.

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


    # The step from modules1.
    def gradient_step(
        model: nn.Module,
        learning_rate: float,
    ) -> None:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= learning_rate * parameter.grad
                parameter.grad = None


    # TODO: Finish the line that sets loss, the mean squared error of the model's
    #   predictions on x against y, so this trains the model and returns the final
    #   loss.
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


    # TODO: Return the mean squared error of model on x against y, without
    #   recording anything for autograd.
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

    def __init__(self) -> None:
        super().__init__()
        self.inner = seeded_mlp(3, (1, 8, 1))
        self.recording: list[bool] = []

    def forward(self, x: Tensor) -> Tensor:
        self.recording.append(torch.is_grad_enabled())

        return self.inner(x)


def check_evaluate_records_nothing(found: object) -> None:
    watcher = GradWatcher()
    found(watcher, CURVE_X.clone(), CURVE_Y.clone())

    if any(watcher.recording):
        raise Mismatch("evaluate ran the model while autograd was recording. Put the work inside with torch.no_grad():.")


MLP3 = Exercise(
    name="mlp3",
    chapter=MLP,
    title="Training an MLP",
    lesson=MLP3_LESSON,
    source=MLP3_SOURCE,
    solution=MLP3_SOLUTION,
    hints={
        "train": "Call the model on x, .squeeze(1) the result, and take the mean squared error against y.",
        "evaluate": "The same mean squared error, inside with torch.no_grad():, returned as a plain number.",
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
    How wide should a hidden layer be? Here MLPs of four widths learned 20
    noisy points of `sin(x)`, 5000 steps each with a learning rate of 0.01.
    Then each was measured on 200 new points. The noise alone gives an error of
    0.09, so no model can beat that on new points.

    | Width | Training error | Validation error |
    |---|---|---|
    | 1 | 0.175 | 0.222 |
    | 4 | 0.106 | 0.233 |
    | 32 | 0.056 | 0.127 |
    | 512 | 0.050 | 0.142 |

    Width 1 is too narrow to bend, and underfits both. Width 32 does best on new
    points. Width 512 fits the training points a little better and new points
    worse, since it starts fitting the noise, as overfitting1's degree 9 did.

    Depth works the same way, with more layers bending more. In practice a
    model is made a bit too big, then held back with weight decay and early
    stopping from Section 2. Its size is a hyperparameter, chosen on the
    validation set. Over many tasks, wider and deeper models held back this way
    tend to do better on new data. The rest of the course leans on that.
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


    # TODO: Finish the line that sets model, an MLP with one hidden layer of the
    #   given width, built after torch.manual_seed(0) so every run starts the same.
    #   train_x has shape (points, 1). It returns the training and validation
    #   errors after training.
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
        train_error = train(model, train_x, train_y, steps, 0.01)

        with torch.no_grad():
            validation_error = ((model(validation_x).squeeze(1) - validation_y) ** 2).mean().item()

        return train_error, validation_error


    # TODO: Return the width from widths with the lowest validation error.
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
        train_error = train(model, train_x, train_y, steps, 0.01)

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
    train_error = train_reference(model, train_x, train_y, steps, 0.01)

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
        "errors_for_width": "nn.Sequential of a Linear from 1 input to width, a ReLU, and a Linear from width to 1.",
        "best_width": "Call errors_for_width for each width and keep the one whose second number, the validation "
        "error, is lowest.",
    },
    introduces=(),
    check=check_mlp4,
    time_limit=60,
)
