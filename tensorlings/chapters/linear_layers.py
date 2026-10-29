"""15_linear, PyTorch's modules, the Linear layer, building it, and why two in a row are one."""

from __future__ import annotations

import math

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
    expect_parts,
    expect_tensor,
    expect_value,
    reject_missing_return,
    small_tensor,
    value_text,
)
from ..text import prose, source_file

LINEAR = Chapter(
    folder="15_linear",
    title="Linear layers",
    readme=source_file(
        """
        # Linear layers

        Section 3 builds the blocks real models are made of. This chapter starts
        with the way PyTorch packages a model, as a module that holds its
        parameters. Then it builds the Linear layer, the block that holds most
        of the parameters of most models, and shows why two of them in a row
        are no better than one.

        ## Further reading

        - [PyTorch, build the neural network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
        - [Dive into Deep Learning, layers and modules](https://d2l.ai/chapter_builders-guide/model-construction.html)
        """,
    ),
)


def expect_module(
    call: str,
    made: object,
) -> nn.Module:
    reject_missing_return(
        call=call,
        actual=made,
    )

    if not isinstance(made, nn.Module):
        raise Mismatch(f"{call} is {value_text(made)}. It should be a module.")

    return made


def expect_parameter(
    module: nn.Module,
    call: str,
    name: str,
    shape: tuple[int, ...],
) -> Tensor:
    parameters = dict(module.named_parameters())

    if name not in parameters:
        found = ", ".join(parameters) or "none"
        raise Mismatch(f"{call} should have a parameter called {name}, made with nn.Parameter. The parameters it has are {found}.")

    parameter = parameters[name]

    if tuple(parameter.shape) != shape:
        raise Mismatch(f"The parameter {name} of {call} has shape {tuple(parameter.shape)}, but it should have shape {shape}.")

    return parameter


# ---------------------------------------------------------------------------
# modules1
# ---------------------------------------------------------------------------

MODULES1_LESSON = prose(
    """
    Section 3 builds the blocks real models are made of. First, the way
    PyTorch packages them.

    Section 2 kept the parameters as loose tensors, and every update named
    each one. A model with millions of parameters needs them gathered up. In
    PyTorch a model is a class that inherits from `nn.Module`. Its parameters
    are tensors wrapped in `nn.Parameter` and set as attributes in
    `__init__`, and its calculation goes in a method called `forward`.
    Calling the model runs `forward`.

    ```python
    import torch
    import torch.nn as nn

    class Scale(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.factor = nn.Parameter(torch.ones(size))

        def forward(self, x):
            return x * self.factor

    model = Scale(3)
    model(torch.tensor([1.0, 2.0, 3.0]))    # tensor([1., 2., 3.], grad_fn=<MulBackward0>)
    ```

    `nn.Parameter` marks a tensor as a parameter and switches its gradient
    on. `super().__init__()` comes first in `__init__`, since it sets up the
    bookkeeping that finds the parameters.

    `model.parameters()` gives every parameter of the model, including those
    of any module stored inside it. Big models are modules of modules, many
    deep, and a training loop updates all their parameters without naming
    one. `p.numel()` gives the number of entries of a tensor `p`.
    """,
)

MODULES1_SOURCE = source_file(
    '''
    # modules1.py

    import torch
    import torch.nn as nn
    from torch import Tensor


    # A module holds its parameters as attributes, and does its work in forward.
    class Scale(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.factor = nn.Parameter(torch.ones(size))

        def forward(self, x: Tensor) -> Tensor:
            return x * self.factor


    # TODO: Finish both methods, so Shift adds a learned offset to x, one number
    #   per entry, starting at zeros. The parameter must be called offset.
    class Shift(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.offset = ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Finish the two lines in __init__, so this holds a Scale and a Shift of
    #   the given size, called scale and shift. Their parameters count as this
    #   module's too.
    class ScaleThenShift(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.scale = ...
            self.shift = ...

        def forward(self, x: Tensor) -> Tensor:
            return self.shift(self.scale(x))


    # TODO: Return how many numbers the parameters of model hold in all.
    def count_parameters(model: nn.Module) -> int:
        ...


    # TODO: Finish the line in the loop, so this takes one step of gradient
    #   descent on every parameter of model, then clears its gradient.
    def gradient_step(
        model: nn.Module,
        learning_rate: float,
    ) -> None:
        with torch.no_grad():
            for parameter in model.parameters():
                ...
                parameter.grad = None
    ''',
)

MODULES1_SOLUTION = source_file(
    '''
    # modules1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor


    class Scale(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.factor = nn.Parameter(torch.ones(size))

        def forward(self, x: Tensor) -> Tensor:
            return x * self.factor


    class Shift(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.offset = nn.Parameter(torch.zeros(size))

        def forward(self, x: Tensor) -> Tensor:
            return x + self.offset


    class ScaleThenShift(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.scale = Scale(size)
            self.shift = Shift(size)

        def forward(self, x: Tensor) -> Tensor:
            return self.shift(self.scale(x))


    def count_parameters(model: nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters())


    # -= changes each parameter in place, so the model keeps the same tensors.
    def gradient_step(
        model: nn.Module,
        learning_rate: float,
    ) -> None:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= learning_rate * parameter.grad
                parameter.grad = None
    ''',
)


def check_shift(found: type) -> None:
    for size in (3, 5):
        call = f"Shift({size})"
        module = expect_module(call, found(size))
        offset = expect_parameter(module, call, "offset", (size,))

        if not torch.equal(offset.detach(), torch.zeros(size)):
            raise Mismatch(f"The offset of {call} starts at {value_text(offset.detach())}, but it should start at zeros.")

        with torch.no_grad():
            offset.copy_(torch.arange(float(size)))

        x = small_tensor(167 + size, 2, size)
        expect_tensor(
            call=f"{call} with offset {value_text(offset.detach())} on {value_text(x)}",
            actual=module(x),
            expected=x + torch.arange(float(size)),
        )


def check_scale_then_shift(found: type) -> None:
    call = "ScaleThenShift(4)"
    module = expect_module(call, found(4))
    names = sorted(name for name, _ in module.named_parameters())

    if names != ["scale.factor", "shift.offset"]:
        raise Mismatch(f"{call} should hold the parameters scale.factor and shift.offset, but it holds {names or 'none'}.")

    x = small_tensor(175, 3, 4)
    expect_tensor(
        call=f"{call} on {value_text(x)}",
        actual=module(x),
        expected=x,
    )

    # A fresh scale of 1 and shift of 0 give x back in either order, so set them to tell the order apart.
    with torch.no_grad():
        module.scale.factor.fill_(2.0)
        module.shift.offset.fill_(1.0)

    expect_tensor(
        call=f"{call}, with its factor set to 2 and its offset to 1, on {value_text(x)}",
        actual=module(x),
        expected=x * 2 + 1,
    )


def check_gradient_step(found: object) -> None:
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    expected = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loss = model(small_tensor(176, 4, 3)).pow(2).mean()
    loss.backward()

    for name, parameter in model.named_parameters():
        expected[name] = expected[name] - 0.1 * parameter.grad

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    found(model, 0.1)

    if all(torch.equal(parameter.detach(), before[name]) for name, parameter in model.named_parameters()):
        raise Mismatch(
            "After gradient_step the weights did not change. parameter = ... makes a new tensor and leaves the "
            "model's own alone, so use -= to change it in place.",
        )

    for name, parameter in model.named_parameters():
        expect_tensor(
            call=f"the {name} of an nn.Linear(3, 2) after gradient_step with learning rate 0.1",
            actual=parameter.detach(),
            expected=expected[name],
        )

        if parameter.grad is not None:
            raise Mismatch(f"After gradient_step, the {name} of the model still has a gradient. Clear it with parameter.grad = None.")


def check_modules1(checker: Checker) -> None:
    checker.test(
        function="Shift",
        description="adds a learned offset that starts at zeros",
        run=check_shift,
    )
    checker.test(
        function="ScaleThenShift",
        description="holds a Scale and a Shift and finds both their parameters",
        run=check_scale_then_shift,
    )
    check_on_inputs(
        checker=checker,
        function="count_parameters",
        description="counts every number the parameters hold",
        inputs=(
            (Described(nn.Linear(3, 4), "nn.Linear(3, 4)"),),
            (Described(nn.Sequential(nn.Linear(784, 100), nn.Linear(100, 10)), "a Linear(784, 100) then a Linear(100, 10)"),),
        ),
        reference=lambda model: sum(parameter.numel() for parameter in model.parameters()),
        expect=expect_value,
    )
    checker.test(
        function="gradient_step",
        description="steps every parameter against its gradient and clears it",
        run=check_gradient_step,
    )


MODULES1 = Exercise(
    name="modules1",
    chapter=LINEAR,
    title="Modules and parameters",
    lesson=MODULES1_LESSON,
    source=MODULES1_SOURCE,
    solution=MODULES1_SOLUTION,
    hints={
        "Shift": "Follow Scale, with torch.zeros(size) for the parameter and + in place of *.",
        "ScaleThenShift": "Set self.scale to a Scale of the given size, and self.shift to a Shift of it.",
        "count_parameters": "Loop over model.parameters() and add up each one's .numel().",
        "gradient_step": "Take the learning rate times parameter.grad away from parameter with -=, which changes "
        "it in place.",
    },
    introduces=(),
    check=check_modules1,
)


# ---------------------------------------------------------------------------
# linear1
# ---------------------------------------------------------------------------

LINEAR1_LESSON = prose(
    """
    A **layer** is one step of a model with parameters of its own. The layer
    models use most is Linear, and it is the scores of softmax regression,
    `x @ W + b`, with one difference in how it keeps its weights.

    PyTorch's `nn.Linear(in_features, out_features)` keeps its weights as a
    matrix of shape `(out_features, in_features)`, one row per output. So it
    computes `input @ weight.T + bias`, with the weights turned by `.T`.

    ```python
    inputs = torch.tensor([[1.0, 2.0, 3.0],
                           [0.0, -1.0, 1.0]])    # 2 examples, 3 features
    weight = torch.tensor([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [1.0, 1.0, 1.0],
                           [0.5, -0.5, 2.0]])    # 4 outputs, 3 inputs
    bias = torch.tensor([0.0, 1.0, -1.0, 0.5])
    inputs @ weight.T + bias    # tensor([[ 1.,  3.,  5.,  6.], [ 0.,  0., -1.,  3.]])
    ```

    Each output is a weighted sum of every input plus a bias of its own. One
    output, with its row of weights and its bias, is a **neuron**. The name
    comes from a loose likeness to cells in the brain. So a Linear layer with
    4 outputs has 4 neurons, and keeping a row per output keeps each neuron's
    weights together. Row `k` of `W` is `W[k]`.

    `nn.Linear(3, 4)` holds 3 * 4 weights and 4 biases. A Linear layer mixes
    every input feature into every output, and in most models nearly all the
    parameters sit in Linear layers.
    """,
)

LINEAR1_SOURCE = source_file(
    '''
    # linear1.py

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0, 2.0, 3.0],
                      [0.0, -1.0, 1.0]])           # 2 examples, 3 features
    W = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [1.0, 1.0, 1.0],
                      [0.5, -0.5, 2.0]])           # 4 outputs, 3 inputs, one row per output
    b = torch.tensor([0.0, 1.0, -1.0, 0.5])

    # A Linear layer keeps one row of weights per output, so its weights are turned.
    outputs = x @ W.T + b                          # tensor([[1., 3., 5., 6.], [0., 0., -1., 3.]])


    # TODO: This crashes when the checker calls it. W has one row per output, the
    #   way a Linear layer keeps it. Read the error, then fix it.
    def linear(
        x: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> Tensor:
        return x @ W + b


    # TODO: Return the output of neuron k alone, for every example in x, one
    #   number per example. Neuron k has row k of W as its weights and entry k of
    #   b as its bias.
    def neuron_output(
        x: Tensor,
        W: Tensor,
        b: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # TODO: Return how many parameters a Linear layer has, for the given numbers
    #   of inputs and outputs.
    def linear_parameters(
        in_features: int,
        out_features: int,
    ) -> int:
        ...
    ''',
)

LINEAR1_SOLUTION = source_file(
    '''
    # linear1.py, a solution for comparison

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0, 2.0, 3.0],
                      [0.0, -1.0, 1.0]])
    W = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [1.0, 1.0, 1.0],
                      [0.5, -0.5, 2.0]])
    b = torch.tensor([0.0, 1.0, -1.0, 0.5])

    outputs = x @ W.T + b


    # W has shape (outputs, inputs), so W.T has the (inputs, outputs) that @ needs.
    def linear(
        x: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> Tensor:
        return x @ W.T + b


    def neuron_output(
        x: Tensor,
        W: Tensor,
        b: Tensor,
        k: int,
    ) -> Tensor:
        return x @ W[k] + b[k]


    def linear_parameters(
        in_features: int,
        out_features: int,
    ) -> int:
        return in_features * out_features + out_features
    ''',
)

LAYERS: tuple[tuple[Tensor, Tensor, Tensor], ...] = (
    (small_tensor(177, 2, 3), small_tensor(178, 4, 3), small_tensor(179, 4)),
    (small_tensor(180, 3, 3), small_tensor(181, 3, 3), small_tensor(182, 3)),
)


def check_linear1(checker: Checker) -> None:
    # The square layer is where x @ W runs without .T and gives other numbers.
    check_on_inputs(
        checker=checker,
        function="linear",
        description="gives x @ W.T + b, one output per row of W",
        inputs=LAYERS,
        reference=lambda x, W, b: F.linear(x, W, b),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="neuron_output",
        description="gives one neuron's output for every example",
        inputs=(
            (*LAYERS[0], 2),
            (*LAYERS[1], 0),
        ),
        reference=lambda x, W, b, k: F.linear(x, W, b)[:, k],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="linear_parameters",
        description="counts the weights and the biases",
        inputs=(
            (3, 4),
            (784, 128),
            (1, 1),
        ),
        reference=lambda in_features, out_features: sum(parameter.numel() for parameter in nn.Linear(in_features, out_features).parameters()),
        expect=expect_value,
    )


LINEAR1 = Exercise(
    name="linear1",
    chapter=LINEAR,
    title="The Linear layer",
    lesson=LINEAR1_LESSON,
    source=LINEAR1_SOURCE,
    solution=LINEAR1_SOLUTION,
    hints={
        "linear": "The error says the middle sizes don't match. W has one row per output, so turn it with .T, as "
        "the line that sets outputs does.",
        "neuron_output": "W[k] is neuron k's row of weights. Each example's weighted sum with it is x @ W[k], and "
        "its bias is b[k].",
        "linear_parameters": "One weight for each pair of an input and an output, and one bias per output.",
    },
    introduces=("layer", "neuron"),
    check=check_linear1,
)


# ---------------------------------------------------------------------------
# linear2
# ---------------------------------------------------------------------------

LINEAR2_LESSON = prose(
    """
    Now build Linear as a module, then match PyTorch's own.

    The weights can't start at zero. Every neuron would then compute the same
    thing and get the same gradient, and they would stay copies of each other
    forever. So they start random. PyTorch draws each weight and each bias at
    random between `-1 / sqrt(in_features)` and `1 / sqrt(in_features)`, with
    every value in between equally likely. That is called a uniform draw. The
    initialization chapter shows why that size.

    `torch.empty(4, 3)` makes a tensor without setting its entries, and
    `.uniform_(low, high)` fills it in place with numbers drawn evenly between
    the two.

    ```python
    bound = 1 / math.sqrt(3)
    torch.empty(4, 3).uniform_(-bound, bound)    # 12 numbers between -0.577 and 0.577
    ```

    To compare with PyTorch's layer, copy its parameters into yours. A
    parameter's numbers can be overwritten in place with `.copy_(other)`.
    Changing a parameter by hand is not part of any calculation to
    differentiate, so it goes inside `with torch.no_grad():`.

    ```python
    layer = nn.Linear(3, 4)
    layer.weight.shape, layer.bias.shape    # (torch.Size([4, 3]), torch.Size([4]))
    ```

    `F.linear(x, W, b)` from `torch.nn.functional` computes `x @ W.T + b`, the
    same thing as the layer, with the parameters passed in.
    """,
)

LINEAR2_SOURCE = source_file(
    '''
    # linear2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Finish the line that sets self.bias and the forward method, so
    #   MyLinear works like nn.Linear. The bias starts between the same bounds as
    #   the weights, one number per output.
    class MyLinear(nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
        ) -> None:
            super().__init__()
            bound = 1 / math.sqrt(in_features)
            self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
            self.bias = ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Finish the two lines, so they copy the weight and the bias of layer, an
    #   nn.Linear, into mine, a MyLinear of the same size.
    def copy_parameters(
        mine: nn.Module,
        layer: nn.Linear,
    ) -> None:
        with torch.no_grad():
            ...
            ...
    ''',
)

LINEAR2_SOLUTION = source_file(
    '''
    # linear2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    class MyLinear(nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
        ) -> None:
            super().__init__()
            bound = 1 / math.sqrt(in_features)
            self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
            self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))

        def forward(self, x: Tensor) -> Tensor:
            return x @ self.weight.T + self.bias


    def copy_parameters(
        mine: nn.Module,
        layer: nn.Linear,
    ) -> None:
        with torch.no_grad():
            mine.weight.copy_(layer.weight)
            mine.bias.copy_(layer.bias)
    ''',
)


def check_my_linear(found: type) -> None:
    for in_features, out_features in ((3, 4), (16, 2)):
        call = f"MyLinear({in_features}, {out_features})"
        module = expect_module(call, found(in_features, out_features))
        bound = 1 / math.sqrt(in_features)

        for name, shape in (("weight", (out_features, in_features)), ("bias", (out_features,))):
            parameter = expect_parameter(module, call, name, shape)

            if parameter.detach().abs().max() > bound or parameter.detach().abs().max() == 0:
                raise Mismatch(f"The {name} of {call} should start random, between -{bound:.3f} and {bound:.3f}.")

        x = small_tensor(183 + in_features, 2, in_features)
        expect_tensor(
            call=f"{call} on {value_text(x)}",
            actual=module(x),
            expected=F.linear(x, module.weight.detach(), module.bias.detach()),
        )


def check_copy_parameters(found: object) -> None:
    torch.manual_seed(1)
    layer = nn.Linear(3, 2)
    mine = nn.Linear(3, 2)
    found(mine, layer)

    for name in ("weight", "bias"):
        expect_tensor(
            call=f"the {name} of mine after copy_parameters",
            actual=getattr(mine, name).detach(),
            expected=getattr(layer, name).detach(),
        )


def check_linear2(checker: Checker) -> None:
    checker.test(
        function="MyLinear",
        description="holds a random weight and bias and computes x @ W.T + b",
        run=check_my_linear,
    )
    checker.test(
        function="copy_parameters",
        description="copies an nn.Linear's weight and bias",
        run=check_copy_parameters,
    )


LINEAR2 = Exercise(
    name="linear2",
    chapter=LINEAR,
    title="Building Linear",
    lesson=LINEAR2_LESSON,
    source=LINEAR2_SOURCE,
    solution=LINEAR2_SOLUTION,
    hints={
        "MyLinear": "The bias follows the line that sets self.weight, with one number per output. forward is "
        "x @ W.T + b with self.weight and self.bias, as in linear1.",
        "copy_parameters": "Call .copy_ on mine.weight with layer.weight, then on mine.bias with layer.bias.",
    },
    introduces=(),
    check=check_linear2,
)


# ---------------------------------------------------------------------------
# linear3
# ---------------------------------------------------------------------------

LINEAR3_LESSON = prose(
    """
    Every output of a Linear layer is a weighted sum of its inputs, which
    limits what one layer can fit. It seems two in a row would do more. They
    don't. Feed one Linear into another and multiply it out.

    ```text
    second(first(x)) = (x @ W1.T + b1) @ W2.T + b2
                     = x @ (W2 @ W1).T + (b1 @ W2.T + b2)
    ```

    That is a single Linear layer with weights `W2 @ W1` and bias
    `b1 @ W2.T + b2`. However many Linear layers are stacked, the result can
    always be written as one. The extra layers add parameters and cost, and no
    power to fit anything new.

    ```python
    W1 = torch.tensor([[1.0, 2.0], [0.0, 1.0], [1.0, -1.0]])    # 2 inputs to 3 outputs
    W2 = torch.tensor([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])       # 3 inputs to 2 outputs
    W2 @ W1    # tensor([[2., 1.], [2., 5.]]), the weights of the one layer they make
    ```

    The way out is to put a step that isn't a weighted sum between them, so
    the product no longer multiplies out. The next chapter builds that step.
    It is the reason models with many layers can learn shapes a single layer
    can't.
    """,
)

LINEAR3_SOURCE = source_file(
    '''
    # linear3.py

    import torch
    from torch import Tensor


    # TODO: Return x passed through the first Linear layer, W1 and b1, then
    #   through the second, W2 and b2.
    def two_layers(
        x: Tensor,
        W1: Tensor,
        b1: Tensor,
        W2: Tensor,
        b2: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish both lines, so this returns the weights and the bias of the
    #   one Linear layer that gives the same output as the two.
    def collapse(
        W1: Tensor,
        b1: Tensor,
        W2: Tensor,
        b2: Tensor,
    ) -> tuple[Tensor, Tensor]:
        W = ...
        b = ...

        return W, b
    ''',
)

LINEAR3_SOLUTION = source_file(
    '''
    # linear3.py, a solution for comparison

    import torch
    from torch import Tensor


    def two_layers(
        x: Tensor,
        W1: Tensor,
        b1: Tensor,
        W2: Tensor,
        b2: Tensor,
    ) -> Tensor:
        hidden = x @ W1.T + b1

        return hidden @ W2.T + b2


    def collapse(
        W1: Tensor,
        b1: Tensor,
        W2: Tensor,
        b2: Tensor,
    ) -> tuple[Tensor, Tensor]:
        W = W2 @ W1
        b = b1 @ W2.T + b2

        return W, b
    ''',
)

STACKS: tuple[tuple[Tensor, ...], ...] = (
    (small_tensor(190, 3, 2), small_tensor(191, 3), small_tensor(192, 2, 3), small_tensor(193, 2)),
    (small_tensor(194, 3, 3), small_tensor(195, 3), small_tensor(196, 3, 3), small_tensor(197, 3)),
)


def check_linear3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="two_layers",
        description="passes x through both layers",
        inputs=(
            (small_tensor(198, 4, 2), *STACKS[0]),
            (small_tensor(199, 2, 3), *STACKS[1]),
        ),
        reference=lambda x, W1, b1, W2, b2: F.linear(F.linear(x, W1, b1), W2, b2),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="collapse",
        description="gives the one layer the two make",
        inputs=STACKS,
        reference=lambda W1, b1, W2, b2: (W2 @ W1, W2 @ b1 + b2),
        expect=expect_parts("W", "b"),
    )


LINEAR3 = Exercise(
    name="linear3",
    chapter=LINEAR,
    title="Two Linears make one",
    lesson=LINEAR3_LESSON,
    source=LINEAR3_SOURCE,
    solution=LINEAR3_SOLUTION,
    hints={
        "two_layers": "Apply the first layer as linear1 does, x @ W1.T + b1, then the second to what it gives.",
        "collapse": "The lesson multiplies it out. W is the second layer's weights times the first's, and b is the "
        "first bias through the second layer's weights, plus the second bias.",
    },
    introduces=(),
    check=check_linear3,
)
