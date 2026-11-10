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

        Section 3 builds the parts that real models are made of. This chapter
        starts with how PyTorch organizes a model, as a module that holds its
        own parameters. Then it builds the Linear layer, which holds most of
        the parameters in most models. Finally it shows why two Linear layers
        in a row can do no more than one.

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
    Section 3 builds the parts that real models are made of. It starts with
    how PyTorch organizes a model in code.

    In Section 2 the parameters were separate tensors, and every update had
    to name each one, like `w` and `b`. A model with millions of parameters
    needs them collected in one place. In PyTorch a model is a class that
    inherits from `nn.Module`, and such a class is called a module. Its
    parameters are tensors wrapped in `nn.Parameter` and stored as attributes
    in `__init__`. Its calculation goes in a method called `forward`. Calling
    the model like a function runs `forward`.

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

    `nn.Parameter` marks a tensor as a parameter of the module, and sets
    `requires_grad=True` on it so autograd computes its gradient.
    `super().__init__()` must come first in `__init__`, because it sets up
    the records that `nn.Module` uses to find the parameters.

    `model.parameters()` gives every parameter of the model, including the
    parameters of any module stored as an attribute inside it. Big models are
    modules inside modules, many levels deep. With `model.parameters()`, a
    training loop can update all their parameters without naming any of
    them. `p.numel()` gives the number of entries in a tensor `p`.
    """,
)

MODULES1_SOURCE = source_file(
    '''
    # modules1.py

    import torch
    import torch.nn as nn
    from torch import Tensor


    # A module stores its parameters as attributes and does its calculation in
    # forward.
    class Scale(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.factor = nn.Parameter(torch.ones(size))

        def forward(self, x: Tensor) -> Tensor:
            return x * self.factor


    # TODO: Finish both methods. Shift has a parameter called offset that holds
    #   size numbers and starts as zeros. forward returns x plus offset.
    class Shift(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.offset = ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Finish the two lines in __init__, so this module holds a Scale called
    #   scale and a Shift called shift, both of the given size. Because they are
    #   stored as attributes, their parameters count as this module's too.
    class ScaleThenShift(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.scale = ...
            self.shift = ...

        def forward(self, x: Tensor) -> Tensor:
            return self.shift(self.scale(x))


    # TODO: Return the total number of entries in all the parameters of model.
    #   Example: count_parameters(Scale(3)) returns 3.
    def count_parameters(model: nn.Module) -> int:
        ...


    # TODO: Finish the line in the loop, so this takes one step of gradient
    #   descent on every parameter of model. The line after it then clears the
    #   parameter's gradient.
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
            "model's own tensor unchanged, so use -= to change it in place.",
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
        description="takes a gradient descent step on every parameter and clears its gradient",
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
        "Shift": "Copy the pattern of Scale, using torch.zeros(size) for the parameter and + instead of *.",
        "ScaleThenShift": "Set self.scale to a Scale of the given size, and self.shift to a Shift of the same size.",
        "count_parameters": "Loop over model.parameters() and add up each one's .numel().",
        "gradient_step": "Subtract the learning rate times parameter.grad from parameter with -=, which changes it "
        "in place.",
    },
    introduces=(),
    check=check_modules1,
)


# ---------------------------------------------------------------------------
# linear1
# ---------------------------------------------------------------------------

LINEAR1_LESSON = prose(
    """
    A **layer** is one step of a model's calculation, with parameters of its
    own. The most common layer is called Linear. It computes the same thing
    as the scores of softmax regression, `x @ W + b`, except that it stores
    its weights the other way around.

    PyTorch's `nn.Linear(in_features, out_features)` stores its weights as a
    matrix of shape `(out_features, in_features)`, with one row per output.
    So it computes `input @ weight.T + bias`. Here `.T` transposes the weight
    matrix, swapping its rows and columns, so it has the shape
    `(in_features, out_features)` that the matrix multiplication needs.

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

    Each output is a weighted sum of all the inputs, plus a bias of its own.
    One output, together with its row of weights and its bias, is called a
    **neuron**. The name comes from a loose similarity to nerve cells in the
    brain. So a Linear layer with 4 outputs has 4 neurons. Storing one row
    per output keeps each neuron's weights together in one row. Row `k` of
    `W` is `W[k]`.

    `nn.Linear(3, 4)` holds 3 * 4 weights and 4 biases. A Linear layer
    combines every input feature into every output. In most models, nearly
    all the parameters are in Linear layers.
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

    # A Linear layer stores one row of weights per output, so W is transposed
    # with .T before the matrix multiplication.
    outputs = x @ W.T + b                          # tensor([[1., 3., 5., 6.], [0., 0., -1., 3.]])


    # TODO: This crashes when the checker calls it. W has one row per output, the
    #   way a Linear layer stores it, so its shape is (outputs, inputs). Read the
    #   error, then fix it.
    def linear(
        x: Tensor,
        W: Tensor,
        b: Tensor,
    ) -> Tensor:
        return x @ W + b


    # TODO: Return the output of neuron k alone for every example in x, as a
    #   tensor with one number per example. Neuron k uses row k of W as its
    #   weights and entry k of b as its bias.
    #   Example: with the x, W and b above, neuron_output(x, W, b, 2) returns
    #   tensor([ 5., -1.]).
    def neuron_output(
        x: Tensor,
        W: Tensor,
        b: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # TODO: Return the total number of weights and biases in a Linear layer with
    #   in_features inputs and out_features outputs.
    #   Example: linear_parameters(3, 4) returns 16.
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


    # W has shape (outputs, inputs), so W.T has shape (inputs, outputs), which
    # is the shape the matrix multiplication with x needs.
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
        "linear": "The error says the inner sizes of the matrix multiplication do not match. W has one row per "
        "output, so transpose it with .T, as the line that sets outputs does.",
        "neuron_output": "W[k] is neuron k's row of weights. Each example's weighted sum with it is x @ W[k], and "
        "its bias is b[k].",
        "linear_parameters": "There is one weight for each pair of an input and an output, and one bias for each "
        "output.",
    },
    introduces=("layer", "neuron"),
    check=check_linear1,
)


# ---------------------------------------------------------------------------
# linear2
# ---------------------------------------------------------------------------

LINEAR2_LESSON = prose(
    """
    Now build Linear as a module, and check that it matches PyTorch's
    `nn.Linear`.

    In planets2 the weights started at zero. That works for a single layer
    like that softmax regression, because each output is compared with its
    own target and so gets its own gradient. It fails when one layer feeds
    into another. All the neurons of the first layer would compute the same
    output and get the same gradient. So they would stay identical copies of
    each other forever. That is why weights start random.

    PyTorch draws each weight and each bias at random between
    `-1 / sqrt(in_features)` and `1 / sqrt(in_features)`, with every value in
    between equally likely. This is called drawing from a uniform
    distribution. The initialization chapter explains why these bounds are
    used.

    `torch.empty(4, 3)` makes a tensor of shape `(4, 3)` without setting its
    entries. `.uniform_(low, high)` then fills it in place with numbers drawn
    uniformly between `low` and `high`.

    ```python
    bound = 1 / math.sqrt(3)
    torch.empty(4, 3).uniform_(-bound, bound)    # 12 numbers between -0.577 and 0.577
    ```

    To compare your layer with PyTorch's, copy PyTorch's parameters into
    yours. `.copy_(other)` overwrites a parameter's numbers in place with the
    numbers of `other`. Setting `mine.weight = layer.weight` instead would
    make both modules share one tensor, so training one would change the
    other. Changing a parameter by hand is not a calculation you want
    gradients for, so it goes inside `with torch.no_grad():`.

    ```python
    layer = nn.Linear(3, 4)
    layer.weight.shape, layer.bias.shape    # (torch.Size([4, 3]), torch.Size([4]))
    ```

    `F.linear(x, W, b)` from `torch.nn.functional` computes `x @ W.T + b`. It
    is the same calculation as the layer, but you pass the parameters in as
    arguments.
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
    #   MyLinear works like nn.Linear. The bias has one number per output, drawn
    #   uniformly between the same bounds as the weights.
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


    # TODO: Finish the two lines, so they copy the numbers in the weight and the
    #   bias of layer, an nn.Linear, into the weight and the bias of mine, a
    #   MyLinear of the same size.
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

    # 2000 even draws all but surely reach past 0.8 of the bound on both sides, which a one sided or tiny bias never does.
    call = "MyLinear(16, 2000)"
    bias = expect_parameter(expect_module(call, found(16, 2000)), call, "bias", (2000,)).detach()
    bound = 1 / math.sqrt(16)

    if bias.min() > -0.8 * bound or bias.max() < 0.8 * bound:
        raise Mismatch(
            f"The bias of {call} has values from {bias.min():.3f} to {bias.max():.3f}, but it should be drawn uniformly "
            f"between -{bound:.3f} and {bound:.3f}, the same bounds as the weights.",
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

        if getattr(mine, name).data_ptr() == getattr(layer, name).data_ptr():
            raise Mismatch(
                f"After copy_parameters, mine.{name} and layer.{name} are the same tensor, so training one would "
                "change the other. Copy the numbers in place with .copy_.",
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
        "MyLinear": "Build the bias the way the line that sets self.weight builds the weight, but with one number "
        "per output. forward computes x @ W.T + b with self.weight and self.bias, as in linear1.",
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
    Every output of a Linear layer is a weighted sum of its inputs plus a
    bias, which limits what one layer can fit. You might expect two Linear
    layers in a row to fit more, but they cannot. To see why, feed the output
    of one Linear layer into another and expand the expression.

    ```text
    second(first(x)) = (x @ W1.T + b1) @ W2.T + b2
                     = x @ (W2 @ W1).T + (b1 @ W2.T + b2)
    ```

    The result is a single Linear layer with weights `W2 @ W1` and bias
    `b1 @ W2.T + b2`. However many Linear layers you stack, the result can
    always be written as one Linear layer. The extra layers add parameters
    and computation, but they cannot fit anything that one layer could not.

    ```python
    W1 = torch.tensor([[1.0, 2.0], [0.0, 1.0], [1.0, -1.0]])    # 2 inputs to 3 outputs
    W2 = torch.tensor([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])       # 3 inputs to 2 outputs
    W2 @ W1    # tensor([[2., 1.], [2., 5.]]), the weights of the one layer they make
    ```

    The fix is to put a step between the layers that is not a weighted sum.
    Then the expression can no longer be collapsed into one layer. The next
    chapter builds that step. It is the reason models with many layers can
    learn shapes that a single layer cannot.
    """,
)

LINEAR3_SOURCE = source_file(
    '''
    # linear3.py

    import torch
    from torch import Tensor


    # TODO: Pass x through the first Linear layer, with weights W1 and bias b1.
    #   Then pass the result through the second Linear layer, with W2 and b2,
    #   and return its output. Each W has one row per output, as in linear1.
    def two_layers(
        x: Tensor,
        W1: Tensor,
        b1: Tensor,
        W2: Tensor,
        b2: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish both lines, so this returns the weights W and the bias b of a
    #   single Linear layer that gives the same output as two_layers.
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
        description="gives the single layer that equals the two",
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
        "two_layers": "Apply the first layer as linear1 does, x @ W1.T + b1, then apply the second layer to its "
        "output.",
        "collapse": "The lesson expands the two layers into one. W is the second layer's weights matrix multiplied "
        "by the first's. b is the first bias passed through the second layer's weights, plus the second bias.",
    },
    introduces=(),
    check=check_linear3,
)
