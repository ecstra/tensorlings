"""19_initialization, how big the starting weights should be, worked out, matched with nn.init, and checked by training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    NOT_STARTED,
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    reject_missing_return,
    seeded,
    small_tensor,
    value_text,
)
from ..text import prose, source_file
from .linear_layers import expect_parameter
from .mlp import GradWatcher, seeded_mlp

INITIALIZATION = Chapter(
    folder="19_initialization",
    title="Initialization",
    readme=source_file(
        """
        # Initialization

        How big a model's weights start decides whether a deep model trains at
        all. Too big and the numbers explode, too small and they vanish. This
        chapter works out the right size, matches the Xavier and Kaiming rules
        with PyTorch's `nn.init`, shows a deep MLP on MNIST that only trains
        from Kaiming's start, and gives each activation its scale.

        ## Further reading

        - [He and others, delving deep into rectifiers, the Kaiming paper](https://arxiv.org/abs/1502.01852)
        - [Glorot and Bengio, the difficulty of training deep networks, the Xavier paper](https://proceedings.mlr.press/v9/glorot10a.html)
        - [Karpathy, a recipe for training neural networks](https://karpathy.github.io/2019/04/25/recipe/)
        - [Karpathy, building makemore part 3, activations and gradients](https://www.youtube.com/watch?v=P6sfmUTpUmc)
        - [PyTorch, nn.init](https://docs.pytorch.org/docs/2.14/nn.init.html)
        """,
    ),
)

ACTIVATIONS = {
    "relu": torch.relu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "none": lambda x: x,
}


def bell_curve(
    seed: int,
    *shape: int,
) -> Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def relative(expected: float) -> float:
    """A tolerance that grows with the number, since a spread can be in the billions."""
    return 1e-5 * max(1.0, abs(expected))


def expect_numbers(
    call: str,
    actual: object,
    expected: list[float],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, (list, tuple)):
        raise Mismatch(f"{call} is {value_text(actual)}. It should be a list of numbers, one per layer.")

    if len(actual) != len(expected):
        raise Mismatch(f"{call} has {len(actual)} numbers, but it should have {len(expected)}, one per layer.")

    for position, (got, wanted) in enumerate(zip(actual, expected)):
        expect_close(
            call=f"number {position + 1} of {call}",
            actual=got,
            expected=wanted,
            tolerance=relative(wanted),
        )


def expect_scaled_weights(
    call: str,
    actual: object,
    expected: list[Tensor],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, (list, tuple)):
        raise Mismatch(f"{call} is {value_text(actual)}. It should be a list of weights.")

    if len(actual) != len(expected):
        raise Mismatch(f"{call} has {len(actual)} weights, but it should have {len(expected)}.")

    for position, (got, wanted) in enumerate(zip(actual, expected)):
        name = f"weight {position + 1} of {call}"

        if isinstance(got, Tensor) and got.shape == wanted.shape and got.numel() > 1:
            spread = got.std().item()
            target = 1 / math.sqrt(wanted.shape[0])

            if abs(spread - target) > 0.3 * target:
                raise Mismatch(f"{name} has spread {spread:.3g}, but it should be about {target:.3g}, 1 / sqrt(width).")

        expect_tensor(
            call=name,
            actual=got,
            expected=wanted,
        )


# ---------------------------------------------------------------------------
# init1
# ---------------------------------------------------------------------------

INIT1_LESSON = prose(
    """
    Every layer so far started from random weights. How big they start
    decides whether a deep model can train at all.

    A neuron adds up n products, one input times one weight each. The square
    of a spread is the **variance**. Variances add up when independent
    numbers are added. Two independent numbers with mean 0, multiplied
    together, give a variance that is the product of theirs.

    So with inputs of spread 1 and weights of spread `s`, each output has
    variance `n * s ** 2`, a spread of `sqrt(n) * s`. Every layer of a stack
    multiplies the spread by that much again.

    Here 1000 rows of 100 numbers with spread 1 went through 10 layers of
    shape `(100, 100)`, with no activation, on 5 seeds. `1e10` is a 1
    followed by 10 zeros, and `1e-10` is 1 divided by that.

    | Weights' spread | After 1 layer | After 10 layers |
    |---|---|---|
    | 1 | 10.0 | about `1e10` |
    | 0.01 | 0.1 | about `1e-10` |
    | 0.1, which is 1 / sqrt(100) | 1.00 | 0.96 to 1.04 |

    Too big, and the numbers explode until they overflow and the loss turns
    `nan`. Too small, and they shrink until the output no longer depends on
    the input. The backward pass multiplies the gradients layer by layer in
    much the same way, so they explode or vanish too.

    Weights with spread `1 / sqrt(n)`, where n is the number of inputs each
    neuron adds up, keep the spread steady at any depth.

    ```python
    x = torch.randn(1000, 100)
    weight = torch.randn(100, 100) / math.sqrt(100)
    (x @ weight).std()    # about 1.0
    ```

    `torch.randn(3, 2, generator=generator)` draws from the generator you pass,
    as `torch.randperm` did in Section 2.
    """,
)

INIT1_SOURCE = source_file(
    '''
    # init1.py

    import math

    import torch
    from torch import Tensor

    x = torch.randn(1000, 100)
    weight = torch.randn(100, 100) / math.sqrt(100)

    # One layer with weights of spread 1 / sqrt(100) keeps the spread at 1.
    spread = (x @ weight).std()                       # about 1.0


    # TODO: Return the spread the lesson predicts after depth layers, each
    #   adding up n inputs, for inputs of spread 1 and weights of spread
    #   weight_spread.
    def predicted_spread(
        n: int,
        weight_spread: float,
        depth: int,
    ) -> float:
        ...


    # TODO: Finish the line in the loop, so this passes x through each weight in
    #   turn and returns the spread after every layer, as a list.
    def spreads_through(
        x: Tensor,
        weights: list[Tensor],
    ) -> list[float]:
        spreads = []

        for weight in weights:
            x = ...
            spreads.append(x.std().item())

        return spreads


    # TODO: Return a list of depth weights of shape (width, width), each drawn
    #   with torch.randn from generator and scaled so the spread stays steady.
    def scaled_weights(
        width: int,
        depth: int,
        generator: torch.Generator,
    ) -> list[Tensor]:
        ...
    ''',
)

INIT1_SOLUTION = source_file(
    '''
    # init1.py, a solution for comparison

    import math

    import torch
    from torch import Tensor

    x = torch.randn(1000, 100)
    weight = torch.randn(100, 100) / math.sqrt(100)

    spread = (x @ weight).std()


    # Each layer multiplies the spread by sqrt(n) * weight_spread.
    def predicted_spread(
        n: int,
        weight_spread: float,
        depth: int,
    ) -> float:
        return (math.sqrt(n) * weight_spread) ** depth


    def spreads_through(
        x: Tensor,
        weights: list[Tensor],
    ) -> list[float]:
        spreads = []

        for weight in weights:
            x = x @ weight
            spreads.append(x.std().item())

        return spreads


    def scaled_weights(
        width: int,
        depth: int,
        generator: torch.Generator,
    ) -> list[Tensor]:
        return [torch.randn(width, width, generator=generator) / math.sqrt(width) for _ in range(depth)]
    ''',
)


def reference_spreads(
    x: Tensor,
    weights: list[Tensor],
) -> list[float]:
    spreads = []

    for weight in weights:
        x = x @ weight
        spreads.append(x.std().item())

    return spreads


def weights_with_spread(
    seed: int,
    width: int,
    depth: int,
    spread: float,
) -> Described:
    generator = torch.Generator().manual_seed(seed)
    weights = [torch.randn(width, width, generator=generator) * spread for _ in range(depth)]

    return Described(weights, f"{depth} weights of shape ({width}, {width}) with spread {spread}")


def check_init1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="predicted_spread",
        description="multiplies the spread by sqrt(n) times the weights' spread per layer",
        inputs=(
            (100, 1.0, 3),
            (64, 0.05, 4),
            (9, 0.5, 2),
            (400, 0.05, 7),
        ),
        reference=lambda n, weight_spread, depth: (math.sqrt(n) * weight_spread) ** depth,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=relative(expected),
        ),
    )
    check_on_inputs(
        checker=checker,
        function="spreads_through",
        description="gives the spread after every layer",
        inputs=(
            (Described(bell_curve(240, 200, 16), "x of shape (200, 16) from the bell curve"), weights_with_spread(241, 16, 4, 1.0)),
            (Described(bell_curve(242, 300, 25), "x of shape (300, 25) from the bell curve"), weights_with_spread(243, 25, 3, 0.2)),
        ),
        reference=reference_spreads,
        expect=expect_numbers,
    )
    check_on_inputs(
        checker=checker,
        function="scaled_weights",
        description="draws depth weights with spread 1 / sqrt(width)",
        inputs=(
            (16, 3, seeded(244)),
            (50, 2, seeded(245)),
        ),
        reference=lambda width, depth, generator: [
            torch.randn(width, width, generator=generator) / math.sqrt(width) for _ in range(depth)
        ],
        expect=expect_scaled_weights,
    )


INIT1 = Exercise(
    name="init1",
    chapter=INITIALIZATION,
    title="Exploding and vanishing",
    lesson=INIT1_LESSON,
    source=INIT1_SOURCE,
    solution=INIT1_SOLUTION,
    hints={
        "predicted_spread": "One layer multiplies the spread by math.sqrt(n) * weight_spread. depth layers do that "
        "depth times, which ** says in one step.",
        "spreads_through": "The missing line is one layer, x matrix multiplied by the weight.",
        "scaled_weights": "Draw each weight with torch.randn(width, width, generator=generator), divide it by the "
        "square root of width, and put depth of them in a list.",
    },
    introduces=("variance",),
    check=check_init1,
)


# ---------------------------------------------------------------------------
# init2
# ---------------------------------------------------------------------------

INIT2_LESSON = prose(
    """
    The **fan in** of a layer is the number of inputs each neuron adds up, its
    `in_features`. The fan out is its `out_features`. init1's rule is spread
    `1 / sqrt(fan_in)`, which suits a layer with no activation after it.

    ReLU sets every negative number to 0, about half of them. That halves the
    average square of the numbers at every layer, so the spread still shrinks.
    Kaiming He and others made up for it in 2015 by doubling the variance.

    - Kaiming: spread `sqrt(2 / fan_in)`, for layers followed by ReLU.
    - Xavier: spread `sqrt(2 / (fan_in + fan_out))`, for tanh, from Xavier Glorot and Yoshua Bengio in 2010.

    The rule is named after Glorot's first name. The forward pass adds up
    fan_in numbers and the backward pass adds up fan_out, and Xavier's rule
    balances the two.

    PyTorch's own Linear draws evenly between `-1 / sqrt(fan_in)` and
    `1 / sqrt(fan_in)`, as linear2 said. Numbers drawn evenly between `-b`
    and `b` have spread `b / sqrt(3)`, so this starts smaller than even
    init1's rule. Shallow models don't mind, but deep ReLU stacks do.

    No paper worked that size out. It is an old rule of thumb, carried over
    from Lush and Torch, the libraries that came before PyTorch. Kaiming's
    size is the one worked out from the variance.

    `nn.init` holds the rules. Each fills a weight in place, which the
    trailing `_` in its name marks.

    ```python
    layer = nn.Linear(100, 50)
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    layer.weight.std()                  # about 0.141, sqrt(2 / 100)
    nn.init.xavier_normal_(layer.weight)
    layer.weight.std()                  # about 0.115, sqrt(2 / 150)
    ```

    By hand, `.normal_(mean, spread)` fills a tensor in place from the bell
    curve, and `.zero_()` fills it with zeros. Changing a parameter by hand
    goes inside `with torch.no_grad():`, as linear2 said.
    """,
)

INIT2_SOURCE = source_file(
    '''
    # init2.py

    import math

    import torch
    import torch.nn as nn

    # Numbers drawn evenly between -bound and bound.
    bound = 1 / math.sqrt(100)
    evenly = torch.empty(100000).uniform_(-bound, bound)
    evenly_spread = evenly.std()                      # about 0.0577, bound / sqrt(3)


    # TODO: Return the spread of the weights of PyTorch's default Linear, for a
    #   layer with fan_in inputs.
    def default_std(fan_in: int) -> float:
        ...


    # TODO: Return Kaiming's spread for a layer with fan_in inputs.
    def kaiming_std(fan_in: int) -> float:
        ...


    # TODO: Return Xavier's spread for a layer with fan_in inputs and fan_out
    #   outputs.
    def xavier_std(
        fan_in: int,
        fan_out: int,
    ) -> float:
        ...


    layer = nn.Linear(100, 50)

    with torch.no_grad():
        layer.bias.zero_()                            # every bias set to 0, in place


    # TODO: Fill layer's weight in place from the bell curve with Kaiming's spread,
    #   and its bias with zeros.
    def kaiming_init_(layer: nn.Linear) -> None:
        ...
    ''',
)

INIT2_SOLUTION = source_file(
    '''
    # init2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn

    bound = 1 / math.sqrt(100)
    evenly = torch.empty(100000).uniform_(-bound, bound)
    evenly_spread = evenly.std()


    def default_std(fan_in: int) -> float:
        return 1 / math.sqrt(fan_in) / math.sqrt(3)


    def kaiming_std(fan_in: int) -> float:
        return math.sqrt(2 / fan_in)


    def xavier_std(
        fan_in: int,
        fan_out: int,
    ) -> float:
        return math.sqrt(2 / (fan_in + fan_out))


    layer = nn.Linear(100, 50)

    with torch.no_grad():
        layer.bias.zero_()


    def kaiming_init_(layer: nn.Linear) -> None:
        with torch.no_grad():
            layer.weight.normal_(0, kaiming_std(layer.in_features))
            layer.bias.zero_()
    ''',
)


def check_kaiming_init(found: object) -> None:
    for fan_in, fan_out in ((1024, 256), (300, 600)):
        call = f"kaiming_init_(nn.Linear({fan_in}, {fan_out}))"
        torch.manual_seed(fan_in)
        layer = nn.Linear(fan_in, fan_out)
        parameters = (layer.weight, layer.bias)
        before = layer.weight.detach().clone()
        returned = found(layer)

        if isinstance(returned, nn.Module) and returned is not layer:
            raise Mismatch(f"{call} returns a new layer. It should change the layer it is given, in place.")

        if layer.weight is not parameters[0] or layer.bias is not parameters[1]:
            raise Mismatch(f"{call} put a new parameter in the layer. Fill the weight and the bias it already has, in place, with .normal_ and .zero_.")

        if torch.equal(layer.weight.detach(), before):
            raise Mismatch(f"{call} left the weight unchanged, which usually means it is {NOT_STARTED}.")

        weight = expect_parameter(layer, call, "weight", (fan_out, fan_in)).detach()
        bias = expect_parameter(layer, call, "bias", (fan_out,)).detach()
        target = math.sqrt(2 / fan_in)
        spread = weight.std().item()

        if abs(spread - target) > 0.03 * target:
            raise Mismatch(f"After {call} the weight's spread is {spread:.4g}, but it should be about {target:.4g}, sqrt(2 / {fan_in}).")

        if abs(weight.mean().item()) > 0.05 * target:
            raise Mismatch(f"After {call} the weight's mean is {weight.mean().item():.4g}, but it should be about 0.")

        if bias.abs().max().item() > 0:
            raise Mismatch(f"After {call} the bias should be all zeros, but its largest number is {bias.abs().max().item():.4g}.")


def check_init2(checker: Checker) -> None:
    exact = lambda call, actual, expected: expect_close(
        call=call,
        actual=actual,
        expected=expected,
        tolerance=1e-9,
    )
    check_on_inputs(
        checker=checker,
        function="default_std",
        description="gives the spread of numbers drawn evenly within 1 / sqrt(fan_in)",
        inputs=(
            (3,),
            (100,),
            (784,),
        ),
        reference=lambda fan_in: (1 / math.sqrt(fan_in)) / math.sqrt(3),
        expect=exact,
    )
    check_on_inputs(
        checker=checker,
        function="kaiming_std",
        description="matches nn.init's Kaiming spread for ReLU",
        inputs=(
            (100,),
            (784,),
            (7,),
        ),
        reference=lambda fan_in: nn.init.calculate_gain("relu") / math.sqrt(fan_in),
        expect=exact,
    )
    check_on_inputs(
        checker=checker,
        function="xavier_std",
        description="matches nn.init's Xavier spread",
        inputs=(
            (100, 50),
            (784, 10),
            (3, 3),
        ),
        reference=lambda fan_in, fan_out: math.sqrt(2 / (fan_in + fan_out)),
        expect=exact,
    )
    checker.test(
        function="kaiming_init_",
        description="fills the weight with Kaiming's spread and the bias with zeros",
        run=check_kaiming_init,
    )


INIT2 = Exercise(
    name="init2",
    chapter=INITIALIZATION,
    title="Xavier and Kaiming",
    lesson=INIT2_LESSON,
    source=INIT2_SOURCE,
    solution=INIT2_SOLUTION,
    hints={
        "default_std": "PyTorch's bound is 1 / sqrt(fan_in), and numbers drawn evenly within a bound have spread "
        "bound / sqrt(3).",
        "kaiming_std": "The square root of 2 / fan_in. math.sqrt takes the square root of a plain number.",
        "xavier_std": "The square root of 2 divided by fan_in and fan_out added together.",
        "kaiming_init_": "Inside a with torch.no_grad() block, call .normal_ on layer.weight with mean 0 and "
        "kaiming_std(layer.in_features), then .zero_() on layer.bias.",
    },
    introduces=("fan in",),
    check=check_init2,
)


# ---------------------------------------------------------------------------
# init3
# ---------------------------------------------------------------------------

INIT3_LESSON = prose(
    """
    Here is what the starting weights do to training. MNIST is a set of
    70,000 handwritten digits, each a grid of 28 by 28 pixels, labelled 0 to
    9. Laid flat, each image is 784 numbers.

    An MLP with 8 hidden layers of 64, each a Linear then ReLU, learned from
    10,000 of those images. It trained for 3 epochs on minibatches of 128, at
    a learning rate of 0.1. Only the starting weights differ, and every bias
    starts at 0. Each range covers 5 seeds.

    | Weights | Starting loss | Training loss after | Validation accuracy |
    |---|---|---|---|
    | spread 1 | 10 to 17 million | `nan` | `nan` |
    | spread 0.01 | 2.303 | 2.300 | 11.0% |
    | PyTorch's default | 2.303 | 2.300 | 11.0% |
    | Kaiming | 2.34 to 2.45 | 0.24 to 0.37 | 87.8% to 91.2% |

    Spread 1 explodes, and the loss overflows to `nan`. Spread 0.01 and the
    default both vanish, and their outputs barely depend on the input. The
    default's gradient at the first layer is 1000 to 3800 times smaller than
    Kaiming's. Spread 0.01's is over 10 billion times smaller.

    So their loss sits near 2.303 the whole time. Their 11.0% is what always
    answering the most common digit scores.

    The starting loss is the first thing to check. A model that knows nothing
    starts near `log(classes)`, as likelihood2 said, which is 2.303 for 10.
    Far above it means the model starts confidently wrong, and the first steps
    go to undoing that. Near it is needed but not enough, since spread 0.01
    started at 2.303 and never learned.

    `model.modules()` gives every module inside a model, the model itself
    included. `isinstance(module, nn.Linear)` is `True` for a Linear layer and
    `False` for the rest, like a ReLU or the `nn.Sequential` itself.

    ```python
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    [type(module).__name__ for module in model.modules()]
    # ['Sequential', 'Linear', 'ReLU', 'Linear']
    ```
    """,
)

INIT3_SOURCE = source_file(
    '''
    # init3.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # kaiming_init_ from init2.
    def kaiming_init_(layer: nn.Linear) -> None:
        with torch.no_grad():
            layer.weight.normal_(0, math.sqrt(2 / layer.in_features))
            layer.bias.zero_()


    # Logits that are all equal, the model that knows nothing.
    logits = torch.zeros(4, 10)
    labels = torch.tensor([3, 1, 4, 1])
    know_nothing = F.cross_entropy(logits, labels)    # tensor(2.3026), log(10)


    # TODO: Return the cross entropy of the model's logits on x against the labels
    #   y, without recording anything for autograd, as a plain number.
    def start_loss(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        ...


    # TODO: Return how far start_loss is above the loss of a model that knows
    #   nothing. The number of classes is the width of the model's output.
    def start_gap(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        ...


    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    names = [type(module).__name__ for module in model.modules()]   # ['Sequential', 'Linear', 'ReLU', 'Linear']


    # TODO: Give every Linear layer in model Kaiming's weights and zero biases,
    #   with kaiming_init_, in the order model.modules() gives them.
    def kaiming_mlp_(model: nn.Module) -> None:
        ...
    ''',
)

INIT3_SOLUTION = source_file(
    '''
    # init3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    def kaiming_init_(layer: nn.Linear) -> None:
        with torch.no_grad():
            layer.weight.normal_(0, math.sqrt(2 / layer.in_features))
            layer.bias.zero_()


    logits = torch.zeros(4, 10)
    labels = torch.tensor([3, 1, 4, 1])
    know_nothing = F.cross_entropy(logits, labels)


    def start_loss(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        with torch.no_grad():
            return F.cross_entropy(model(x), y).item()


    def start_gap(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        with torch.no_grad():
            classes = model(x).shape[1]

        return start_loss(model, x, y) - math.log(classes)


    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    names = [type(module).__name__ for module in model.modules()]


    def kaiming_mlp_(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                kaiming_init_(module)
    ''',
)


def scaled_mlp(
    seed: int,
    sizes: tuple[int, ...],
    scale: float,
) -> nn.Sequential:
    model = seeded_mlp(seed, sizes)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter *= scale

    return model


def reference_kaiming_mlp_(model: nn.Module) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.normal_(0, math.sqrt(2 / module.in_features))
                module.bias.zero_()


def check_kaiming_mlp(found: object) -> None:
    for sizes in ((6, 16, 16, 3), (10, 32, 32, 32, 4)):
        call = f"kaiming_mlp_ on an MLP with sizes {list(sizes)}"
        model = seeded_mlp(0, sizes)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        torch.manual_seed(9)
        found(model)

        if all(torch.equal(old, new.detach()) for old, new in zip(before, model.parameters())):
            raise Mismatch(f"{call} left the model unchanged, which usually means it is {NOT_STARTED}.")

        reference = seeded_mlp(0, sizes)
        torch.manual_seed(9)
        reference_kaiming_mlp_(reference)

        linears = [module for module in model.modules() if isinstance(module, nn.Linear)]
        wanted = [module for module in reference.modules() if isinstance(module, nn.Linear)]

        for position, (mine, theirs) in enumerate(zip(linears, wanted)):
            if torch.allclose(mine.weight, theirs.weight) and torch.allclose(mine.bias, theirs.bias):
                continue

            spread = mine.weight.detach().std().item()
            target = math.sqrt(2 / mine.in_features)
            zero_bias = mine.bias.abs().max().item() == 0
            right_spread = abs(spread - target) < 0.25 * target

            if right_spread and zero_bias:
                raise Mismatch(f"After {call}, Linear layer {position + 1} has the right spread but different numbers. Fill the layers in the order model.modules() gives them.")

            raise Mismatch(
                f"After {call}, Linear layer {position + 1} doesn't hold Kaiming's weights and zero biases. "
                f"Its weight's spread is {spread:.3g}, where sqrt(2 / {mine.in_features}) is {target:.3g}, "
                f"and its bias {'is' if zero_bias else 'is not'} all zeros.",
            )


def check_start_loss_records_nothing(found: object) -> None:
    watcher = GradWatcher((4, 8, 3))
    found(watcher, small_tensor(250, 6, 4), torch.tensor([0, 2, 1, 1, 0, 2]))

    if any(watcher.recording):
        raise Mismatch("start_loss ran the model while autograd was recording. Put the work inside a with torch.no_grad() block.")


def check_init3(checker: Checker) -> None:
    close = lambda call, actual, expected: expect_close(
        call=call,
        actual=actual,
        expected=expected,
        tolerance=relative(expected),
    )
    data = (
        (
            Described(seeded_mlp(3, (4, 8, 3)), "an MLP from 4 inputs to 3 classes"),
            small_tensor(250, 6, 4),
            torch.tensor([0, 2, 1, 1, 0, 2]),
        ),
        (
            Described(scaled_mlp(4, (5, 16, 16, 4), 4.0), "an MLP from 5 inputs to 4 classes with its weights scaled up 4 times"),
            small_tensor(251, 8, 5),
            torch.tensor([3, 0, 1, 2, 2, 0, 1, 3]),
        ),
        # Classes 2 and 3 never appear in the labels, so a class count taken from y comes out wrong.
        (
            Described(seeded_mlp(5, (3, 8, 4)), "an MLP from 3 inputs to 4 classes"),
            small_tensor(257, 4, 3),
            torch.tensor([0, 1, 1, 0]),
        ),
    )
    check_on_inputs(
        checker=checker,
        function="start_loss",
        description="gives the cross entropy of the model's logits",
        inputs=data,
        reference=lambda model, x, y: F.cross_entropy(model(x), y).item(),
        expect=close,
    )
    checker.test(
        function="start_loss",
        description="runs the model without recording for autograd",
        run=check_start_loss_records_nothing,
    )
    check_on_inputs(
        checker=checker,
        function="start_gap",
        description="gives how far the loss is above log of the class count",
        inputs=data,
        reference=lambda model, x, y: F.cross_entropy(model(x), y).item() - math.log(model(x).shape[1]),
        expect=close,
    )
    checker.test(
        function="kaiming_mlp_",
        description="gives every Linear layer Kaiming's weights and zero biases",
        run=check_kaiming_mlp,
    )


INIT3 = Exercise(
    name="init3",
    chapter=INITIALIZATION,
    title="Why the start matters",
    lesson=INIT3_LESSON,
    source=INIT3_SOURCE,
    solution=INIT3_SOLUTION,
    hints={
        "start_loss": "Inside a with torch.no_grad() block, take F.cross_entropy of model(x) and y, and turn it "
        "into a plain number with .item().",
        "start_gap": "model(x).shape[1] is the number of classes. Take math.log of it away from start_loss(model, x, "
        "y).",
        "kaiming_mlp_": "Loop over model.modules(), and call kaiming_init_ on each module that isinstance says is an "
        "nn.Linear.",
    },
    introduces=(),
    check=check_init3,
)


# ---------------------------------------------------------------------------
# init4
# ---------------------------------------------------------------------------

INIT4_LESSON = prose(
    """
    Each activation needs its own scale. Here 1000 rows went through layers
    of width 100, each a matrix multiply then the activation, on 5 seeds.
    Each cell is the spread at the end, the median of the seeds. Where seeds
    differ a lot, the lowest and highest follow in brackets.

    | Activation | Weights | 10 layers | 30 layers | 50 layers |
    |---|---|---|---|---|
    | none | `1 / sqrt(fan_in)` | 1.00 (0.96 to 1.04) | 0.89 (0.75 to 1.03) | 0.71 (0.63 to 1.26) |
    | ReLU | `1 / sqrt(fan_in)` | 0.024 | 2e-5 | 1e-8 |
    | ReLU | PyTorch's default | 1e-4 | 2e-12 | 3e-20 |
    | ReLU | Kaiming | 0.76 (0.53 to 1.05) | 0.60 (0.30 to 1.16) | 0.42 (0.26 to 1.12) |
    | tanh | spread 1 | 0.96, 78% past 0.99 | 0.96 | 0.96 |
    | tanh | PyTorch's default | 0.003 | 5e-8 | 9e-13 |
    | tanh | Xavier | 0.22 | 0.12 | 0.08 |
    | tanh | Xavier times 5/3 | 0.65 | 0.65 | 0.65 |
    | GELU | Kaiming | 0.42 (0.30 to 0.70) | 0.04 (0.001 to 0.36) | 1e-4 (1e-6 to 0.09) |

    - ReLU: Kaiming, `sqrt(2 / fan_in)`.
    - tanh: Xavier times tanh's gain of 5/3, as `nn.init.calculate_gain("tanh")` gives.
    - no activation after the layer: `1 / sqrt(fan_in)`, from init1.

    Spread 1 saturates tanh, as activations1 showed. Glorot and Bengio
    measured Xavier against the start PyTorch now uses by default, and it
    shrinks far slower. With the gain it holds, and under 2% of the numbers
    pass 0.99.

    Kaiming keeps the average square of the numbers at 1, on average across
    runs. Over 20 seeds it averaged 1.28, 0.97 and 0.93 at 10, 30 and 50
    layers. A ReLU's outputs are never negative, so the spread around their
    mean is less, about `sqrt(1 - 1 / pi)`, or 0.83.

    One run still drifts as it goes deeper. At 50 layers its average square
    ran from 0.09 to 4.1 over those seeds. Hanin and Rolnick showed in 2018
    that this drift grows with depth divided by width.

    GELU and SiLU match ReLU for large inputs, so Kaiming is the nearest plain
    scale. Near 0 they pass less, so a deep stack under Kaiming shrinks. A
    little more, `1.5 / sqrt(fan_in)`, made it grow to 2.7 to 14 by 50 layers.
    Normalization, two chapters on, holds the spread steady at any depth.

    The last layer gets one more rule, from Karpathy's recipe for training. Set
    its bias so the model starts out predicting the average target. For
    classes, that is each class's share of the labels. Softmax of the logs of
    shares that add up to 1 gives the shares back.

    ```python
    y = torch.tensor([0, 0, 2, 1, 0])
    torch.bincount(y, minlength=3)    # tensor([3, 1, 1]), how often each class shows up
    ```

    With the last weights small, the model then starts at the loss of
    guessing by the shares. On the labels the shares came from, that is never
    worse than `log(classes)`.

    A class missing from those labels gets bias `-inf`, the log of 0. The
    model can then never predict or learn that class, and its loss turns
    `inf` if the class shows up. So take the shares from labels that hold
    every class.
    """,
)

INIT4_SOURCE = source_file(
    '''
    # init4.py

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    # Each activation by its name, with "none" passing x through unchanged.
    ACTIVATIONS = {
        "relu": torch.relu,
        "gelu": F.gelu,
        "tanh": torch.tanh,
        "none": lambda x: x,
    }

    kaiming_100 = math.sqrt(2 / 100)                  # 0.1414, Kaiming for fan_in 100


    # TODO: Return the spread for a layer's weights, given the activation after
    #   it, "relu", "gelu", "tanh" or "none", and the layer's fan_in and fan_out.
    def init_std(
        activation: str,
        fan_in: int,
        fan_out: int,
    ) -> float:
        ...


    # TODO: Finish the two lines in the loop, so this passes x through depth
    #   layers of width x.shape[1]. Each draws its weight with torch.randn from
    #   generator at init_std's spread, then sets x to the activation of
    #   x @ weight. It returns the spread at the end.
    def final_spread(
        x: Tensor,
        activation: str,
        depth: int,
        generator: torch.Generator,
    ) -> float:
        width = x.shape[1]

        for _ in range(depth):
            weight = ...
            x = ...

        return x.std().item()


    y = torch.tensor([0, 0, 2, 1, 0])
    counts = torch.bincount(y, minlength=3)           # tensor([3, 1, 1])


    # TODO: Return the log of each class's share of the labels y, one number
    #   per class. As the last layer's bias, it makes the model start out
    #   predicting those shares.
    def last_bias(
        y: Tensor,
        classes: int,
    ) -> Tensor:
        ...
    ''',
)

INIT4_SOLUTION = source_file(
    '''
    # init4.py, a solution for comparison

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    ACTIVATIONS = {
        "relu": torch.relu,
        "gelu": F.gelu,
        "tanh": torch.tanh,
        "none": lambda x: x,
    }

    kaiming_100 = math.sqrt(2 / 100)


    def init_std(
        activation: str,
        fan_in: int,
        fan_out: int,
    ) -> float:
        if activation in ("relu", "gelu"):
            return math.sqrt(2 / fan_in)

        if activation == "tanh":
            return 5 / 3 * math.sqrt(2 / (fan_in + fan_out))

        return 1 / math.sqrt(fan_in)


    def final_spread(
        x: Tensor,
        activation: str,
        depth: int,
        generator: torch.Generator,
    ) -> float:
        width = x.shape[1]

        for _ in range(depth):
            weight = torch.randn(width, width, generator=generator) * init_std(activation, width, width)
            x = ACTIVATIONS[activation](x @ weight)

        return x.std().item()


    y = torch.tensor([0, 0, 2, 1, 0])
    counts = torch.bincount(y, minlength=3)


    # Softmax of the log of shares gives the shares back.
    def last_bias(
        y: Tensor,
        classes: int,
    ) -> Tensor:
        counts = torch.bincount(y, minlength=classes).float()

        return (counts / len(y)).log()
    ''',
)


def reference_init_std(
    activation: str,
    fan_in: int,
    fan_out: int,
) -> float:
    if activation in ("relu", "gelu"):
        return math.sqrt(2 / fan_in)

    if activation == "tanh":
        return 5 / 3 * math.sqrt(2 / (fan_in + fan_out))

    return 1 / math.sqrt(fan_in)


def reference_final_spread(
    x: Tensor,
    activation: str,
    depth: int,
    generator: torch.Generator,
) -> float:
    width = x.shape[1]

    for _ in range(depth):
        weight = torch.randn(width, width, generator=generator) * reference_init_std(activation, width, width)
        x = ACTIVATIONS[activation](x @ weight)

    return x.std().item()


def check_init4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="init_std",
        description="picks the spread that suits the activation",
        inputs=(
            ("relu", 100, 50),
            ("gelu", 64, 64),
            ("tanh", 100, 50),
            ("none", 30, 10),
        ),
        reference=reference_init_std,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-9,
        ),
    )
    x = Described(bell_curve(252, 500, 64), "x of shape (500, 64) from the bell curve")
    check_on_inputs(
        checker=checker,
        function="final_spread",
        description="passes x through layers set up for the activation",
        inputs=(
            (x, "relu", 6, seeded(253)),
            (x, "tanh", 6, seeded(254)),
            (x, "none", 4, seeded(255)),
            (x, "gelu", 5, seeded(256)),
        ),
        reference=reference_final_spread,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=relative(expected),
        ),
    )
    check_on_inputs(
        checker=checker,
        function="last_bias",
        description="gives the log of each class's share of the labels",
        inputs=(
            (torch.tensor([0, 0, 2, 1, 0]), 3),
            (torch.tensor([1, 1, 1, 0, 2, 3, 3, 1]), 4),
            (torch.tensor([2, 2, 0, 1, 2, 2, 2, 0, 2, 1]), 3),
            # Class 2 never shows up, so the counts need minlength to give 3 numbers, the last -inf.
            (torch.tensor([0, 1, 1, 0]), 3),
        ),
        reference=lambda y, classes: (torch.bincount(y, minlength=classes).float() / len(y)).log(),
        expect=expect_tensor,
    )


INIT4 = Exercise(
    name="init4",
    chapter=INITIALIZATION,
    title="A scale for each activation",
    lesson=INIT4_LESSON,
    source=INIT4_SOURCE,
    solution=INIT4_SOLUTION,
    hints={
        "init_std": "Use Kaiming's sqrt(2 / fan_in) for relu and gelu, and 1 / sqrt(fan_in) for none. For tanh, "
        "multiply Xavier's sqrt(2 / (fan_in + fan_out)) by the gain 5/3.",
        "final_spread": "Each weight is torch.randn(width, width, generator=generator) times init_std(activation, "
        "width, width). Then x goes through x @ weight and the function ACTIVATIONS[activation] gives.",
        "last_bias": "torch.bincount(y, minlength=classes) counts each class. Turn the counts into floats, divide by "
        "len(y) for the shares, and take their .log().",
    },
    introduces=(),
    check=check_init4,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

def measure_stack(seeds: range = range(5)) -> dict[float, list[tuple[float, float]]]:
    """init1's table: for each weights' spread, the spread after 1 layer and after 10 on every seed. 1000 rows of 100
    numbers from the bell curve go through 10 weights of shape (100, 100), with no activation."""
    results: dict[float, list[tuple[float, float]]] = {}

    for weight_spread in (1.0, 0.01, 0.1):
        results[weight_spread] = []

        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)
            x = torch.randn(1000, 100, generator=generator)
            spreads = []

            for _ in range(10):
                x = x @ (torch.randn(100, 100, generator=generator) * weight_spread)
                spreads.append(x.std().item())

            results[weight_spread].append((spreads[0], spreads[-1]))

    return results


# init3's rows: how each Linear layer's weights start. Every bias starts at 0.
STARTS = ("spread 1", "spread 0.01", "default", "kaiming")


def started_mlp(
    start: str,
    seed: int,
) -> nn.Sequential:
    """init3's MLP from 784 pixels through 8 hidden layers of 64 to 10 classes, drawn after torch.manual_seed(seed)."""
    model = seeded_mlp(seed, (784, *[64] * 8, 10))

    with torch.no_grad():
        for layer in model.modules():
            if not isinstance(layer, nn.Linear):
                continue

            if start == "spread 1":
                layer.weight.normal_(0, 1.0)
            elif start == "spread 0.01":
                layer.weight.normal_(0, 0.01)
            elif start == "kaiming":
                layer.weight.normal_(0, math.sqrt(2 / layer.in_features))

            layer.bias.zero_()

    return model


def measure_starts(seeds: range = range(5)) -> dict[tuple[str, int], dict[str, float]]:
    """init3's table, keyed by start and seed. Each run gives the starting loss on the 10,000 training images, the
    size of the first layer's gradient for that loss, then the training loss and validation accuracy after 3 epochs
    in minibatches of 128 at a rate of 0.1, from train_on_mnist."""
    from .mnist import mnist_sets
    from .normalization import pinned_threads, train_on_mnist

    sets = mnist_sets()
    x, y = sets["train_x"][:10000], sets["train_y"][:10000]
    results: dict[tuple[str, int], dict[str, float]] = {}

    with pinned_threads():
        for start in STARTS:
            for seed in seeds:
                model = started_mlp(start, seed)
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                gradient = model[0].weight.grad.norm().item()
                model.zero_grad()
                run = train_on_mnist(model, 0.1, seed, images=10000, epochs=3, batch=128)
                results[(start, seed)] = {
                    "start": loss.item(),
                    "gradient": gradient,
                    "loss": run["loss"],
                    "accuracy": run["final"],
                }

    return results


# init4's rows: the activation, and the weights' spread for fan_in and fan_out of 100. None stands for PyTorch's
# default, drawn evenly between -1 / sqrt(fan_in) and 1 / sqrt(fan_in). GELU at 1.5 / sqrt(fan_in) backs the lesson's
# line that a scale a little above Kaiming's makes a GELU stack grow.
SPREAD_SETUPS: dict[str, tuple[str, float | None]] = {
    "none, 1 / sqrt(fan_in)": ("none", 1 / math.sqrt(100)),
    "relu, 1 / sqrt(fan_in)": ("relu", 1 / math.sqrt(100)),
    "relu, PyTorch's default": ("relu", None),
    "relu, Kaiming": ("relu", math.sqrt(2 / 100)),
    "gelu, Kaiming": ("gelu", math.sqrt(2 / 100)),
    "gelu, 1.5 / sqrt(fan_in)": ("gelu", 1.5 / math.sqrt(100)),
    "tanh, spread 1": ("tanh", 1.0),
    "tanh, PyTorch's default": ("tanh", None),
    "tanh, Xavier": ("tanh", math.sqrt(2 / 200)),
    "tanh, Xavier times 5/3": ("tanh", 5 / 3 * math.sqrt(2 / 200)),
}


def spread_after(
    activation: str,
    std: float | None,
    depth: int,
    seed: int,
) -> tuple[float, float, float]:
    """The spread of 1000 rows of width 100 after depth layers, the share of numbers past 0.99 in size, and the
    average square of the numbers."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1000, 100, generator=generator)

    for _ in range(depth):
        if std is None:
            weight = (torch.rand(100, 100, generator=generator) * 2 - 1) / math.sqrt(100)
        else:
            weight = torch.randn(100, 100, generator=generator) * std

        x = ACTIVATIONS[activation](x @ weight)

    return x.std().item(), (x.abs() > 0.99).float().mean().item(), (x**2).mean().item()


def measure_spreads(
    seeds: range = range(5),
    depths: tuple[int, ...] = (10, 30, 50),
) -> dict[tuple[str, int], list[tuple[float, float, float]]]:
    """init4's table: for each setup and depth, the spread, the share past 0.99 and the average square on every
    seed. The lesson's line on ReLU with Kaiming over 20 seeds comes from measure_spreads(range(20))."""
    results: dict[tuple[str, int], list[tuple[float, float, float]]] = {}

    for name, (activation, std) in SPREAD_SETUPS.items():
        for depth in depths:
            results[(name, depth)] = [
                spread_after(
                    activation=activation,
                    std=std,
                    depth=depth,
                    seed=seed,
                )
                for seed in seeds
            ]

    return results
