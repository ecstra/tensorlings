"""22_residuals, adding a block's input to its output, why deep stacks need it, and the block orders of the papers.

quiz4 and quiz5 close the blocks of Section 3 here."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import QUIZZES, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .dropout import labelled_points
from .initialization import bell_curve, expect_numbers, relative
from .linear_layers import expect_module, expect_parameter
from .normalization import with_random_affine

RESIDUALS = Chapter(
    folder="22_residuals",
    title="Residual connections",
    readme=source_file(
        """
        # Residual connections

        A residual connection adds a block's input back to its output, so the
        gradient reaches the first layer however deep the model is. This
        chapter builds residual blocks, shows a deep plain MLP that won't train
        next to a residual one that does, starts each block as "do nothing",
        and sets out the block orders of the papers.

        ## Further reading

        - [He and others, deep residual learning, the ResNet paper](https://arxiv.org/abs/1512.03385)
        - [He and others, identity mappings in deep residual networks](https://arxiv.org/abs/1603.05027)
        - [Xiong and others, on layer normalization in the transformer architecture](https://arxiv.org/abs/2002.04745)
        - [Dive into Deep Learning, residual networks](https://d2l.ai/chapter_convolutional-modern/resnet.html)
        """,
    ),
)


def seeded_module(
    seed: int,
    make: type,
    *arguments: int,
) -> nn.Module:
    torch.manual_seed(seed)

    return make(*arguments)


def randomize(
    module: nn.Module,
    seed: int,
) -> nn.Module:
    """Every parameter drawn small and random, so a check sees a layer the learner's code skipped or zeroed."""
    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(0.3 * torch.randn(parameter.shape, generator=generator))

    return module


class ReferenceResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.branch(x)


class ReferencePreNormBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, width)
        self.down = nn.Linear(width, width)

        with torch.no_grad():
            self.down.weight.zero_()
            self.down.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.down(torch.relu(self.up(self.norm(x))))


# ---------------------------------------------------------------------------
# residual1
# ---------------------------------------------------------------------------

RESIDUAL1_LESSON = prose(
    """
    A **residual connection** adds a block's input back to its output, so the
    block computes `f(x)` plus `x`. The block `f` only has to learn the change
    to make. If no change helps, `f(x) = 0` hands x on as it is.

    The derivative of x plus `f(x)` is `1 + f'(x)`. Through a stack of blocks,
    the chain rule multiplies these. When every `f'` is tiny, each factor
    still stays near 1, so the gradient reaches the first layer whole. In a
    plain stack the factors are the `f'` alone, and 50 small factors multiply
    to nearly nothing.

    ```python
    x = torch.tensor(1.0, requires_grad=True)
    (x + 0.1 * x).backward()
    x.grad    # tensor(1.1000), 1 from the residual and 0.1 from f
    ```

    He and others brought it in for image models in 2015, and trained one 152
    layers deep. Nearly every deep model since uses it, including the language
    models of later sections.

    The addition needs `f(x)` to have the same shape as `x`. In a module, `f`
    is the block's branch, and `forward` adds the input back.
    """,
)

RESIDUAL1_SOURCE = source_file(
    '''
    # residual1.py

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.tensor(1.0, requires_grad=True)
    (x + 0.1 * x).backward()
    slope = x.grad                                   # tensor(1.1000)


    # TODO: Return x with f(x) added to it.
    def residual(
        x: Tensor,
        f: Callable[[Tensor], Tensor],
    ) -> Tensor:
        ...


    # TODO: Finish the line in the loop, so each layer multiplies h by scale and
    #   adds h back when use_residual is True. This returns the gradient at x.
    def input_gradient(
        scale: float,
        depth: int,
        use_residual: bool,
    ) -> float:
        x = torch.tensor(1.0, requires_grad=True)
        h = x

        for _ in range(depth):
            if use_residual:
                h = ...
            else:
                h = scale * h

        h.backward()

        return x.grad.item()


    # TODO: Finish forward, so the block returns x plus its branch of x.
    class ResidualBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.branch = nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, width),
            )

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

RESIDUAL1_SOLUTION = source_file(
    '''
    # residual1.py, a solution for comparison

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.tensor(1.0, requires_grad=True)
    (x + 0.1 * x).backward()
    slope = x.grad


    def residual(
        x: Tensor,
        f: Callable[[Tensor], Tensor],
    ) -> Tensor:
        return x + f(x)


    def input_gradient(
        scale: float,
        depth: int,
        use_residual: bool,
    ) -> float:
        x = torch.tensor(1.0, requires_grad=True)
        h = x

        for _ in range(depth):
            if use_residual:
                h = residual(h, lambda value: scale * value)
            else:
                h = scale * h

        h.backward()

        return x.grad.item()


    class ResidualBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.branch = nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, width),
            )

        def forward(self, x: Tensor) -> Tensor:
            return x + self.branch(x)
    ''',
)


def check_residual_block(found: type) -> None:
    for width in (3, 5):
        call = f"ResidualBlock({width})"
        block = expect_module(call, seeded_module(360 + width, found, width))
        x = small_tensor(362 + width, 4, width)

        with torch.no_grad():
            expected = x + block.branch(x)

        expect_tensor(
            call=f"{call} on x of shape (4, {width})",
            actual=block(x),
            expected=expected,
        )


def check_residual1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="residual",
        description="adds f(x) to x",
        inputs=(
            (small_tensor(355, 2, 3), Described(lambda x: 2 * x, "f doubling x")),
            (small_tensor(356, 4), Described(torch.relu, "f as torch.relu")),
        ),
        reference=lambda x, f: x + f(x),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="input_gradient",
        description="gives the gradient through a plain or a residual stack",
        inputs=(
            (0.1, 10, True),
            (0.1, 10, False),
            (0.5, 4, True),
            (2.0, 3, False),
        ),
        reference=lambda scale, depth, use_residual: (1 + scale) ** depth if use_residual else scale ** depth,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=relative(expected) if expected > 1e-3 else 1e-15,
        ),
    )
    checker.test_method(
        class_name="ResidualBlock",
        method="forward",
        description="adds the input back to the branch's output",
        run=check_residual_block,
    )


RESIDUAL1 = Exercise(
    name="residual1",
    chapter=RESIDUALS,
    title="Adding the input back",
    lesson=RESIDUAL1_LESSON,
    source=RESIDUAL1_SOURCE,
    solution=RESIDUAL1_SOLUTION,
    hints={
        "residual": "Call f on x and add x to the result.",
        "input_gradient": "Use residual with h and a lambda that multiplies its input by scale.",
        "ResidualBlock.forward": "Add x to self.branch(x).",
    },
    introduces=(),
    check=check_residual1,
)


# ---------------------------------------------------------------------------
# residual2
# ---------------------------------------------------------------------------

RESIDUAL2_LESSON = prose(
    """
    Here MLPs of width 64 learned MNIST from 10,000 images for 3 epochs, at a
    learning rate of 0.05. The plain ones started from Kaiming, their best
    start. The residual ones had a first layer, then blocks of two Linear
    layers like residual1's, from PyTorch's default start.

    | Hidden layers | Plain | Residual |
    |---|---|---|
    | 5 | 79.7% | 84.8% |
    | 17 | 72.1% | 86.1% |
    | 51 | 11.0%, never learned | 79.5% |

    In init3 the plain MLP of 8 layers got 85.9%, at a learning rate of 0.1
    against the 0.05 here.

    The plain MLP gets worse with depth, and its training loss rises with it,
    0.56 at 5 layers and 0.79 at 17. So this isn't overfitting. He and others
    found the same, deeper plain models training worse, which is what led them
    to residual connections.

    With PyTorch's default start, the average gradient at the first layer of
    51 layers was about `5e-24` plain and 0.002 residual.

    The cost is that each block adds to the main path. Its spread grew from
    0.14 to 0.74 over 25 default blocks. With Kaiming in both layers of every
    block, it grew past 100,000 and the loss turned `nan`. residual3 fixes
    that.

    `.abs().mean()` gives the average size of the numbers in a tensor.
    """,
)

RESIDUAL2_SOURCE = source_file(
    '''
    # residual2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # The block from residual1.
    class ResidualBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.branch = nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, width),
            )

        def forward(self, x: Tensor) -> Tensor:
            return x + self.branch(x)


    # TODO: Finish forward. It passes x through stem and a ReLU, then through
    #   every block in turn, then through head.
    class ResidualMLP(nn.Module):
        def __init__(
            self,
            inputs: int,
            width: int,
            outputs: int,
            blocks: int,
        ) -> None:
            super().__init__()
            self.stem = nn.Linear(inputs, width)
            self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(blocks)])
            self.head = nn.Linear(width, outputs)

        def forward(self, x: Tensor) -> Tensor:
            ...


    size = torch.tensor([-2.0, 1.0, 3.0]).abs().mean()    # tensor(2.)


    # TODO: Return the average size of the gradient of first's weight after one
    #   backward pass of the cross entropy of model on x against y.
    def first_layer_gradient(
        model: nn.Module,
        first: nn.Linear,
        x: Tensor,
        y: Tensor,
    ) -> float:
        ...


    # TODO: Return the spread of x after each block in blocks, as a list, without
    #   recording anything for autograd.
    def spreads_through_blocks(
        blocks: nn.ModuleList,
        x: Tensor,
    ) -> list[float]:
        ...
    ''',
)

RESIDUAL2_SOLUTION = source_file(
    '''
    # residual2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    class ResidualBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.branch = nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, width),
            )

        def forward(self, x: Tensor) -> Tensor:
            return x + self.branch(x)


    class ResidualMLP(nn.Module):
        def __init__(
            self,
            inputs: int,
            width: int,
            outputs: int,
            blocks: int,
        ) -> None:
            super().__init__()
            self.stem = nn.Linear(inputs, width)
            self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(blocks)])
            self.head = nn.Linear(width, outputs)

        def forward(self, x: Tensor) -> Tensor:
            x = torch.relu(self.stem(x))

            for block in self.blocks:
                x = block(x)

            return self.head(x)


    size = torch.tensor([-2.0, 1.0, 3.0]).abs().mean()


    def first_layer_gradient(
        model: nn.Module,
        first: nn.Linear,
        x: Tensor,
        y: Tensor,
    ) -> float:
        F.cross_entropy(model(x), y).backward()

        return first.weight.grad.abs().mean().item()


    def spreads_through_blocks(
        blocks: nn.ModuleList,
        x: Tensor,
    ) -> list[float]:
        spreads = []

        with torch.no_grad():
            for block in blocks:
                x = block(x)
                spreads.append(x.std().item())

        return spreads
    ''',
)


def residual_mlp_forward(
    model: nn.Module,
    x: Tensor,
) -> Tensor:
    with torch.no_grad():
        x = torch.relu(model.stem(x))

        for block in model.blocks:
            x = x + block.branch(x)

        return model.head(x)


def check_residual_mlp(found: type) -> None:
    for inputs, width, outputs, blocks in ((6, 8, 3, 2), (4, 5, 2, 3)):
        call = f"ResidualMLP({inputs}, {width}, {outputs}, {blocks})"
        model = expect_module(call, seeded_module(370 + blocks, found, inputs, width, outputs, blocks))
        x = small_tensor(372 + blocks, 3, inputs)
        expect_tensor(
            call=f"{call} on x of shape (3, {inputs})",
            actual=model(x),
            expected=residual_mlp_forward(model, x),
        )


def plain_model(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    layers: list[nn.Module] = [nn.Linear(20, 32), nn.ReLU()]

    for _ in range(12):
        layers.extend([nn.Linear(32, 32), nn.ReLU()])

    return nn.Sequential(*layers, nn.Linear(32, 4))


class ReferenceResidualMLP(nn.Module):
    def __init__(
        self,
        inputs: int,
        width: int,
        outputs: int,
        blocks: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Linear(inputs, width)
        self.blocks = nn.ModuleList([ReferenceResidualBlock(width) for _ in range(blocks)])
        self.head = nn.Linear(width, outputs)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.stem(x))

        for block in self.blocks:
            x = block(x)

        return self.head(x)


def residual_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)

    return ReferenceResidualMLP(20, 32, 4, 6)


def check_first_layer_gradient(found: object) -> None:
    x, y = labelled_points(375, 64, 20, 4)
    cases = (
        ("a plain MLP with 13 hidden layers", plain_model, lambda model: model[0]),
        ("a residual MLP with 6 blocks", residual_model, lambda model: model.stem),
    )

    for name, make, first in cases:
        call = f"first_layer_gradient on {name} and 64 labelled points"
        mine = make(376)
        actual = found(mine, first(mine), x, y)

        theirs = make(376)
        F.cross_entropy(theirs(x), y).backward()
        expected = first(theirs).weight.grad.abs().mean().item()

        # The plain model's gradient is tiny, so the tolerance is a share of it rather than a fixed amount.
        expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-4 * abs(expected),
        )


def reference_spreads_through_blocks(
    blocks: nn.ModuleList,
    x: Tensor,
) -> list[float]:
    spreads = []

    with torch.no_grad():
        for block in blocks:
            x = block(x)
            spreads.append(x.std().item())

    return spreads


def seeded_blocks(
    seed: int,
    width: int,
    count: int,
) -> Described:
    torch.manual_seed(seed)

    return Described(nn.ModuleList([ReferenceResidualBlock(width) for _ in range(count)]), f"{count} residual blocks of width {width}")


def check_residual2(checker: Checker) -> None:
    checker.test_method(
        class_name="ResidualMLP",
        method="forward",
        description="passes x through the stem, every block and the head",
        run=check_residual_mlp,
    )
    checker.test(
        function="first_layer_gradient",
        description="gives the average size of the first layer's gradient",
        run=check_first_layer_gradient,
    )
    check_on_inputs(
        checker=checker,
        function="spreads_through_blocks",
        description="gives the spread after every block",
        inputs=(
            (seeded_blocks(377, 16, 4), Described(bell_curve(378, 50, 16), "x of shape (50, 16) from the bell curve")),
            (seeded_blocks(379, 8, 2), Described(bell_curve(380, 30, 8), "x of shape (30, 8) from the bell curve")),
        ),
        reference=reference_spreads_through_blocks,
        expect=expect_numbers,
    )


RESIDUAL2 = Exercise(
    name="residual2",
    chapter=RESIDUALS,
    title="Why deep models need them",
    lesson=RESIDUAL2_LESSON,
    source=RESIDUAL2_SOURCE,
    solution=RESIDUAL2_SOLUTION,
    hints={
        "ResidualMLP.forward": "Set x to torch.relu(self.stem(x)), then loop over self.blocks setting x to block(x), "
        "and return self.head(x).",
        "first_layer_gradient": "Call .backward() on F.cross_entropy(model(x), y). Then take first.weight.grad, "
        ".abs().mean() it, and return it as a plain number.",
        "spreads_through_blocks": "Inside a with torch.no_grad() block, loop over blocks, set x to block(x), and "
        "append x.std().item() each time.",
    },
    introduces=(),
    check=check_residual2,
    time_limit=10,
)


# ---------------------------------------------------------------------------
# residual3
# ---------------------------------------------------------------------------

RESIDUAL3_LESSON = prose(
    """
    Each block adds its branch's output to the main path, so their variances
    add up and the main path keeps growing. Modern models use two fixes
    together.

    - pre norm: the branch normalizes its input first, `x + f(norm(x))`. It then sees spread 1 however large the main path has grown.
    - a zero start: the branch's last layer starts at zero, so every block starts as `x + 0` and does nothing.

    With a zero start, a model of any depth starts as its first and last
    layers alone. Each block grows in once training finds it useful. Goyal and
    others zeroed the last scale of each ResNet block in 2017, and GPT-2
    scales its blocks down for the same reason.

    The main path is never normalized inside the stack, so pre norm models
    put one more norm after the last block, before the head.

    From Kaiming starts with zeroed last layers, pre norm MLPs trained to
    84.4%, 78.9% and 81.2% at 5, 17 and 51 hidden layers. That was in the
    same setup as residual2, where the plain MLP of 51 layers never learned.

    Those scores are no better than residual2's residual MLPs, and worse at 17
    layers. Three epochs are too few to show an accuracy gain, and the differences
    between residual versions are within noise. What the fixes buy here is a
    safe start from Kaiming. The main path's spread stays steady with no
    `nan`, where Kaiming in both layers of residual2's blocks blew up.

    A zeroed layer still learns. Its gradient isn't zero, since the layer
    before it gives it nonzero inputs.
    """,
)

RESIDUAL3_SOURCE = source_file(
    '''
    # residual3.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    layer = nn.Linear(4, 4)

    with torch.no_grad():
        layer.weight.zero_()                          # every weight set to 0, in place


    # TODO: Two things to finish. In __init__, zero down's weight and bias, so
    #   the block starts doing nothing. forward returns x plus down of the ReLU
    #   of up of the norm of x.
    class PreNormBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.up = nn.Linear(width, width)
            self.down = nn.Linear(width, width)

            with torch.no_grad():
                ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return whether block hands x back unchanged, up to rounding.
    def does_nothing(
        block: nn.Module,
        x: Tensor,
    ) -> bool:
        ...


    # TODO: Finish forward. It passes x through stem and a ReLU, then every block
    #   in turn, then norm, then head.
    class PreNormMLP(nn.Module):
        def __init__(
            self,
            inputs: int,
            width: int,
            outputs: int,
            blocks: int,
        ) -> None:
            super().__init__()
            self.stem = nn.Linear(inputs, width)
            self.blocks = nn.ModuleList([PreNormBlock(width) for _ in range(blocks)])
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, outputs)

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

RESIDUAL3_SOLUTION = source_file(
    '''
    # residual3.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    layer = nn.Linear(4, 4)

    with torch.no_grad():
        layer.weight.zero_()


    class PreNormBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.up = nn.Linear(width, width)
            self.down = nn.Linear(width, width)

            with torch.no_grad():
                self.down.weight.zero_()
                self.down.bias.zero_()

        def forward(self, x: Tensor) -> Tensor:
            return x + self.down(torch.relu(self.up(self.norm(x))))


    def does_nothing(
        block: nn.Module,
        x: Tensor,
    ) -> bool:
        return torch.allclose(block(x), x)


    class PreNormMLP(nn.Module):
        def __init__(
            self,
            inputs: int,
            width: int,
            outputs: int,
            blocks: int,
        ) -> None:
            super().__init__()
            self.stem = nn.Linear(inputs, width)
            self.blocks = nn.ModuleList([PreNormBlock(width) for _ in range(blocks)])
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, outputs)

        def forward(self, x: Tensor) -> Tensor:
            x = torch.relu(self.stem(x))

            for block in self.blocks:
                x = block(x)

            return self.head(self.norm(x))
    ''',
)


def check_pre_norm_start(found: type) -> None:
    for width in (4, 6):
        call = f"PreNormBlock({width})"
        block = expect_module(call, found(width))

        for name in ("weight", "bias"):
            parameter = getattr(block.down, name).detach()

            if parameter.abs().max().item() > 0:
                raise Mismatch(f"The {name} of down in {call} should start at zero, but its largest number is {parameter.abs().max().item():.4g}.")


def check_pre_norm_forward(found: type) -> None:
    for width in (4, 6):
        call = f"PreNormBlock({width})"
        block = randomize(expect_module(call, found(width)), 385 + width)
        x = small_tensor(387 + width, 3, width)

        with torch.no_grad():
            expected = x + block.down(torch.relu(block.up(block.norm(x))))

        expect_tensor(
            call=f"{call}, with every layer given random numbers, on x of shape (3, {width})",
            actual=block(x),
            expected=expected,
        )


def pre_norm_mlp_forward(
    model: nn.Module,
    x: Tensor,
) -> Tensor:
    with torch.no_grad():
        x = torch.relu(model.stem(x))

        for block in model.blocks:
            x = x + block.down(torch.relu(block.up(block.norm(x))))

        return model.head(model.norm(x))


def check_pre_norm_mlp(found: type) -> None:
    for inputs, width, outputs, blocks in ((6, 8, 3, 2), (4, 5, 2, 3)):
        call = f"PreNormMLP({inputs}, {width}, {outputs}, {blocks})"
        model = randomize(expect_module(call, found(inputs, width, outputs, blocks)), 390 + blocks)
        x = small_tensor(392 + blocks, 3, inputs)
        expect_tensor(
            call=f"{call}, with every layer given random numbers, on x of shape (3, {inputs})",
            actual=model(x),
            expected=pre_norm_mlp_forward(model, x),
        )


def check_residual3(checker: Checker) -> None:
    checker.test_method(
        class_name="PreNormBlock",
        method="__init__",
        description="starts with down's weight and bias at zero",
        run=check_pre_norm_start,
    )
    checker.test_method(
        class_name="PreNormBlock",
        method="forward",
        description="adds x to the branch of its normalized self",
        run=check_pre_norm_forward,
    )
    torch.manual_seed(395)
    x = small_tensor(396, 3, 4)
    check_on_inputs(
        checker=checker,
        function="does_nothing",
        description="says whether a block hands x back unchanged",
        inputs=(
            (Described(ReferencePreNormBlock(4), "a pre norm block with down at zero"), x),
            (Described(ReferenceResidualBlock(4), "a residual block with random layers"), x),
            (Described(nn.Identity(), "nn.Identity()"), x),
        ),
        reference=lambda block, x: torch.allclose(block(x), x),
        expect=expect_value,
    )
    checker.test_method(
        class_name="PreNormMLP",
        method="forward",
        description="passes x through the stem, the blocks, the last norm and the head",
        run=check_pre_norm_mlp,
    )


RESIDUAL3 = Exercise(
    name="residual3",
    chapter=RESIDUALS,
    title="Starting as do nothing",
    lesson=RESIDUAL3_LESSON,
    source=RESIDUAL3_SOURCE,
    solution=RESIDUAL3_SOLUTION,
    hints={
        "PreNormBlock.__init__": "Call .zero_() on self.down.weight and on self.down.bias, in place of the ... line.",
        "PreNormBlock.forward": "Apply self.norm, then self.up, then torch.relu, then self.down, and add x to that.",
        "does_nothing": "Compare block(x) with x using torch.allclose.",
        "PreNormMLP.forward": "Like ResidualMLP's forward in residual2, with self.norm applied right before self.head.",
    },
    introduces=(),
    check=check_residual3,
)


# ---------------------------------------------------------------------------
# residual4
# ---------------------------------------------------------------------------

RESIDUAL4_LESSON = prose(
    """
    Where the norm sits next to the residual has changed over the years.

    | Paper | Where the norm goes |
    |---|---|
    | He and others 2015, ResNet | inside the branch, with a ReLU after the addition |
    | He and others 2016, ResNet v2 | at the start of the branch, and nothing after the addition |
    | Vaswani and others 2017 | after the addition, called post norm |
    | GPT-2, 2019 | at the start of the branch, called pre norm, and once more after the last block |

    The second ResNet paper kept the main path clean, with nothing on it but
    additions. That trained a 1001 layer model better than the first design.
    Post norm puts a norm on the main path at every block. Xiong and others
    showed in 2020 that post norm needs the learning rate raised slowly at the
    start, and pre norm trains without that.

    So most models since GPT-2 use pre norm. Use it by default, and expect
    post norm when you read the 2017 paper.

    When a block changes the width, `x` and `f(x)` no longer line up. ResNet
    then put a Linear layer on the main path of that one block, to match the
    width.

    A list of pairs can be unpacked in a loop, `for norm, f in pairs:`.
    """,
)

RESIDUAL4_SOURCE = source_file(
    '''
    # residual4.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    pairs = [("a", 1), ("b", 2)]
    names = [name for name, _ in pairs]              # ['a', 'b']


    # TODO: Return the post norm block of Vaswani and others, the norm of the
    #   sum, x plus f of x.
    def post_norm(
        x: Tensor,
        f: nn.Module,
        norm: nn.Module,
    ) -> Tensor:
        ...


    # TODO: Return the pre norm block of GPT-2, x plus f of the norm of x.
    def pre_norm(
        x: Tensor,
        f: nn.Module,
        norm: nn.Module,
    ) -> Tensor:
        ...


    # TODO: Finish the line in the loop, so x goes through a pre norm block for
    #   each pair of a norm and an f, then through final_norm.
    def pre_norm_stack(
        x: Tensor,
        blocks: list[tuple[nn.Module, nn.Module]],
        final_norm: nn.Module,
    ) -> Tensor:
        for norm, f in blocks:
            x = ...

        return final_norm(x)
    ''',
)

RESIDUAL4_SOLUTION = source_file(
    '''
    # residual4.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    pairs = [("a", 1), ("b", 2)]
    names = [name for name, _ in pairs]


    def post_norm(
        x: Tensor,
        f: nn.Module,
        norm: nn.Module,
    ) -> Tensor:
        return norm(x + f(x))


    def pre_norm(
        x: Tensor,
        f: nn.Module,
        norm: nn.Module,
    ) -> Tensor:
        return x + f(norm(x))


    def pre_norm_stack(
        x: Tensor,
        blocks: list[tuple[nn.Module, nn.Module]],
        final_norm: nn.Module,
    ) -> Tensor:
        for norm, f in blocks:
            x = pre_norm(x, f, norm)

        return final_norm(x)
    ''',
)


def norm_and_layer(
    seed: int,
    width: int,
) -> tuple[Described, Described]:
    torch.manual_seed(seed)
    norm = with_random_affine(nn.LayerNorm(width), seed)
    layer = nn.Linear(width, width)

    return Described(layer, f"a Linear layer of width {width}"), Described(norm, f"a LayerNorm of width {width}")


def stack_of(
    seed: int,
    width: int,
    count: int,
) -> tuple[Described, Described]:
    torch.manual_seed(seed)
    pairs = [(with_random_affine(nn.LayerNorm(width), seed + index), nn.Linear(width, width)) for index in range(count)]
    final = with_random_affine(nn.LayerNorm(width), seed + count)

    counted = "1 pair" if count == 1 else f"{count} pairs"

    return Described(pairs, f"{counted} of a LayerNorm and a Linear layer"), Described(final, "a final LayerNorm")


def check_residual4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="post_norm",
        description="normalizes x plus f of x",
        inputs=(
            (small_tensor(400, 3, 4), *norm_and_layer(401, 4)),
            (small_tensor(402, 2, 5), *norm_and_layer(403, 5)),
        ),
        reference=lambda x, f, norm: norm(x + f(x)),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="pre_norm",
        description="adds x to f of the norm of x",
        inputs=(
            (small_tensor(404, 3, 4), *norm_and_layer(405, 4)),
            (small_tensor(406, 2, 5), *norm_and_layer(407, 5)),
        ),
        reference=lambda x, f, norm: x + f(norm(x)),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="pre_norm_stack",
        description="runs pre norm blocks, then the final norm",
        inputs=(
            (small_tensor(408, 3, 4), *stack_of(409, 4, 3)),
            (small_tensor(410, 2, 5), *stack_of(411, 5, 1)),
        ),
        reference=reference_pre_norm_stack,
        expect=expect_tensor,
    )


def reference_pre_norm_stack(
    x: Tensor,
    blocks: list[tuple[nn.Module, nn.Module]],
    final_norm: nn.Module,
) -> Tensor:
    for norm, f in blocks:
        x = x + f(norm(x))

    return final_norm(x)


RESIDUAL4 = Exercise(
    name="residual4",
    chapter=RESIDUALS,
    title="Where the norm goes",
    lesson=RESIDUAL4_LESSON,
    source=RESIDUAL4_SOURCE,
    solution=RESIDUAL4_SOLUTION,
    hints={
        "post_norm": "Add x and f(x) first, then pass the sum through norm.",
        "pre_norm": "Pass x through norm, then through f, and add x to that.",
        "pre_norm_stack": "Each step is pre_norm with this pair's f and norm, in that order of arguments.",
    },
    introduces=(),
    check=check_residual4,
)


# ---------------------------------------------------------------------------
# quiz4
# ---------------------------------------------------------------------------

QUIZ4_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines moves
    from Section 3 so far. If a step is unclear, the exercise in the table
    shows it.

    | Function | Moves | Shown in |
    |---|---|---|
    | `kaiming_dropout_mlp` | an MLP from a list of sizes with dropout after each hidden ReLU, Kaiming weights and zero biases | dropout3, init3 |
    | `batchnorm_mlp` | hidden layers of a Linear without a bias, BatchNorm and ReLU | norm1, mlp1 |
    | `eval_accuracy` | eval mode, no autograd, the share of right answers | dropout2, mnist2 |
    | `normed_embeddings` | the rows of a table for ids, then layer normalization of each | embedding1, layernorm1 |
    """,
)

QUIZ4_SOURCE = source_file(
    '''
    # quiz4.py
    #
    # No examples and no new tools. Each function combines moves from Section 3.

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor


    # TODO: Return an nn.Sequential for the layer sizes in sizes. Each hidden
    #   layer is a Linear, a ReLU and dropout of p, and a last Linear gives the
    #   outputs. Every Linear starts with Kaiming's weights and zero biases.
    def kaiming_dropout_mlp(
        sizes: list[int],
        p: float,
    ) -> nn.Sequential:
        ...


    # TODO: Return an nn.Sequential for the layer sizes in sizes. Each hidden
    #   layer is a Linear with no bias, a BatchNorm1d and a ReLU, and a last
    #   Linear, with its bias, gives the outputs.
    def batchnorm_mlp(sizes: list[int]) -> nn.Sequential:
        ...


    # TODO: Return the share of rows of x the model labels as y, as a plain number,
    #   measured in eval mode without recording anything for autograd.
    def eval_accuracy(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        ...


    # TODO: Return the rows of table for ids, each normalized over its own
    #   features with eps, the variance dividing by n.
    def normed_embeddings(
        table: Tensor,
        ids: Tensor,
        eps: float,
    ) -> Tensor:
        ...
    ''',
)

QUIZ4_SOLUTION = source_file(
    '''
    # quiz4.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor


    def kaiming_dropout_mlp(
        sizes: list[int],
        p: float,
    ) -> nn.Sequential:
        layers = []

        for inputs, outputs in zip(sizes[:-2], sizes[1:-1]):
            layers.extend([nn.Linear(inputs, outputs), nn.ReLU(), nn.Dropout(p)])

        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        model = nn.Sequential(*layers)

        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.weight.normal_(0, math.sqrt(2 / module.in_features))
                    module.bias.zero_()

        return model


    def batchnorm_mlp(sizes: list[int]) -> nn.Sequential:
        layers = []

        for inputs, outputs in zip(sizes[:-2], sizes[1:-1]):
            layers.extend([nn.Linear(inputs, outputs, bias=False), nn.BatchNorm1d(outputs), nn.ReLU()])

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        return nn.Sequential(*layers)


    def eval_accuracy(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        model.eval()

        with torch.no_grad():
            return (model(x).argmax(dim=1) == y).float().mean().item()


    def normed_embeddings(
        table: Tensor,
        ids: Tensor,
        eps: float,
    ) -> Tensor:
        rows = table[ids]
        mean = rows.mean(dim=-1, keepdim=True)
        variance = rows.var(dim=-1, keepdim=True, unbiased=False)

        return (rows - mean) / torch.sqrt(variance + eps)
    ''',
)


def layer_parts(module: nn.Module) -> tuple[str, object]:
    if isinstance(module, nn.Linear):
        bias = "" if module.bias is not None else ", bias=False"

        return "Linear", f"({module.in_features}, {module.out_features}{bias})"

    if isinstance(module, nn.Dropout):
        return "Dropout", f"({module.p})"

    if isinstance(module, nn.BatchNorm1d):
        return "BatchNorm1d", f"({module.num_features})"

    if isinstance(module, nn.LayerNorm):
        return "LayerNorm", f"({tuple(module.normalized_shape)[0]})"

    return type(module).__name__, ""


def expect_layer_list(
    call: str,
    actual: object,
    expected: nn.Sequential,
) -> nn.Sequential:
    model = expect_module(call, actual)

    if not isinstance(model, nn.Sequential):
        raise Mismatch(f"{call} is a {type(model).__name__}. It should be an nn.Sequential.")

    got = [layer_parts(layer) for layer in model]
    wanted = [layer_parts(layer) for layer in expected]

    if got != wanted:
        raise Mismatch(
            f"{call} should hold {', '.join(name + detail for name, detail in wanted)}, "
            f"but it holds {', '.join(name + detail for name, detail in got)}.",
        )

    return model


def reference_dropout_layers(
    sizes: list[int],
    p: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []

    for inputs, outputs in zip(sizes[:-2], sizes[1:-1]):
        layers.extend([nn.Linear(inputs, outputs), nn.ReLU(), nn.Dropout(p)])

    layers.append(nn.Linear(sizes[-2], sizes[-1]))

    return nn.Sequential(*layers)


def expect_kaiming_dropout_mlp(
    call: str,
    actual: object,
    expected: nn.Sequential,
) -> None:
    model = expect_layer_list(call, actual, expected)

    for position, layer in enumerate(model):
        if not isinstance(layer, nn.Linear):
            continue

        spread = layer.weight.detach().std().item()
        target = math.sqrt(2 / layer.in_features)

        if abs(spread - target) > 0.1 * target:
            raise Mismatch(f"Layer {position} of {call} has weights with spread {spread:.4g}, but Kaiming's is {target:.4g}.")

        if layer.bias.detach().abs().max().item() > 0:
            raise Mismatch(f"Layer {position} of {call} should start with its bias all zeros.")


def reference_batchnorm_layers(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []

    for inputs, outputs in zip(sizes[:-2], sizes[1:-1]):
        layers.extend([nn.Linear(inputs, outputs, bias=False), nn.BatchNorm1d(outputs), nn.ReLU()])

    layers.append(nn.Linear(sizes[-2], sizes[-1]))

    return nn.Sequential(*layers)


def batchnorm_classifier(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    model = reference_batchnorm_layers([6, 16, 3])

    with torch.no_grad():
        for step in range(4):
            model(small_tensor(seed + step, 20, 6) + 1)

    return model


def check_eval_accuracy(found: object) -> None:
    for seed in (420, 430):
        x, y = labelled_points(seed, 40, 6, 3)
        call = "eval_accuracy on a model with BatchNorm, passed in still in training mode, and 40 labelled points"
        model = batchnorm_classifier(seed)
        before = model[1].running_mean.clone()
        actual = found(model, x, y)

        if not torch.equal(model[1].running_mean, before):
            raise Mismatch(f"{call} changed the model's running mean, so it measured in training mode. Call model.eval() first.")

        reference = batchnorm_classifier(seed)
        reference.eval()

        with torch.no_grad():
            expected = (reference(x).argmax(dim=1) == y).float().mean().item()

        expect_close(
            call=call,
            actual=actual,
            expected=expected,
        )


def check_quiz4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="kaiming_dropout_mlp",
        description="builds the MLP with dropout and Kaiming starts",
        inputs=(
            ([784, 256, 256, 10], 0.5),
            ([100, 300, 5], 0.2),
        ),
        reference=reference_dropout_layers,
        expect=expect_kaiming_dropout_mlp,
    )
    check_on_inputs(
        checker=checker,
        function="batchnorm_mlp",
        description="builds hidden layers of Linear without bias, BatchNorm and ReLU",
        inputs=(
            ([784, 64, 64, 10],),
            ([8, 16, 2],),
        ),
        reference=reference_batchnorm_layers,
        expect=expect_layer_list,
    )
    checker.test(
        function="eval_accuracy",
        description="measures the share of right answers in eval mode",
        run=check_eval_accuracy,
    )
    check_on_inputs(
        checker=checker,
        function="normed_embeddings",
        description="looks up rows and normalizes each over its features",
        inputs=(
            (small_tensor(415, 5, 4), torch.tensor([[0, 3], [4, 4]]), 1e-5),
            (small_tensor(416, 3, 6), torch.tensor([2, 0, 1, 2]), 0.1),
        ),
        reference=lambda table, ids, eps: F.layer_norm(table[ids], (table.shape[1],), eps=eps),
        expect=expect_tensor,
    )


QUIZ4 = Exercise(
    name="quiz4",
    chapter=QUIZZES,
    title="Blocks, one at a time",
    lesson=QUIZ4_LESSON,
    source=QUIZ4_SOURCE,
    solution=QUIZ4_SOLUTION,
    hints={
        "kaiming_dropout_mlp": "Build the list of layers as dropout3's dropout_mlp does, without the input dropout. "
        "Then give every nn.Linear in model.modules() Kaiming's start, as init3 does.",
        "batchnorm_mlp": "Each hidden layer adds nn.Linear(inputs, outputs, bias=False), nn.BatchNorm1d(outputs) and "
        "nn.ReLU().",
        "eval_accuracy": "Call model.eval(). Then, inside a with torch.no_grad() block, compare "
        "model(x).argmax(dim=1) with y and take the mean of the matches as floats.",
        "normed_embeddings": "Index table with ids, then normalize along dim=-1 as layer_normalize does in "
        "layernorm1.",
    },
    introduces=(),
    check=check_quiz4,
)


# ---------------------------------------------------------------------------
# quiz5
# ---------------------------------------------------------------------------

QUIZ5_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function or class
    combines moves from Section 3. If a step is unclear, the exercise in the
    table shows it.

    | Name | Moves | Shown in |
    |---|---|---|
    | `rms_pre_norm` | RMSNorm with a weight, inside a pre norm block | rmsnorm1, residual4 |
    | `DropoutBlock` | a pre norm block starting at zero, with dropout on the branch's output | residual3, dropout1 |
    | `block_stack` | blocks in an nn.Sequential, then a final LayerNorm | residual3, mlp1 |
    | `block_parameters` | the parameters of one DropoutBlock, worked out by hand | linear1, layernorm1 |

    Dropout on a residual branch goes on the branch's output, before the add,
    as in nanoGPT and GPT-2. The branch's last layer isn't the model's last
    layer, so dropout3's rule still holds.
    """,
)

QUIZ5_SOURCE = source_file(
    '''
    # quiz5.py
    #
    # No examples and no new tools. Each function and class combines moves from
    # Section 3.

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    from torch import Tensor


    # TODO: Return x plus f of the RMSNorm of x, that is x divided by the root
    #   mean square of its rows with eps inside the root, then scaled by weight.
    def rms_pre_norm(
        x: Tensor,
        f: Callable[[Tensor], Tensor],
        weight: Tensor,
        eps: float,
    ) -> Tensor:
        ...


    # TODO: Finish both methods. The block holds four modules. norm is a
    #   LayerNorm of width. up and down are Linear layers from width to width.
    #   dropout is an nn.Dropout of p. down's weight and bias both start at
    #   zero. forward returns x plus the dropout of down of the ReLU of up of
    #   the norm of x.
    class DropoutBlock(nn.Module):
        def __init__(
            self,
            width: int,
            p: float,
        ) -> None:
            super().__init__()
            ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return an nn.Sequential of `blocks` DropoutBlocks, each of width
    #   and p, followed by a LayerNorm of width.
    def block_stack(
        width: int,
        blocks: int,
        p: float,
    ) -> nn.Sequential:
        ...


    # TODO: Return how many parameters one DropoutBlock of width holds, worked
    #   out without building it.
    def block_parameters(width: int) -> int:
        ...
    ''',
)

QUIZ5_SOLUTION = source_file(
    '''
    # quiz5.py, a solution for comparison

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    from torch import Tensor


    def rms_pre_norm(
        x: Tensor,
        f: Callable[[Tensor], Tensor],
        weight: Tensor,
        eps: float,
    ) -> Tensor:
        normed = x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps) * weight

        return x + f(normed)


    class DropoutBlock(nn.Module):
        def __init__(
            self,
            width: int,
            p: float,
        ) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.up = nn.Linear(width, width)
            self.down = nn.Linear(width, width)
            self.dropout = nn.Dropout(p)

            with torch.no_grad():
                self.down.weight.zero_()
                self.down.bias.zero_()

        def forward(self, x: Tensor) -> Tensor:
            return x + self.dropout(self.down(torch.relu(self.up(self.norm(x)))))


    def block_stack(
        width: int,
        blocks: int,
        p: float,
    ) -> nn.Sequential:
        return nn.Sequential(*[DropoutBlock(width, p) for _ in range(blocks)], nn.LayerNorm(width))


    # The LayerNorm's weight and bias, then up's and down's weights and biases.
    def block_parameters(width: int) -> int:
        return 2 * width + 2 * (width * width + width)
    ''',
)


def check_dropout_block_start(found: type) -> None:
    for width, p in ((4, 0.1), (6, 0.5)):
        call = f"DropoutBlock({width}, {p})"
        block = expect_module(call, found(width, p))

        for name in ("norm", "up", "down", "dropout"):
            if not isinstance(getattr(block, name, None), nn.Module):
                raise Mismatch(f"{call} should have a module called {name}.")

        if not isinstance(block.dropout, nn.Dropout) or block.dropout.p != p:
            raise Mismatch(f"The dropout of {call} should be nn.Dropout({p}).")

        expect_parameter(block.up, f"up in {call}", "weight", (width, width))

        for name in ("weight", "bias"):
            parameter = getattr(block.down, name).detach()

            if parameter.abs().max().item() > 0:
                raise Mismatch(f"The {name} of down in {call} should start at zero, but its largest number is {parameter.abs().max().item():.4g}.")


def check_dropout_block_forward(found: type) -> None:
    for width in (4, 6):
        x = small_tensor(440 + width, 3, width)

        call = f"DropoutBlock({width}, 0.3), with every layer given random numbers, in eval mode"
        block = randomize(expect_module(call, found(width, 0.3)), 442 + width)
        block.eval()

        with torch.no_grad():
            expected = x + block.down(torch.relu(block.up(block.norm(x))))

        expect_tensor(
            call=f"{call}, on x of shape (3, {width})",
            actual=block(x),
            expected=expected,
        )

        # Dropout of 1 zeroes all it gets, so only dropout on the branch's output leaves x alone.
        call = f"DropoutBlock({width}, 1.0), with every layer given random numbers, in training mode"
        block = randomize(expect_module(call, found(width, 1.0)), 444 + width)
        block.train()
        expect_tensor(
            call=f"{call}, on x of shape (3, {width}), where dropout should zero the whole branch",
            actual=block(x),
            expected=x,
        )


def check_block_stack(found: object) -> None:
    for width, blocks, p in ((8, 3, 0.1), (4, 1, 0.2)):
        call = f"block_stack({width}, {blocks}, {p})"
        model = expect_module(call, found(width, blocks, p))

        if not isinstance(model, nn.Sequential):
            raise Mismatch(f"{call} is a {type(model).__name__}. It should be an nn.Sequential.")

        names = [type(layer).__name__ for layer in model]
        wanted = ["DropoutBlock"] * blocks + ["LayerNorm"]

        if names != wanted:
            raise Mismatch(f"{call} should hold {', '.join(wanted)}, but it holds {', '.join(names)}.")

        if tuple(model[-1].normalized_shape) != (width,):
            raise Mismatch(f"The LayerNorm at the end of {call} should have width {width}.")


def check_quiz5(checker: Checker) -> None:
    torch.manual_seed(435)
    check_on_inputs(
        checker=checker,
        function="rms_pre_norm",
        description="adds x to f of the RMSNorm of x",
        inputs=(
            (small_tensor(436, 3, 4), Described(nn.Linear(4, 4), "a Linear layer of width 4"), small_tensor(437, 4), 1e-6),
            (small_tensor(438, 2, 5), Described(torch.tanh, "torch.tanh"), small_tensor(439, 5), 0.1),
        ),
        reference=lambda x, f, weight, eps: x + f(x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps) * weight),
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="DropoutBlock",
        method="__init__",
        description="holds norm, up, down and dropout, with down at zero",
        run=check_dropout_block_start,
    )
    checker.test_method(
        class_name="DropoutBlock",
        method="forward",
        description="adds x to the dropout of the branch",
        run=check_dropout_block_forward,
    )
    checker.test(
        function="block_stack",
        description="stacks the blocks and ends with a LayerNorm",
        run=check_block_stack,
    )
    check_on_inputs(
        checker=checker,
        function="block_parameters",
        description="counts one block's parameters without building it",
        inputs=(
            (4,),
            (64,),
            (768,),
        ),
        reference=lambda width: sum(parameter.numel() for parameter in ReferencePreNormBlock(width).parameters()),
        expect=expect_value,
    )


QUIZ5 = Exercise(
    name="quiz5",
    chapter=QUIZZES,
    title="Deep blocks",
    lesson=QUIZ5_LESSON,
    source=QUIZ5_SOURCE,
    solution=QUIZ5_SOLUTION,
    hints={
        "rms_pre_norm": "Normalize x as MyRMSNorm's forward does in rmsnorm1, with weight, then add x to f of that.",
        "DropoutBlock.__init__": "Make norm, up and down as PreNormBlock does in residual3 and zero down the same "
        "way. Then add self.dropout = nn.Dropout(p).",
        "DropoutBlock.forward": "PreNormBlock's forward, with self.dropout around self.down(...).",
        "block_stack": "Make a list of DropoutBlock(width, p), one per block, then pass it with * and nn.LayerNorm(width) "
        "to nn.Sequential.",
        "block_parameters": "The LayerNorm holds a weight and a bias of width numbers each. up and down each hold "
        "width * width weights and width biases.",
    },
    introduces=(),
    check=check_quiz5,
)
