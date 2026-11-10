"""22_residuals, adding a block's input to its output, why deep stacks need it, and the block orders of the papers.

quiz4 and quiz5 close the blocks of Section 3 here."""

from __future__ import annotations

import math
from collections.abc import Callable

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
from .initialization import bell_curve, relative
from .linear_layers import expect_module, expect_parameter
from .mnist import mnist_sets
from .normalization import expect_spreads, kaiming_, pinned_threads, train_on_mnist, with_random_affine

RESIDUALS = Chapter(
    folder="22_residuals",
    title="Residual connections",
    readme=source_file(
        """
        # Residual connections

        A residual connection adds a block's input back to its output. That
        way the gradient reaches the first layer, however deep the model is.
        This chapter builds residual blocks, and shows a deep plain MLP that
        fails to train next to a residual MLP that trains well. It then trains
        stacks of 201 layers using pre norm or a zero start. Finally it
        compares where different papers put the norm in each block.

        ## Further reading

        - [He and others, deep residual learning, the ResNet paper](https://arxiv.org/abs/1512.03385)
        - [He and others, identity mappings in deep residual networks](https://arxiv.org/abs/1603.05027)
        - [Zhang and others, Fixup, residual learning without normalization](https://arxiv.org/abs/1901.09321)
        - [Xiong and others, on layer normalization in the transformer architecture](https://arxiv.org/abs/2002.04745)
        - [Liu and others, understanding the difficulty of training transformers](https://arxiv.org/abs/2004.08249)
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

    def forward(self, x: Tensor) -> Tensor:
        return x + self.down(torch.relu(self.up(self.norm(x))))


class ReferencePostNormBlock(nn.Module):
    """residual4's post norm, the norm of x plus the branch of x, where Vaswani and others put it."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, width)
        self.down = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.down(torch.relu(self.up(x))))


def reference_zero_start_(
    up: nn.Linear,
    down: nn.Linear,
    blocks: int,
) -> None:
    """Fixup's start for a branch of two layers in a stack of blocks with no norm."""
    with torch.no_grad():
        down.weight.zero_()
        down.bias.zero_()
        up.weight.mul_(1 / math.sqrt(blocks))


# ---------------------------------------------------------------------------
# residual1
# ---------------------------------------------------------------------------

RESIDUAL1_LESSON = prose(
    """
    A **residual connection** adds a block's input back to its output, so the
    block computes `f(x)` plus `x` instead of `f(x)` alone. Here `f` stands
    for the block's own layers, which only have to learn how to change `x`.
    If no change helps, `f` can learn to output 0, and then the block passes
    `x` through unchanged.

    The derivative of `x` plus `f(x)` is `1 + f'(x)`. Through a stack of
    blocks, the chain rule multiplies these derivatives together. Even when
    every `f'` is tiny, each factor `1 + f'` stays near 1. So the gradient
    reaches the first layer without shrinking away. In a plain stack without
    residuals, the factors are the `f'` alone, and 50 small factors multiply
    to nearly 0.

    ```python
    x = torch.tensor(1.0, requires_grad=True)
    (x + 0.1 * x).backward()
    x.grad    # tensor(1.1000), 1 from the residual and 0.1 from f
    ```

    He and others introduced residual connections for image models in 2015,
    and trained a model 152 layers deep. Nearly every deep model since then
    uses them, including the language models of later sections.

    The addition needs `f(x)` to have the same shape as `x`. In a module, the
    layers that compute `f` are called the block's branch. The block's
    `forward` adds the input back to the branch's output.
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


    # TODO: Return x plus f(x), the input plus the output of f on that input.
    #   Example: residual(torch.tensor([1.0, -2.0]), torch.relu) returns
    #   tensor([ 2., -2.]).
    def residual(
        x: Tensor,
        f: Callable[[Tensor], Tensor],
    ) -> Tensor:
        ...


    # TODO: Finish the line in the loop. Each of the depth layers multiplies h
    #   by scale, and when use_residual is True it also adds h back.
    #   input_gradient returns the derivative of the final h with respect to x.
    #   Example: input_gradient(0.1, 10, True) returns about 2.594, and
    #   input_gradient(0.1, 10, False) returns about 1e-10.
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


    # TODO: Finish forward, so the block returns x plus the output of its branch
    #   on x.
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
    Here MLPs of width 64 were trained on MNIST's 50,000 training images for 2
    epochs, at a learning rate of 0.05, with three seeds. The plain MLPs
    started from Kaiming initialization, the best start for them. The residual
    MLPs had a first layer followed by blocks of two Linear layers like
    residual1's, and started from PyTorch's default initialization.

    | Hidden layers | Plain | Residual |
    |---|---|---|
    | 17 | 93.2% to 93.6% | 93.9% to 94.5% |
    | 51 | 10% to 21%, never learned | 95.1% to 95.4% |

    The residual MLP gets better with more layers, and the plain one stops
    learning. The plain MLP's training loss also rises with depth, from 0.22
    to 0.24 at 17 layers to 1.8 to 2.3 at 51 layers. So the deep plain MLP is
    not overfitting, since it fits even the training data badly. He and
    others found the same on ImageNet. Their residual model with 34 layers
    beat the one with 18, with 25.0% error against 27.9%. Their plain model
    with 34 layers did worse than the plain one with 18.

    With PyTorch's default start and 51 layers, the first layer's gradient was
    tiny in the plain MLP. Its average size was about `6e-24`, against 0.002
    in the residual MLP.

    Residual connections have a cost. Each block adds its output to the main
    path, the running sum that flows from block to block. So the values on
    the main path keep growing. Their standard deviation grew from 0.14 to
    0.53 over 25 blocks with PyTorch's default start. With Kaiming
    initialization in both layers of every block, it grew to about 59,000,
    and training produced `nan` on every seed. residual3 shows two ways to
    fix this.

    `.abs().mean()` gives the average size of the numbers in a tensor,
    ignoring their signs. `x.std()` gives the standard deviation of all the
    numbers in `x`.
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


    # TODO: Finish forward. It passes x through self.stem and then a ReLU, then
    #   through every block in self.blocks in turn, then through self.head, and
    #   returns the result.
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


    # TODO: Compute the cross entropy of model(x) against y and call backward on
    #   it. Then return the average absolute value of the gradient of
    #   first.weight, as a float. first is the model's first Linear layer.
    def first_layer_gradient(
        model: nn.Module,
        first: nn.Linear,
        x: Tensor,
        y: Tensor,
    ) -> float:
        ...


    # TODO: Pass x through each block in blocks in turn. After each block,
    #   record the standard deviation of x as a float. Return the list of those
    #   values. Run it inside torch.no_grad(), so autograd records nothing.
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
        description="gives the standard deviation after every block",
        inputs=(
            (seeded_blocks(377, 16, 4), Described(bell_curve(378, 200, 16), "x of shape (200, 16) from the bell curve")),
            (seeded_blocks(379, 8, 2), Described(bell_curve(380, 200, 8), "x of shape (200, 8) from the bell curve")),
        ),
        reference=reference_spreads_through_blocks,
        expect=expect_spreads,
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
        "apply .abs().mean() to it, and return the result as a float with .item().",
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
    Each block adds its branch's output to the main path, so the values on
    the main path keep growing. residual2 measured their standard deviation
    going from 0.14 to 0.53 over 25 blocks. In deeper stacks they grow
    further, until training breaks. Two fixes come from different papers, and
    each is useful in a different situation.

    The first fix is **pre norm**. The branch normalizes its input before
    using it, so the block computes `x + f(norm(x))`. The branch then always
    sees inputs with a standard deviation of about 1, however large the main
    path has grown. The main path itself is never normalized inside the
    blocks. So one more norm goes after the last block, before the head.
    GPT-2 placed its norms this way in 2019.

    Here is residual2's setup at a learning rate of 0.05, with three seeds.
    Both kinds of block start from PyTorch's default initialization.

    | Hidden layers | 51 | 101 | 201 |
    |---|---|---|---|
    | residual2's blocks | 95.1% to 95.4% | `nan` on every seed | `nan` on every seed |
    | pre norm blocks | 95.6% to 96.1% | 95.7% to 96.0% | 95.8% to 95.9% |

    The second fix is a **zero start**, for a deep stack with no norms at all.
    The last layer of each branch starts with all its weights and biases at
    zero. So every block starts as `x + 0` and passes `x` through unchanged.
    At the start, the model behaves like its first and last layers alone.
    Each branch then grows away from zero as training finds it useful.

    Fixup, by Zhang and others in 2019, also scales down the weights of the
    branch's first layer, here by `1 / sqrt(blocks)`. With starts like this,
    Fixup and ReZero trained residual stacks 10,000 layers deep with no norms.
    Here are residual2's blocks at 201 hidden layers, with three seeds each.

    | Start | Rate 0.05 | Rate 0.2 |
    |---|---|---|
    | PyTorch's default | `nan` on every seed | `nan` on every seed |
    | zero start | 90.1% to 91.0% | 92.9% to 93.6% |

    A layer that starts at zero still learns. The gradient of a layer's
    weights depends on the layer's input. The layer before it gives it
    nonzero inputs, so that gradient is not zero.

    So use pre norm wherever a block can have a norm, as language models do.
    The zero start suits a stack with no norms. combine3 uses it again, to add
    a branch to a model that already works without changing the model's
    outputs.

    `.mul_(s)` multiplies every number of a tensor by `s`, in place.
    """,
)

RESIDUAL3_SOURCE = source_file(
    '''
    # residual3.py

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor


    # TODO: Finish forward, so the block returns x plus its branch applied to
    #   self.norm(x). The branch is self.up, then a ReLU, then self.down.
    class PreNormBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.up = nn.Linear(width, width)
            self.down = nn.Linear(width, width)

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Finish forward. It passes x through self.stem and a ReLU, then
    #   through every block in turn, then through self.norm and finally
    #   self.head.
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


    layer = nn.Linear(4, 4)

    with torch.no_grad():
        layer.weight.zero_()                          # every weight set to 0, in place
        layer.bias.mul_(0.5)                          # every bias halved, in place


    # TODO: Finish the three lines. They give a branch of up, a ReLU and down a
    #   zero start for a stack of `blocks` blocks, so the branch outputs 0 at
    #   first. Set down's weight and bias to zero, and multiply up's weight by
    #   1 / sqrt(blocks), all in place.
    def zero_start_(
        up: nn.Linear,
        down: nn.Linear,
        blocks: int,
    ) -> None:
        with torch.no_grad():
            ...
            ...
            ...


    # TODO: Return True if block(x) equals x up to rounding error, and False
    #   otherwise.
    def does_nothing(
        block: nn.Module,
        x: Tensor,
    ) -> bool:
        ...
    ''',
)

RESIDUAL3_SOLUTION = source_file(
    '''
    # residual3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor


    class PreNormBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.up = nn.Linear(width, width)
            self.down = nn.Linear(width, width)

        def forward(self, x: Tensor) -> Tensor:
            return x + self.down(torch.relu(self.up(self.norm(x))))


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


    layer = nn.Linear(4, 4)

    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.mul_(0.5)


    def zero_start_(
        up: nn.Linear,
        down: nn.Linear,
        blocks: int,
    ) -> None:
        with torch.no_grad():
            down.weight.zero_()
            down.bias.zero_()
            up.weight.mul_(1 / math.sqrt(blocks))


    def does_nothing(
        block: nn.Module,
        x: Tensor,
    ) -> bool:
        return torch.allclose(block(x), x)
    ''',
)


def check_zero_start(found: object) -> None:
    for width, blocks in ((4, 9), (6, 100)):
        torch.manual_seed(380 + width)
        up = nn.Linear(width, width)
        down = nn.Linear(width, width)
        weight = up.weight.detach().clone()
        bias = up.bias.detach().clone()
        call = f"zero_start_ on an up and a down of width {width}, for {blocks} blocks"
        found(up, down, blocks)

        for name in ("weight", "bias"):
            parameter = getattr(down, name).detach()

            if parameter.abs().max().item() > 0:
                raise Mismatch(f"After {call}, down's {name} should be all zeros, but its largest number is {parameter.abs().max().item():.4g}.")

        expect_tensor(
            call=f"up's weight after {call}, which should be its old weight times 1 / sqrt({blocks})",
            actual=up.weight.detach(),
            expected=weight / math.sqrt(blocks),
        )
        expect_tensor(
            call=f"up's bias after {call}, which should stay as it was",
            actual=up.bias.detach(),
            expected=bias,
        )


def zero_started_block(width: int) -> nn.Module:
    block = ReferenceResidualBlock(width)
    reference_zero_start_(block.branch[0], block.branch[2], 4)

    return block


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
        method="forward",
        description="adds x to the branch applied to norm(x)",
        run=check_pre_norm_forward,
    )
    checker.test_method(
        class_name="PreNormMLP",
        method="forward",
        description="passes x through the stem, the blocks, the last norm and the head",
        run=check_pre_norm_mlp,
    )
    checker.test(
        function="zero_start_",
        description="zeroes down and scales up's weight by 1 / sqrt(blocks)",
        run=check_zero_start,
    )
    torch.manual_seed(395)
    x = small_tensor(396, 3, 4)
    check_on_inputs(
        checker=checker,
        function="does_nothing",
        description="says whether a block returns x unchanged",
        inputs=(
            (Described(zero_started_block(4), "residual2's block after a zero start"), x),
            (Described(ReferencePreNormBlock(4), "a pre norm block from PyTorch's default start"), x),
            (Described(nn.Identity(), "nn.Identity()"), x),
        ),
        reference=lambda block, x: torch.allclose(block(x), x),
        expect=expect_value,
    )


RESIDUAL3 = Exercise(
    name="residual3",
    chapter=RESIDUALS,
    title="Pre norm and the zero start",
    lesson=RESIDUAL3_LESSON,
    source=RESIDUAL3_SOURCE,
    solution=RESIDUAL3_SOLUTION,
    hints={
        "PreNormBlock.forward": "Apply self.norm, then self.up, then torch.relu, then self.down, and add x to that.",
        "PreNormMLP.forward": "Write it like ResidualMLP's forward in residual2, but apply self.norm right before "
        "self.head.",
        "zero_start_": "Call .zero_() on down.weight and on down.bias. Then call .mul_ on up.weight with 1 divided by "
        "math.sqrt(blocks).",
        "does_nothing": "Compare block(x) with x using torch.allclose.",
    },
    introduces=(),
    check=check_residual3,
)


# ---------------------------------------------------------------------------
# residual4
# ---------------------------------------------------------------------------

RESIDUAL4_LESSON = prose(
    """
    Where the norm goes in a residual block has changed over the years.

    | Paper | Where the norm goes |
    |---|---|
    | He and others 2015, ResNet | inside the branch, with a ReLU after the addition |
    | He and others 2016, ResNet v2 | at the start of the branch, and nothing after the addition |
    | Vaswani and others 2017, the paper later sections' language models come from | after the addition, called post norm |
    | GPT-2, 2019 | at the start of the branch, called pre norm, and once more after the last block |

    The second ResNet paper kept the main path clean, with nothing on it but
    the additions. With that design, a 1001 layer model trained better than
    with the first design. Post norm does the opposite, and puts a norm on
    the main path in every block.

    Here is residual3's setup with post norm blocks against residual3's pre
    norm blocks, with three seeds each. A post norm block adds x and the
    branch's output, then normalizes the sum. In the rows with warmup, the
    learning rate rose in a straight line to its full value over the first
    200 of the 780 steps.

    | Hidden layers and rate | Post norm | Pre norm |
    |---|---|---|
    | 51, 0.05 | 95.9% to 96.4% | 95.6% to 96.1% |
    | 51, 0.5 | 11%, 89% and 95% | 94.7% to 95.9% |
    | 51, 0.5 with warmup | 92.6% to 95.7% | 95.4% to 95.6% |
    | 201, 0.05 | 89.8% to 95.3% | 95.8% to 95.9% |
    | 201, 0.2 | 11% on every seed | 95.1% to 95.4% |

    At 51 layers and a small learning rate, both reach about 96%. Liu and
    others found in 2020 that post norm often ends a little higher than pre
    norm, when it trains at all. At a large learning rate post norm failed on
    one seed, and warmup fixed that, as Xiong and others showed in 2020. At
    201 layers post norm became unreliable, and at a rate of 0.2 it never
    learned, even with warmup. Pre norm trained in every row.

    That is why most models since GPT-2 use pre norm. Use it by default, and
    expect to see post norm when you read the 2017 paper.

    When a block changes the width, `x` and `f(x)` have different shapes, so
    they cannot be added. For that one block, ResNet changed the width of `x`
    on the main path to match. It used either a learned layer, or zeros to
    fill the extra width. In an MLP the learned layer is a Linear.

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


    # TODO: Return the output of the post norm block of Vaswani and others. Add
    #   x and f(x), then apply norm to the sum.
    def post_norm(
        x: Tensor,
        f: nn.Module,
        norm: nn.Module,
    ) -> Tensor:
        ...


    # TODO: Return the output of the pre norm block of GPT-2, which is x plus f
    #   applied to norm(x).
    def pre_norm(
        x: Tensor,
        f: nn.Module,
        norm: nn.Module,
    ) -> Tensor:
        ...


    # TODO: Finish the line in the loop. blocks is a list of (norm, f) pairs. x
    #   goes through one pre norm block for each pair, in order, and then
    #   through final_norm.
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
        "pre_norm_stack": "Each pass through the loop calls pre_norm with x, this pair's f and this pair's norm, in "
        "that order.",
    },
    introduces=(),
    check=check_residual4,
)


# ---------------------------------------------------------------------------
# quiz4
# ---------------------------------------------------------------------------

QUIZ4_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines steps
    you have already done in Section 3. If a step is unclear, the exercise
    named in the table shows how to do it.

    | Function | Steps | Shown in |
    |---|---|---|
    | `kaiming_dropout_mlp` | an MLP from a list of sizes with dropout after each hidden ReLU, Kaiming weights and zero biases | dropout3, init3 |
    | `batch_normed_hidden` | a Linear without a bias, batch normalization with a scale and a shift, then ReLU | linear1, batchnorm1, relu1 |
    | `eval_confusion_matrix` | eval mode without autograd, each row's predicted class, the confusion matrix | dropout2, softmax3, planets3 |
    | `normed_embeddings` | the rows of a table for ids, then layer normalization of each | embedding1, layernorm1 |
    """,
)

QUIZ4_SOURCE = source_file(
    '''
    # quiz4.py
    #
    # No examples and no new tools. Each function combines steps from Section 3.

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor


    # TODO: Return an nn.Sequential for the layer sizes in sizes. Each hidden
    #   layer is a Linear, a ReLU and nn.Dropout(p), and a last Linear gives the
    #   outputs. Every Linear starts with Kaiming's weights and zero biases.
    #   Example: sizes [4, 8, 2] gives Linear(4, 8), ReLU, Dropout(p) and
    #   Linear(8, 2).
    def kaiming_dropout_mlp(
        sizes: list[int],
        p: float,
    ) -> nn.Sequential:
        ...


    # TODO: Pass x through a Linear layer with weights W and no bias, then batch
    #   normalization, then a ReLU, and return the result. W has one row per
    #   output, the way nn.Linear stores its weights. Batch normalization gives
    #   each feature mean 0 and standard deviation 1 over the batch, then
    #   multiplies by weight and adds bias. Compute the variance by dividing by
    #   n, and add eps to it before the square root.
    def batch_normed_hidden(
        x: Tensor,
        W: Tensor,
        weight: Tensor,
        bias: Tensor,
        eps: float,
    ) -> Tensor:
        ...


    # TODO: Return the confusion matrix of the model's predicted classes for x
    #   against the labels y, of shape (classes, classes). Entry [i, j] counts
    #   the examples whose true class is i and whose predicted class is j. Run
    #   the model in eval mode, inside torch.no_grad().
    def eval_confusion_matrix(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        classes: int,
    ) -> Tensor:
        ...


    # TODO: Look up the rows of table for ids, which can have any shape. Then
    #   normalize each row over its own features to mean 0 and standard
    #   deviation 1, and return the result. Compute the variance by dividing by
    #   n, and add eps to it before the square root.
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


    # A bias added before the normalization would be removed again when the mean
    # is subtracted. So the Linear has no bias, and the normalization's own bias
    # does the shifting.
    def batch_normed_hidden(
        x: Tensor,
        W: Tensor,
        weight: Tensor,
        bias: Tensor,
        eps: float,
    ) -> Tensor:
        hidden = x @ W.T
        mean = hidden.mean(dim=0)
        variance = hidden.var(dim=0, unbiased=False)
        normalized = (hidden - mean) / torch.sqrt(variance + eps)

        return torch.relu(normalized * weight + bias)


    def eval_confusion_matrix(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        classes: int,
    ) -> Tensor:
        model.eval()

        with torch.no_grad():
            predicted = model(x).argmax(dim=1)

        counts = torch.zeros(classes, classes, dtype=torch.long)

        for true_class, guess in zip(y.tolist(), predicted.tolist()):
            counts[true_class, guess] += 1

        return counts


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
            raise Mismatch(f"Layer {position} of {call} has weights with standard deviation {spread:.4g}, but Kaiming's is {target:.4g}.")

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


def check_eval_confusion_matrix(found: Callable[..., object]) -> None:
    for seed in (420, 430):
        x, y = labelled_points(seed, 40, 6, 3)
        call = (
            "eval_confusion_matrix on a model with BatchNorm, passed in still in training mode, 40 labelled points "
            "and 3 classes"
        )
        model = batchnorm_classifier(seed)
        before = model[1].running_mean.clone()
        actual = found(model, x, y, 3)

        if not torch.equal(model[1].running_mean, before):
            raise Mismatch(f"{call} changed the model's running mean, so it ran the model in training mode. Call model.eval() first.")

        reference = batchnorm_classifier(seed)
        reference.eval()

        with torch.no_grad():
            predicted = reference(x).argmax(dim=1)

        expected = torch.bincount(y * 3 + predicted, minlength=9).view(3, 3)
        swapped = (
            isinstance(actual, Tensor)
            and actual.shape == expected.shape
            and not torch.equal(expected, expected.T)
            and torch.equal(actual.double(), expected.T.double())
        )

        if swapped:
            raise Mismatch(
                f"{call} has the true class along the columns. Make the true class the row and the predicted class "
                "the column.",
            )

        expect_tensor(
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
    # Batches of 5 and 6 rows keep the n and n - 1 variances apart, and eps of 0.1 keeps it inside the root or
    # outside apart.
    check_on_inputs(
        checker=checker,
        function="batch_normed_hidden",
        description="runs the Linear, normalizes each feature over the batch, scales, shifts and applies ReLU",
        inputs=(
            (
                small_tensor(417, 5, 4),
                small_tensor(418, 3, 4),
                torch.tensor([1.5, 0.5, -1.0]),
                torch.tensor([0.2, -0.3, 0.1]),
                1e-5,
            ),
            (
                small_tensor(421, 6, 3),
                small_tensor(422, 5, 3),
                torch.tensor([2.0, -0.5, 1.0, 0.75, 1.25]),
                torch.tensor([-0.5, 0.25, 0.5, 0.0, 1.0]),
                0.1,
            ),
        ),
        reference=lambda x, W, weight, bias, eps: F.relu(
            F.batch_norm(
                x @ W.T,
                None,
                None,
                weight,
                bias,
                training=True,
                eps=eps,
            ),
        ),
        expect=expect_tensor,
    )
    checker.test(
        function="eval_confusion_matrix",
        description="counts each pair of true and predicted class in eval mode",
        run=check_eval_confusion_matrix,
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
        "batch_normed_hidden": "Run x through W as linear does in linear1, leaving out the bias. Normalize each "
        "column of the result over the batch as MyBatchNorm does in batchnorm1, with its scale and shift. Finish "
        "with ReLU, as in relu1.",
        "eval_confusion_matrix": "Get the predicted classes in eval mode without autograd, as evaluate does in "
        "dropout2. Then count each pair of a true class and its prediction, as confusion_matrix does in planets3.",
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
    combines steps you have already done in Section 3. If a step is unclear,
    the exercise named in the table shows how to do it.

    | Name | Steps | Shown in |
    |---|---|---|
    | `rms_pre_norm` | RMSNorm with a weight, inside a pre norm block | rmsnorm1, residual4 |
    | `DropoutBlock` | a pre norm block with dropout on the branch's output | residual3, dropout1 |
    | `block_stack` | blocks in an nn.Sequential, then a final LayerNorm | residual3, mlp1 |
    | `block_parameters` | the parameters of one DropoutBlock, worked out by hand | linear1, layernorm1 |

    Dropout on a residual branch goes on the branch's output, before the
    addition, as in Vaswani and others' 2017 paper and in nanoGPT. This does
    not break dropout3's rule of no dropout after the last layer, because the
    branch's last layer is not the model's last layer.
    """,
)

QUIZ5_SOURCE = source_file(
    '''
    # quiz5.py
    #
    # No examples and no new tools. Each function and class combines steps from
    # Section 3.

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    from torch import Tensor


    # TODO: Return x plus f applied to the RMSNorm of x. RMSNorm divides each
    #   row of x by the root mean square of that row, with eps added inside the
    #   square root, then multiplies the result by weight.
    def rms_pre_norm(
        x: Tensor,
        f: Callable[[Tensor], Tensor],
        weight: Tensor,
        eps: float,
    ) -> Tensor:
        ...


    # TODO: Finish both methods. The block holds four modules. norm is a
    #   LayerNorm of width. up and down are Linear layers from width to width.
    #   dropout is an nn.Dropout of p. forward returns x plus dropout applied to
    #   the branch's output. The branch is norm, then up, then a ReLU, then
    #   down.
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


    # TODO: Return an nn.Sequential of `blocks` DropoutBlocks, each made with
    #   width and p, followed by one LayerNorm of width.
    def block_stack(
        width: int,
        blocks: int,
        p: float,
    ) -> nn.Sequential:
        ...


    # TODO: Return the number of parameters in one DropoutBlock of width,
    #   computed with arithmetic instead of building the block.
    #   Example: block_parameters(4) returns 48.
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

        if not isinstance(block.norm, nn.LayerNorm) or tuple(block.norm.normalized_shape) != (width,):
            raise Mismatch(f"The norm of {call} should be nn.LayerNorm({width}).")

        for name in ("up", "down"):
            expect_parameter(getattr(block, name), f"{name} in {call}", "weight", (width, width))


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
        description="holds norm, up, down and dropout",
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
        "DropoutBlock.__init__": "Make norm, up and down as PreNormBlock does in residual3. Then add a fourth module "
        "called dropout, an nn.Dropout of p.",
        "DropoutBlock.forward": "Write PreNormBlock's forward, but wrap self.down(...) in self.dropout.",
        "block_stack": "Make a list of DropoutBlock(width, p), one per block, then pass it with * and nn.LayerNorm(width) "
        "to nn.Sequential.",
        "block_parameters": "The LayerNorm holds a weight and a bias of width numbers each. up and down each hold "
        "width * width weights and width biases.",
    },
    introduces=(),
    check=check_quiz5,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def plain_mnist_mlp(hidden: int) -> nn.Sequential:
    layers: list[nn.Module] = []

    for inputs in [784] + [64] * (hidden - 1):
        layers.extend([nn.Linear(inputs, 64), nn.ReLU()])

    return nn.Sequential(*layers, nn.Linear(64, 10))


class StackedMLP(nn.Module):
    def __init__(
        self,
        blocks: list[nn.Module],
        final_norm: bool,
    ) -> None:
        super().__init__()
        self.stem = nn.Linear(784, 64)
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(64) if final_norm else nn.Identity()
        self.head = nn.Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.stem(x))

        for block in self.blocks:
            x = block(x)

        return self.head(self.norm(x))


def depth_model(
    kind: str,
    hidden: int,
) -> nn.Module:
    """An MLP of width 64 for MNIST with hidden layers in all.

    - plain: Linear and ReLU layers from Kaiming, their best start.
    - plain_default: the same from PyTorch's default start.
    - residual: residual2's blocks from PyTorch's default start.
    - kaiming_both: residual2's blocks with Kaiming in both layers.
    - zero_start: residual2's blocks with residual3's zero start, down at zero and up scaled down.
    - pre_norm: residual3's pre norm blocks from PyTorch's default start, then one norm before the head.
    - post_norm: residual4's post norm blocks from PyTorch's default start."""
    if kind == "plain":
        return kaiming_(plain_mnist_mlp(hidden))

    if kind == "plain_default":
        return plain_mnist_mlp(hidden)

    blocks = (hidden - 1) // 2

    if kind == "residual":
        return StackedMLP([ReferenceResidualBlock(64) for _ in range(blocks)], final_norm=False)

    if kind == "kaiming_both":
        return StackedMLP([kaiming_(ReferenceResidualBlock(64)) for _ in range(blocks)], final_norm=False)

    if kind == "zero_start":
        model = StackedMLP([ReferenceResidualBlock(64) for _ in range(blocks)], final_norm=False)

        for block in model.blocks:
            reference_zero_start_(block.branch[0], block.branch[2], blocks)

        return model

    if kind == "post_norm":
        return StackedMLP([ReferencePostNormBlock(64) for _ in range(blocks)], final_norm=False)

    return StackedMLP([ReferencePreNormBlock(64) for _ in range(blocks)], final_norm=True)


def depth_run(
    kind: str,
    hidden: int,
    rate: float,
    seed: int,
    warmup: int = 0,
) -> dict[str, float]:
    torch.manual_seed(seed)

    return train_on_mnist(
        model=depth_model(kind, hidden),
        rate=rate,
        seed=seed,
        warmup=warmup,
    )


# Each run is a kind, hidden layers, a rate and warmup steps.
RESIDUAL2_RUNS: tuple[tuple[str, int, float, int], ...] = (
    ("plain", 17, 0.05, 0),
    ("residual", 17, 0.05, 0),
    ("plain", 51, 0.05, 0),
    ("residual", 51, 0.05, 0),
    ("kaiming_both", 51, 0.05, 0),
)

# residual2's blocks turn nan within a few steps at 101 and 201 layers, so those runs take no time.
RESIDUAL3_RUNS: tuple[tuple[str, int, float, int], ...] = (
    ("pre_norm", 51, 0.05, 0),
    ("residual", 101, 0.05, 0),
    ("pre_norm", 101, 0.05, 0),
    ("residual", 201, 0.05, 0),
    ("pre_norm", 201, 0.05, 0),
    ("residual", 201, 0.2, 0),
    ("zero_start", 201, 0.05, 0),
    ("zero_start", 201, 0.2, 0),
)

# residual4's pre norm runs at 0.05 are residual3's.
RESIDUAL4_RUNS: tuple[tuple[str, int, float, int], ...] = (
    ("post_norm", 51, 0.05, 0),
    ("post_norm", 51, 0.5, 0),
    ("post_norm", 51, 0.5, 200),
    ("post_norm", 201, 0.05, 0),
    ("post_norm", 201, 0.2, 0),
    ("post_norm", 201, 0.2, 200),
    ("pre_norm", 51, 0.5, 0),
    ("pre_norm", 51, 0.5, 200),
    ("pre_norm", 201, 0.2, 0),
)


def measure_depths(
    runs: tuple[tuple[str, int, float, int], ...] = RESIDUAL2_RUNS + RESIDUAL3_RUNS + RESIDUAL4_RUNS,
    seeds: range = range(3),
) -> dict[tuple[str, int, float, int, int], dict[str, float]]:
    """residual2's, residual3's and residual4's tables, keyed by kind, hidden layers, rate, warmup steps and seed. Each
    run is 2 epochs of plain SGD at batch 128 on MNIST's 50,000 training images, 780 steps. A run of 201 layers that
    trains takes about half a minute to a minute and a half on 4 threads."""
    with pinned_threads():
        return {
            (kind, hidden, rate, warmup, seed): depth_run(kind, hidden, rate, seed, warmup)
            for kind, hidden, rate, warmup in runs
            for seed in seeds
        }


def measure_first_gradients() -> dict[str, float]:
    """residual2's first layer gradients, the average size after one backward pass on 128 MNIST images, for 51 hidden
    layers from PyTorch's default start."""
    sets = mnist_sets()
    x, y = sets["train_x"][:128], sets["train_y"][:128]
    results = {}

    with pinned_threads():
        for kind in ("plain_default", "residual"):
            torch.manual_seed(0)
            model = depth_model(kind, 51)
            F.cross_entropy(model(x), y).backward()
            first = model[0] if kind == "plain_default" else model.stem
            results[kind] = first.weight.grad.abs().mean().item()

    return results


def measure_spreads() -> dict[str, list[float]]:
    """residual2's main path, its spread after each of 25 blocks on 128 MNIST images, from PyTorch's default start and
    with Kaiming in both layers of every block."""
    x = mnist_sets()["train_x"][:128]
    results = {}

    with pinned_threads(), torch.no_grad():
        for kind in ("residual", "kaiming_both"):
            torch.manual_seed(0)
            model = depth_model(kind, 51)
            path = torch.relu(model.stem(x))
            spreads = []

            for block in model.blocks:
                path = block(path)
                spreads.append(path.std().item())

            results[kind] = spreads

    return results
