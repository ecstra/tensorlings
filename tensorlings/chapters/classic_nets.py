"""33_classic_nets, LeNet-5, VGG's stacks of 3 by 3 convolutions, and ResNet's basic block, each from its paper."""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import load_cifar10
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from ..training import best_device
from .linear_layers import expect_module
from .mnist import mnist_sets

CLASSIC_NETS = Chapter(
    folder="33_classic_nets",
    title="Classic convolutional nets",
    readme=source_file(
        """
        # Classic convolutional nets

        Three papers set the design that most image models still follow.
        LeNet-5 put convolutions, pooling and Linear layers in a row. VGG built
        the whole network from 3 by 3 convolutions. ResNet added the residual
        connection and combined it with BatchNorm, which was published the same
        year. Together they let a network have many more layers and still train
        well. This chapter builds each network from its paper. It also shows
        why ResNet's block makes extra layers help instead of hurt.

        ## Further reading

        - [LeCun and others, gradient based learning applied to document recognition, LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), which the site serves over plain http only, or [the same paper at IEEE](https://doi.org/10.1109/5.726791)
        - [Simonyan and Zisserman, very deep convolutional networks, VGG](https://arxiv.org/abs/1409.1556)
        - [He and others, deep residual learning, ResNet](https://arxiv.org/abs/1512.03385)
        - [Dive into Deep Learning, modern convolutional networks](https://d2l.ai/chapter_convolutional-modern/index.html)
        """,
    ),
)


@functools.cache
def mnist_images() -> dict[str, Tensor]:
    """MNIST's shared splits, as images of shape (1, 28, 28)."""
    sets = mnist_sets()

    return {
        "train_x": sets["train_x"].reshape(-1, 1, 28, 28),
        "train_y": sets["train_y"],
        "validation_x": sets["validation_x"].reshape(-1, 1, 28, 28),
        "validation_y": sets["validation_y"],
    }


class ReferenceLeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.avg_pool2d(torch.tanh(self.conv1(x)), 2)
        x = F.avg_pool2d(torch.tanh(self.conv2(x)), 2)
        x = torch.tanh(self.fc1(x.flatten(1)))
        x = torch.tanh(self.fc2(x))

        return self.fc3(x)


def reference_train_lenet(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    steps: int,
    generator: torch.Generator,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for _ in range(steps):
        batch = torch.randint(0, len(x), (64,), generator=generator)
        loss = F.cross_entropy(model(x[batch]), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# lenet1
# ---------------------------------------------------------------------------

LENET1_LESSON = prose(
    """
    LeNet-5, from LeCun and others in 1998, read handwritten digits on bank
    cheques. Its layout is the one that image models kept using. Convolutions
    and pooling make the image smaller while giving it more channels. Then
    Linear layers at the end turn the remaining numbers into one score per
    class.

    | Layer | Output shape |
    |---|---|
    | 6 kernels of 5 by 5, padding 2, tanh | `(6, 28, 28)` |
    | average pool 2 | `(6, 14, 14)` |
    | 16 kernels of 5 by 5, tanh | `(16, 10, 10)` |
    | average pool 2 | `(16, 5, 5)` |
    | flatten, Linear to 120, tanh | `(120,)` |
    | Linear to 84, tanh | `(84,)` |
    | Linear to 10 | `(10,)` |

    The paper padded each 28 by 28 digit to 32 by 32. The padding of 2 in the
    first layer has the same effect. The paper's version differs from this one
    in a few ways. Its pooling multiplied by a learned scale. Its second
    convolution connected each kernel to only some of the 6 maps that the
    first convolution makes. Its output layer was a special kind of layer. This
    version keeps tanh and average pooling. It connects every kernel to all 6
    maps and ends with a plain Linear. It has 61,706 weights and biases in all.

    Trained with Adam at a learning rate of 0.001 on minibatches of 64 images,
    it got 92.4% of MNIST's validation images right after 300 steps. After
    1500 steps it got 97.2% right.
    """,
)

LENET1_SOURCE = source_file(
    '''
    # lenet1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Finish forward, following the table in the lesson. Pass x through
    #   conv1, then tanh, then an average pool of 2, and do the same with
    #   conv2. Flatten the result to shape (batch, 400) and pass it through
    #   fc1, fc2 and fc3, with tanh after fc1 and after fc2. x has shape
    #   (batch, 1, 28, 28), and forward returns scores of shape (batch, 10).
    class LeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return how many numbers LeNet learns, counting every weight and
    #   bias, when fc3 outputs classes scores instead of 10. Work it out from
    #   the layer sizes. lenet_parameters(10) counts the LeNet above.
    #   Example: lenet_parameters(10) returns 61706.
    def lenet_parameters(classes: int) -> int:
        ...


    # TODO: Finish the line that sets loss, so each step trains on 64 random
    #   images. batch holds 64 random indices into x. loss is the cross entropy
    #   between the model's scores for x[batch] and the labels y[batch].
    def train_lenet(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()

        for _ in range(steps):
            batch = torch.randint(0, len(x), (64,), generator=generator)
            loss = ...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
    ''',
)

LENET1_SOLUTION = source_file(
    '''
    # lenet1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    class LeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x: Tensor) -> Tensor:
            x = F.avg_pool2d(torch.tanh(self.conv1(x)), 2)
            x = F.avg_pool2d(torch.tanh(self.conv2(x)), 2)
            x = torch.tanh(self.fc1(x.flatten(1)))
            x = torch.tanh(self.fc2(x))

            return self.fc3(x)


    # Count the two convolutions, then the three Linear layers. fc3 has one output per class.
    def lenet_parameters(classes: int) -> int:
        convolutions = (6 * 1 * 5 * 5 + 6) + (16 * 6 * 5 * 5 + 16)
        linears = (400 * 120 + 120) + (120 * 84 + 84) + (84 * classes + classes)

        return convolutions + linears


    def train_lenet(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()

        for _ in range(steps):
            batch = torch.randint(0, len(x), (64,), generator=generator)
            loss = F.cross_entropy(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
    ''',
)


def check_lenet_forward(found: type) -> None:
    torch.manual_seed(790)
    model = expect_module("LeNet()", found())
    reference = ReferenceLeNet()
    reference.load_state_dict(model.state_dict())
    x = torch.rand(3, 1, 28, 28, generator=torch.Generator().manual_seed(791))

    with torch.no_grad():
        expected = reference(x)

    expect_tensor(
        call="LeNet() on 3 images of 28 by 28",
        actual=model(x),
        expected=expected,
    )


def check_train_lenet(found: object) -> None:
    images = mnist_images()
    x, y = images["train_x"][:2000], images["train_y"][:2000]
    torch.manual_seed(792)
    mine = ReferenceLeNet()
    torch.manual_seed(792)
    theirs = ReferenceLeNet()
    expect_close(
        call="train_lenet for 20 steps on 2000 MNIST images",
        actual=found(mine, x, y, 20, torch.Generator().manual_seed(793)),
        expected=reference_train_lenet(theirs, x, y, 20, torch.Generator().manual_seed(793)),
        tolerance=1e-4,
    )


def check_lenet_learns(found: object) -> None:
    images = mnist_images()
    torch.manual_seed(0)
    model = ReferenceLeNet()
    found(model, images["train_x"], images["train_y"], 300, torch.Generator().manual_seed(1))
    model.eval()

    with torch.no_grad():
        accuracy = (model(images["validation_x"]).argmax(dim=1) == images["validation_y"]).float().mean().item()

    if accuracy < 0.9:
        raise Mismatch(f"After 300 steps of train_lenet, LeNet gets {accuracy:.1%} of the validation images right, but it should get over 90%.")


def reference_lenet_parameters(classes: int) -> int:
    model = ReferenceLeNet()
    model.fc3 = nn.Linear(84, classes)

    return sum(parameter.numel() for parameter in model.parameters())


def check_lenet1(checker: Checker) -> None:
    checker.test_method(
        class_name="LeNet",
        method="forward",
        description="follows the layers in the lesson's table",
        run=check_lenet_forward,
    )
    # 47 is EMNIST's balanced set of letters and digits, so a count typed in from the lesson fails.
    check_on_inputs(
        checker=checker,
        function="lenet_parameters",
        description="counts LeNet's weights and biases for any number of classes",
        inputs=(
            (10,),
            (47,),
        ),
        reference=reference_lenet_parameters,
        expect=expect_value,
    )
    checker.test(
        function="train_lenet",
        description="trains on random minibatches of 64",
        run=check_train_lenet,
    )
    checker.test(
        function="train_lenet",
        description="gets LeNet past 90% on MNIST in 300 steps",
        run=check_lenet_learns,
    )


LENET1 = Exercise(
    name="lenet1",
    chapter=CLASSIC_NETS,
    title="LeNet-5",
    lesson=LENET1_LESSON,
    source=LENET1_SOURCE,
    solution=LENET1_SOLUTION,
    hints={
        "LeNet.forward": "Pass x through conv1, then torch.tanh, then F.avg_pool2d with a window of 2. Do the same "
        "with conv2. Then pass x.flatten(1) through fc1 and fc2, each followed by torch.tanh, and finish with fc3.",
        "lenet_parameters": "A convolution has out * in * k * k weights and out biases, as channels1 counted. A "
        "Linear layer has in * out weights and out biases. fc3 has classes outputs.",
        "train_lenet": "Use F.cross_entropy on the scores model(x[batch]) and the labels y[batch].",
    },
    introduces=(),
    check=check_lenet1,
    time_limit=150,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# vgg1
# ---------------------------------------------------------------------------

VGG1_LESSON = prose(
    """
    VGG-16, from Simonyan and Zisserman in 2014, uses only one kernel size, 3
    by 3. Two stacked 3 by 3 convolutions have a receptive field of 5 by 5, as
    dilation1 counted. That means each output of the second layer depends on a
    5 by 5 patch of the input. Three stacked 3 by 3 convolutions have a
    receptive field of 7 by 7.

    A stack of small kernels needs fewer weights than one big kernel with the
    same receptive field. With `c` channels in and out, three 3 by 3 layers
    learn `3 * 9 * c * c = 27 * c * c` weights. One 7 by 7 layer learns
    `49 * c * c`. The stack also has a ReLU after each layer, so it can compute
    more complicated functions than a single convolution can.

    Simonyan and Zisserman tested this on ImageNet, a dataset of over a million
    photos in 1,000 classes. In one of their nets, they replaced each pair of
    3 by 3 layers with a single 5 by 5 layer. That net's error, the fraction of
    photos whose top guess was wrong, came out 7% higher.

    VGG-16 is built from blocks. Each block is two or three 3 by 3 convolutions
    with padding 1, each followed by a ReLU. A max pool of 2 at the end of the
    block halves the height and the width. The five blocks have 64, 128, 256,
    512 and 512 channels. After them, three Linear layers turn the result into
    class scores. VGG-16 has 138 million weights, and most of them are in
    those Linear layers.

    `nn.MaxPool2d(2)` is max pooling with a window of 2, written as a module.
    """,
)

VGG1_SOURCE = source_file(
    '''
    # vgg1.py

    import torch
    import torch.nn as nn

    pool = nn.MaxPool2d(2)                           # halves the height and the width


    # TODO: Return the receptive field, as a width in pixels, of a stack of
    #   3 by 3 convolutions at stride 1. layers is the number of convolutions in
    #   the stack. Example: stack_view(2) returns 5.
    def stack_view(layers: int) -> int:
        ...


    # TODO: Return two weight counts, both without biases. The first is the
    #   number of weights in a stack of layers 3 by 3 convolutions, each from c
    #   channels to c channels. The second is the number of weights in a single
    #   convolution from c to c channels with the same receptive field as the
    #   stack. Example: weights_compared(64, 3) returns (110592, 200704).
    def weights_compared(
        c: int,
        layers: int,
    ) -> tuple[int, int]:
        ...


    # TODO: Return a VGG block as an nn.Sequential. It holds convs 3 by 3
    #   convolutions with padding 1, each followed by a ReLU, and then a max
    #   pool of 2. The first convolution goes from in_channels to out_channels,
    #   and the rest go from out_channels to out_channels.
    def vgg_block(
        in_channels: int,
        out_channels: int,
        convs: int,
    ) -> nn.Sequential:
        ...
    ''',
)

VGG1_SOLUTION = source_file(
    '''
    # vgg1.py, a solution for comparison

    import torch
    import torch.nn as nn

    pool = nn.MaxPool2d(2)


    def stack_view(layers: int) -> int:
        return 2 * layers + 1


    def weights_compared(
        c: int,
        layers: int,
    ) -> tuple[int, int]:
        view = stack_view(layers)

        return layers * 9 * c * c, view * view * c * c


    def vgg_block(
        in_channels: int,
        out_channels: int,
        convs: int,
    ) -> nn.Sequential:
        layers = []

        for index in range(convs):
            layers.extend([nn.Conv2d(in_channels if index == 0 else out_channels, out_channels, 3, padding=1), nn.ReLU()])

        layers.append(nn.MaxPool2d(2))

        return nn.Sequential(*layers)
    ''',
)


def padding_text(layer: nn.Conv2d) -> object:
    """padding="same" at stride 1 is the (k - 1) // 2 of padding1, so it reads as that number."""
    if layer.padding == "same":
        return (layer.kernel_size[0] - 1) // 2

    return layer.padding if isinstance(layer.padding, str) else layer.padding[0]


def reject_shared_layers(
    call: str,
    stack: nn.Module,
) -> None:
    """A list multiplied by a number repeats one layer object, so the stack's layers share one set of weights."""
    for layer in stack:
        if not any(True for _ in layer.parameters()):
            continue

        repeats = sum(1 for other in stack if other is layer)

        if repeats > 1:
            raise Mismatch(
                f"{call} holds one {type(layer).__name__} {repeats} times, so those layers share one set of weights. "
                "A list multiplied by a number repeats the same layer. Make a new one each time round the loop.",
            )


def block_summary(block: nn.Module) -> list[str]:
    parts = []

    for layer in block:
        if isinstance(layer, nn.Conv2d):
            parts.append(f"Conv2d({layer.in_channels}, {layer.out_channels}, {layer.kernel_size[0]}, padding={padding_text(layer)})")
            continue

        if isinstance(layer, nn.MaxPool2d):
            parts.append(f"MaxPool2d({layer.kernel_size})")
            continue

        parts.append(type(layer).__name__)

    return parts


def reference_vgg_block(
    in_channels: int,
    out_channels: int,
    convs: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []

    for index in range(convs):
        layers.extend([nn.Conv2d(in_channels if index == 0 else out_channels, out_channels, 3, padding=1), nn.ReLU()])

    layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)


def expect_block(
    call: str,
    actual: object,
    expected: nn.Sequential,
) -> None:
    block = expect_module(call, actual)

    if not isinstance(block, nn.Sequential):
        raise Mismatch(f"{call} is a {type(block).__name__}. It should be an nn.Sequential.")

    got, wanted = block_summary(block), block_summary(expected)

    if got != wanted:
        raise Mismatch(f"{call} should hold {', '.join(wanted)}, but it holds {', '.join(got)}.")

    reject_shared_layers(
        call=call,
        stack=block,
    )

    # The summary leaves out strides, so a stride of 2 in a convolution or a pool at stride 1 shows in the shape.
    in_channels, out_channels = expected[0].in_channels, expected[0].out_channels

    with torch.no_grad():
        shape = tuple(block(torch.zeros(1, in_channels, 16, 16)).shape)

    if shape != (1, out_channels, 8, 8):
        raise Mismatch(
            f"{call} turns zeros of shape (1, {in_channels}, 16, 16) into {shape}, but it should give (1, {out_channels}, 8, 8). "
            "Each convolution keeps the size at stride 1, and only the max pool of 2 halves it.",
        )


def check_vgg1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="stack_view",
        description="gives the receptive field of stacked 3 by 3 layers",
        inputs=(
            (1,),
            (2,),
            (3,),
        ),
        reference=lambda layers: 2 * layers + 1,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="weights_compared",
        description="counts the weights of the stack and of one big kernel",
        inputs=(
            (64, 3),
            (10, 2),
        ),
        reference=lambda c, layers: (layers * 9 * c * c, (2 * layers + 1) ** 2 * c * c),
        expect=expect_parts("the stack's weights", "the big kernel's weights"),
    )
    check_on_inputs(
        checker=checker,
        function="vgg_block",
        description="builds convolutions with ReLUs, then a max pool",
        inputs=(
            (3, 64, 2),
            (128, 256, 3),
        ),
        reference=reference_vgg_block,
        expect=expect_block,
    )


VGG1 = Exercise(
    name="vgg1",
    chapter=CLASSIC_NETS,
    title="VGG's 3 by 3 stacks",
    lesson=VGG1_LESSON,
    source=VGG1_SOURCE,
    solution=VGG1_SOLUTION,
    hints={
        "stack_view": "With no layers the receptive field is 1 pixel. Each 3 by 3 layer adds 2 to the receptive field "
        "of the layers before it.",
        "weights_compared": "Each 3 by 3 layer learns 9 * c * c weights. The single big kernel is stack_view(layers) "
        "pixels wide and tall.",
        "vgg_block": "Loop convs times. Each time, add a Conv2d with kernel size 3 and padding=1, then an nn.ReLU(). "
        "After the loop, add nn.MaxPool2d(2).",
    },
    introduces=(),
    check=check_vgg1,
)


# ---------------------------------------------------------------------------
# resnet1
# ---------------------------------------------------------------------------

RESNET1_LESSON = prose(
    """
    ResNet's basic block, from He and others in 2015, is the residual block
    from Section 3, built from convolutions. Its branch is a 3 by 3
    convolution, BatchNorm and ReLU, followed by another 3 by 3 convolution and
    BatchNorm. The block then adds its input to the branch's output and applies
    a ReLU to the sum. ResNet-18 and ResNet-34 are stacks of this block.

    ```text
    out = relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x))
    ```

    In a block that makes the image smaller, the first convolution has stride
    2 and more output channels. Then the branch's output has a different shape
    from `x`, so the two cannot be added. For its ImageNet nets the paper fixes
    this with a projection shortcut. The projection shortcut passes `x` through
    a 1 by 1 convolution with the same stride and the new number of channels,
    then through BatchNorm. In a block where the shape stays the same, the
    shortcut returns `x` unchanged, using `nn.Identity()`.

    A convolution right before BatchNorm needs no bias. BatchNorm subtracts the
    mean, which would cancel any bias, so ResNet uses `bias=False`.
    `nn.BatchNorm2d(c)` is BatchNorm for images. It computes one mean and one
    standard deviation for each of the `c` channels.
    """,
)

RESNET1_SOURCE = source_file(
    '''
    # resnet1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    shortcut = nn.Identity()
    same = shortcut(torch.ones(2))                   # tensor([1., 1.])


    # TODO: Finish two things. In __init__, set self.shortcut to nn.Identity()
    #   when stride is 1 and in_channels equals out_channels. Otherwise make it
    #   an nn.Sequential of a 1 by 1 nn.Conv2d from in_channels to out_channels,
    #   with the block's stride and no bias, followed by an nn.BatchNorm2d.
    #   Then finish forward so it computes and returns the formula in the
    #   lesson.
    class BasicBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
        ) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.shortcut = ...

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

RESNET1_SOLUTION = source_file(
    '''
    # resnet1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    shortcut = nn.Identity()
    same = shortcut(torch.ones(2))


    class BasicBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
        ) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            if stride == 1 and in_channels == out_channels:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x: Tensor) -> Tensor:
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

            return torch.relu(out + self.shortcut(x))
    ''',
)


class ReferenceBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return torch.relu(out + self.shortcut(x))


def shortcut_text(shortcut: object) -> str:
    if isinstance(shortcut, nn.Identity):
        return "nn.Identity()"

    if isinstance(shortcut, nn.Sequential):
        return "an nn.Sequential of " + ", ".join(type(layer).__name__ for layer in shortcut)

    return repr(shortcut)


def check_shortcut(found: type) -> None:
    for in_channels, out_channels, stride in ((8, 8, 1), (8, 16, 2), (4, 8, 1), (8, 8, 2)):
        call = f"BasicBlock({in_channels}, {out_channels}, {stride})"
        block = expect_module(call, found(in_channels, out_channels, stride))
        wanted = ReferenceBasicBlock(in_channels, out_channels, stride).shortcut

        if shortcut_text(block.shortcut) != shortcut_text(wanted):
            raise Mismatch(f"The shortcut of {call} is {shortcut_text(block.shortcut)}, but it should be {shortcut_text(wanted)}.")

        if isinstance(wanted, nn.Sequential):
            conv = block.shortcut[0]

            if conv.kernel_size != (1, 1) or conv.stride != (stride, stride) or conv.bias is not None or conv.out_channels != out_channels:
                raise Mismatch(f"The shortcut's convolution in {call} should be 1 by 1, with stride {stride}, {out_channels} channels out and no bias.")


def check_block_forward(found: type) -> None:
    for in_channels, out_channels, stride in ((4, 4, 1), (4, 8, 2)):
        call = f"BasicBlock({in_channels}, {out_channels}, {stride})"
        torch.manual_seed(795)
        block = expect_module(call, found(in_channels, out_channels, stride))
        reference = ReferenceBasicBlock(in_channels, out_channels, stride)

        try:
            reference.load_state_dict(block.state_dict())
        except RuntimeError:
            raise Mismatch(f"{call} holds different layers than the lesson's block. Check the shortcut first.") from None

        x = small_tensor(796, 2, in_channels, 6, 6)
        expect_tensor(
            call=f"{call} on x of shape (2, {in_channels}, 6, 6)",
            actual=block(x),
            expected=reference(x),
        )


def check_resnet1(checker: Checker) -> None:
    checker.test_method(
        class_name="BasicBlock",
        method="__init__",
        description="uses a projection shortcut only where the shape changes",
        run=check_shortcut,
    )
    checker.test_method(
        class_name="BasicBlock",
        method="forward",
        description="adds the shortcut to the branch, then applies a ReLU",
        run=check_block_forward,
    )


RESNET1 = Exercise(
    name="resnet1",
    chapter=CLASSIC_NETS,
    title="ResNet's basic block",
    lesson=RESNET1_LESSON,
    source=RESNET1_SOURCE,
    solution=RESNET1_SOLUTION,
    hints={
        "BasicBlock.__init__": "Use an if that tests stride == 1 and in_channels == out_channels. In the other case, "
        "build an nn.Sequential. It holds a 1 by 1 nn.Conv2d from in_channels to out_channels, with the block's "
        "stride and no bias, then an nn.BatchNorm2d.",
        "BasicBlock.forward": "Compute the branch in two lines. The first applies conv1, bn1 and a ReLU, and the "
        "second applies conv2 and bn2. Return the ReLU of the branch's output plus self.shortcut(x).",
    },
    introduces=(),
    check=check_resnet1,
)


# ---------------------------------------------------------------------------
# resnet2
# ---------------------------------------------------------------------------

RESNET2_LESSON = prose(
    """
    He and others found something surprising about depth, which is the number
    of layers in a net. CIFAR-10 is a dataset of 60,000 colour photos, each 32
    by 32, in 10 classes such as planes and ships. A plain net is one with no
    residual connections. On CIFAR-10, a plain net of 56 layers had a higher
    test error than a plain net of 20 layers, even with BatchNorm. The deeper
    net's training error was higher too, so the problem was not overfitting.
    They called this degradation.

    The table below comes from the paper's CIFAR-10 nets, trained for a short
    time. Each net starts with a 3 by 3 convolution. Then come three stages
    with 16, 32 and 64 channels, and each new stage halves the height and the
    width of the image. An average pool and a Linear layer finish the net. The
    ResNet fills each stage with basic blocks, as `residual_stack` does. The
    plain net of the same depth replaces each block with two of the layers
    that `plain_stack` makes.

    Each net trained for 1,000 steps of SGD with momentum, far fewer than the
    paper's 64,000. Each batch held 128 photos, and a random half of them were
    flipped left to right. Each kind of net was trained with five different
    random seeds. Every run was then scored on 5,000 validation photos.

    | Layers | Plain, training loss | Plain, validation accuracy | ResNet, training loss | ResNet, validation accuracy |
    |---|---|---|---|---|
    | 20 | 0.79 to 0.88 | 66.3% to 70.4% | 0.66 to 0.70 | 73.3% to 74.6% |
    | 32 | 1.19 to 1.31 | 50.6% to 54.3% | 0.63 to 0.66 | 74.5% to 75.6% |

    Going from 20 to 32 layers lowered the plain net's validation accuracy by
    13 to 18 percentage points on each seed. Its training loss went up too,
    and that is degradation. For the ResNet, the same twelve extra layers
    lowered the training loss on every seed. They also raised its validation
    accuracy by 0.3 to 2.1 points. In the paper, ResNets went from 8.75% test
    error at 20 layers to 6.61% at 110 layers, averaged over five runs.

    In principle the 32 layer plain net could do as well as the 20 layer one.
    Its twelve extra layers would only need to pass their input through
    unchanged. In practice, training does not find weights that do this. A
    residual block passes its input through unchanged as soon as its branch
    outputs 0. Training finds weights like that far more often, so extra depth
    stops hurting.

    In these ResNets, the learned scale of each block's second BatchNorm also
    started at 0, a trick that Goyal and others use. BatchNorm multiplies its
    normalized output by this scale, then adds a learned shift that starts at
    0. So each branch starts out outputting 0. A block whose shortcut is
    `nn.Identity()` then starts out computing `relu(x)`. The block's input came
    out of a ReLU and has no negative values, so `relu(x)` equals `x`. The
    block passes its input through unchanged from the first step.

    With PyTorch's default starting values instead, a new block with 16
    channels changes its input a lot. The ratio
    `(block(x) - x).norm() / x.norm()` comes out about 1.1. So the shortcut
    makes passing the input through possible, and the zero scale makes it the
    starting point.
    """,
)

RESNET2_SOURCE = source_file(
    '''
    # resnet2.py

    import torch
    import torch.nn as nn
    from torch import Tensor


    # The basic block from resnet1.
    class BasicBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
        ) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            if stride == 1 and in_channels == out_channels:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x: Tensor) -> Tensor:
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

            return torch.relu(out + self.shortcut(x))


    norm = nn.BatchNorm2d(4)
    scale = norm.weight                              # the learned scale, starting at 1


    # TODO: Return an nn.Sequential that holds a number of plain layers given
    #   by layers. Each plain layer is a 3 by 3 nn.Conv2d from channels to
    #   channels with padding 1 and no bias, followed by an nn.BatchNorm2d and
    #   an nn.ReLU, as three separate modules in the Sequential.
    #   Example: plain_stack(8, 2) holds 6 modules.
    def plain_stack(
        channels: int,
        layers: int,
    ) -> nn.Sequential:
        ...


    # TODO: Return an nn.Sequential that holds a number of BasicBlocks given by
    #   blocks. Each block goes from channels to channels at stride 1.
    def residual_stack(
        channels: int,
        blocks: int,
    ) -> nn.Sequential:
        ...


    # TODO: In every BasicBlock inside model, set the learned scale of bn2,
    #   which is bn2.weight, to 0. Change the weights in place and return
    #   nothing. model can hold BasicBlocks at any depth, inside other modules.
    #   With the scale at 0, each block's branch starts out outputting 0.
    def zero_last_norms_(model: nn.Module) -> None:
        ...
    ''',
)

RESNET2_SOLUTION = source_file(
    '''
    # resnet2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor


    class BasicBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
        ) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            if stride == 1 and in_channels == out_channels:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x: Tensor) -> Tensor:
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

            return torch.relu(out + self.shortcut(x))


    norm = nn.BatchNorm2d(4)
    scale = norm.weight


    def plain_stack(
        channels: int,
        layers: int,
    ) -> nn.Sequential:
        modules = []

        for _ in range(layers):
            modules.extend([nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU()])

        return nn.Sequential(*modules)


    def residual_stack(
        channels: int,
        blocks: int,
    ) -> nn.Sequential:
        return nn.Sequential(*[BasicBlock(channels, channels, 1) for _ in range(blocks)])


    def zero_last_norms_(model: nn.Module) -> None:
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, BasicBlock):
                    module.bn2.weight.zero_()
    ''',
)


def stack_summary(stack: nn.Module) -> list[str]:
    parts = []

    for layer in stack:
        if isinstance(layer, nn.Conv2d):
            parts.append(f"Conv2d({layer.in_channels}, {layer.out_channels}, {layer.kernel_size[0]}, padding={padding_text(layer)}, bias={layer.bias is not None})")
            continue

        if isinstance(layer, nn.BatchNorm2d):
            parts.append(f"BatchNorm2d({layer.num_features})")
            continue

        parts.append(type(layer).__name__)

    return parts


def check_plain_stack(found: object) -> None:
    for channels, layers in ((8, 3), (4, 1)):
        call = f"plain_stack({channels}, {layers})"
        stack = expect_module(call, found(channels, layers))

        if not isinstance(stack, nn.Sequential):
            raise Mismatch(f"{call} is a {type(stack).__name__}. It should be an nn.Sequential.")

        wanted = [f"Conv2d({channels}, {channels}, 3, padding=1, bias=False)", f"BatchNorm2d({channels})", "ReLU"] * layers
        got = stack_summary(stack)

        if got != wanted:
            raise Mismatch(f"{call} should hold {', '.join(wanted)}, but it holds {', '.join(got)}.")

        reject_shared_layers(
            call=call,
            stack=stack,
        )


def check_residual_stack(found: object) -> None:
    for channels, blocks in ((8, 3), (4, 2)):
        call = f"residual_stack({channels}, {blocks})"
        stack = expect_module(call, found(channels, blocks))
        names = [type(layer).__name__ for layer in stack]

        if not isinstance(stack, nn.Sequential) or names != ["BasicBlock"] * blocks:
            raise Mismatch(f"{call} should be an nn.Sequential of {blocks} BasicBlocks, but it holds {', '.join(names) or 'nothing'}.")

        reject_shared_layers(
            call=call,
            stack=stack,
        )

        for block in stack:
            if block.conv1.in_channels != channels or block.conv2.out_channels != channels or block.conv1.stride != (1, 1):
                raise Mismatch(f"Every block of {call} should go from {channels} channels to {channels} at stride 1.")


def check_zero_with_learner_blocks(found: object) -> None:
    # The function looks for the file's own BasicBlock class, so the model is built from that class.
    block_class = found.__globals__.get("BasicBlock")

    if not isinstance(block_class, type):
        raise Mismatch("resnet2.py should keep its BasicBlock class, which zero_last_norms_ looks for.")

    # Blocks sit at different depths, as they do inside a residual_stack, so only a walk over every module finds them all.
    torch.manual_seed(797)
    model = nn.Sequential(
        block_class(4, 4, 1),
        nn.ReLU(),
        nn.Sequential(block_class(4, 8, 2), block_class(8, 8, 1)),
        nn.Sequential(nn.Sequential(block_class(8, 8, 1))),
    )
    found(model)
    blocks = [module for module in model.modules() if isinstance(module, block_class)]

    for index, block in enumerate(blocks):
        if block.bn2.weight.detach().abs().max().item() != 0:
            raise Mismatch(
                f"After zero_last_norms_, the scale of bn2 in block {index} of {len(blocks)} is not all 0. "
                "Some blocks sit inside other Sequentials, and model.modules() reaches every one.",
            )

        if not torch.all(block.bn1.weight.detach() == 1):
            raise Mismatch("zero_last_norms_ should leave bn1 alone and set only bn2's scale to 0.")


def check_resnet2(checker: Checker) -> None:
    checker.test(
        function="plain_stack",
        description="stacks convolution, BatchNorm and ReLU",
        run=check_plain_stack,
    )
    checker.test(
        function="residual_stack",
        description="stacks basic blocks",
        run=check_residual_stack,
    )
    checker.test(
        function="zero_last_norms_",
        description="sets each block's second BatchNorm scale to 0",
        run=check_zero_with_learner_blocks,
    )


RESNET2 = Exercise(
    name="resnet2",
    chapter=CLASSIC_NETS,
    title="Why ResNet's blocks help",
    lesson=RESNET2_LESSON,
    source=RESNET2_SOURCE,
    solution=RESNET2_SOLUTION,
    hints={
        "plain_stack": "Loop layers times, adding the Conv2d, the BatchNorm2d and an nn.ReLU() to a list each time. "
        "Then pass the list to nn.Sequential, unpacked with *.",
        "residual_stack": "Build a list with one BasicBlock(channels, channels, 1) per block, and pass it to "
        "nn.Sequential unpacked with *.",
        "zero_last_norms_": "Inside a with torch.no_grad() block, loop over model.modules(). For each module that "
        "isinstance says is a BasicBlock, call .zero_() on module.bn2.weight.",
    },
    introduces=(),
    check=check_resnet2,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def pinned_threads(count: int = 4) -> Iterator[None]:
    """CPU results move with the thread count, so every measured table runs on the same number of threads."""
    before = torch.get_num_threads()
    torch.set_num_threads(count)

    try:
        yield
    finally:
        torch.set_num_threads(before)


@functools.cache
def cifar_splits() -> dict[str, Tensor]:
    """CIFAR-10 for the measured tables of Section 5: 45,000 training images and 5,000 for validation, cut by one fixed
    shuffle, and the 10,000 test images. Every channel is standardized with the training part's mean and spread."""
    images, labels = load_cifar10("train")
    test_images, test_labels = load_cifar10("test")
    x = images.float() / 255
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))
    train_x = x[order[:45000]]
    mean = train_x.mean(dim=(0, 2, 3), keepdim=True)
    spread = train_x.std(dim=(0, 2, 3), keepdim=True)

    return {
        "train_x": (train_x - mean) / spread,
        "train_y": labels[order[:45000]],
        "validation_x": (x[order[45000:]] - mean) / spread,
        "validation_y": labels[order[45000:]],
        "test_x": (test_images.float() / 255 - mean) / spread,
        "test_y": test_labels,
    }


def plain_layer(
    in_channels: int,
    out_channels: int,
    stride: int,
) -> list[nn.Module]:
    return [nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()]


def cifar_depth_net(
    blocks: int,
    residual: bool,
) -> nn.Sequential:
    """He and colleagues' CIFAR-10 nets of 6n + 2 layers: a 3 by 3 convolution to 16 channels, three stages of n blocks
    at widths 16, 32 and 64, each new stage halving the image, then an average pool and a Linear to 10 classes. The
    ResNet's blocks are basic blocks with bn2's scale at 0. Where a stage starts they use resnet1's projection shortcut,
    where the paper's CIFAR-10 nets padded the shortcut with zeros. The plain twin swaps each block for two plain
    layers."""
    layers: list[nn.Module] = plain_layer(3, 16, 1)
    in_channels = 16

    for stage, width in enumerate((16, 32, 64)):
        for index in range(blocks):
            stride = 2 if stage > 0 and index == 0 else 1

            if residual:
                block = ReferenceBasicBlock(in_channels, width, stride)
                nn.init.zeros_(block.bn2.weight)
                layers.append(block)
            else:
                layers.extend(plain_layer(in_channels, width, stride) + plain_layer(width, width, 1))

            in_channels = width

    layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)])

    return nn.Sequential(*layers)


class CifarNet(nn.Module):
    """The photo model of transfer2 and augment2's measured runs: five plain layers of widths 32, 32, 64, 64 and 128
    with two max pools and an average pool as its body, then a Linear head."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            *plain_layer(3, 32, 1),
            *plain_layer(32, 32, 1),
            nn.MaxPool2d(2),
            *plain_layer(32, 64, 1),
            *plain_layer(64, 64, 1),
            nn.MaxPool2d(2),
            *plain_layer(64, 128, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(128, classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))


def flipped_half(
    images: Tensor,
    generator: torch.Generator,
) -> Tensor:
    flip = torch.rand(len(images), 1, 1, 1, generator=generator) < 0.5

    return torch.where(flip.to(images.device), images.flip(-1), images)


def accuracy_on(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    device: torch.device,
) -> float:
    model.eval()
    right = 0

    with torch.no_grad():
        for start in range(0, len(x), 1000):
            scores = model(x[start : start + 1000].to(device))
            right += (scores.argmax(dim=1).cpu() == y[start : start + 1000]).sum().item()

    model.train()

    return right / len(x)


def loss_on(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0

    with torch.no_grad():
        for start in range(0, len(x), 1000):
            scores = model(x[start : start + 1000].to(device))
            total += F.cross_entropy(scores.cpu(), y[start : start + 1000], reduction="sum").item()

    model.train()

    return total / len(x)


def depth_run(
    blocks: int,
    residual: bool,
    seed: int,
    steps: int = 1000,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """One net of resnet2's table, trained for 1,000 steps with SGD, Nesterov momentum 0.9, weight decay 5e-4 and
    batches of 128 with flips, on one cycle up to 0.1. He and colleagues used plain momentum 0.9, weight decay 1e-4 and a
    rate of 0.1 cut tenfold twice, over 64,000 steps. Gives the loss on 10,000 training images and the accuracy on the
    5,000 validation images."""
    device = device or best_device()
    sets = cifar_splits()
    x = sets["train_x"].to(device)
    y = sets["train_y"].to(device)
    torch.manual_seed(seed)
    model = cifar_depth_net(blocks, residual).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=steps, pct_start=0.15)
    generator = torch.Generator().manual_seed(seed + 100)

    for _ in range(steps):
        batch = torch.randint(0, len(x), (128,), generator=generator).to(device)
        loss = F.cross_entropy(model(flipped_half(x[batch], generator)), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()

    return (
        loss_on(model, sets["train_x"][:10000], sets["train_y"][:10000], device),
        accuracy_on(model, sets["validation_x"], sets["validation_y"], device),
    )


def measure_depths(
    depths: tuple[int, ...] = (3, 5),
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[tuple[str, int], list[tuple[float, float]]]:
    """resnet2's table, keyed by the kind of net and its layers, 20 and 32 for 3 and 5 blocks a stage. The 20 runs
    take about 40 minutes on MPS on a busy machine."""
    results: dict[tuple[str, int], list[tuple[float, float]]] = {}

    with pinned_threads():
        for blocks in depths:
            for residual in (False, True):
                for seed in seeds:
                    results.setdefault(("resnet" if residual else "plain", 6 * blocks + 2), []).append(depth_run(blocks, residual, seed))

    return results


def measure_fresh_change(seeds: tuple[int, ...] = (0, 1, 2)) -> list[float]:
    """How much a fresh basic block of width 16, at PyTorch's default start, changes inputs that came out of a ReLU, as
    a share of their own size. With bn2's scale at 0 the share is exactly 0."""
    shares = []

    with pinned_threads():
        for seed in seeds:
            torch.manual_seed(seed)
            block = ReferenceBasicBlock(16, 16, 1)
            x = torch.relu(torch.randn(64, 16, 32, 32, generator=torch.Generator().manual_seed(seed + 50)))

            with torch.no_grad():
                shares.append(((block(x) - x).norm() / x.norm()).item())

    return shares
