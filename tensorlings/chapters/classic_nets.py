"""33_classic_nets, LeNet-5, VGG's stacks of 3 by 3 convolutions, and ResNet's basic block, each from its paper."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
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
from .linear_layers import expect_module
from .mnist import mnist_sets

CLASSIC_NETS = Chapter(
    folder="33_classic_nets",
    title="Classic convolutional nets",
    readme=source_file(
        """
        # Classic convolutional nets

        Three papers set the shape of image models. LeNet-5 put convolutions,
        pooling and Linear layers in a row. VGG built everything from 3 by 3
        convolutions. ResNet added the residual connection, with BatchNorm, so
        the stack could go deep. This chapter builds each from its paper and
        shows why ResNet's block lets depth help.

        ## Further reading

        - [LeCun and others, gradient based learning applied to document recognition, LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
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
    cheques. Its shape is the one image models kept. Convolutions and pooling
    shrink the image while adding channels, and Linear layers at the end turn
    what is left into scores.

    | Layer | Output shape |
    |---|---|
    | 6 kernels of 5 by 5, padding 2, tanh | `(6, 28, 28)` |
    | average pool 2 | `(6, 14, 14)` |
    | 16 kernels of 5 by 5, tanh | `(16, 10, 10)` |
    | average pool 2 | `(16, 5, 5)` |
    | flatten, Linear to 120, tanh | `(120,)` |
    | Linear to 84, tanh | `(84,)` |
    | Linear to 10 | `(10,)` |

    The paper padded each digit to 32 by 32. The padding of 2 in the first
    layer does the same. The paper's pooling had a learned scale. Its second
    layer connected each kernel to only some of the 6 maps, and its output
    layer was a special one. This version keeps tanh and average pooling,
    connects everything and ends with a plain Linear, 61,706 weights in all.

    Trained with Adam at 0.001 on minibatches of 64, it reached 92.4% on
    MNIST's validation images after 300 steps and 97.2% after 1500.
    """,
)

LENET1_SOURCE = source_file(
    '''
    # lenet1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Finish forward, following the table in the lesson. Each convolution
    #   is followed by tanh and an average pool of 2, then the maps are
    #   flattened into fc1, fc2 and fc3, with tanh after fc1 and fc2.
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


    # TODO: Return how many numbers LeNet learns, worked out from its layer
    #   sizes, weights and biases.
    def lenet_parameters() -> int:
        ...


    # TODO: Finish the line that sets loss, so each step trains on 64 random
    #   images of x with their labels y.
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


    # The two convolutions, then the three Linear layers.
    def lenet_parameters() -> int:
        convolutions = (6 * 1 * 5 * 5 + 6) + (16 * 6 * 5 * 5 + 16)
        linears = (400 * 120 + 120) + (120 * 84 + 84) + (84 * 10 + 10)

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


def check_lenet1(checker: Checker) -> None:
    checker.test_method(
        class_name="LeNet",
        method="forward",
        description="follows the paper's layers",
        run=check_lenet_forward,
    )
    checker.test(
        function="lenet_parameters",
        description="counts LeNet's weights and biases",
        run=lambda found: expect_value(
            call="lenet_parameters()",
            actual=found(),
            expected=sum(parameter.numel() for parameter in ReferenceLeNet().parameters()),
        ),
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
        "LeNet.forward": "x = F.avg_pool2d(torch.tanh(self.conv1(x)), 2), the same with conv2, then x.flatten(1) "
        "through fc1 and fc2 with tanh, and fc3 at the end.",
        "lenet_parameters": "A convolution has out * in * k * k weights and out biases, as channels1 counted. A "
        "Linear has in * out weights and out biases.",
        "train_lenet": "F.cross_entropy of model(x[batch]) against y[batch].",
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
    VGG-16, from Simonyan and Zisserman in 2014, used one kernel size only, 3
    by 3. Two of them stacked see a 5 by 5 patch, as dilation1 counted, and
    three see 7 by 7.

    A stack is cheaper than one big kernel. With `c` channels in and out,
    three 3 by 3 layers learn `27 * c * c` weights, where one 7 by 7 layer
    learns `49 * c * c`. The stack also puts a ReLU between each pair, so it
    can bend more.

    VGG-16 is built of blocks. Each block is two or three 3 by 3 convolutions
    with padding 1 and ReLU, then a max pool of 2 that halves the size. The
    channels go 64, 128, 256, 512 and 512, then three Linear layers give the
    scores. It has 138 million weights, most of them in those Linear layers.

    `nn.MaxPool2d(2)` is max pooling as a module.
    """,
)

VGG1_SOURCE = source_file(
    '''
    # vgg1.py

    import torch
    import torch.nn as nn

    pool = nn.MaxPool2d(2)                           # halves the height and the width


    # TODO: Return the receptive field of layers 3 by 3 convolutions stacked at
    #   stride 1.
    def stack_view(layers: int) -> int:
        ...


    # TODO: Return how many weights a stack of layers 3 by 3 convolutions from
    #   c channels to c channels learns, and how many one convolution with the
    #   same receptive field learns, both without biases.
    def weights_compared(
        c: int,
        layers: int,
    ) -> tuple[int, int]:
        ...


    # TODO: Return a VGG block as an nn.Sequential. It holds convs 3 by 3
    #   convolutions with padding 1, each followed by a ReLU, then a max pool of
    #   2. The first convolution goes from in_channels to out_channels, the rest
    #   from out_channels to out_channels.
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


def block_summary(block: nn.Module) -> list[str]:
    parts = []

    for layer in block:
        if isinstance(layer, nn.Conv2d):
            parts.append(f"Conv2d({layer.in_channels}, {layer.out_channels}, {layer.kernel_size[0]}, padding={layer.padding[0]})")
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


def check_vgg1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="stack_view",
        description="gives what stacked 3 by 3 layers see",
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
        description="counts the stack's weights and the big kernel's",
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
        "stack_view": "Each 3 by 3 layer adds 2 to what the one before saw, starting from 1.",
        "weights_compared": "Each 3 by 3 layer learns 9 * c * c. The single kernel is stack_view(layers) wide and "
        "tall.",
        "vgg_block": "Loop convs times, adding a Conv2d with kernel 3 and padding=1 and an nn.ReLU() each time, "
        "then add nn.MaxPool2d(2).",
    },
    introduces=(),
    check=check_vgg1,
)


# ---------------------------------------------------------------------------
# resnet1
# ---------------------------------------------------------------------------

RESNET1_LESSON = prose(
    """
    ResNet's basic block, from He and others in 2015, is the residual block of
    Section 3 made of convolutions. Its branch is a 3 by 3 convolution,
    BatchNorm and ReLU, then another 3 by 3 convolution and BatchNorm. The
    block adds its input back, then takes a ReLU. ResNet-18 and ResNet-34 are
    stacks of it.

    ```text
    out = relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x))
    ```

    Where the image shrinks, the first convolution has stride 2 and more
    channels out. Then `x` no longer fits what the branch gives. The paper's
    fix, the projection shortcut, sends `x` through a 1 by 1 convolution with
    the same stride and the new channels, then BatchNorm. Where nothing
    changes, the shortcut hands `x` on as it is, with `nn.Identity()`.

    A convolution right before BatchNorm needs no bias, since BatchNorm takes
    the mean away, so ResNet uses `bias=False`. `nn.BatchNorm2d(c)` is
    BatchNorm for images, with one mean and spread per channel.
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


    # TODO: Two things to finish. In __init__, set self.shortcut to nn.Identity()
    #   when stride is 1 and the channels stay the same. Otherwise make it an
    #   nn.Sequential of a 1 by 1 nn.Conv2d with that stride and no bias, then
    #   an nn.BatchNorm2d. forward follows the formula in the lesson.
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
        description="adds the shortcut to the branch, then takes a ReLU",
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
        "BasicBlock.__init__": "An if on stride == 1 and in_channels == out_channels. The other case is "
        "nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), "
        "nn.BatchNorm2d(out_channels)).",
        "BasicBlock.forward": "Work out the branch in two lines, conv1, bn1 and a ReLU, then conv2 and bn2. Return the "
        "ReLU of that plus self.shortcut(x).",
    },
    introduces=(),
    check=check_resnet1,
)


# ---------------------------------------------------------------------------
# resnet2
# ---------------------------------------------------------------------------

RESNET2_LESSON = prose(
    """
    He and others found something odd about depth. On CIFAR-10, a plain net of
    56 layers had a higher test error than one of 20, BatchNorm and all. The
    training error was worse too, so it was not overfitting. They called it
    degradation.

    Here small nets of width 8 learned MNIST shrunk to 14 by 14, from 5000
    images, in 800 steps of SGD with momentum.

    | Net | Training loss | Validation accuracy |
    |---|---|---|
    | plain, 4 layers | 0.185 | 94.0% |
    | plain, 24 layers | 0.806 | 75.1% |
    | ResNet, 24 layers in 12 basic blocks | 0.203 | 93.8% |

    The 24 layer plain net could copy the 4 layer one, its extra layers
    handing their input on. Training doesn't find that. A residual block hands
    its input on as soon as its own branch gives 0, which is far easier to
    reach, so depth stops hurting.

    These ResNet blocks also had the scale of their second BatchNorm set to 0,
    which Goyal and others use. Then each block starts out as `relu(x)`,
    handing its input on from the first step. With the default start a fresh
    block of width 8 changes its input by about 1.1 times the input's own
    size. The skip makes handing on possible, and the zero scale makes it the
    starting point.

    On ImageNet, ResNet-18, 34, 50, 101 and 152 got better with every step of
    depth.
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


    # TODO: Return an nn.Sequential of layers plain layers, each a 3 by 3
    #   nn.Conv2d from channels to channels with padding 1 and no bias, an
    #   nn.BatchNorm2d and a ReLU.
    def plain_stack(
        channels: int,
        layers: int,
    ) -> nn.Sequential:
        ...


    # TODO: Return an nn.Sequential of blocks BasicBlocks, each from channels to
    #   channels at stride 1.
    def residual_stack(
        channels: int,
        blocks: int,
    ) -> nn.Sequential:
        ...


    # TODO: Set the scale of bn2 in every BasicBlock inside model to 0, in place,
    #   so each block starts as relu(x).
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
            parts.append(f"Conv2d({layer.in_channels}, {layer.out_channels}, {layer.kernel_size[0]}, padding={layer.padding[0]}, bias={layer.bias is not None})")
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


def check_residual_stack(found: object) -> None:
    for channels, blocks in ((8, 3), (4, 2)):
        call = f"residual_stack({channels}, {blocks})"
        stack = expect_module(call, found(channels, blocks))
        names = [type(layer).__name__ for layer in stack]

        if not isinstance(stack, nn.Sequential) or names != ["BasicBlock"] * blocks:
            raise Mismatch(f"{call} should be an nn.Sequential of {blocks} BasicBlocks, but it holds {', '.join(names) or 'nothing'}.")

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
    title="Why ResNet's blocks",
    lesson=RESNET2_LESSON,
    source=RESNET2_SOURCE,
    solution=RESNET2_SOLUTION,
    hints={
        "plain_stack": "Loop layers times, adding the Conv2d, the BatchNorm2d and an nn.ReLU() each time, then pass "
        "the list to nn.Sequential with *.",
        "residual_stack": "nn.Sequential of a list of BasicBlock(channels, channels, 1), one per block, spread with *.",
        "zero_last_norms_": "Inside a with torch.no_grad() block, loop over model.modules(). For each module that "
        "isinstance says is a BasicBlock, call .zero_() on module.bn2.weight.",
    },
    introduces=(),
    check=check_resnet2,
)
