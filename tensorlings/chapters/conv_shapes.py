"""31_conv_shapes, channels, padding, stride and dilation, and picking them for the shape a model needs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    NOT_STARTED,
    Checker,
    Mismatch,
    check_on_inputs,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

CONV_SHAPES = Chapter(
    folder="31_conv_shapes",
    title="Channels, padding, stride and dilation",
    readme=source_file(
        """
        # Channels, padding, stride and dilation

        A convolution layer has more settings than its kernel size. This
        chapter covers the channels it reads and writes, and the padding that
        adds a border of zeros around its input. It also covers the stride
        that makes the output smaller, and the dilation that widens what each
        output sees. It ends with choosing these settings to get the shapes a
        model needs, and with the 1 by 1 convolution.

        ## Further reading

        - [Dumoulin and Visin, a guide to convolution arithmetic](https://arxiv.org/abs/1603.07285)
        - [PyTorch, nn.Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html)
        - [CS231n, convolutional networks](https://cs231n.github.io/convolutional-networks/)
        """,
    ),
)


def conv_one(
    image: Tensor,
    kernel: Tensor,
) -> Tensor:
    return F.conv2d(image[None, None], kernel[None, None])[0, 0]


def reference_output_size(
    n: int,
    k: int,
    p: int,
    s: int,
    d: int = 1,
) -> int:
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


# ---------------------------------------------------------------------------
# channels1
# ---------------------------------------------------------------------------

CHANNELS1_LESSON = prose(
    """
    In one hot form, DNA has 4 numbers per base, one each for A, C, G and T.
    The DNA project at the end of this section reads DNA in this form. A
    colour image has 3 numbers per pixel, one each for red, green and blue.
    Each of these numbers per position is a **channel**. So one hot DNA is a
    tensor of shape `(channels, length)`, and an image is a tensor of shape
    `(channels, height, width)`.

    ```python
    dna = torch.tensor([[1.0, 0.0, 0.0],     # A
                        [0.0, 0.0, 1.0],     # C
                        [0.0, 1.0, 0.0],     # G
                        [0.0, 0.0, 0.0]])    # T, so the 3 bases read A, G, C
    ```

    The layers inside a model produce many more channels. Each of those
    channels is a map of where one pattern appeared in the input.

    A kernel reads every input channel at once. For 3 input channels it has
    shape `(3, kh, kw)`, which is one 2D kernel per input channel. Its output
    at a position is the sum of the three 2D convolutions at that position. A
    layer with many output channels has one such kernel for each output
    channel. So the layer's weight has shape
    `(out channels, in channels, kh, kw)`, and the layer also has one bias
    per output channel.

    `nn.Conv2d(3, 16, 3)` is such a layer, with 16 kernels of shape
    `(3, 3, 3)`. It learns `16 * 3 * 3 * 3 + 16` numbers, which is 448. A
    Linear layer from a 32 by 32 colour image to 16 channels of the same size
    would learn over 50 million.

    Python's `sum(...)` adds up the tensors from a loop, the same way it adds
    up numbers.
    """,
)

CHANNELS1_SOURCE = source_file(
    '''
    # channels1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # One 2D convolution, of one channel with one kernel.
    def conv_one(
        image: Tensor,
        kernel: Tensor,
    ) -> Tensor:
        return F.conv2d(image[None, None], kernel[None, None])[0, 0]


    layer = nn.Conv2d(3, 16, 3)
    weight_shape = layer.weight.shape                # torch.Size([16, 3, 3, 3])


    # TODO: Finish the line in the loop, so this returns the convolution of
    #   image with kernels. image has shape (in channels, h, w), and kernels
    #   has shape (out channels, in channels, kh, kw). Output channel out is
    #   the sum over every input channel c of conv_one of that channel with
    #   its own 2D kernel, kernels[out, c].
    def multi_channel_conv(
        image: Tensor,
        kernels: Tensor,
    ) -> Tensor:
        outputs = []

        for out in range(kernels.shape[0]):
            outputs.append(...)

        return torch.stack(outputs)


    # TODO: Return how many numbers nn.Conv2d(in_channels, out_channels, k)
    #   learns, counting its weights and its biases.
    #   Example: conv_parameters(3, 16, 3) returns 448.
    def conv_parameters(
        in_channels: int,
        out_channels: int,
        k: int,
    ) -> int:
        ...


    # TODO: Return how many numbers a Linear layer learns, counting its weights
    #   and its biases, when it maps a flattened image of shape (channels, h, w)
    #   to out_channels channels of the same h and w.
    #   Example: linear_parameters(1, 4, 5, 2) returns 840.
    def linear_parameters(
        channels: int,
        h: int,
        w: int,
        out_channels: int,
    ) -> int:
        ...
    ''',
)

CHANNELS1_SOLUTION = source_file(
    '''
    # channels1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    def conv_one(
        image: Tensor,
        kernel: Tensor,
    ) -> Tensor:
        return F.conv2d(image[None, None], kernel[None, None])[0, 0]


    layer = nn.Conv2d(3, 16, 3)
    weight_shape = layer.weight.shape


    def multi_channel_conv(
        image: Tensor,
        kernels: Tensor,
    ) -> Tensor:
        outputs = []

        for out in range(kernels.shape[0]):
            outputs.append(sum(conv_one(image[c], kernels[out, c]) for c in range(image.shape[0])))

        return torch.stack(outputs)


    def conv_parameters(
        in_channels: int,
        out_channels: int,
        k: int,
    ) -> int:
        return out_channels * in_channels * k * k + out_channels


    def linear_parameters(
        channels: int,
        h: int,
        w: int,
        out_channels: int,
    ) -> int:
        inputs = channels * h * w
        outputs = out_channels * h * w

        return inputs * outputs + outputs
    ''',
)


def check_channels1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="multi_channel_conv",
        description="adds up every input channel for each output channel",
        inputs=(
            (small_tensor(730, 3, 5, 5), small_tensor(731, 2, 3, 3, 3)),
            (small_tensor(732, 2, 4, 6), small_tensor(733, 4, 2, 2, 3)),
        ),
        reference=lambda image, kernels: F.conv2d(image[None], kernels)[0],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="conv_parameters",
        description="counts nn.Conv2d's weights and biases",
        inputs=(
            (3, 16, 3),
            (64, 128, 1),
        ),
        reference=lambda in_channels, out_channels, k: sum(p.numel() for p in nn.Conv2d(in_channels, out_channels, k).parameters()),
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="linear_parameters",
        description="counts the Linear that maps the same shapes",
        inputs=(
            (3, 32, 32, 16),
            (1, 4, 5, 2),
        ),
        reference=lambda channels, h, w, out_channels: channels * h * w * out_channels * h * w + out_channels * h * w,
        expect=expect_value,
    )


CHANNELS1 = Exercise(
    name="channels1",
    chapter=CONV_SHAPES,
    title="Channels",
    lesson=CHANNELS1_LESSON,
    source=CHANNELS1_SOURCE,
    solution=CHANNELS1_SOLUTION,
    hints={
        "multi_channel_conv": "Add up conv_one(image[c], kernels[out, c]) for every input channel c, with sum over "
        "a loop of range(image.shape[0]).",
        "conv_parameters": "Each of the out_channels kernels holds in_channels * k * k weights, and each output "
        "channel has one bias.",
        "linear_parameters": "The Linear has channels * h * w inputs and out_channels * h * w outputs. Count one "
        "weight for each pair of an input and an output, and one bias per output.",
    },
    introduces=(),
    check=check_channels1,
)


# ---------------------------------------------------------------------------
# padding1
# ---------------------------------------------------------------------------

PADDING1_LESSON = prose(
    """
    Without padding, each layer makes the image `k - 1` pixels smaller in
    each dimension. The pixels at the edges also appear in fewer outputs than
    the pixels in the middle. **Padding** adds a border of zeros around the
    input before the convolution. With `p` zeros on every side, an input of
    size `n` gives an output of size `n + 2p - k + 1`.

    For an odd kernel size, `p = (k - 1) / 2` keeps the output the same size
    as the input. That is the usual choice, for example padding 1 for a 3 by 3
    kernel. It lets you stack many layers without the image shrinking away.

    ```python
    image = torch.ones(2, 2)
    F.pad(image, (1, 1, 1, 1))    # a 4 by 4 tensor, a border of zeros around the ones
    ```

    `F.pad` takes the number of zeros to add on each side as
    `(left, right, top, bottom)`. The last dimension comes first, so left and
    right pad the width. `nn.Conv2d(3, 16, 3, padding=1)` does the padding
    inside the layer.
    """,
)

PADDING1_SOURCE = source_file(
    '''
    # padding1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    image = torch.ones(2, 2)
    bordered = F.pad(image, (1, 1, 1, 1))            # shape (4, 4), zeros around the ones


    # TODO: Return image, of shape (h, w), with p zeros added on every side.
    #   The result has shape (h + 2 * p, w + 2 * p).
    def pad_zeros(
        image: Tensor,
        p: int,
    ) -> Tensor:
        ...


    # TODO: Return the output size of a convolution with kernel size k and
    #   padding p on an input of size n.
    #   Example: padded_size(32, 3, 1) returns 32.
    def padded_size(
        n: int,
        k: int,
        p: int,
    ) -> int:
        ...


    # TODO: Return the padding that keeps the output the same size as the
    #   input, for an odd kernel size k. Return it as an int.
    #   Example: same_padding(3) returns 1.
    def same_padding(k: int) -> int:
        ...
    ''',
)

PADDING1_SOLUTION = source_file(
    '''
    # padding1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    image = torch.ones(2, 2)
    bordered = F.pad(image, (1, 1, 1, 1))


    def pad_zeros(
        image: Tensor,
        p: int,
    ) -> Tensor:
        return F.pad(image, (p, p, p, p))


    def padded_size(
        n: int,
        k: int,
        p: int,
    ) -> int:
        return n + 2 * p - k + 1


    def same_padding(k: int) -> int:
        return (k - 1) // 2
    ''',
)


def check_padding1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="pad_zeros",
        description="adds a border of zeros",
        inputs=(
            (small_tensor(740, 2, 3), 1),
            (small_tensor(741, 3, 2), 2),
        ),
        reference=lambda image, p: F.pad(image, (p, p, p, p)),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="padded_size",
        description="matches nn.Conv2d's output size",
        inputs=(
            (32, 3, 1),
            (28, 5, 0),
            (10, 7, 2),
        ),
        reference=lambda n, k, p: reference_output_size(n, k, p, 1),
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="same_padding",
        description="keeps the size for odd kernels",
        inputs=(
            (3,),
            (7,),
            (1,),
        ),
        reference=lambda k: (k - 1) // 2,
        expect=expect_whole_number,
    )


def expect_whole_number(
    call: str,
    actual: object,
    expected: int,
) -> None:
    """nn.Conv2d refuses a padding like 1.0, so a float fails here even when it equals the right number."""
    if isinstance(actual, float):
        raise Mismatch(f"{call} is {actual!r}, a float, but nn.Conv2d takes its padding as a whole number. Divide with // so it comes out an int.")

    expect_value(
        call=call,
        actual=actual,
        expected=expected,
    )


PADDING1 = Exercise(
    name="padding1",
    chapter=CONV_SHAPES,
    title="Padding",
    lesson=PADDING1_LESSON,
    source=PADDING1_SOURCE,
    solution=PADDING1_SOLUTION,
    hints={
        "pad_zeros": "Call F.pad with p zeros on each of the four sides, (p, p, p, p).",
        "padded_size": "Padding adds p to each side, so the padded input has length n + 2 * p. The kernel fits in "
        "that length minus k, plus 1, positions.",
        "same_padding": "Split k - 1 zeros evenly between the two sides. Use // so the result is an int.",
    },
    introduces=("padding",),
    check=check_padding1,
)


# ---------------------------------------------------------------------------
# stride1
# ---------------------------------------------------------------------------

STRIDE1_LESSON = prose(
    """
    The kernel can move more than one position at a time. With a **stride** of
    `s`, it moves `s` positions between one output and the next, so the output
    is about `s` times smaller. A stride of 2 halves the height and the width.
    Most image models shrink the image this way as their layers go deeper.

    With padding `p`, the output size is `(n + 2p - k) // s + 1`. The `//`
    rounds down, because a position where the kernel would stick out past the
    end of the input is skipped.

    ```python
    x = torch.arange(7.0)
    x.unfold(0, 3, 2)    # windows of 3 starting 2 apart, tensor([[0., 1., 2.], [2., 3., 4.], [4., 5., 6.]])
    ```

    `nn.Conv2d(16, 32, 3, stride=2, padding=1)` turns a 32 by 32 image into a
    16 by 16 one, with twice the channels.
    """,
)

STRIDE1_SOURCE = source_file(
    '''
    # stride1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.arange(7.0)
    windows = x.unfold(0, 3, 2)                      # 3 windows of 3, starting 2 apart


    # TODO: Return the output size of a convolution with kernel size k, padding p
    #   and stride s on an input of size n.
    #   Example: strided_size(32, 3, 1, 2) returns 16.
    def strided_size(
        n: int,
        k: int,
        p: int,
        s: int,
    ) -> int:
        ...


    # TODO: Return the convolution of x with w at stride s. Get the windows of x
    #   from x.unfold, s positions apart, and take the dot product of each
    #   window with w.
    def strided_conv1d(
        x: Tensor,
        w: Tensor,
        s: int,
    ) -> Tensor:
        ...


    # TODO: Return an nn.Conv2d from in_channels to out_channels, with a 3 by 3
    #   kernel, that halves the height and the width of an image whose height
    #   and width are even.
    def halving_conv(
        in_channels: int,
        out_channels: int,
    ) -> nn.Conv2d:
        ...
    ''',
)

STRIDE1_SOLUTION = source_file(
    '''
    # stride1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.arange(7.0)
    windows = x.unfold(0, 3, 2)


    def strided_size(
        n: int,
        k: int,
        p: int,
        s: int,
    ) -> int:
        return (n + 2 * p - k) // s + 1


    def strided_conv1d(
        x: Tensor,
        w: Tensor,
        s: int,
    ) -> Tensor:
        return x.unfold(0, len(w), s) @ w


    def halving_conv(
        in_channels: int,
        out_channels: int,
    ) -> nn.Conv2d:
        return nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
    ''',
)


def check_halving_conv(found: object) -> None:
    for in_channels, out_channels, size in ((3, 8, 32), (16, 32, 10)):
        call = f"halving_conv({in_channels}, {out_channels})"
        layer = expect_module(call, found(in_channels, out_channels))

        if not isinstance(layer, nn.Conv2d) or layer.kernel_size != (3, 3):
            raise Mismatch(f"{call} should be an nn.Conv2d with a 3 by 3 kernel.")

        with torch.no_grad():
            shape = tuple(layer(torch.zeros(1, in_channels, size, size)).shape)

        wanted = (1, out_channels, size // 2, size // 2)

        if shape != wanted:
            raise Mismatch(f"{call} turns an image of shape (1, {in_channels}, {size}, {size}) into {shape}, but it should give {wanted}.")


def check_stride1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="strided_size",
        description="matches nn.Conv2d's output size with a stride",
        inputs=(
            (32, 3, 1, 2),
            (7, 3, 0, 2),
            (224, 7, 3, 2),
            (10, 4, 0, 3),
        ),
        reference=reference_output_size,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="strided_conv1d",
        description="matches F.conv1d with a stride",
        inputs=(
            (small_tensor(745, 9), small_tensor(746, 3), 2),
            (small_tensor(747, 10), small_tensor(748, 2), 3),
        ),
        reference=lambda x, w, s: F.conv1d(x.view(1, 1, -1), w.view(1, 1, -1), stride=s).flatten(),
        expect=expect_tensor,
    )
    checker.test(
        function="halving_conv",
        description="halves the height and the width",
        run=check_halving_conv,
    )


STRIDE1 = Exercise(
    name="stride1",
    chapter=CONV_SHAPES,
    title="Stride",
    lesson=STRIDE1_LESSON,
    source=STRIDE1_SOURCE,
    solution=STRIDE1_SOLUTION,
    hints={
        "strided_size": "Follow the formula in the lesson, with // for the rounding down.",
        "strided_conv1d": "x.unfold(0, len(w), s) gives the windows s apart. Multiply them by w with @.",
        "halving_conv": "nn.Conv2d with kernel size 3, stride=2 and padding=1.",
    },
    introduces=(),
    check=check_stride1,
)


# ---------------------------------------------------------------------------
# dilation1
# ---------------------------------------------------------------------------

DILATION1_LESSON = prose(
    """
    Each weight of a kernel reads one input position, called a tap.
    **Dilation** spreads the taps out. With a dilation of `d`, the taps are `d`
    positions apart. So a kernel of size 3 with dilation 2 reads positions 0,
    2 and 4. It spans 5 positions with the same 3 weights.

    The **receptive field** of an output is the range of input positions that
    can affect it. One layer with kernel size `k` and dilation 1 sees `k`
    positions. At stride 1, each layer stacked on top adds `k - 1` more
    positions, times that layer's dilation. So three layers of size 3 see 7
    positions. Three layers of size 3 with dilations 1, 2 and 4 see 15.

    Dilation widens the receptive field without adding weights, and without
    shrinking the output the way a stride does. Models of sound and of long
    sequences rely on it, as Section 6 shows.

    `x[::2]` takes every second entry, starting at the first.
    `F.conv1d(x, w, dilation=2)` and `nn.Conv2d(..., dilation=2)` take the
    dilation as an argument.
    """,
)

DILATION1_SOURCE = source_file(
    '''
    # dilation1.py

    import torch
    from torch import Tensor

    x = torch.arange(6.0)
    every_second = x[::2]                            # tensor([0., 2., 4.])


    # TODO: Return how many input positions a kernel of size k spans with
    #   dilation d, from its first tap to its last.
    #   Example: spanned(3, 2) returns 5.
    def spanned(
        k: int,
        d: int,
    ) -> int:
        ...


    # TODO: Finish the last line, so this returns the convolution of x with w
    #   at dilation d. Each window covers spanned(k, d) positions, and the
    #   kernel reads every d-th position of it, starting with the first.
    def dilated_conv1d(
        x: Tensor,
        w: Tensor,
        d: int,
    ) -> Tensor:
        windows = x.unfold(0, spanned(len(w), d), 1)

        return ...


    # TODO: layers describes a stack of layers at stride 1, as a list of (k, d)
    #   pairs, one per layer. Return the receptive field of the stack.
    #   Example: receptive_field([(3, 1), (3, 2), (3, 4)]) returns 15.
    def receptive_field(layers: list[tuple[int, int]]) -> int:
        ...
    ''',
)

DILATION1_SOLUTION = source_file(
    '''
    # dilation1.py, a solution for comparison

    import torch
    from torch import Tensor

    x = torch.arange(6.0)
    every_second = x[::2]


    def spanned(
        k: int,
        d: int,
    ) -> int:
        return d * (k - 1) + 1


    def dilated_conv1d(
        x: Tensor,
        w: Tensor,
        d: int,
    ) -> Tensor:
        windows = x.unfold(0, spanned(len(w), d), 1)

        return windows[:, ::d] @ w


    def receptive_field(layers: list[tuple[int, int]]) -> int:
        return 1 + sum((k - 1) * d for k, d in layers)
    ''',
)


def check_dilation1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="spanned",
        description="gives the positions a dilated kernel spans",
        inputs=(
            (3, 1),
            (3, 2),
            (5, 4),
        ),
        reference=lambda k, d: d * (k - 1) + 1,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="dilated_conv1d",
        description="matches F.conv1d with a dilation",
        inputs=(
            (small_tensor(750, 9), small_tensor(751, 3), 2),
            (small_tensor(752, 12), small_tensor(753, 2), 4),
        ),
        reference=lambda x, w, d: F.conv1d(x.view(1, 1, -1), w.view(1, 1, -1), dilation=d).flatten(),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="receptive_field",
        description="adds up what each layer adds",
        inputs=(
            ([(3, 1), (3, 1), (3, 1)],),
            ([(3, 1), (3, 2), (3, 4)],),
            ([(2, 1), (2, 2), (2, 4), (2, 8)],),
        ),
        reference=lambda layers: 1 + sum((k - 1) * d for k, d in layers),
        expect=expect_value,
    )


DILATION1 = Exercise(
    name="dilation1",
    chapter=CONV_SHAPES,
    title="Dilation and the receptive field",
    lesson=DILATION1_LESSON,
    source=DILATION1_SOURCE,
    solution=DILATION1_SOLUTION,
    hints={
        "spanned": "A kernel of size k has k taps, with d - 1 empty positions between each neighbouring pair of "
        "taps. Add up the taps and the empty positions.",
        "dilated_conv1d": "windows[:, ::d] keeps every d-th position of each window. Multiply that by w with @.",
        "receptive_field": "Start from 1 and add (k - 1) * d for each layer.",
    },
    introduces=(),
    check=check_dilation1,
)


# ---------------------------------------------------------------------------
# conv5
# ---------------------------------------------------------------------------

CONV5_LESSON = prose(
    """
    A **1 by 1 convolution** reads one pixel at a time, with all of that
    pixel's channels. It is a Linear layer from the input channels to the
    output channels, applied at every pixel with the same weights. Models use
    it to change the number of channels cheaply.

    ResNet-50 uses a 1 by 1 convolution to shrink 256 channels to 64. It runs
    its 3 by 3 convolution on those 64 channels, then a second 1 by 1
    convolution grows them back to 256. Without biases, those three layers
    learn 69,632 weights. A single 3 by 3 layer from 256 channels to 256
    would learn 589,824, over 8 times as many.

    Here are the usual settings for a kernel of 3 by 3 or bigger.

    - keep the size: stride 1, padding `(k - 1) // 2`, as in VGG.
    - halve the size: stride 2, padding `(k - 1) // 2`.
    - ResNet's first layer: a 7 by 7 kernel, stride 2, padding 3, so 224 becomes 112.

    To plan a model, follow the size through each layer with the formula from
    stride1. To make a layer give a target size, find a padding `p` that
    solves `(n + 2p - k) // s + 1 = target`.

    For `x` of shape `(c, h, w)`, `x.reshape(c, -1)` has one column per
    pixel. So `weight @ x.reshape(c, -1)` multiplies a weight of shape
    `(out, c)` by every pixel's channels at once.
    """,
)

CONV5_SOURCE = source_file(
    '''
    # conv5.py

    import torch
    from torch import Tensor

    x = torch.ones(3, 2, 2)
    pixels = x.reshape(3, -1)                        # shape (3, 4), one column per pixel


    # TODO: x has shape (c, h, w), and weight has shape (out, c). Return the
    #   1 by 1 convolution of x with weight, with no bias, as a tensor of shape
    #   (out, h, w).
    def pointwise(
        x: Tensor,
        weight: Tensor,
    ) -> Tensor:
        ...


    # TODO: An input of size n goes through the layers in order. layers is a
    #   list of (k, s, p) triples, one per layer, giving its kernel size,
    #   stride and padding. Return a list of the size after each layer.
    #   Example: sizes_through(224, [(7, 2, 3), (3, 2, 1)]) returns [112, 56].
    def sizes_through(
        n: int,
        layers: list[tuple[int, int, int]],
    ) -> list[int]:
        ...


    # TODO: Return the smallest padding p from 0 to k, including k, that makes
    #   a convolution with kernel size k and stride s turn an input of size n
    #   into an output of size target. Return None if no such padding exists.
    #   Example: padding_for(224, 112, 7, 2) returns 3.
    def padding_for(
        n: int,
        target: int,
        k: int,
        s: int,
    ) -> int | None:
        ...
    ''',
)

CONV5_SOLUTION = source_file(
    '''
    # conv5.py, a solution for comparison

    import torch
    from torch import Tensor

    x = torch.ones(3, 2, 2)
    pixels = x.reshape(3, -1)


    def pointwise(
        x: Tensor,
        weight: Tensor,
    ) -> Tensor:
        c, h, w = x.shape

        return (weight @ x.reshape(c, -1)).reshape(-1, h, w)


    def sizes_through(
        n: int,
        layers: list[tuple[int, int, int]],
    ) -> list[int]:
        sizes = []

        for k, s, p in layers:
            n = (n + 2 * p - k) // s + 1
            sizes.append(n)

        return sizes


    def padding_for(
        n: int,
        target: int,
        k: int,
        s: int,
    ) -> int | None:
        for p in range(k + 1):
            if (n + 2 * p - k) // s + 1 == target:
                return p

        return None
    ''',
)


def reference_sizes(
    n: int,
    layers: list[tuple[int, int, int]],
) -> list[int]:
    sizes = []

    for k, s, p in layers:
        n = reference_output_size(n, k, p, s)
        sizes.append(n)

    return sizes


def reference_padding(
    n: int,
    target: int,
    k: int,
    s: int,
) -> int | None:
    for p in range(k + 1):
        if reference_output_size(n, k, p, s) == target:
            return p

    return None


def check_conv5(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="pointwise",
        description="matches a 1 by 1 F.conv2d",
        inputs=(
            (small_tensor(760, 3, 4, 5), small_tensor(761, 2, 3)),
            (small_tensor(762, 4, 2, 2), small_tensor(763, 6, 4)),
        ),
        reference=lambda x, weight: F.conv2d(x[None], weight[:, :, None, None])[0],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="sizes_through",
        description="follows the size through each layer",
        inputs=(
            (224, [(7, 2, 3), (3, 1, 1), (3, 2, 1)]),
            (32, [(3, 1, 0), (3, 1, 0), (2, 2, 0)]),
        ),
        reference=reference_sizes,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="padding_for",
        description="finds the padding that gives the target size",
        inputs=(
            (32, 32, 3, 1),
            (32, 16, 3, 2),
            (224, 112, 7, 2),
            (10, 3, 3, 1),
            # Paddings 1 and 2 both give 3 here, and the smallest is wanted.
            (8, 3, 2, 4),
            # Only a padding of 1, which is k itself, gives 6 here, so the search has to reach k.
            (4, 6, 1, 1),
        ),
        reference=reference_padding,
        expect=expect_padding,
    )


def expect_padding(
    call: str,
    actual: object,
    expected: int | None,
) -> None:
    if expected is not None and actual is None:
        raise Mismatch(
            f"{call} is None, but a padding of {expected} gives that size. A None means it is {NOT_STARTED}, "
            "or the search stopped before it reached that padding.",
        )

    if expected is not None:
        expect_value(
            call=call,
            actual=actual,
            expected=expected,
        )
        return

    if actual is not None:
        raise Mismatch(f"{call} is {actual!r}, but no padding gives that size, so it should be None.")


CONV5 = Exercise(
    name="conv5",
    chapter=CONV_SHAPES,
    title="Choosing the shapes",
    lesson=CONV5_LESSON,
    source=CONV5_SOURCE,
    solution=CONV5_SOLUTION,
    hints={
        "pointwise": "Take c, h, w from x.shape, multiply weight by x.reshape(c, -1), then reshape the result to "
        "(-1, h, w).",
        "sizes_through": "For each (k, s, p), compute the new n with stride1's formula and append it.",
        "padding_for": "Try each p in range(k + 1) and return the first whose output size equals target. After the "
        "loop, return None.",
    },
    introduces=(),
    check=check_conv5,
)
