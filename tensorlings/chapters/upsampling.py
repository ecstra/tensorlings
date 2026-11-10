"""34_upsampling, growing maps back with interpolation and transposed convolution, and the U-Net with its skips."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .classic_nets import pinned_threads
from .linear_layers import expect_module

UPSAMPLING = Chapter(
    folder="34_upsampling",
    title="Upsampling and U-Net",
    readme=source_file(
        """
        # Upsampling and U-Net

        Some models output an image, such as a mask that marks which pixels
        belong to an object. Such a model makes its maps smaller on the way in,
        so it has to make them full size again on the way out. This chapter
        builds nearest and bilinear upsampling and the transposed convolution.
        Then it builds the U-Net, whose skip connections pass fine detail from
        the early layers to the late ones.

        ## Further reading

        - [Ronneberger, Fischer and Brox, U-Net](https://arxiv.org/abs/1505.04597)
        - [Odena, Dumoulin and Olah, deconvolution and checkerboard artifacts](https://distill.pub/2016/deconv-checkerboard/)
        - [Dumoulin and Visin, a guide to convolution arithmetic](https://arxiv.org/abs/1603.07285)
        """,
    ),
)


def reference_transposed(
    x: Tensor,
    w: Tensor,
    s: int,
) -> Tensor:
    return F.conv_transpose1d(x.view(1, 1, -1), w.view(1, 1, -1), stride=s).flatten()


def block(
    inputs: int,
    outputs: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(outputs, outputs, 3, padding=1),
        nn.ReLU(),
    )


class ReferenceUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down1 = block(1, 8)
        self.down2 = block(8, 16)
        self.bottom = block(16, 32)
        self.up2 = block(32 + 16, 16)
        self.up1 = block(16 + 8, 8)
        self.out = nn.Conv2d(8, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        a = self.down1(x)
        b = self.down2(F.max_pool2d(a, 2))
        c = self.bottom(F.max_pool2d(b, 2))
        u = self.up2(torch.cat([F.interpolate(c, scale_factor=2), b], dim=1))
        u = self.up1(torch.cat([F.interpolate(u, scale_factor=2), a], dim=1))

        return self.out(u)


# ---------------------------------------------------------------------------
# upsample1
# ---------------------------------------------------------------------------

UPSAMPLE1_LESSON = prose(
    """
    Some models output an image, such as a mask that says which pixels belong
    to a cell or a road. The maps inside such a model get smaller in the early
    layers, so the later layers have to make them bigger again. Making a map
    bigger is called **upsampling**.

    Nearest upsampling by a factor `s` repeats every pixel `s` times
    vertically and `s` times horizontally. So each pixel becomes an `s` by `s`
    square of copies. Linear interpolation takes a weighted average of the two
    nearest inputs instead. A point a quarter of the way from value `a` to
    value `b` gets `0.75 * a + 0.25 * b`. Bilinear upsampling does linear
    interpolation along the rows, then along the columns.

    `repeat_interleave(n)` repeats each entry `n` times in a row.

    ```python
    torch.tensor([1, 2]).repeat_interleave(3)    # tensor([1, 1, 1, 2, 2, 2])
    ```

    `F.interpolate(x, scale_factor=2, mode="bilinear")` does bilinear
    upsampling by 2. It expects `x` of shape `(batch, channels, h, w)`. With
    `mode="nearest"` it repeats pixels instead.

    With `align_corners=True`, the first output lands exactly on the first
    input, and the last output exactly on the last input. Say you also pass
    `size=(n - 1) * s + 1` for `n` inputs. Then output `j` is the value
    interpolated at position `j / s` along the input. With `scale_factor=s`
    instead, it gives `n * s` outputs, spaced a little closer together. The
    default for `align_corners` is `False`.

    In `linear_1d` below, `.floor()` rounds each position down. `.long()` then
    turns the result into integers, so it can index a tensor.
    """,
)

UPSAMPLE1_SOURCE = source_file(
    '''
    # upsample1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    repeated = torch.tensor([1, 2]).repeat_interleave(3)    # tensor([1, 1, 1, 2, 2, 2])


    # TODO: Return image upsampled by a factor of s with nearest upsampling, by
    #   repeating every pixel s times along each dimension. image has shape
    #   (h, w), and the result has shape (h * s, w * s).
    #   Example: nearest(torch.tensor([[1., 2.], [3., 4.]]), 2) returns a 4 by 4
    #   tensor whose first row is [1., 1., 2., 2.].
    def nearest(
        image: Tensor,
        s: int,
    ) -> Tensor:
        ...


    # TODO: Return x upsampled by a factor of s with bilinear upsampling. Use
    #   F.interpolate and leave align_corners at its default. x has shape
    #   (batch, channels, h, w), and the result has shape
    #   (batch, channels, h * s, w * s).
    def bilinear(
        x: Tensor,
        s: int,
    ) -> Tensor:
        ...


    # TODO: Finish the last line, so this returns the 1D tensor x upsampled by
    #   a factor of s with linear interpolation and aligned corners. Output j is
    #   a weighted average of x[left[j]] and the input after it. The input
    #   after it gets weight fraction[j], and x[left[j]] gets 1 - fraction[j].
    #   Example: linear_1d(torch.tensor([0., 4., 2.]), 2) returns
    #   tensor([0., 2., 4., 3., 2.]).
    def linear_1d(
        x: Tensor,
        s: int,
    ) -> Tensor:
        positions = torch.arange((len(x) - 1) * s + 1) / s
        left = positions.floor().long().clamp(max=len(x) - 2)
        fraction = positions - left

        return ...
    ''',
)

UPSAMPLE1_SOLUTION = source_file(
    '''
    # upsample1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    repeated = torch.tensor([1, 2]).repeat_interleave(3)


    def nearest(
        image: Tensor,
        s: int,
    ) -> Tensor:
        return image.repeat_interleave(s, dim=0).repeat_interleave(s, dim=1)


    def bilinear(
        x: Tensor,
        s: int,
    ) -> Tensor:
        return F.interpolate(x, scale_factor=s, mode="bilinear")


    def linear_1d(
        x: Tensor,
        s: int,
    ) -> Tensor:
        positions = torch.arange((len(x) - 1) * s + 1) / s
        left = positions.floor().long().clamp(max=len(x) - 2)
        fraction = positions - left

        return x[left] * (1 - fraction) + x[left + 1] * fraction
    ''',
)


def check_upsample1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="nearest",
        description="repeats every pixel",
        inputs=(
            (small_tensor(800, 2, 3), 2),
            (small_tensor(801, 3, 2), 3),
        ),
        reference=lambda image, s: F.interpolate(image[None, None], scale_factor=s, mode="nearest")[0, 0],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="bilinear",
        description="matches F.interpolate's bilinear mode",
        inputs=(
            (small_tensor(802, 1, 2, 3, 3), 2),
            (small_tensor(803, 2, 1, 2, 4), 3),
        ),
        reference=lambda x, s: F.interpolate(x, scale_factor=s, mode="bilinear"),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="linear_1d",
        description="matches F.interpolate with the corners aligned",
        inputs=(
            (torch.tensor([0.0, 4.0, 2.0]), 4),
            (small_tensor(804, 5), 3),
        ),
        reference=lambda x, s: F.interpolate(x.view(1, 1, -1), size=(len(x) - 1) * s + 1, mode="linear", align_corners=True).flatten(),
        expect=expect_tensor,
    )


UPSAMPLE1 = Exercise(
    name="upsample1",
    chapter=UPSAMPLING,
    title="Nearest and bilinear",
    lesson=UPSAMPLE1_LESSON,
    source=UPSAMPLE1_SOURCE,
    solution=UPSAMPLE1_SOLUTION,
    hints={
        "nearest": "Use .repeat_interleave(s, dim=0) for the rows, then .repeat_interleave(s, dim=1) for the "
        "columns.",
        "bilinear": "Call F.interpolate with scale_factor=s and mode=\"bilinear\".",
        "linear_1d": "Take a weighted average of x[left] and x[left + 1], with weight fraction on the second and "
        "1 - fraction on the first.",
    },
    introduces=(),
    check=check_upsample1,
)


# ---------------------------------------------------------------------------
# transposed1
# ---------------------------------------------------------------------------

TRANSPOSED1_LESSON = prose(
    """
    A **transposed convolution** makes a map bigger using learned weights. For
    each input value, it multiplies the kernel by that value and adds the
    result into the output. This scaled copy of the kernel is called a stamp
    here. With stride `s`, the stamp for input `i` starts at output position
    `i * s`. Where stamps overlap, their values add up. The output ends where
    the last stamp ends.

    ```python
    # x = [1, 2], w = [1, 1, 1], s = 2
    # stamp of 1 at 0: [1, 1, 1, 0, 0]
    # stamp of 2 at 2: [0, 0, 2, 2, 2]
    # sum:             [1, 1, 3, 2, 2]
    ```

    A transposed convolution computes the same thing as the backward pass of a
    convolution. In a convolution, each output is computed from `k` inputs. So
    in the backward pass, the gradient of each output flows back to those same
    `k` inputs, multiplied by the kernel's weights. That pattern is exactly the
    stamp. Some people call a transposed convolution a deconvolution, but it
    does not undo a convolution.

    Sometimes you want the backward pass to start from a chosen gradient `g`
    for the outputs `y`, instead of from a loss. To do that, call `.backward()`
    on `(y * g).sum()`. The derivative of that sum with respect to each `y[i]`
    is `g[i]`. So the gradient that flows back from the outputs is exactly
    `g`.

    `nn.ConvTranspose2d(in, out, k, stride=2)` is the layer version. When the
    kernel size is not a multiple of the stride, like kernel 3 at stride 2,
    the stamps overlap unevenly. You can see this by using all ones as both
    the input and the kernel. Then each output equals the number of stamps
    that cover it.

    ```python
    ones = torch.ones(1, 1, 6)
    F.conv_transpose1d(ones, torch.ones(1, 1, 3), stride=2)    # 1 1 2 1 2 1 2 1 2 1 2 1 1
    F.conv_transpose1d(ones, torch.ones(1, 1, 4), stride=2)    # 1 1 2 2 2 2 2 2 2 2 2 2 1 1
    grown = F.interpolate(ones, scale_factor=2)                # 12 ones
    F.conv1d(grown, torch.ones(1, 1, 3), padding=1)            # 2 3 3 3 3 3 3 3 3 3 3 2
    ```

    With kernel 3, the output alternates between 1 and 2. In 2D that pattern
    shows up as a checkerboard. Kernel 4 covers the middle evenly. Upsampling
    with interpolation and then applying a plain convolution gives an even
    result everywhere except the padded ends. Odena and others recommended
    this last approach.
    """,
)

TRANSPOSED1_SOURCE = source_file(
    '''
    # transposed1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    output = torch.zeros(5)
    output[2:5] += 2 * torch.tensor([1.0, 1.0, 1.0])    # tensor([0., 0., 2., 2., 2.])


    # TODO: Return how many outputs a transposed convolution with kernel size k
    #   and stride s gives for n inputs.
    #   Example: transposed_size(2, 3, 2) returns 5.
    def transposed_size(
        n: int,
        k: int,
        s: int,
    ) -> int:
        ...


    # TODO: Finish the line in the loop. For each input i, add w times x[i] into
    #   output, starting at position i * s and covering k positions. Where
    #   stamps overlap, their values add up.
    #   Example: for x = [1., 2.], w = [1., 1., 1.] and s = 2, transposed_1d
    #   returns tensor([1., 1., 3., 2., 2.]).
    def transposed_1d(
        x: Tensor,
        w: Tensor,
        s: int,
    ) -> Tensor:
        k = len(w)
        output = torch.zeros(transposed_size(len(x), k, s))

        for i in range(len(x)):
            ...

        return output


    # TODO: Return the gradient with respect to x of a stride 1 convolution of
    #   x with w, when grad is the gradient arriving at the convolution's
    #   outputs. x, w and grad are 1D, and grad has one entry per output. Use
    #   autograd and the (y * g).sum() trick from the lesson.
    def conv_input_gradient(
        x: Tensor,
        w: Tensor,
        grad: Tensor,
    ) -> Tensor:
        ...
    ''',
)

TRANSPOSED1_SOLUTION = source_file(
    '''
    # transposed1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    output = torch.zeros(5)
    output[2:5] += 2 * torch.tensor([1.0, 1.0, 1.0])


    def transposed_size(
        n: int,
        k: int,
        s: int,
    ) -> int:
        return (n - 1) * s + k


    def transposed_1d(
        x: Tensor,
        w: Tensor,
        s: int,
    ) -> Tensor:
        k = len(w)
        output = torch.zeros(transposed_size(len(x), k, s))

        for i in range(len(x)):
            output[i * s:i * s + k] += x[i] * w

        return output


    # Calling .backward() on (outputs * grad).sum() sends exactly grad back into the outputs.
    def conv_input_gradient(
        x: Tensor,
        w: Tensor,
        grad: Tensor,
    ) -> Tensor:
        x = x.clone().requires_grad_()
        outputs = F.conv1d(x.view(1, 1, -1), w.view(1, 1, -1)).flatten()
        (outputs * grad).sum().backward()

        return x.grad
    ''',
)


def check_transposed1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="transposed_size",
        description="gives the number of outputs",
        inputs=(
            (2, 3, 2),
            (5, 4, 2),
            (7, 3, 1),
        ),
        reference=lambda n, k, s: (n - 1) * s + k,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="transposed_1d",
        description="matches F.conv_transpose1d",
        inputs=(
            (torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0, 1.0]), 2),
            (small_tensor(810, 4), small_tensor(811, 3), 2),
            (small_tensor(812, 5), small_tensor(813, 2), 1),
        ),
        reference=reference_transposed,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="conv_input_gradient",
        description="equals a transposed convolution of the gradient",
        inputs=(
            (small_tensor(814, 6), small_tensor(815, 3), small_tensor(816, 4)),
            (small_tensor(817, 8), small_tensor(818, 2), small_tensor(819, 7)),
        ),
        reference=lambda x, w, grad: reference_transposed(grad, w, 1),
        expect=expect_tensor,
    )


TRANSPOSED1 = Exercise(
    name="transposed1",
    chapter=UPSAMPLING,
    title="Transposed convolution",
    lesson=TRANSPOSED1_LESSON,
    source=TRANSPOSED1_SOURCE,
    solution=TRANSPOSED1_SOLUTION,
    hints={
        "transposed_size": "The last stamp starts at position (n - 1) * s and is k positions long.",
        "transposed_1d": "Add x[i] * w to the slice output[i * s:i * s + k]. The example at the top of the file adds "
        "into output[2:5] the same way.",
        "conv_input_gradient": "Clone x and make it require a gradient. Run F.conv1d on it, with x and w reshaped to "
        "(1, 1, length) as in conv2. Flatten the outputs, multiply them by grad, sum, and call .backward(). Then "
        "return the .grad of the clone.",
    },
    introduces=(),
    check=check_transposed1,
)


# ---------------------------------------------------------------------------
# unet1
# ---------------------------------------------------------------------------

UNET1_LESSON = prose(
    """
    U-Net, from Ronneberger, Fischer and Brox in 2015, was built to segment
    cells in microscope images. To segment an image means to label every pixel
    with the class it belongs to. The network has two halves, usually drawn as
    the two arms of a U.

    - the way down: blocks of two 3 by 3 convolutions, each block followed by a max pool of 2.
    - the way up: upsampling by 2, each time followed by a block, until the maps are full size again.
    - the skips: the maps from each level of the way down are joined onto the maps of the same size on the way up.

    At the end, a 1 by 1 convolution gives each pixel a score for each class.
    The way down learns what is in the image, but pooling throws away exactly
    where things are. The skips pass the earlier maps, which still have that
    detail, to the way up.

    Joining means concatenating along the channel dimension. So an upsampled
    map with 32 channels and a skip with 16 channels give a map with 48
    channels. For tensors of shape `(batch, channels, h, w)`,
    `torch.cat([upsampled, skip], dim=1)` does this, with the upsampled
    channels first. The order matters, because each weight in the next block
    applies to one particular input channel.

    The paper used convolutions without padding, so its maps got a little
    smaller in each block. That made each skip's map larger than the map it
    was joined to. So the paper first cropped the skip's map to its middle
    part. Most U-Nets today use padding instead.

    The paper also followed each upsampling with a 2 by 2 convolution that
    halved the number of channels. `TinyUNet` below leaves that out.
    `TinyUNet` marks only one kind of object, so it outputs one score per
    pixel.
    """,
)

UNET1_SOURCE = source_file(
    '''
    # unet1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    upsampled = torch.zeros(1, 32, 8, 8)
    skip = torch.zeros(1, 16, 8, 8)
    joined = torch.cat([upsampled, skip], dim=1)     # shape (1, 48, 8, 8), the upsampled channels first


    # TODO: Return the middle size by size square of x, the way the paper
    #   cropped its skips. x has shape (batch, channels, h, w), and the result
    #   has shape (batch, channels, size, size). Assume h - size and w - size
    #   are even.
    #   Example: for h = w = 6 and size = 2, it keeps rows 2 and 3 and columns
    #   2 and 3.
    def center_crop(
        x: Tensor,
        size: int,
    ) -> Tensor:
        ...


    def block(
        inputs: int,
        outputs: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.ReLU(),
        )


    # TODO: Finish forward. On the way down, a is down1(x). b is down2 applied
    #   to a after a max pool of 2, and c is bottom applied to b after another
    #   max pool of 2. On the way up, upsample c by 2 with F.interpolate in its
    #   default nearest mode. Join b after it with torch.cat([upsampled, b],
    #   dim=1), and pass the result through up2. Then upsample up2's output the
    #   same way, join a after it, and pass the result through up1. Return out
    #   applied to up1's output. x has shape (batch, 1, h, w), and so does the
    #   result.
    class TinyUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down1 = block(1, 8)
            self.down2 = block(8, 16)
            self.bottom = block(16, 32)
            self.up2 = block(32 + 16, 16)
            self.up1 = block(16 + 8, 8)
            self.out = nn.Conv2d(8, 1, 1)

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

UNET1_SOLUTION = source_file(
    '''
    # unet1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    upsampled = torch.zeros(1, 32, 8, 8)
    skip = torch.zeros(1, 16, 8, 8)
    joined = torch.cat([upsampled, skip], dim=1)


    def center_crop(
        x: Tensor,
        size: int,
    ) -> Tensor:
        top = (x.shape[2] - size) // 2
        left = (x.shape[3] - size) // 2

        return x[:, :, top:top + size, left:left + size]


    def block(
        inputs: int,
        outputs: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.ReLU(),
        )


    class TinyUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down1 = block(1, 8)
            self.down2 = block(8, 16)
            self.bottom = block(16, 32)
            self.up2 = block(32 + 16, 16)
            self.up1 = block(16 + 8, 8)
            self.out = nn.Conv2d(8, 1, 1)

        def forward(self, x: Tensor) -> Tensor:
            a = self.down1(x)
            b = self.down2(F.max_pool2d(a, 2))
            c = self.bottom(F.max_pool2d(b, 2))
            u = self.up2(torch.cat([F.interpolate(c, scale_factor=2), b], dim=1))
            u = self.up1(torch.cat([F.interpolate(u, scale_factor=2), a], dim=1))

            return self.out(u)
    ''',
)


def check_unet_forward(found: type) -> None:
    torch.manual_seed(820)
    model = expect_module("TinyUNet()", found())
    reference = ReferenceUNet()
    reference.load_state_dict(model.state_dict())
    x = torch.rand(2, 1, 16, 16, generator=torch.Generator().manual_seed(821))

    with torch.no_grad():
        expected = reference(x)

    expect_tensor(
        call="TinyUNet() on 2 images of 16 by 16",
        actual=model(x),
        expected=expected,
    )


def check_unet1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="center_crop",
        description="keeps the middle part",
        inputs=(
            (small_tensor(822, 1, 2, 6, 6), 2),
            (small_tensor(823, 2, 1, 8, 6), 4),
        ),
        reference=lambda x, size: x[:, :, (x.shape[2] - size) // 2 : (x.shape[2] - size) // 2 + size, (x.shape[3] - size) // 2 : (x.shape[3] - size) // 2 + size],
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="TinyUNet",
        method="forward",
        description="goes down, up, and joins the skips",
        run=check_unet_forward,
    )


UNET1 = Exercise(
    name="unet1",
    chapter=UPSAMPLING,
    title="U-Net",
    lesson=UNET1_LESSON,
    source=UNET1_SOURCE,
    solution=UNET1_SOLUTION,
    hints={
        "center_crop": "The crop starts (h - size) // 2 rows down and (w - size) // 2 columns in. Slice the last two "
        "dimensions from there.",
        "TinyUNet.forward": "Compute a, b and c with down1, down2 and bottom, with F.max_pool2d(..., 2) between them. "
        "Then compute self.up2(torch.cat([F.interpolate(c, scale_factor=2), b], dim=1)), and do the same with up1 "
        "and a.",
    },
    introduces=(),
    check=check_unet1,
)


# ---------------------------------------------------------------------------
# unet2
# ---------------------------------------------------------------------------

UNET2_LESSON = prose(
    """
    For the table below, two models learned to mark made up shapes, circles
    and squares, in noisy images of 32 by 32. One model was unet1's U-Net. The
    other was the same network with no skips, so its way up gets input only
    from the layer below it. Each model trained for 400 steps on 1000
    examples, with three different seeds. Each was then scored on 500 new
    examples.

    **IoU**, short for intersection over union, measures how well a predicted
    mask matches the true mask. It is the number of pixels that both masks
    mark, divided by the number of pixels that at least one of them marks. A
    perfect match gives 1. On tensors of `True` and `False`, `&` computes and,
    and `|` computes or, entry by entry.

    | Model | Pixels correct | Edge pixels correct | IoU |
    |---|---|---|---|
    | no skips | 97.5% to 97.6% | 81.8% to 82.5% | 0.885 to 0.892 |
    | U-Net | 98.7% to 99.0% | 90.0% to 92.3% | 0.936 to 0.950 |

    Most pixels are easy because they are far inside or far outside a shape.
    So the fraction of all pixels correct hides the difference between the
    models. On the pixels next to an edge, the U-Net does 7.6 to 9.8 points
    better on every seed. The way down shrank the maps to 8 by 8, so each
    value there covers a 4 by 4 block of the image. Without skips, the way up
    has to guess where inside that 4 by 4 block each edge falls.

    `F.max_pool2d(mask, 3, stride=1, padding=1)` grows a mask by one pixel in
    every direction. Each output pixel is the maximum of the 3 by 3 square
    around it, so any pixel next to a marked one becomes marked. Negating the
    mask, taking the same max pool and negating the result gives the minimum
    instead. That shrinks the mask by one pixel in every direction. The edge
    is the set of pixels where the grown mask and the shrunk mask differ.
    """,
)

UNET2_SOURCE = source_file(
    '''
    # unet2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    mask = torch.zeros(1, 1, 5, 5)
    mask[0, 0, 2, 2] = 1.0
    grown = F.max_pool2d(mask, 3, stride=1, padding=1)    # now the middle 3 by 3 square is marked


    def block(
        inputs: int,
        outputs: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.ReLU(),
        )


    # TODO: Finish forward like TinyUNet's in unet1, but with no skips. The way
    #   down is the same. On the way up, upsample c by 2 with F.interpolate in
    #   its default nearest mode and pass it through up2. Then upsample up2's
    #   output the same way and pass it through up1. Return out applied to
    #   up1's output. Nothing is concatenated.
    class NoSkipUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down1 = block(1, 8)
            self.down2 = block(8, 16)
            self.bottom = block(16, 32)
            self.up2 = block(32, 16)
            self.up1 = block(16, 8)
            self.out = nn.Conv2d(8, 1, 1)

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return the IoU between the model's predicted mask and the true
    #   masks, as a Python float. The predicted mask marks the pixels where
    #   logits is above 0. masks holds 1 for marked pixels and 0 for the rest.
    #   Both tensors have shape (batch, 1, h, w). Count the pixels over the
    #   whole batch at once, not image by image.
    #   Example: if the model marks all 9 pixels of a 3 by 3 image and the true
    #   mask marks 3 of them, iou returns about 0.3333.
    def iou(
        logits: Tensor,
        masks: Tensor,
    ) -> float:
        ...


    # TODO: Return a tensor of True and False with the same shape as masks,
    #   which is (batch, 1, h, w). An entry is True where the mask grown by one
    #   pixel and the mask shrunk by one pixel differ. Those are the edge
    #   pixels.
    def edge_pixels(masks: Tensor) -> Tensor:
        ...
    ''',
)

UNET2_SOLUTION = source_file(
    '''
    # unet2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    mask = torch.zeros(1, 1, 5, 5)
    mask[0, 0, 2, 2] = 1.0
    grown = F.max_pool2d(mask, 3, stride=1, padding=1)


    def block(
        inputs: int,
        outputs: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.ReLU(),
        )


    class NoSkipUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down1 = block(1, 8)
            self.down2 = block(8, 16)
            self.bottom = block(16, 32)
            self.up2 = block(32, 16)
            self.up1 = block(16, 8)
            self.out = nn.Conv2d(8, 1, 1)

        def forward(self, x: Tensor) -> Tensor:
            a = self.down1(x)
            b = self.down2(F.max_pool2d(a, 2))
            c = self.bottom(F.max_pool2d(b, 2))
            u = self.up2(F.interpolate(c, scale_factor=2))
            u = self.up1(F.interpolate(u, scale_factor=2))

            return self.out(u)


    def iou(
        logits: Tensor,
        masks: Tensor,
    ) -> float:
        marked = logits > 0
        truth = masks > 0.5

        return ((marked & truth).sum() / (marked | truth).sum()).item()


    def edge_pixels(masks: Tensor) -> Tensor:
        grown = F.max_pool2d(masks, 3, stride=1, padding=1)
        shrunk = -F.max_pool2d(-masks, 3, stride=1, padding=1)

        return grown != shrunk
    ''',
)


class ReferenceNoSkip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down1 = block(1, 8)
        self.down2 = block(8, 16)
        self.bottom = block(16, 32)
        self.up2 = block(32, 16)
        self.up1 = block(16, 8)
        self.out = nn.Conv2d(8, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        a = self.down1(x)
        b = self.down2(F.max_pool2d(a, 2))
        c = self.bottom(F.max_pool2d(b, 2))
        u = self.up2(F.interpolate(c, scale_factor=2))
        u = self.up1(F.interpolate(u, scale_factor=2))

        return self.out(u)


def check_no_skip_forward(found: type) -> None:
    torch.manual_seed(830)
    model = expect_module("NoSkipUNet()", found())
    reference = ReferenceNoSkip()
    reference.load_state_dict(model.state_dict())
    x = torch.rand(2, 1, 16, 16, generator=torch.Generator().manual_seed(831))

    with torch.no_grad():
        expected = reference(x)

    expect_tensor(
        call="NoSkipUNet() on 2 images of 16 by 16",
        actual=model(x),
        expected=expected,
    )


def square_masks() -> Tensor:
    masks = torch.zeros(2, 1, 6, 6)
    masks[0, 0, 1:4, 1:4] = 1.0
    masks[1, 0, 2:6, 0:3] = 1.0

    return masks


def check_unet2(checker: Checker) -> None:
    from ..engine import expect_close

    checker.test_method(
        class_name="NoSkipUNet",
        method="forward",
        description="goes down and up with nothing joined on",
        run=check_no_skip_forward,
    )
    check_on_inputs(
        checker=checker,
        function="iou",
        description="divides the overlap by the pixels either mask marks",
        inputs=(
            (small_tensor(832, 2, 1, 6, 6), square_masks()),
            (torch.ones(1, 1, 3, 3), torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]])),
            # Logits between 0 and 1 catch a cut at 0.5, which whole numbers can't tell from a cut at 0.
            (small_tensor(833, 2, 1, 6, 6) / 4, square_masks()),
        ),
        reference=lambda logits, masks: (((logits > 0) & (masks > 0.5)).sum() / ((logits > 0) | (masks > 0.5)).sum()).item(),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="edge_pixels",
        description="marks where the grown and shrunk masks differ",
        inputs=(
            (square_masks(),),
        ),
        reference=lambda masks: F.max_pool2d(masks, 3, stride=1, padding=1) != -F.max_pool2d(-masks, 3, stride=1, padding=1),
        expect=expect_tensor,
    )


UNET2 = Exercise(
    name="unet2",
    chapter=UPSAMPLING,
    title="Why U-Net's skips help",
    lesson=UNET2_LESSON,
    source=UNET2_SOURCE,
    solution=UNET2_SOLUTION,
    hints={
        "NoSkipUNet.forward": "Compute a, b and c as in TinyUNet. Then compute self.up2(F.interpolate(c, "
        "scale_factor=2)), do the same with up1, and apply self.out at the end.",
        "iou": "Compute logits > 0 and masks > 0.5. Divide the sum of their & by the sum of their |, then call "
        ".item() to get a float.",
        "edge_pixels": "Grow the masks with F.max_pool2d(masks, 3, stride=1, padding=1). Shrink them by negating "
        "the same max pool of -masks. Compare the two results with !=.",
    },
    introduces=(),
    check=check_unet2,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def made_up_shapes(
    count: int,
    generator: torch.Generator,
    noise: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """count images of 32 by 32, each with 1 to 3 circles or squares of radius 3 to 8 at 1 on a background of 0,
    plus noise of the given spread, and their masks of 0 and 1."""
    rows = torch.arange(32).view(32, 1).float()
    columns = torch.arange(32).view(1, 32).float()
    masks = torch.zeros(count, 1, 32, 32)

    for index in range(count):
        for _ in range(int(torch.randint(1, 4, (1,), generator=generator))):
            radius = float(torch.randint(3, 9, (1,), generator=generator))
            middle_row, middle_column = (torch.rand(2, generator=generator) * (32 - 2 * radius) + radius).tolist()

            if torch.rand(1, generator=generator) < 0.5:
                shape = (rows - middle_row) ** 2 + (columns - middle_column) ** 2 <= radius**2
            else:
                shape = ((rows - middle_row).abs() <= radius) & ((columns - middle_column).abs() <= radius)

            masks[index, 0] = torch.maximum(masks[index, 0], shape.float())

    images = masks + noise * torch.randn(count, 1, 32, 32, generator=generator)

    return images, masks


def mask_scores(
    model: nn.Module,
    images: Tensor,
    masks: Tensor,
) -> tuple[float, float, float]:
    """The share of pixels right, the share of edge pixels right, and the IoU, over all of images at once."""
    model.eval()

    with torch.no_grad():
        marked = model(images) > 0

    truth = masks > 0.5
    edge = F.max_pool2d(masks, 3, stride=1, padding=1) != -F.max_pool2d(-masks, 3, stride=1, padding=1)
    right = marked == truth

    return (
        right.float().mean().item(),
        right[edge].float().mean().item(),
        ((marked & truth).sum() / (marked | truth).sum()).item(),
    )


def measure_unet(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, list[tuple[float, float, float]]]:
    """unet2's table: unet1's U-Net and the same with no skips, each trained 400 Adam steps at 0.001 on batches of 32
    from 1000 made up images, with binary cross entropy, then scored on 500 new ones, on the CPU with 4 threads. Gives
    the pixels right, the edge pixels right and the IoU, one triple per seed."""
    results: dict[str, list[tuple[float, float, float]]] = {}

    with pinned_threads():
        for seed in seeds:
            train_x, train_masks = made_up_shapes(1000, torch.Generator().manual_seed(seed))
            held_x, held_masks = made_up_shapes(500, torch.Generator().manual_seed(1000 + seed))

            for name, make in (("no skips", ReferenceNoSkip), ("U-Net", ReferenceUNet)):
                torch.manual_seed(seed)
                model = make()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                generator = torch.Generator().manual_seed(seed + 50)

                for _ in range(400):
                    batch = torch.randint(0, 1000, (32,), generator=generator)
                    loss = F.binary_cross_entropy_with_logits(model(train_x[batch]), train_masks[batch])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                results.setdefault(name, []).append(mask_scores(model, held_x, held_masks))

    return results
