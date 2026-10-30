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
from .linear_layers import expect_module

UPSAMPLING = Chapter(
    folder="34_upsampling",
    title="Upsampling and U-Net",
    readme=source_file(
        """
        # Upsampling and U-Net

        A model that answers with an image, like a mask of which pixels belong
        to an object, has to grow its small maps back to full size. This
        chapter builds nearest and bilinear upsampling and the transposed
        convolution, then the U-Net, whose skip connections carry the fine
        detail across.

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
    Some models answer with an image, like a mask that says which pixels
    belong to a cell or a road. Their maps shrink on the way in, so on the way
    out they have to grow back. That is **upsampling**.

    Nearest upsampling by `s` repeats every pixel `s` times down and across.
    Linear interpolation blends instead. A point a quarter of the way from `a`
    to `b` gets `0.75 * a + 0.25 * b`. Bilinear does that along the rows,
    then along the columns.

    ```python
    torch.tensor([1, 2]).repeat_interleave(3)    # tensor([1, 1, 1, 2, 2, 2])
    ```

    `F.interpolate(x, scale_factor=2, mode="bilinear")` takes `x` of shape
    `(batch, channels, h, w)`. Its `mode="nearest"` repeats instead.

    `align_corners=True` makes the first and last outputs land exactly on the
    first and last inputs. Then the output of `n` inputs grown `s` times has
    `(n - 1) * s + 1` points, and output `j` sits at input position `j / s`.
    `.floor()` rounds down and `.long()` makes whole numbers usable as
    positions.
    """,
)

UPSAMPLE1_SOURCE = source_file(
    '''
    # upsample1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    repeated = torch.tensor([1, 2]).repeat_interleave(3)    # tensor([1, 1, 1, 2, 2, 2])


    # TODO: Return image, of shape (h, w), grown s times down and across by
    #   repeating every pixel.
    def nearest(
        image: Tensor,
        s: int,
    ) -> Tensor:
        ...


    # TODO: Return x, of shape (batch, channels, h, w), grown s times with
    #   bilinear upsampling, using F.interpolate.
    def bilinear(
        x: Tensor,
        s: int,
    ) -> Tensor:
        ...


    # TODO: Finish the last line, so this returns x grown s times by linear
    #   interpolation, with the ends on the ends. Each output blends the input
    #   at left and the one after it, by fraction.
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
        "bilinear": "F.interpolate with scale_factor=s and mode=\"bilinear\".",
        "linear_1d": "Blend x[left] and x[left + 1], weighting the second by fraction and the first by 1 - fraction.",
    },
    introduces=(),
    check=check_upsample1,
)


# ---------------------------------------------------------------------------
# transposed1
# ---------------------------------------------------------------------------

TRANSPOSED1_LESSON = prose(
    """
    A **transposed convolution** grows a map with learned weights. Each input
    value stamps a copy of the kernel, scaled by that value, into the output.
    The stamp for input `i` starts at position `i * s`, and where stamps
    overlap they add up. The output ends where the last stamp ends.

    ```python
    # x = [1, 2], w = [1, 1, 1], s = 2
    # stamp of 1 at 0: [1, 1, 1, 0, 0]
    # stamp of 2 at 2: [0, 0, 2, 2, 2]
    # sum:             [1, 1, 3, 2, 2]
    ```

    It is the backward pass of a convolution. Each convolution output read
    `k` inputs, so its gradient flows back to those same `k` inputs, weighted
    by the kernel. That is the stamp. Some call it deconvolution, though it
    undoes nothing.

    To hand a chosen gradient `g` to outputs `y`, call `.backward()` on
    `(y * g).sum()`. The derivative of that sum by each output is its entry of
    `g`, so exactly `g` flows back from the outputs.

    `nn.ConvTranspose2d(in, out, k, stride=2)` is the layer. With a kernel
    that the stride doesn't divide, like 3 at stride 2, the stamps overlap
    unevenly and leave a checkerboard pattern. So many models upsample with
    interpolation, then a plain convolution, as Odena and others advised.
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
    def transposed_size(
        n: int,
        k: int,
        s: int,
    ) -> int:
        ...


    # TODO: Finish the line in the loop, so each input stamps w, scaled by its
    #   value, into output from position i * s, adding where stamps overlap.
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


    # TODO: Return the gradient of a convolution of x with w, at stride 1, with
    #   respect to x, when the gradient handed down to the outputs is grad.
    #   Work it out with autograd and the sum the lesson shows.
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


    # The gradient handed down, times the outputs, summed, gives that gradient back through .backward().
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
        "transposed_size": "The last stamp starts at (n - 1) * s and is k long.",
        "transposed_1d": "Add x[i] * w to output[i * s:i * s + k], as the line that sets output does.",
        "conv_input_gradient": "Clone x and make it require a gradient. Run F.conv1d on it with the shapes of conv2, "
        "multiply the flat outputs by grad, sum, call .backward(), and return x's .grad.",
    },
    introduces=(),
    check=check_transposed1,
)


# ---------------------------------------------------------------------------
# unet1
# ---------------------------------------------------------------------------

UNET1_LESSON = prose(
    """
    U-Net, from Ronneberger, Fischer and Brox in 2015, segments cells in
    microscope images. It has two halves, drawn as the two arms of a U.

    - the way down: blocks of two 3 by 3 convolutions, each block followed by a max pool of 2.
    - the way up: upsampling by 2, then a block, back to full size, then a 1 by 1 convolution to one score per pixel.
    - the skips: each level's maps on the way down are joined onto the same level's maps on the way up.

    The way down learns what is in the image but loses where exactly. The
    skips hand the exact places back to the way up.

    Joining is concatenating along the channels, so an upsampled map of 32
    channels and a skip of 16 give 48. `torch.cat([upsampled, skip], dim=1)`
    does that for tensors of shape `(batch, channels, h, w)`, the upsampled
    channels first. The order matters, since the next block's weights expect
    each channel in its place.

    The paper used convolutions without padding, so its maps shrank a little
    in each block. A skip's map was then larger than the map it joined, and
    was cut down to the middle part first. Most versions now pad instead.
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


    # TODO: Return the middle size by size part of x, of shape
    #   (batch, channels, h, w), the way the paper cut its skips. Assume h - size
    #   and w - size are even.
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


    # TODO: Finish forward. a is down1 of x. b is down2 of a after a max pool of
    #   2, and c is bottom of b after another. The way up upsamples c by 2 with
    #   F.interpolate in its default nearest mode, joins b after it with
    #   torch.cat([upsampled, b], dim=1), and runs up2. It then upsamples that
    #   the same way, joins a after it, runs up1, and returns out of it.
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
        "TinyUNet.forward": "a, b and c come from down1, down2 and bottom with F.max_pool2d(..., 2) between. Then "
        "self.up2(torch.cat([F.interpolate(c, scale_factor=2), b], dim=1)), and the same with up1 and a.",
    },
    introduces=(),
    check=check_unet1,
)


# ---------------------------------------------------------------------------
# unet2
# ---------------------------------------------------------------------------

UNET2_LESSON = prose(
    """
    Here two models learned to mark made up shapes, circles and squares on
    noisy images of 32 by 32, from 1000 examples for 400 steps. One was
    unet1's U-Net. The other was the same with no skips, its way up fed only
    from below.

    | Model | Pixels right | Pixels on an edge right | IoU |
    |---|---|---|---|
    | no skips | 98.5% | 86.2% | 0.876 |
    | U-Net | 99.1% | 92.2% | 0.927 |

    Most pixels are easy, far inside or far outside a shape, so the share of
    all pixels right hides the gap. On the pixels next to an edge the skips
    win by 6 points. The way down shrank the maps to 8 by 8. Without skips,
    the way up has to guess where each edge falls inside a block of 4 by 4.

    **IoU**, intersection over union, scores a mask. It is the count of pixels
    both masks mark, divided by the count of pixels either marks, 1 for a
    perfect match. `&` and `|` are and and or on tensors of `True` and
    `False`.

    `F.max_pool2d(mask, 3, stride=1, padding=1)` grows a mask by one pixel all
    around, since a pixel next to a marked one becomes marked. Minus the max
    pool of minus the mask shrinks it the same way. The pixels where the grown
    and the shrunk masks differ are the edge.
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
    grown = F.max_pool2d(mask, 3, stride=1, padding=1)    # the middle 3 by 3 marked


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


    # TODO: Finish forward like TinyUNet's in unet1, with no skips. The way up
    #   upsamples c by 2 with F.interpolate in its default nearest mode and runs
    #   up2, then upsamples that the same way and runs up1. Nothing is joined on.
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


    # TODO: Return the IoU of the mask the model marks, where logits are above 0,
    #   against masks of 0 and 1, as a plain number. Count the pixels over the
    #   whole batch at once, not image by image.
    def iou(
        logits: Tensor,
        masks: Tensor,
    ) -> float:
        ...


    # TODO: Return a tensor of True and False, the same shape as masks, of shape
    #   (batch, 1, h, w), True on the pixels where the grown and the shrunk masks
    #   differ.
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
        description="divides the overlap by what either marks",
        inputs=(
            (small_tensor(832, 2, 1, 6, 6), square_masks()),
            (torch.ones(1, 1, 3, 3), torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]])),
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
    title="Why the skips",
    lesson=UNET2_LESSON,
    source=UNET2_SOURCE,
    solution=UNET2_SOLUTION,
    hints={
        "NoSkipUNet.forward": "The same a, b and c as TinyUNet. Then self.up2(F.interpolate(c, scale_factor=2)), "
        "the same with up1, and self.out at the end.",
        "iou": "Take logits > 0 and masks > 0.5. Divide the sum of their & by the sum of their |, and use .item().",
        "edge_pixels": "Grow with F.max_pool2d(masks, 3, stride=1, padding=1), shrink with minus that of -masks, "
        "and compare the two with !=.",
    },
    introduces=(),
    check=check_unet2,
)
