"""44_causal_conv, convolutions that never see the future, WaveNet's dilated stacks and why they suit long signals, and the short time Fourier transform."""

from __future__ import annotations

import textwrap
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
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
from .bidirectional import expect_nothing_recorded
from .linear_layers import expect_module

CAUSAL_CONV = Chapter(
    folder="44_causal_conv",
    title="Causal convolutions",
    readme=source_file(
        """
        # Causal convolutions

        A convolution can read a sequence too, every position at once. This
        chapter pads it so no output sees the future, stacks dilated layers
        the way WaveNet does to see far back with few layers, and shows why
        that suits long signals like audio. It ends with the short time
        Fourier transform, the spectra of a signal's overlapping windows,
        which most audio models read.

        ## Further reading

        - [van den Oord and colleagues, WaveNet](https://arxiv.org/abs/1609.03499)
        - [Bai, Kolter and Koltun, convolutional and recurrent networks for sequence modeling](https://arxiv.org/abs/1803.01271)
        - [PyTorch, torch.stft](https://docs.pytorch.org/docs/2.14/generated/torch.stft.html)
        """,
    ),
)


def fixed_kernel(
    seed: int,
    k: int,
) -> Tensor:
    """A random kernel of shape (1, 1, k) with no entry near 0, so every input it reads gets a gradient."""
    signs = torch.randint(0, 2, (1, 1, k), generator=torch.Generator().manual_seed(seed)) * 2 - 1

    return signs * (0.5 + torch.rand(1, 1, k, generator=torch.Generator().manual_seed(seed + 1)))


def reference_inputs_used(
    conv: Callable[[Tensor], Tensor],
    length: int,
    position: int,
) -> list[int]:
    x = torch.ones(1, 1, length, requires_grad=True)
    conv(x)[..., position].sum().backward()

    return x.grad.flatten().nonzero().flatten().tolist()


# ---------------------------------------------------------------------------
# causal1
# ---------------------------------------------------------------------------

CAUSAL1_LESSON = prose(
    """
    The models of Section 6 so far read a sequence with a loop, one step at a
    time. A convolution reads it too, every position at once. To predict what
    comes next, though, no output may see a later input.

    `F.conv1d` with a kernel of 3 and `padding=1` makes output t from inputs
    t - 1, t and t + 1. So output t sees input t + 1, the future. A model
    trained this way to predict the next step would copy it and learn nothing
    useful.

    A **causal convolution** adds its k - 1 zeros on the left only. Then
    output t comes from inputs t - k + 1 up to t, and the output keeps the
    input's length.

    ```python
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    F.pad(x, (2, 0))    # tensor([[[0., 0., 1., 2., 3.]]])
    ```

    `F.pad` takes how many zeros to add at the start and at the end of the
    last dimension.

    Autograd can check which inputs an output uses. Make the input require
    grad, backprop one output, and the input's gradient is 0 everywhere that
    output never looked.

    | Convolution, k = 3, on 10 inputs | Inputs output 5 uses |
    |---|---|
    | `padding=1`, a zero on each side | 4, 5, 6 |
    | causal, two zeros on the left | 3, 4, 5 |

    `.nonzero()` gives the positions of the entries that aren't 0, one row
    each.

    ```python
    torch.tensor([0.0, 2.0, 0.0, -1.0]).nonzero()    # tensor([[1], [3]])
    ```

    This check is worth keeping. A model that sees the future trains to a
    loss that looks wonderful, then fails the moment it has to predict.
    """,
)

CAUSAL1_SOURCE = source_file(
    '''
    # causal1.py

    from collections.abc import Callable

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    padded = F.pad(x, (2, 0))    # tensor([[[0., 0., 1., 2., 3.]]])

    positions = torch.tensor([0.0, 2.0, 0.0, -1.0]).nonzero().flatten().tolist()    # [1, 3]


    # TODO: Return the causal convolution of x, of shape (batch, channels,
    #   length), with weight, of shape (out_channels, channels, k), and bias.
    #   Add zeros on the left only, so the output keeps the input's length.
    def causal_conv1d(
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the positions of the inputs that output position of conv
    #   uses, as a sorted list. Start from torch.ones(1, 1, length) that
    #   requires grad, run conv on it, and backprop that one output position.
    def inputs_used(
        conv: Callable[[Tensor], Tensor],
        length: int,
        position: int,
    ) -> list[int]:
        ...


    # TODO: Return True when some output of conv uses an input after its own
    #   position, for inputs of the given length, and False otherwise.
    def sees_future(
        conv: Callable[[Tensor], Tensor],
        length: int,
    ) -> bool:
        ...
    ''',
)

CAUSAL1_SOLUTION = source_file(
    '''
    # causal1.py, a solution for comparison

    from collections.abc import Callable

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    padded = F.pad(x, (2, 0))

    positions = torch.tensor([0.0, 2.0, 0.0, -1.0]).nonzero().flatten().tolist()


    def causal_conv1d(
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        k = weight.shape[2]

        return F.conv1d(F.pad(x, (k - 1, 0)), weight, bias)


    def inputs_used(
        conv: Callable[[Tensor], Tensor],
        length: int,
        position: int,
    ) -> list[int]:
        x = torch.ones(1, 1, length, requires_grad=True)
        conv(x)[..., position].sum().backward()

        return x.grad.flatten().nonzero().flatten().tolist()


    def sees_future(
        conv: Callable[[Tensor], Tensor],
        length: int,
    ) -> bool:
        for position in range(length):
            if max(inputs_used(conv, length, position)) > position:
                return True

        return False
    ''',
)

KERNEL3 = fixed_kernel(440, 3)
KERNEL4 = fixed_kernel(442, 4)

CONVS: tuple[Described, ...] = (
    Described(
        lambda x: F.conv1d(x, KERNEL3, padding=1),
        "a convolution with k = 3 and padding=1",
    ),
    Described(
        lambda x: F.conv1d(F.pad(x, (2, 0)), KERNEL3),
        "a causal convolution with k = 3",
    ),
    Described(
        lambda x: F.conv1d(F.pad(x, (3, 0)), KERNEL4, dilation=1),
        "a causal convolution with k = 4",
    ),
    Described(
        lambda x: F.conv1d(F.pad(x, (1, 2)), KERNEL4),
        "a convolution with k = 4, one zero on the left and two on the right",
    ),
)


LATE_LEAK = Described(
    lambda x: torch.cat(
        [
            F.conv1d(F.pad(x, (2, 0)), KERNEL3)[..., :4],
            F.conv1d(x, KERNEL3, padding=1)[..., 4:],
        ],
        dim=-1,
    ),
    "a convolution with k = 3, causal for outputs 0 to 3 and padding=1 from output 4 on",
)


def check_causal1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="causal_conv1d",
        description="adds zeros on the left and keeps the length",
        inputs=(
            (small_tensor(441, 2, 1, 6), small_tensor(443, 1, 1, 3), torch.tensor([0.5])),
            (small_tensor(444, 1, 2, 7), small_tensor(445, 3, 2, 4), small_tensor(446, 3)),
        ),
        reference=lambda x, weight, bias: F.conv1d(F.pad(x, (weight.shape[2] - 1, 0)), weight, bias),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="inputs_used",
        description="finds the inputs one output uses through its gradient",
        inputs=(
            (CONVS[0], 10, 5),
            (CONVS[1], 10, 5),
            (CONVS[2], 9, 2),
            (CONVS[3], 8, 0),
        ),
        reference=reference_inputs_used,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="sees_future",
        description="tells whether any output uses a later input",
        inputs=tuple((conv, 8) for conv in CONVS + (LATE_LEAK,)),
        reference=lambda conv, length: any(
            max(reference_inputs_used(conv, length, position)) > position for position in range(length)
        ),
        expect=expect_value,
    )


CAUSAL1 = Exercise(
    name="causal1",
    chapter=CAUSAL_CONV,
    title="A convolution that can't see ahead",
    lesson=CAUSAL1_LESSON,
    source=CAUSAL1_SOURCE,
    solution=CAUSAL1_SOLUTION,
    hints={
        "causal_conv1d": "k is weight.shape[2]. F.pad with (k - 1, 0) adds the zeros on the left, then call F.conv1d "
        "with weight and bias.",
        "inputs_used": "Make x with requires_grad=True. Call .sum() on conv(x)[..., position], then .backward(). "
        "x.grad.flatten().nonzero() gives the positions, and .flatten().tolist() makes them a list.",
        "sees_future": "Loop over every position. If the largest input inputs_used finds is after the position, "
        "return True. Return False after the loop.",
    },
    introduces=(),
    check=check_causal1,
)


# ---------------------------------------------------------------------------
# dilated1
# ---------------------------------------------------------------------------

DILATED1_LESSON = prose(
    """
    A causal convolution with a kernel of 2 sees 2 steps. Stack 10 of them
    and the top layer sees 11. Seeing 1,000 steps back would take 999 layers.

    A **dilated** convolution spaces its kernel's taps apart. With a dilation
    of d, a kernel of 2 reads inputs t - d and t. `nn.Conv1d` takes
    `dilation=d`, and the zeros on the left grow to (k - 1) times d.

    ```python
    conv = nn.Conv1d(1, 1, 2, dilation=4)    # reads inputs t - 4 and t
    ```

    WaveNet stacks such layers with dilations 1, 2, 4, 8 and so on. Each
    layer doubles how far back the top can see.

    | Layers | Dilations | Steps the top sees |
    |---|---|---|
    | 1 | 1 | 2 |
    | 2 | 1, 2 | 4 |
    | 3 | 1, 2, 4 | 8 |
    | 10 | 1 up to 512 | 1,024 |

    The span of inputs one output depends on is its **receptive field**. A
    layer with kernel k and dilation d widens it by (k - 1) times d. So a
    stack's receptive field is 1 plus that amount summed over its layers.

    Each layer here also adds its input back, as `x + relu(layer(x))`. That is
    the residual of Section 3, so the signal and its gradient pass through a
    deep stack.

    `nn.ModuleList` holds a list of layers, so their weights count as the
    model's own and `.parameters()` finds them. Loop over it in `forward`.
    """,
)

DILATED1_SOURCE = source_file(
    '''
    # dilated1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    widening = (2 - 1) * 4      # 4, what a layer with k = 2 and dilation 4 adds
    layers = nn.ModuleList([nn.Conv1d(1, 1, 2), nn.Conv1d(1, 1, 2)])    # two layers the model owns


    # TODO: Return the receptive field of a stack of causal convolutions, each
    #   with kernel_size taps, with the given dilations, one per layer.
    def receptive_field(
        kernel_size: int,
        dilations: list[int],
    ) -> int:
        ...


    # TODO: Set self.left to the zeros a causal convolution with this
    #   kernel_size and dilation needs on the left. Then finish forward, which
    #   adds them and runs self.conv, so the output keeps the input's length.
    class CausalConv1d(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
        ) -> None:
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.left = ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Set self.layers to an nn.ModuleList holding `layers` CausalConv1d
    #   layers from channels to channels with a kernel of 2, the first with
    #   dilation 1 and each next one doubling it. forward sets x to x + relu(layer(x)) for
    #   each layer in turn, and returns x.
    class DilatedStack(nn.Module):
        def __init__(
            self,
            channels: int,
            layers: int,
        ) -> None:
            super().__init__()
            self.layers = ...

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

DILATED1_SOLUTION = source_file(
    '''
    # dilated1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    widening = (2 - 1) * 4
    layers = nn.ModuleList([nn.Conv1d(1, 1, 2), nn.Conv1d(1, 1, 2)])


    def receptive_field(
        kernel_size: int,
        dilations: list[int],
    ) -> int:
        return 1 + (kernel_size - 1) * sum(dilations)


    class CausalConv1d(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
        ) -> None:
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.left = (kernel_size - 1) * dilation

        def forward(self, x: Tensor) -> Tensor:
            return self.conv(F.pad(x, (self.left, 0)))


    class DilatedStack(nn.Module):
        def __init__(
            self,
            channels: int,
            layers: int,
        ) -> None:
            super().__init__()
            self.layers = nn.ModuleList([CausalConv1d(channels, channels, 2, 2 ** index) for index in range(layers)])

        def forward(self, x: Tensor) -> Tensor:
            for layer in self.layers:
                x = x + torch.relu(layer(x))

            return x
    ''',
)


class ReferenceCausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.left = (kernel_size - 1) * dilation

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.left, 0)))


class ReferenceDilatedStack(nn.Module):
    def __init__(
        self,
        channels: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ReferenceCausalConv1d(channels, channels, 2, 2 ** index) for index in range(layers)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = x + torch.relu(layer(x))

        return x


def seeded_module(
    make: type,
    *arguments: int,
) -> nn.Module:
    torch.manual_seed(450)

    return make(*arguments)


def check_causal_layer(found: type) -> None:
    for arguments, length in (((1, 2, 2, 1), 6), ((2, 3, 3, 2), 9)):
        call = f"CausalConv1d{arguments}"
        layer = expect_module(call, seeded_module(found, *arguments))
        x = small_tensor(451, 2, arguments[0], length)

        with torch.no_grad():
            expected = seeded_module(ReferenceCausalConv1d, *arguments)(x)

        expect_tensor(
            call=f"{call} on input of shape (2, {arguments[0]}, {length})",
            actual=layer(x),
            expected=expected,
        )


def check_stack_forward(found: type) -> None:
    for channels, layers in ((2, 3), (3, 4)):
        call = f"DilatedStack({channels}, {layers})"
        stack = expect_module(call, seeded_module(found, channels, layers))
        x = small_tensor(452, 2, channels, 20) / 3

        with torch.no_grad():
            expected = seeded_module(ReferenceDilatedStack, channels, layers)(x)

        expect_tensor(
            call=f"{call} on input of shape (2, {channels}, 20)",
            actual=stack(x),
            expected=expected,
        )


def check_stack_view(found: type) -> None:
    stack = expect_module("DilatedStack(1, 4)", seeded_module(found, 1, 4))
    layers = getattr(stack, "layers", None)
    dilations = [layer.conv.dilation[0] for layer in layers] if isinstance(layers, nn.ModuleList) else None

    if dilations != [1, 2, 4, 8]:
        raise Mismatch(
            f"DilatedStack(1, 4) should hold an nn.ModuleList of layers with dilations [1, 2, 4, 8], so the top sees "
            f"16 steps, but its dilations are {dilations}.",
        )


def check_dilated1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="receptive_field",
        description="adds (k - 1) times each dilation to 1",
        inputs=(
            (2, [1, 2, 4, 8]),
            (3, [1, 2, 4]),
            (2, [1, 1, 1]),
            (4, [1, 3, 9]),
        ),
        reference=lambda kernel_size, dilations: 1 + (kernel_size - 1) * sum(dilations),
        expect=expect_value,
    )
    checker.test_method(
        class_name="CausalConv1d",
        method="forward",
        description="adds (k - 1) times d zeros on the left, then convolves",
        run=check_causal_layer,
    )
    checker.test_method(
        class_name="DilatedStack",
        method="forward",
        description="runs each dilated layer with its input added back",
        run=check_stack_forward,
    )
    checker.test_method(
        class_name="DilatedStack",
        method="forward",
        description="sees 16 inputs back with 4 layers",
        run=check_stack_view,
    )


DILATED1 = Exercise(
    name="dilated1",
    chapter=CAUSAL_CONV,
    title="Dilated stacks",
    lesson=DILATED1_LESSON,
    source=DILATED1_SOURCE,
    solution=DILATED1_SOLUTION,
    hints={
        "receptive_field": "Add up the dilations, multiply by kernel_size - 1, and add 1.",
        "CausalConv1d.forward": "self.left is (kernel_size - 1) * dilation. In forward, F.pad x with (self.left, 0) "
        "and pass it to self.conv.",
        "DilatedStack.forward": "The dilation of layer number index is 2 ** index. In forward, loop over self.layers "
        "and set x to x + torch.relu(layer(x)).",
    },
    introduces=(),
    check=check_dilated1,
)


# ---------------------------------------------------------------------------
# wavenet1
# ---------------------------------------------------------------------------

WAVENET1_LESSON = prose(
    """
    Two claims make WaveNet's stack worth its parts. It sees far back with
    few layers, and it trains on every position at once.

    The task here shifts a signal of noise, so output t should equal input
    t - 10. A model has to see 10 steps back to do that. Here are two stacks
    of 4 layers, each trained for 150 steps.

    | Stack | Steps the top sees | Mean squared error |
    |---|---|---|
    | dilations 1, 2, 4, 8 | 16 | 0.0006 |
    | dilations 1, 1, 1, 1 | 5 | 1.007 |

    An error of 1 is what always guessing 0 gets on this noise, so the plain
    stack learned nothing. Input t - 10 is outside its view, and no amount of
    training brings it in. Seeing 1,000 steps back takes 999 plain layers, or
    10 dilated ones.

    An RNN can see that far too, but it computes step t only once step t - 1
    is done. Over 1,000 steps that is 1,000 small steps one after another,
    forward and again backward. A dilated stack that sees 1,024 steps has 10
    layers, and each one handles every position in a single call.

    So causal convolutions suit long signals, like audio at 16,000 samples a
    second, which is what WaveNet generated.
    """,
)

SHIFTER_CODE = '''
# CausalConv1d from dilated1.
class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.left = (kernel_size - 1) * dilation

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.left, 0)))


# A stack of 8 channels that reads a signal of shape (batch, length) and writes
# one of the same shape, with one layer per dilation.
class Shifter(nn.Module):
    def __init__(self, dilations: list[int]) -> None:
        super().__init__()
        self.read = nn.Conv1d(1, 8, 1)
        self.layers = nn.ModuleList([CausalConv1d(8, 8, 2, dilation) for dilation in dilations])
        self.write = nn.Conv1d(8, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.read(x.unsqueeze(1))

        for layer in self.layers:
            h = h + torch.relu(layer(h))

        return self.write(h).squeeze(1)
'''

INDENTED_SHIFTER = textwrap.indent(SHIFTER_CODE.strip("\n"), "    ")

WAVENET1_SOURCE = source_file(
    '''
    # wavenet1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


''' + INDENTED_SHIFTER + '''


    fourth = 2 ** 3                                  # 8, the dilation of the fourth layer
    later = F.pad(torch.tensor([[1.0, 2.0, 3.0]]), (1, 0))[:, :3]    # tensor([[0., 1., 2.]])


    # TODO: Return how many causal layers with kernel_size taps it takes for
    #   the top to see at least view steps. Plain layers all have dilation 1.
    #   Dilated ones have 1, 2, 4 and so on.
    def layers_for_view(
        view: int,
        kernel_size: int,
        dilated: bool,
    ) -> int:
        ...


    # TODO: Return x, of shape (batch, length), shifted delay steps later, so
    #   position t holds x[:, t - delay] and the first delay positions hold 0.
    def delayed(
        x: Tensor,
        delay: int,
    ) -> Tensor:
        ...


    # TODO: Finish the line that sets loss, the mean squared error between the
    #   model's output on x and x delayed by delay, so this trains with Adam.
    def train_shifter(
        model: nn.Module,
        delay: int,
        steps: int,
        generator: torch.Generator,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        for _ in range(steps):
            x = torch.randn(16, 32, generator=generator)
            loss = ...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()


    # TODO: Return the mean squared error between the model's output on x and x
    #   delayed by delay, over positions delay onwards only, as a plain number,
    #   without recording for autograd.
    def shift_error(
        model: nn.Module,
        x: Tensor,
        delay: int,
    ) -> float:
        ...
    ''',
)

WAVENET1_SOLUTION = source_file(
    '''
    # wavenet1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


''' + INDENTED_SHIFTER + '''


    fourth = 2 ** 3
    later = F.pad(torch.tensor([[1.0, 2.0, 3.0]]), (1, 0))[:, :3]


    def layers_for_view(
        view: int,
        kernel_size: int,
        dilated: bool,
    ) -> int:
        layers = 0
        seen = 1

        while seen < view:
            seen += (kernel_size - 1) * (2 ** layers if dilated else 1)
            layers += 1

        return layers


    def delayed(
        x: Tensor,
        delay: int,
    ) -> Tensor:
        return F.pad(x, (delay, 0))[:, : x.shape[1]]


    def train_shifter(
        model: nn.Module,
        delay: int,
        steps: int,
        generator: torch.Generator,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        for _ in range(steps):
            x = torch.randn(16, 32, generator=generator)
            loss = F.mse_loss(model(x), delayed(x, delay))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()


    def shift_error(
        model: nn.Module,
        x: Tensor,
        delay: int,
    ) -> float:
        with torch.no_grad():
            return F.mse_loss(model(x)[:, delay:], delayed(x, delay)[:, delay:]).item()
    ''',
)


class ReferenceShifter(nn.Module):
    def __init__(self, dilations: list[int]) -> None:
        super().__init__()
        self.read = nn.Conv1d(1, 8, 1)
        self.layers = nn.ModuleList([ReferenceCausalConv1d(8, 8, 2, dilation) for dilation in dilations])
        self.write = nn.Conv1d(8, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.read(x.unsqueeze(1))

        for layer in self.layers:
            h = h + torch.relu(layer(h))

        return self.write(h).squeeze(1)


def seeded_shifter(dilations: list[int]) -> nn.Module:
    torch.manual_seed(0)

    return ReferenceShifter(dilations)


def reference_layers_for_view(
    view: int,
    kernel_size: int,
    dilated: bool,
) -> int:
    layers = 0
    seen = 1

    while seen < view:
        seen += (kernel_size - 1) * (2 ** layers if dilated else 1)
        layers += 1

    return layers


def reference_delayed(
    x: Tensor,
    delay: int,
) -> Tensor:
    return F.pad(x, (delay, 0))[:, : x.shape[1]]


def reference_train_shifter(
    model: nn.Module,
    delay: int,
    steps: int,
    generator: torch.Generator,
) -> float:
    """The run the lesson's table comes from, with steps=150 and a generator seeded with 1."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    for _ in range(steps):
        x = torch.randn(16, 32, generator=generator)
        loss = F.mse_loss(model(x), reference_delayed(x, delay))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def reference_shift_error(
    model: nn.Module,
    x: Tensor,
    delay: int,
) -> float:
    with torch.no_grad():
        return F.mse_loss(model(x)[:, delay:], reference_delayed(x, delay)[:, delay:]).item()


def check_shifter_learns(found: object) -> None:
    model = seeded_shifter([1, 2, 4, 8])
    found(model, 10, 150, torch.Generator().manual_seed(1))
    error = reference_shift_error(model, torch.randn(256, 32, generator=torch.Generator().manual_seed(99)), 10)

    if error > 0.05:
        raise Mismatch(
            f"After train_shifter for 150 steps, the dilated stack's error on new signals is {error:.3f}, "
            "but it should be below 0.05.",
        )


def check_wavenet1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="layers_for_view",
        description="counts the layers it takes to see that far",
        inputs=(
            (1000, 2, False),
            (1000, 2, True),
            (16, 2, True),
            (17, 2, True),
            (50, 3, False),
            (50, 3, True),
            (1, 2, True),
        ),
        reference=reference_layers_for_view,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="delayed",
        description="shifts each row later, with zeros at the start",
        inputs=(
            (small_tensor(460, 2, 6), 2),
            (small_tensor(461, 3, 5), 4),
            (small_tensor(462, 1, 4), 0),
        ),
        reference=reference_delayed,
        expect=expect_tensor,
    )
    checker.test(
        function="train_shifter",
        description="trains with Adam on the mean squared error",
        run=lambda found: expect_close(
            call="train_shifter for 5 steps with delay 3",
            actual=found(seeded_shifter([1, 2]), 3, 5, torch.Generator().manual_seed(463)),
            expected=reference_train_shifter(seeded_shifter([1, 2]), 3, 5, torch.Generator().manual_seed(463)),
            tolerance=1e-5,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="shift_error",
        description="measures the error from position delay onwards",
        inputs=(
            (Described(seeded_shifter([1, 2]), "an untrained Shifter([1, 2])"), small_tensor(464, 3, 12), 4),
            (Described(seeded_shifter([1]), "an untrained Shifter([1])"), small_tensor(465, 2, 8), 1),
        ),
        reference=reference_shift_error,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )
    watched = seeded_shifter([1, 2])
    checker.test(
        function="shift_error",
        description="records nothing for autograd",
        run=lambda found: expect_nothing_recorded(
            function="shift_error",
            modules=[watched],
            run=lambda: found(watched, small_tensor(464, 3, 12), 4),
        ),
    )
    checker.test(
        function="train_shifter",
        description="trains a dilated stack that shifts by 10",
        run=check_shifter_learns,
    )


WAVENET1 = Exercise(
    name="wavenet1",
    chapter=CAUSAL_CONV,
    title="Why dilated stacks",
    lesson=WAVENET1_LESSON,
    source=WAVENET1_SOURCE,
    solution=WAVENET1_SOLUTION,
    hints={
        "layers_for_view": "Start with layers = 0 and seen = 1. While seen is below view, add (kernel_size - 1) "
        "times the next layer's dilation to seen, and count the layer.",
        "delayed": "F.pad with (delay, 0) adds the zeros at the start. Then keep the first x.shape[1] positions.",
        "train_shifter": "F.mse_loss takes model(x) and delayed(x, delay).",
        "shift_error": "Inside a with torch.no_grad() block, slice both the output and the target with "
        "[:, delay:] before F.mse_loss, then call .item().",
    },
    introduces=(),
    check=check_wavenet1,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# stft1
# ---------------------------------------------------------------------------

STFT1_LESSON = prose(
    """
    fft1's spectrum covers a whole signal at once. It says which frequencies
    the signal holds, but not when. Swap the two halves of a signal and the
    strengths in its spectrum don't change at all.

    Take 512 samples of a wave with a period of 8, then 512 with a period of
    4. That is a low note followed by a high one. With the halves swapped,
    the high note comes first. `torch.fft.rfft(...).abs()` gives exactly the same
    strengths for both, with peaks at frequencies 128 and 256.

    The **short time Fourier transform**, or STFT, cuts the signal into short
    windows and takes the spectrum of each one.

    ```python
    torch.arange(6.0).unfold(0, 4, 2)    # tensor([[0., 1., 2., 3.], [2., 3., 4., 5.]])
    ```

    `x.unfold(0, size, step)` gives windows of `size` along dimension 0, one
    starting every `step`.

    A wave cut off sharply at a window's edges smears over many frequencies.
    So each window is first multiplied by a Hann window, which rises from 0
    to 1 and falls back. `torch.hann_window(n)` gives one of n samples. Take a
    wave of 10.5 cycles in 64 samples. Its four nearest frequencies hold 90.2
    percent of its energy without the Hann window, and 99.9 percent with it.

    The taper costs something. It shrinks each window's edges to near 0, so
    windows that only touched would all but drop the samples at every cut.
    Windows of 64 that start every 32 overlap by half. Their Hann windows add
    up to exactly 1 at every sample, so every sample counts the same.

    `torch.stft` does all of this and gives one column per window. A window
    of 64 has frequencies 0 to 32, so a period of 8 is frequency 8 and a
    period of 4 is frequency 16.

    | Signal | Loudest frequency of each window |
    |---|---|
    | low then high | 8 in the first 15 windows, then 16 in the last 16 |
    | high then low | 16 in the first 16 windows, then 8 in the last 15 |

    The STFT shows what and when. Its strengths form a spectrogram, an image
    of frequency against time. Most audio models read that image, or a
    version of it, and the echo canceller project later works on STFT
    windows too.
    """,
)

STFT1_SOURCE = source_file(
    '''
    # stft1.py

    import torch
    from torch import Tensor

    windows = torch.arange(6.0).unfold(0, 4, 2)    # tensor([[0., 1., 2., 3.], [2., 3., 4., 5.]])
    taper = torch.hann_window(8)                    # rises from 0 up to 1 at the middle, then falls


    # TODO: Return the windows of signal, each frame_length samples long, one
    #   starting every hop samples, as rows of shape (count, frame_length).
    def frames(
        signal: Tensor,
        frame_length: int,
        hop: int,
    ) -> Tensor:
        ...


    # TODO: Return the STFT of signal without torch.stft. Multiply each window
    #   by a Hann window, take its spectrum with torch.fft.rfft, and give one
    #   column per window, of shape (frame_length // 2 + 1, count).
    def stft_by_hand(
        signal: Tensor,
        frame_length: int,
        hop: int,
    ) -> Tensor:
        ...


    # TODO: Return the strengths of the STFT of signal from torch.stft, with
    #   n_fft=n_fft, hop_length=hop, a Hann window of n_fft samples,
    #   center=False and return_complex=True.
    def spectrogram(
        signal: Tensor,
        n_fft: int,
        hop: int,
    ) -> Tensor:
        ...


    # TODO: Return the loudest frequency of each window of strengths, whose
    #   columns are the windows, as a tensor of one number per window.
    def loudest_frequencies(strengths: Tensor) -> Tensor:
        ...
    ''',
)

STFT1_SOLUTION = source_file(
    '''
    # stft1.py, a solution for comparison

    import torch
    from torch import Tensor

    windows = torch.arange(6.0).unfold(0, 4, 2)
    taper = torch.hann_window(8)


    def frames(
        signal: Tensor,
        frame_length: int,
        hop: int,
    ) -> Tensor:
        return signal.unfold(0, frame_length, hop)


    def stft_by_hand(
        signal: Tensor,
        frame_length: int,
        hop: int,
    ) -> Tensor:
        tapered = frames(signal, frame_length, hop) * torch.hann_window(frame_length)

        return torch.fft.rfft(tapered, dim=1).T


    def spectrogram(
        signal: Tensor,
        n_fft: int,
        hop: int,
    ) -> Tensor:
        window = torch.hann_window(n_fft)

        return torch.stft(signal, n_fft=n_fft, hop_length=hop, window=window, center=False, return_complex=True).abs()


    def loudest_frequencies(strengths: Tensor) -> Tensor:
        return strengths.argmax(dim=0)
    ''',
)


def two_notes(
    first_period: int,
    second_period: int,
    half: int,
) -> Tensor:
    t = torch.arange(half, dtype=torch.float64)

    return torch.cat([torch.sin(2 * torch.pi * t / first_period), torch.sin(2 * torch.pi * t / second_period)])


def reference_stft(
    signal: Tensor,
    n_fft: int,
    hop: int,
) -> Tensor:
    return torch.stft(signal, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), center=False, return_complex=True)


def expect_complex(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    """Complex spectra, compared through their real and imaginary parts side by side."""
    if isinstance(actual, Tensor) and not actual.is_complex():
        raise Mismatch(f"{call} holds real numbers, but a spectrum from torch.fft.rfft holds complex ones.")

    expect_tensor(
        call=call,
        actual=torch.view_as_real(actual) if isinstance(actual, Tensor) else actual,
        expected=torch.view_as_real(expected),
    )


def double_noise(
    seed: int,
    length: int,
) -> Tensor:
    return torch.randn(length, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def check_stft1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="frames",
        description="cuts overlapping windows, one per row",
        inputs=(
            (torch.arange(10.0), 4, 2),
            (torch.arange(12.0), 5, 3),
            (torch.arange(8.0), 4, 4),
        ),
        reference=lambda signal, frame_length, hop: signal.unfold(0, frame_length, hop),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="stft_by_hand",
        description="matches torch.stft, one column per window",
        inputs=(
            (Described(double_noise(470, 40), "40 samples of noise"), 16, 8),
            (Described(double_noise(471, 50), "50 samples of noise"), 12, 5),
        ),
        reference=reference_stft,
        expect=expect_complex,
    )
    check_on_inputs(
        checker=checker,
        function="spectrogram",
        description="gives the strengths of torch.stft",
        inputs=(
            (Described(double_noise(472, 60), "60 samples of noise"), 16, 4),
            (Described(two_notes(8, 4, 64), "a low note then a high one, 128 samples"), 32, 16),
        ),
        reference=lambda signal, n_fft, hop: reference_stft(signal, n_fft, hop).abs(),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="loudest_frequencies",
        description="gives each window's loudest frequency",
        inputs=(
            (Described(reference_stft(two_notes(8, 4, 256), 64, 32).abs(), "the strengths of a low note then a high one"),),
            (Described(reference_stft(double_noise(473, 80), 16, 8).abs(), "the strengths of 80 samples of noise"),),
        ),
        reference=lambda strengths: strengths.argmax(dim=0),
        expect=expect_tensor,
    )


STFT1 = Exercise(
    name="stft1",
    chapter=CAUSAL_CONV,
    title="Spectra over time",
    lesson=STFT1_LESSON,
    source=STFT1_SOURCE,
    solution=STFT1_SOLUTION,
    hints={
        "frames": "Call .unfold on signal along dimension 0, with frame_length as the size and hop as the step.",
        "stft_by_hand": "Multiply frames(signal, frame_length, hop) by torch.hann_window(frame_length). Then "
        "torch.fft.rfft with dim=1 gives one row per window, and .T makes them columns.",
        "spectrogram": "Call torch.stft with the arguments the TODO names and window=torch.hann_window(n_fft), then "
        "take .abs().",
        "loudest_frequencies": "The frequencies run down each column, so take the argmax over dim=0.",
    },
    introduces=(),
    check=check_stft1,
)
