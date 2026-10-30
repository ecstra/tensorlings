"""30_convolution, convolution as a Linear with shared weights, as a sliding dot product in 1D and 2D, and why it finds a pattern anywhere."""

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
    expect_close,
    expect_parts,
    expect_tensor,
    reject_missing_return,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

CONVOLUTION = Chapter(
    folder="30_convolution",
    title="Convolution",
    readme=source_file(
        """
        # Convolution

        Section 5 is about images. A convolution slides one small set of
        weights along its input and takes a dot product at every place. This
        chapter shows it is a Linear layer with most weights zero and the rest
        shared, builds it in 1D and 2D, matches PyTorch's, and shows why a
        model built from it finds a pattern wherever it sits. It ends with the
        convolution theorem, which turns a long convolution into a product of
        spectra.

        ## Further reading

        - [CS231n, convolutional networks](https://cs231n.github.io/convolutional-networks/)
        - [Dive into Deep Learning, from fully connected layers to convolutions](https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html)
        - [Dumoulin and Visin, a guide to convolution arithmetic](https://arxiv.org/abs/1603.07285)
        """,
    ),
)

PATTERN = torch.tensor([1.0, -1.0, 1.0])


def pattern_signals(
    count: int,
    length: int,
    low: int,
    high: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Noisy signals, half of them with PATTERN added at a random start from low up to high."""
    x = 0.3 * torch.randn(count, length, generator=generator)
    y = (torch.rand(count, generator=generator) < 0.5).float()
    starts = torch.randint(low, high, (count,), generator=generator)

    for index in range(count):
        if y[index] == 1:
            x[index, starts[index] : starts[index] + 3] += 2 * PATTERN

    return x, y


def reference_conv_matrix(
    w: Tensor,
    n: int,
) -> Tensor:
    k = len(w)
    matrix = torch.zeros(n - k + 1, n)

    for row in range(n - k + 1):
        matrix[row, row : row + k] = w

    return matrix


# ---------------------------------------------------------------------------
# conv1
# ---------------------------------------------------------------------------

CONV1_LESSON = prose(
    """
    Section 5 is about images, and its block is the **convolution**. Start in
    one dimension, with a signal `x` of length `n` and a small set of weights
    `w` of length `k`, the **kernel**. The output at position `i` is the dot
    product of `w` with the `k` numbers of `x` starting at `i`.

    ```python
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w = torch.tensor([1.0, 0.0, -1.0])
    # position 0: 1 * 1 + 0 * 2 - 1 * 3 = -2
    # position 1: 1 * 2 + 0 * 3 - 1 * 4 = -2
    ```

    The kernel fits in `n - k + 1` places, so that is the output's length.

    Every output is a weighted sum of inputs, so a convolution is a Linear
    layer. Its weight matrix has one row per output. Row `i` holds `w` in
    columns `i` to `i + k - 1` and zeros everywhere else. So most weights are
    zero, and the rest are the same few numbers shared by every row.

    That is why convolutions are cheap. A Linear from 32 inputs to 30 outputs
    learns 960 weights. A convolution of the same shapes learns 3.

    `matrix[i, a:b] = w` writes `w` into row `i`, columns `a` up to `b`.
    """,
)

CONV1_SOURCE = source_file(
    '''
    # conv1.py

    import torch
    from torch import Tensor

    matrix = torch.zeros(2, 4)
    matrix[1, 1:3] = torch.tensor([5.0, 6.0])        # row 1 becomes [0., 5., 6., 0.]


    # TODO: Finish the line in the loop, so this returns the weight matrix of a
    #   convolution with kernel w on inputs of length n, one row per output, w
    #   placed at columns row to row + k - 1.
    def conv_matrix(
        w: Tensor,
        n: int,
    ) -> Tensor:
        k = len(w)
        matrix = torch.zeros(n - k + 1, n)

        for row in range(n - k + 1):
            ...

        return matrix


    # TODO: Return the convolution of x with the kernel w, worked out as
    #   conv_matrix times x.
    def conv_as_linear(
        x: Tensor,
        w: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return how many weights a Linear layer from n inputs to the
    #   convolution's n - k + 1 outputs learns, without a bias, and how many the
    #   convolution learns.
    def weight_counts(
        n: int,
        k: int,
    ) -> tuple[int, int]:
        ...
    ''',
)

CONV1_SOLUTION = source_file(
    '''
    # conv1.py, a solution for comparison

    import torch
    from torch import Tensor

    matrix = torch.zeros(2, 4)
    matrix[1, 1:3] = torch.tensor([5.0, 6.0])


    def conv_matrix(
        w: Tensor,
        n: int,
    ) -> Tensor:
        k = len(w)
        matrix = torch.zeros(n - k + 1, n)

        for row in range(n - k + 1):
            matrix[row, row:row + k] = w

        return matrix


    def conv_as_linear(
        x: Tensor,
        w: Tensor,
    ) -> Tensor:
        return conv_matrix(w, len(x)) @ x


    def weight_counts(
        n: int,
        k: int,
    ) -> tuple[int, int]:
        return (n - k + 1) * n, k
    ''',
)


def check_conv1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="conv_matrix",
        description="places the kernel along the diagonal, one row per output",
        inputs=(
            (torch.tensor([1.0, 0.0, -1.0]), 5),
            (small_tensor(700, 2), 4),
        ),
        reference=reference_conv_matrix,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="conv_as_linear",
        description="slides the kernel along x",
        inputs=(
            (torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([1.0, 0.0, -1.0])),
            (small_tensor(701, 7), small_tensor(702, 3)),
        ),
        reference=lambda x, w: F.conv1d(x.view(1, 1, -1), w.view(1, 1, -1)).flatten(),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="weight_counts",
        description="counts the Linear's weights and the convolution's",
        inputs=(
            (32, 3),
            (100, 5),
        ),
        reference=lambda n, k: ((n - k + 1) * n, k),
        expect=expect_parts("the Linear's count", "the convolution's count"),
    )


CONV1 = Exercise(
    name="conv1",
    chapter=CONVOLUTION,
    title="A Linear with shared weights",
    lesson=CONV1_LESSON,
    source=CONV1_SOURCE,
    solution=CONV1_SOLUTION,
    hints={
        "conv_matrix": "Write w into this row, from column row up to row + k, as the line that sets matrix[1, 1:3] "
        "does.",
        "conv_as_linear": "Build conv_matrix(w, len(x)) and multiply it by x with @.",
        "weight_counts": "The Linear has one weight per input for each output. The convolution has one per kernel "
        "entry.",
    },
    introduces=("convolution",),
    check=check_conv1,
)


# ---------------------------------------------------------------------------
# conv2
# ---------------------------------------------------------------------------

CONV2_LESSON = prose(
    """
    Nobody builds that matrix in practice. A convolution is computed as a
    sliding dot product, the kernel moved one place at a time.

    A loop over the positions is the plainest way. `x.unfold(0, k, 1)` does it
    without a loop. It cuts `x` into every window of length `k`, one step
    apart, as rows of a matrix. Each row times the kernel is then one output.

    ```python
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x.unfold(0, 3, 1)    # tensor([[1., 2., 3.], [2., 3., 4.]])
    ```

    PyTorch's `F.conv1d(x, w)` works on batches with channels, which the next
    chapter explains. For now, an input of shape `(batch, 1, length)` and a
    weight of shape `(1, 1, k)` give an output of shape
    `(batch, 1, length - k + 1)`. `.view(1, 1, -1)` gives a tensor those
    extra dimensions of size 1.

    In maths, a convolution flips the kernel before sliding it. Deep learning
    skips the flip, which makes no difference to a kernel that is learned, and
    still calls it convolution.
    """,
)

CONV2_SOURCE = source_file(
    '''
    # conv2.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    windows = x.unfold(0, 3, 1)                      # tensor([[1., 2., 3.], [2., 3., 4.]])


    # TODO: Finish the line in the loop, so this returns the convolution of x
    #   with w, one dot product of w with each window of x.
    def slide(
        x: Tensor,
        w: Tensor,
    ) -> Tensor:
        k = len(w)
        outputs = []

        for start in range(len(x) - k + 1):
            outputs.append(...)

        return torch.stack(outputs)


    # TODO: Return the same convolution with no loop, from the windows that
    #   x.unfold gives.
    def unfold_conv(
        x: Tensor,
        w: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the convolution of every row of x, of shape (batch, length),
    #   with w, using F.conv1d. The result has shape (batch, length - k + 1).
    def batch_conv(
        x: Tensor,
        w: Tensor,
    ) -> Tensor:
        ...
    ''',
)

CONV2_SOLUTION = source_file(
    '''
    # conv2.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    windows = x.unfold(0, 3, 1)


    def slide(
        x: Tensor,
        w: Tensor,
    ) -> Tensor:
        k = len(w)
        outputs = []

        for start in range(len(x) - k + 1):
            outputs.append((x[start:start + k] * w).sum())

        return torch.stack(outputs)


    def unfold_conv(
        x: Tensor,
        w: Tensor,
    ) -> Tensor:
        return x.unfold(0, len(w), 1) @ w


    # F.conv1d wants (batch, channels, length) and a weight of (out channels, in channels, k).
    def batch_conv(
        x: Tensor,
        w: Tensor,
    ) -> Tensor:
        return F.conv1d(x.unsqueeze(1), w.view(1, 1, -1)).squeeze(1)
    ''',
)


def check_conv2(checker: Checker) -> None:
    reference = lambda x, w: F.conv1d(x.view(1, 1, -1), w.view(1, 1, -1)).flatten()
    inputs = (
        (torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([1.0, 0.0, -1.0])),
        (small_tensor(703, 8), small_tensor(704, 4)),
    )
    check_on_inputs(
        checker=checker,
        function="slide",
        description="takes a dot product at every position",
        inputs=inputs,
        reference=reference,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="unfold_conv",
        description="matches the loop with windows from unfold",
        inputs=inputs,
        reference=reference,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="batch_conv",
        description="convolves every row with F.conv1d",
        inputs=(
            (small_tensor(705, 3, 6), small_tensor(706, 3)),
            (small_tensor(707, 2, 9), small_tensor(708, 5)),
        ),
        reference=lambda x, w: torch.stack([F.conv1d(row.view(1, 1, -1), w.view(1, 1, -1)).flatten() for row in x]),
        expect=expect_tensor,
    )


CONV2 = Exercise(
    name="conv2",
    chapter=CONVOLUTION,
    title="A sliding dot product",
    lesson=CONV2_LESSON,
    source=CONV2_SOURCE,
    solution=CONV2_SOLUTION,
    hints={
        "slide": "The window is x[start:start + k]. Multiply it by w and take the sum.",
        "unfold_conv": "x.unfold(0, len(w), 1) gives one window per row. Multiply that matrix by w with @.",
        "batch_conv": "Give x a channel dimension with .unsqueeze(1), and give w the shape (1, 1, k) with .view. "
        "Call F.conv1d, then take the channel dimension away with .squeeze(1).",
    },
    introduces=(),
    check=check_conv2,
)


# ---------------------------------------------------------------------------
# conv3
# ---------------------------------------------------------------------------

CONV3_LESSON = prose(
    """
    An image is two dimensional, and so is its kernel. A kernel of shape
    `(kh, kw)` slides across the image's rows and columns. At each place it
    multiplies the patch under it by its weights and adds them all up. An image
    of shape `(h, w)` gives an output with `h - kh + 1` rows and `w - kw + 1`
    columns.

    Each output says how much the patch at that place looks like the kernel.
    This kernel answers to a vertical edge, bright on the left and dark on
    the right.

    ```python
    edge = torch.tensor([[1.0, 0.0, -1.0],
                         [1.0, 0.0, -1.0],
                         [1.0, 0.0, -1.0]])
    ```

    `image[i:i + kh, j:j + kw]` is the patch at row `i` and column `j`.

    `F.conv2d` takes images of shape `(batch, channels, height, width)` and a
    weight of shape `(out channels, in channels, kh, kw)`. With one channel in
    and out, `image[None, None]` and `kernel[None, None]` add the two size 1
    dimensions in front.
    """,
)

CONV3_SOURCE = source_file(
    '''
    # conv3.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    image = torch.arange(16.0).reshape(4, 4)
    patch = image[1:3, 0:2]                          # tensor([[4., 5.], [8., 9.]])


    # TODO: Return the output shape of a convolution of an image of shape (h, w)
    #   with a kernel of shape (kh, kw), as a tuple.
    def output_shape(
        h: int,
        w: int,
        kh: int,
        kw: int,
    ) -> tuple[int, int]:
        ...


    # TODO: Finish the line in the loops, so this returns the convolution of
    #   image with kernel, one sum of patch times kernel at every place.
    def conv2d_loop(
        image: Tensor,
        kernel: Tensor,
    ) -> Tensor:
        kh, kw = kernel.shape
        rows, columns = output_shape(image.shape[0], image.shape[1], kh, kw)
        output = torch.zeros(rows, columns)

        for i in range(rows):
            for j in range(columns):
                output[i, j] = ...

        return output


    # TODO: Return the convolution of every image in images, of shape
    #   (batch, h, w), with kernel, using F.conv2d, as shape
    #   (batch, h - kh + 1, w - kw + 1).
    def batch_conv2d(
        images: Tensor,
        kernel: Tensor,
    ) -> Tensor:
        ...
    ''',
)

CONV3_SOLUTION = source_file(
    '''
    # conv3.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    image = torch.arange(16.0).reshape(4, 4)
    patch = image[1:3, 0:2]


    def output_shape(
        h: int,
        w: int,
        kh: int,
        kw: int,
    ) -> tuple[int, int]:
        return h - kh + 1, w - kw + 1


    def conv2d_loop(
        image: Tensor,
        kernel: Tensor,
    ) -> Tensor:
        kh, kw = kernel.shape
        rows, columns = output_shape(image.shape[0], image.shape[1], kh, kw)
        output = torch.zeros(rows, columns)

        for i in range(rows):
            for j in range(columns):
                output[i, j] = (image[i:i + kh, j:j + kw] * kernel).sum()

        return output


    def batch_conv2d(
        images: Tensor,
        kernel: Tensor,
    ) -> Tensor:
        return F.conv2d(images.unsqueeze(1), kernel[None, None]).squeeze(1)
    ''',
)


def expect_pair(
    call: str,
    actual: object,
    expected: tuple[int, int],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, (tuple, list)) or tuple(actual) != expected:
        raise Mismatch(f"{call} is {actual!r}, but it should be {expected!r}.")


def check_conv3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="output_shape",
        description="gives the output shape of a convolution",
        inputs=(
            (4, 4, 3, 3),
            (28, 20, 5, 2),
        ),
        reference=lambda h, w, kh, kw: (h - kh + 1, w - kw + 1),
        expect=expect_pair,
    )
    check_on_inputs(
        checker=checker,
        function="conv2d_loop",
        description="matches F.conv2d on one image",
        inputs=(
            (torch.arange(16.0).reshape(4, 4), torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])),
            (small_tensor(710, 5, 6), small_tensor(711, 2, 3)),
        ),
        reference=lambda image, kernel: F.conv2d(image[None, None], kernel[None, None])[0, 0],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="batch_conv2d",
        description="convolves every image with F.conv2d",
        inputs=(
            (small_tensor(712, 2, 5, 5), small_tensor(713, 3, 3)),
            (small_tensor(714, 3, 4, 6), small_tensor(715, 2, 2)),
        ),
        reference=lambda images, kernel: torch.stack([F.conv2d(image[None, None], kernel[None, None])[0, 0] for image in images]),
        expect=expect_tensor,
    )


CONV3 = Exercise(
    name="conv3",
    chapter=CONVOLUTION,
    title="Convolution in 2D",
    lesson=CONV3_LESSON,
    source=CONV3_SOURCE,
    solution=CONV3_SOLUTION,
    hints={
        "output_shape": "The kernel fits in h - kh + 1 rows and w - kw + 1 columns.",
        "conv2d_loop": "Take the patch image[i:i + kh, j:j + kw], multiply it by kernel and take the sum.",
        "batch_conv2d": "Give images a channel dimension with .unsqueeze(1), give kernel two with [None, None], "
        "call F.conv2d, then .squeeze(1).",
    },
    introduces=(),
    check=check_conv3,
)


# ---------------------------------------------------------------------------
# conv4
# ---------------------------------------------------------------------------

CONV4_LESSON = prose(
    """
    Here are signals of 32 numbers, noise plus, in half of them, a short
    pattern of 3. The task is to say whether the pattern is there. In
    training it only ever started in the first 13 positions. The test moved it
    to positions 16 to 28.

    | Model | Weights | Training | Pattern where it was | Pattern moved |
    |---|---|---|---|---|
    | MLP, 32 to 64 to 1 | 2177 | 100% | 100% | 47.3% |
    | convolution of 4 kernels of 3, then the largest response | 21 | 100% | 100% | 100% |

    Both learned. The MLP has a separate weight for every position, and the
    weights at the right never saw the pattern. So for a moved pattern it says
    "no" every time. Its 47.3% is the share of moved signals that had no
    pattern at all. The convolution slides the same kernel everywhere,
    so it answers to the pattern wherever it sits. The largest response over
    all positions then says whether it showed up anywhere.

    That is the idea of Section 5. A cat is a cat anywhere in a picture, and a
    convolution doesn't have to learn that again for every place.

    `nn.Conv1d(1, 4, 3)` holds 4 kernels of 3. `.amax(dim=2)` keeps the
    largest number along the positions. `F.binary_cross_entropy_with_logits`
    from activations2 is the loss for a yes or no.
    """,
)

CONV4_SOURCE = source_file(
    '''
    # conv4.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    responses = torch.tensor([[[0.1, 2.0, 0.3]]])
    largest = responses.amax(dim=2)                  # tensor([[2.]])


    # TODO: Finish forward. It gives x a channel dimension, runs conv, then a
    #   ReLU, keeps each kernel's largest response along the positions, and
    #   runs head. The result is one score per signal, shape (batch,).
    class Detector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(1, 4, 3)
            self.head = nn.Linear(4, 1)

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return the share of signals the model labels right, a score above
    #   0 meaning the pattern is there, as a plain number, without recording for
    #   autograd.
    def accuracy(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        ...


    # TODO: Finish the line that sets loss, the binary cross entropy of the
    #   model's scores on x against y, so this trains with Adam.
    def train_detector(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for _ in range(steps):
            loss = ...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
    ''',
)

CONV4_SOLUTION = source_file(
    '''
    # conv4.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    responses = torch.tensor([[[0.1, 2.0, 0.3]]])
    largest = responses.amax(dim=2)


    class Detector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(1, 4, 3)
            self.head = nn.Linear(4, 1)

        def forward(self, x: Tensor) -> Tensor:
            responses = torch.relu(self.conv(x.unsqueeze(1)))

            return self.head(responses.amax(dim=2)).squeeze(1)


    def accuracy(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        with torch.no_grad():
            return ((model(x) > 0).float() == y).float().mean().item()


    def train_detector(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for _ in range(steps):
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
    ''',
)


class ReferenceDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 4, 3)
        self.head = nn.Linear(4, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(torch.relu(self.conv(x.unsqueeze(1))).amax(dim=2)).squeeze(1)


def seeded_detector(
    seed: int,
    make: type = ReferenceDetector,
) -> nn.Module:
    torch.manual_seed(seed)

    return make()


def reference_train_detector(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    steps: int,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def check_detector(found: type) -> None:
    model = expect_module("Detector()", seeded_detector(720, found))
    reference = seeded_detector(720)
    x, _ = pattern_signals(5, 12, 0, 9, torch.Generator().manual_seed(721))

    with torch.no_grad():
        expected = reference(x)

    expect_tensor(
        call="Detector() on 5 signals of 12",
        actual=model(x),
        expected=expected,
    )


def check_finds_moved_patterns(found: object) -> None:
    generator = torch.Generator().manual_seed(722)
    x, y = pattern_signals(2000, 32, 0, 13, generator)
    moved_x, moved_y = pattern_signals(1000, 32, 16, 29, generator)
    model = seeded_detector(0)
    found(model, x, y, 500)

    with torch.no_grad():
        moved = ((model(moved_x) > 0).float() == moved_y).float().mean().item()

    if moved < 0.95:
        raise Mismatch(f"After train_detector for 500 steps, the detector finds {moved:.1%} of the moved patterns, but it should find over 95%.")


def check_accuracy(
    found: object,
    x: Tensor,
    y: Tensor,
) -> None:
    model = seeded_detector(724)
    recording: list[bool] = []
    hook = model.register_forward_pre_hook(lambda module, inputs: recording.append(torch.is_grad_enabled()))

    try:
        actual = found(model, x, y)
    finally:
        hook.remove()

    if any(recording):
        raise Mismatch("accuracy ran the model while recording for autograd. Run it inside torch.no_grad().")

    with torch.no_grad():
        expected = ((model(x) > 0).float() == y).float().mean().item()

    expect_close(
        call="accuracy(a fresh detector, 50 signals, their labels)",
        actual=actual,
        expected=expected,
    )


def check_conv4(checker: Checker) -> None:
    checker.test_method(
        class_name="Detector",
        method="forward",
        description="convolves, keeps the largest response and scores it",
        run=check_detector,
    )
    x, y = pattern_signals(50, 16, 0, 13, torch.Generator().manual_seed(723))
    checker.test(
        function="accuracy",
        description="gives the share labelled right, recording nothing",
        run=lambda found: check_accuracy(found, x, y),
    )
    checker.test(
        function="train_detector",
        description="trains with Adam on the binary cross entropy",
        run=lambda found: expect_close(
            call="train_detector for 20 steps on 50 signals",
            actual=found(seeded_detector(725), x, y, 20),
            expected=reference_train_detector(seeded_detector(725), x, y, 20),
            tolerance=1e-5,
        ),
    )
    checker.test(
        function="train_detector",
        description="trains a detector that finds the pattern where it never saw it",
        run=check_finds_moved_patterns,
    )


CONV4 = Exercise(
    name="conv4",
    chapter=CONVOLUTION,
    title="Why convolution",
    lesson=CONV4_LESSON,
    source=CONV4_SOURCE,
    solution=CONV4_SOLUTION,
    hints={
        "Detector.forward": "x.unsqueeze(1) gives the channel. Pass it through self.conv and torch.relu, take "
        ".amax(dim=2), pass that through self.head, and .squeeze(1) the result.",
        "accuracy": "Inside a with torch.no_grad() block, compare (model(x) > 0).float() with y, and take the mean of "
        "the matches as floats.",
        "train_detector": "F.binary_cross_entropy_with_logits takes the model's scores on x and the labels y.",
    },
    introduces=(),
    check=check_conv4,
    time_limit=90,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# fft1
# ---------------------------------------------------------------------------

FFT1_LESSON = prose(
    """
    A long kernel is slow to slide. An echo in a room lasts thousands of
    samples, and sliding k weights along n samples costs about n times k
    multiplications. The Fourier transform gives a faster road, and a new way
    to see what a kernel does.

    Any signal of n samples is a sum of waves, one for each frequency from 0
    up to n / 2 cycles across the signal. `torch.fft.rfft` says how much of
    each wave the signal holds. It gives one complex number per frequency,
    which holds the wave's strength and its shift, and `.abs()` is the
    strength. `torch.fft.irfft` adds the waves back up into the signal.

    ```python
    t = torch.arange(16)
    signal = torch.cos(2 * torch.pi * 3 * t / 16)    # 3 cycles across 16 samples
    torch.fft.rfft(signal).abs().round()             # tensor([0., 0., 0., 8., 0., 0., 0., 0., 0.])
    ```

    These strengths, one per frequency, are the signal's **spectrum**.

    The convolution theorem says that convolving two signals multiplies their
    spectra. Take both spectra, multiply them frequency by frequency, and turn
    the product back with `irfft`. That costs about n log n. For 100,000
    samples and a kernel of 10,000, the spectra were 9 to 13 times faster than
    `F.conv1d` here, over five runs.

    Two details make it match `F.conv1d`.

    The waves wrap around, so the plain product gives a circular convolution.
    There the end of the kernel spills onto the start of the signal. Zeros
    added to both, up to the full output length of n + k - 1, stop that.
    `rfft(x, n=length)` adds the zeros, and `irfft(spectrum, n=length)` gives
    that length back.

    The theorem's convolution flips the kernel, and `F.conv1d` doesn't, as
    conv2 said. So flip the kernel with `.flip(0)` first. Then keep the
    outputs where it fits fully, from k - 1 up to n.

    A kernel's spectrum shows what it does to each frequency. Here is an
    average of 4 samples, with zeros added up to 16.

    | Frequency | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
    |---|---|---|---|---|---|---|---|---|---|
    | Strength kept | 1.00 | 0.91 | 0.65 | 0.32 | 0.00 | 0.21 | 0.27 | 0.18 | 0.00 |

    It keeps slow changes and mostly wipes out fast ones, so it smooths. A
    learned kernel is a filter like this one, and its spectrum says which
    frequencies it listens to. The echo project later works on spectra like
    these.
    """,
)

FFT1_SOURCE = source_file(
    '''
    # fft1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    t = torch.arange(16)
    signal = torch.cos(2 * torch.pi * 3 * t / 16)
    strengths = torch.fft.rfft(signal).abs()    # 8 at frequency 3, about 0 elsewhere


    # TODO: Return how strongly kernel keeps each frequency, from 0 up to n // 2,
    #   once zeros fill it out to n samples.
    def frequency_response(
        kernel: Tensor,
        n: int,
    ) -> Tensor:
        ...


    # TODO: Return the circular convolution of a and b, two signals of the same
    #   length, by multiplying their spectra. The result has that length too.
    def circular_convolve(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the full convolution of signal and kernel by multiplying their
    #   spectra, with zeros added so nothing wraps around. It has length
    #   len(signal) + len(kernel) - 1.
    def fft_convolve(
        signal: Tensor,
        kernel: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return what F.conv1d gives for signal and kernel, as one row of length
    #   len(signal) - len(kernel) + 1, using fft_convolve.
    def conv1d_by_fft(
        signal: Tensor,
        kernel: Tensor,
    ) -> Tensor:
        ...
    ''',
)

FFT1_SOLUTION = source_file(
    '''
    # fft1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    t = torch.arange(16)
    signal = torch.cos(2 * torch.pi * 3 * t / 16)
    strengths = torch.fft.rfft(signal).abs()


    def frequency_response(
        kernel: Tensor,
        n: int,
    ) -> Tensor:
        return torch.fft.rfft(kernel, n=n).abs()


    def circular_convolve(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=len(a))


    def fft_convolve(
        signal: Tensor,
        kernel: Tensor,
    ) -> Tensor:
        length = len(signal) + len(kernel) - 1
        product = torch.fft.rfft(signal, n=length) * torch.fft.rfft(kernel, n=length)

        return torch.fft.irfft(product, n=length)


    # F.conv1d skips the flip, so flipping first cancels the theorem's own flip.
    def conv1d_by_fft(
        signal: Tensor,
        kernel: Tensor,
    ) -> Tensor:
        k = len(kernel)

        return fft_convolve(signal, kernel.flip(0))[k - 1 : len(signal)]
    ''',
)


def reference_circular_convolve(
    a: Tensor,
    b: Tensor,
) -> Tensor:
    n = len(a)
    positions = torch.arange(n)

    return torch.stack([(a * b[(i - positions) % n]).sum() for i in range(n)])


def reference_full_convolve(
    signal: Tensor,
    kernel: Tensor,
) -> Tensor:
    k = len(kernel)

    return F.conv1d(signal.view(1, 1, -1), kernel.flip(0).view(1, 1, -1), padding=k - 1).flatten()


def double_signal(
    seed: int,
    length: int,
) -> Tensor:
    """Doubles, so the spectra's rounding stays far below what the check can see."""
    return small_tensor(seed, length).double()


def check_fft1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="frequency_response",
        description="gives the strength of each frequency of the padded kernel",
        inputs=(
            (torch.full((4,), 0.25, dtype=torch.float64), 16),
            (double_signal(301, 5), 12),
            (double_signal(302, 3), 9),
        ),
        reference=lambda kernel, n: torch.fft.rfft(kernel, n=n).abs(),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="circular_convolve",
        description="wraps around the end, the same length as its inputs",
        inputs=(
            (double_signal(303, 6), double_signal(304, 6)),
            (double_signal(305, 7), double_signal(306, 7)),
        ),
        reference=reference_circular_convolve,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="fft_convolve",
        description="gives the full convolution, with nothing wrapped around",
        inputs=(
            (double_signal(307, 8), double_signal(308, 3)),
            (double_signal(309, 11), double_signal(310, 4)),
        ),
        reference=reference_full_convolve,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="conv1d_by_fft",
        description="matches F.conv1d",
        inputs=(
            (double_signal(311, 9), double_signal(312, 3)),
            (double_signal(313, 12), double_signal(314, 5)),
        ),
        reference=lambda signal, kernel: F.conv1d(signal.view(1, 1, -1), kernel.view(1, 1, -1)).flatten(),
        expect=expect_tensor,
    )


FFT1 = Exercise(
    name="fft1",
    chapter=CONVOLUTION,
    title="Convolution through spectra",
    lesson=FFT1_LESSON,
    source=FFT1_SOURCE,
    solution=FFT1_SOLUTION,
    hints={
        "frequency_response": "torch.fft.rfft with n=n adds the zeros, then .abs() gives the strengths.",
        "circular_convolve": "Multiply torch.fft.rfft(a) by torch.fft.rfft(b), then torch.fft.irfft the product "
        "with n=len(a), which an odd length needs.",
        "fft_convolve": "Pass n=length to both rfft calls and to irfft, with length the full output length.",
        "conv1d_by_fft": "Call fft_convolve with kernel.flip(0), then slice from k - 1 up to len(signal).",
    },
    introduces=(),
    check=check_fft1,
)
