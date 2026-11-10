"""32_pooling, max and average pooling built and matched, and why pooling after a convolution gives one answer for a pattern anywhere."""

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
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module
from .tensors import pinned_threads

POOLING = Chapter(
    folder="32_pooling",
    title="Pooling",
    readme=source_file(
        """
        # Pooling

        Pooling replaces a patch of a feature map with one number, either the
        largest value in the patch or its average. This chapter builds max and
        average pooling and checks them against PyTorch's versions. Then it
        shows why pooling the responses of a convolution turns "this pattern
        is here" into "this pattern is somewhere".

        ## Further reading

        - [CS231n, pooling layer](https://cs231n.github.io/convolutional-networks/#pool)
        - [Dive into Deep Learning, pooling](https://d2l.ai/chapter_convolutional-neural-networks/pooling.html)
        """,
    ),
)

CROSS = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])


def cross_images(
    count: int,
    size: int,
    low: int,
    high: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Noisy square images, half of them with CROSS added at a random row and a column from low up to high."""
    x = 0.3 * torch.randn(count, size, size, generator=generator)
    y = (torch.rand(count, generator=generator) < 0.5).float()
    rows = torch.randint(0, size - 2, (count,), generator=generator)
    columns = torch.randint(low, high, (count,), generator=generator)

    for index in range(count):
        if y[index] == 1:
            x[index, rows[index] : rows[index] + 3, columns[index] : columns[index] + 3] += 1.5 * CROSS

    return x, y


# ---------------------------------------------------------------------------
# pooling1
# ---------------------------------------------------------------------------

POOLING1_LESSON = prose(
    """
    A feature map is one channel of a convolution's output, like the maps in
    channels1. **Pooling** cuts a feature map into patches and keeps one
    number per patch. Max pooling keeps the largest number in each patch, and
    average pooling keeps the mean. With patches of `k` by `k` that don't
    overlap, the height and the width each become `k` times smaller. Pooling
    has no weights to learn.

    ```python
    image = torch.tensor([[1.0, 2.0, 0.0, 0.0],
                          [3.0, 4.0, 0.0, 1.0],
                          [0.0, 0.0, 5.0, 0.0],
                          [0.0, 2.0, 0.0, 0.0]])
    # max pooling with k = 2: tensor([[4., 1.], [2., 5.]])
    ```

    Max pooling records whether a pattern appeared somewhere in a patch, but
    not exactly where. A convolution with a stride of 2 also shrinks the
    image, and many newer models use that instead of pooling.

    Pooling over a whole map at once is called global pooling. Global average
    pooling turns each channel of shape `(h, w)` into one number, its mean.
    ResNet uses it before its last Linear layer.

    `F.max_pool2d` and `F.avg_pool2d` take `x` of shape
    `(batch, channels, h, w)` as their first argument and the patch size as
    their second. `.mean(dim=(2, 3))` averages over dimensions 2 and 3 at
    once.
    """,
)

POOLING1_SOURCE = source_file(
    '''
    # pooling1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    image = torch.tensor([[1.0, 2.0, 0.0, 0.0],
                          [3.0, 4.0, 0.0, 1.0],
                          [0.0, 0.0, 5.0, 0.0],
                          [0.0, 2.0, 0.0, 0.0]])
    corner = image[0:2, 0:2].max()                   # tensor(4.)


    # TODO: Finish the line in the loops, so this returns max pooling of image,
    #   of shape (h, w), with patches of k by k that don't overlap. output[i, j]
    #   is the largest value in the patch at patch row i and patch column j.
    #   Example: max_pool(image, 2) returns tensor([[4., 1.], [2., 5.]]).
    def max_pool(
        image: Tensor,
        k: int,
    ) -> Tensor:
        rows, columns = image.shape[0] // k, image.shape[1] // k
        output = torch.zeros(rows, columns)

        for i in range(rows):
            for j in range(columns):
                output[i, j] = ...

        return output


    # TODO: x has shape (batch, channels, h, w). Return average pooling of x
    #   with patches of k by k that don't overlap, using F.avg_pool2d.
    def average_pool(
        x: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # TODO: x has shape (batch, channels, h, w). Return global average pooling
    #   of x, the mean of each channel's map, as a tensor of shape
    #   (batch, channels).
    def global_average(x: Tensor) -> Tensor:
        ...
    ''',
)

POOLING1_SOLUTION = source_file(
    '''
    # pooling1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    image = torch.tensor([[1.0, 2.0, 0.0, 0.0],
                          [3.0, 4.0, 0.0, 1.0],
                          [0.0, 0.0, 5.0, 0.0],
                          [0.0, 2.0, 0.0, 0.0]])
    corner = image[0:2, 0:2].max()


    def max_pool(
        image: Tensor,
        k: int,
    ) -> Tensor:
        rows, columns = image.shape[0] // k, image.shape[1] // k
        output = torch.zeros(rows, columns)

        for i in range(rows):
            for j in range(columns):
                output[i, j] = image[i * k:(i + 1) * k, j * k:(j + 1) * k].max()

        return output


    def average_pool(
        x: Tensor,
        k: int,
    ) -> Tensor:
        return F.avg_pool2d(x, k)


    def global_average(x: Tensor) -> Tensor:
        return x.mean(dim=(2, 3))
    ''',
)


def check_pooling1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="max_pool",
        description="matches F.max_pool2d",
        inputs=(
            (torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 1.0], [0.0, 0.0, 5.0, 0.0], [0.0, 2.0, 0.0, 0.0]]), 2),
            (small_tensor(770, 6, 9), 3),
        ),
        reference=lambda image, k: F.max_pool2d(image[None, None], k)[0, 0],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="average_pool",
        description="averages each patch",
        inputs=(
            (small_tensor(771, 2, 3, 4, 4), 2),
            (small_tensor(772, 1, 2, 6, 6), 3),
        ),
        reference=lambda x, k: F.avg_pool2d(x, k),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="global_average",
        description="gives one number per channel",
        inputs=(
            (small_tensor(773, 2, 3, 4, 5),),
            (small_tensor(774, 1, 4, 2, 2),),
        ),
        reference=lambda x: x.mean(dim=(2, 3)),
        expect=expect_tensor,
    )


POOLING1 = Exercise(
    name="pooling1",
    chapter=POOLING,
    title="Max and average pooling",
    lesson=POOLING1_LESSON,
    source=POOLING1_SOURCE,
    solution=POOLING1_SOLUTION,
    hints={
        "max_pool": "The patch at row i and column j covers rows i * k up to (i + 1) * k. Its columns go the same "
        "way with j. Take its .max().",
        "average_pool": "F.avg_pool2d takes x and k.",
        "global_average": "Take the mean over the height and the width, dim=(2, 3).",
    },
    introduces=(),
    check=check_pooling1,
)


# ---------------------------------------------------------------------------
# pooling2
# ---------------------------------------------------------------------------

POOLING2_LESSON = prose(
    """
    A convolution's responses move with the pattern. If you shift the image
    two positions right, the map of responses shifts two positions right too.
    So the output of a convolution still says where the pattern is. The head
    after the convolution decides whether the final answer depends on the
    position.

    Here each image is 16 by 16 and holds random noise. Half of the images
    also hold a small cross. In the training images the cross was in the left
    part, and in the test images it was moved to the right. Every model
    starts with 4 kernels of 3 by 3 and a ReLU. Each model was trained three
    times from different seeds, and the table gives its accuracy.

    | After the ReLU | Weights | Training | Cross where it was | Cross moved |
    |---|---|---|---|---|
    | flatten, then a Linear head | 825 | 100% | 99.8% to 99.9% | 47.5% to 47.7% |
    | global max pool, then a Linear head | 45 | 100% | 100% | 100% |
    | global average pool, then a Linear head | 45 | 97.0% to 98.8% | 96.3% to 98.3% | 95.6% to 97.6% |

    Flattening gives the head a separate weight for every position again, so
    the model works like the MLP of conv4. For a moved cross it says no. So
    its accuracy is close to 47.4%, the fraction of moved images that had no
    cross at all. The global max keeps each kernel's strongest response,
    wherever that response was. So the head sees the same numbers wherever
    the cross is.

    The global average also gives the same numbers wherever the cross is. But
    it averages over all 196 positions of the map, so one small cross changes
    the average only a little. Max pooling suits a small pattern that appears
    once. Average pooling suits features spread over the whole map, which is
    why ResNet ends with it.

    Real models pool a little at a time, with a 2 by 2 max pool or a stride of
    2 after every few layers. That lets the model tolerate small shifts, and
    it shrinks the map. A global pool at the end then gives one vector per
    image.

    `.flatten(1)` flattens every dimension after the batch dimension, giving
    one row per example. `.amax(dim=(2, 3))` takes the largest value over
    dimensions 2 and 3.
    """,
)

POOLING2_SOURCE = source_file(
    '''
    # pooling2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    maps = torch.zeros(2, 4, 3, 3)
    rows = maps.flatten(1)                           # shape (2, 36)


    # TODO: Finish forward. x has shape (batch, h, w). Give x a channel
    #   dimension, then run self.conv and a ReLU. Keep each channel's largest
    #   response over the whole map, then run self.head. Return one score per
    #   image, with shape (batch,).
    class PooledDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3)
            self.head = nn.Linear(4, 1)

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Finish forward. Run self.conv and a ReLU as in PooledDetector. Then
    #   flatten the maps into one row per image, run self.head on those rows,
    #   and return one score per image, with shape (batch,).
    class FlatDetector(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3)
            self.head = nn.Linear(4 * (size - 2) * (size - 2), 1)

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

POOLING2_SOLUTION = source_file(
    '''
    # pooling2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    maps = torch.zeros(2, 4, 3, 3)
    rows = maps.flatten(1)


    class PooledDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3)
            self.head = nn.Linear(4, 1)

        def forward(self, x: Tensor) -> Tensor:
            responses = torch.relu(self.conv(x.unsqueeze(1)))

            return self.head(responses.amax(dim=(2, 3))).squeeze(1)


    class FlatDetector(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3)
            self.head = nn.Linear(4 * (size - 2) * (size - 2), 1)

        def forward(self, x: Tensor) -> Tensor:
            responses = torch.relu(self.conv(x.unsqueeze(1)))

            return self.head(responses.flatten(1)).squeeze(1)
    ''',
)


def pooled_reference(
    model: nn.Module,
    x: Tensor,
) -> Tensor:
    with torch.no_grad():
        return model.head(torch.relu(model.conv(x.unsqueeze(1))).amax(dim=(2, 3))).squeeze(1)


def flat_reference(
    model: nn.Module,
    x: Tensor,
) -> Tensor:
    with torch.no_grad():
        return model.head(torch.relu(model.conv(x.unsqueeze(1))).flatten(1)).squeeze(1)


def check_pooled(found: type) -> None:
    torch.manual_seed(780)
    model = expect_module("PooledDetector()", found())
    x, _ = cross_images(5, 10, 0, 7, torch.Generator().manual_seed(781))
    expect_tensor(
        call="PooledDetector() on 5 images of 10 by 10",
        actual=model(x),
        expected=pooled_reference(model, x),
    )


def check_flat(found: type) -> None:
    torch.manual_seed(782)
    model = expect_module("FlatDetector(10)", found(10))
    x, _ = cross_images(5, 10, 0, 7, torch.Generator().manual_seed(783))
    expect_tensor(
        call="FlatDetector(10) on 5 images of 10 by 10",
        actual=model(x),
        expected=flat_reference(model, x),
    )


def check_pooled_finds_moved(found: type) -> None:
    generator = torch.Generator().manual_seed(784)
    x, y = cross_images(2000, 16, 0, 6, generator)
    moved_x, moved_y = cross_images(1000, 16, 8, 14, generator)
    torch.manual_seed(0)
    model = expect_module("PooledDetector()", found())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        moved = ((model(moved_x) > 0).float() == moved_y).float().mean().item()

    if moved < 0.95:
        raise Mismatch(f"PooledDetector, trained for 300 steps, finds {moved:.1%} of the moved crosses, but it should find over 95%.")


def check_pooling2(checker: Checker) -> None:
    checker.test_method(
        class_name="PooledDetector",
        method="forward",
        description="keeps each channel's largest response",
        run=check_pooled,
    )
    checker.test_method(
        class_name="FlatDetector",
        method="forward",
        description="flattens the maps for the head",
        run=check_flat,
    )
    checker.test_method(
        class_name="PooledDetector",
        method="forward",
        description="once trained, finds the cross in positions it never saw",
        run=check_pooled_finds_moved,
    )


POOLING2 = Exercise(
    name="pooling2",
    chapter=POOLING,
    title="Why pooling",
    lesson=POOLING2_LESSON,
    source=POOLING2_SOURCE,
    solution=POOLING2_SOLUTION,
    hints={
        "PooledDetector.forward": "Follow conv4's Detector, but take .amax(dim=(2, 3)) over the height and the width.",
        "FlatDetector.forward": "Pass torch.relu(self.conv(x.unsqueeze(1))).flatten(1) to self.head, then "
        ".squeeze(1).",
    },
    introduces=(),
    check=check_pooling2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


class CrossDetector(nn.Module):
    """pooling2's models: 4 kernels of 3 by 3 and a ReLU, then a global max, a global average or a flatten, then a
    Linear head."""

    def __init__(
        self,
        then: str,
        size: int = 16,
    ) -> None:
        super().__init__()
        self.then = then
        self.conv = nn.Conv2d(1, 4, 3)
        self.head = nn.Linear(4 * (size - 2) * (size - 2) if then == "flatten" else 4, 1)

    def forward(self, x: Tensor) -> Tensor:
        responses = torch.relu(self.conv(x.unsqueeze(1)))

        if self.then == "max":
            return self.head(responses.amax(dim=(2, 3))).squeeze(1)

        if self.then == "average":
            return self.head(responses.mean(dim=(2, 3))).squeeze(1)

        return self.head(responses.flatten(1)).squeeze(1)


def labelled_right(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> float:
    with torch.no_grad():
        return ((model(x) > 0).float() == y).float().mean().item()


def measure_pooling(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, list[tuple[int, float, float, float]]]:
    """pooling2's table, the data the check trains on, 300 full batch steps of Adam at 0.01 per seed. Gives, for each
    head, the weights and the share labelled right in training, with the cross where it was, and with it moved."""
    generator = torch.Generator().manual_seed(784)
    x, y = cross_images(2000, 16, 0, 6, generator)
    moved_x, moved_y = cross_images(1000, 16, 8, 14, generator)
    same_x, same_y = cross_images(1000, 16, 0, 6, generator)
    results: dict[str, list[tuple[int, float, float, float]]] = {}

    with pinned_threads():
        for then in ("flatten", "max", "average"):
            for seed in seeds:
                torch.manual_seed(seed)
                model = CrossDetector(then)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                for _ in range(300):
                    loss = F.binary_cross_entropy_with_logits(model(x), y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                results.setdefault(then, []).append(
                    (
                        sum(parameter.numel() for parameter in model.parameters()),
                        labelled_right(model, x, y),
                        labelled_right(model, same_x, same_y),
                        labelled_right(model, moved_x, moved_y),
                    ),
                )

    return results
