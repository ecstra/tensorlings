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

        Pooling sums up a patch of a feature map in one number, its largest or
        its average. This chapter builds max and average pooling, matches
        PyTorch's, and shows why pooling the responses of a convolution turns
        "this pattern is here" into "this pattern is somewhere".

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
    **Pooling** cuts a feature map into patches and keeps one number per
    patch. Max pooling keeps the largest, average pooling the mean. With
    patches of `k` by `k` that don't overlap, the height and the width shrink
    `k` times, and there are no weights to learn.

    ```python
    image = torch.tensor([[1.0, 2.0, 0.0, 0.0],
                          [3.0, 4.0, 0.0, 1.0],
                          [0.0, 0.0, 5.0, 0.0],
                          [0.0, 2.0, 0.0, 0.0]])
    # max pooling with k = 2: tensor([[4., 1.], [2., 5.]])
    ```

    Max pooling keeps whether a pattern showed up in a patch and forgets
    exactly where. A stride of 2 in a convolution shrinks the image too, and
    many newer models use that in place of pooling.

    Pooling over a whole map at once is global pooling. Global average
    pooling turns each channel of shape `(h, w)` into one number, and ResNet
    uses it before its last Linear layer.

    `F.max_pool2d` and `F.avg_pool2d` take `x` of shape
    `(batch, channels, h, w)`, then the patch size. `.mean(dim=(2, 3))`
    averages over two dimensions at once.
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
    #   of shape (h, w), with patches of k by k that don't overlap.
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


    # TODO: Return average pooling of x, of shape (batch, channels, h, w), with
    #   patches of k by k, using F.avg_pool2d.
    def average_pool(
        x: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # TODO: Return global average pooling of x, of shape
    #   (batch, channels, h, w), one number per channel, shape (batch, channels).
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
    A convolution's responses move with the pattern. Shift the image two
    places right and the map of responses shifts two places right too. So a
    convolution alone still says where. The head after it decides whether the
    answer depends on the place.

    Here images of 16 by 16 held noise and, in half of them, a small cross.
    In training the cross sat in the left part, and the test moved it right.
    Every model starts with 4 kernels of 3 by 3 and a ReLU, and each was
    trained three times from different seeds.

    | Then | Weights | Training | Cross where it was | Cross moved |
    |---|---|---|---|---|
    | flatten, then a Linear head | 825 | 100% | 99.8% to 99.9% | 47.5% to 47.7% |
    | global max pool, then a Linear head | 45 | 100% | 100% | 100% |
    | global average pool, then a Linear head | 45 | 97.0% to 98.8% | 96.3% to 98.3% | 95.6% to 97.6% |

    Flattening hands the head one weight per place again, so it is back to
    the MLP of conv4. For a moved cross it says no, so its score sits at the
    47.4% of moved images that had no cross at all. The global max keeps each
    kernel's strongest response wherever it was, so the head sees the same
    numbers wherever the cross is.

    The global average gives the same numbers wherever the cross is too. But
    it spreads one small cross over all 196 places of the map, so the cross
    counts for little. Max suits a small pattern that shows up once. Average
    suits features spread over the whole map, which is why ResNet ends with
    one.

    Real models pool a little at a time, a 2 by 2 max or a stride of 2 after
    a few layers. That tolerates small shifts and shrinks the map. A global
    pool at the end then gives one vector per image.

    `.flatten(1)` lays out everything after the batch dimension in one row
    per example. `.amax(dim=(2, 3))` takes the largest over two dimensions.
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


    # TODO: Finish forward. It gives x, of shape (batch, h, w), a channel
    #   dimension, runs conv and a ReLU, keeps each channel's largest response
    #   over the whole map, and runs head. The result has shape (batch,).
    class PooledDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3)
            self.head = nn.Linear(4, 1)

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Finish forward. It runs conv and a ReLU like PooledDetector, then
    #   flattens the maps into one row per image for head. The result has shape
    #   (batch,).
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
        description="finds the cross where it never saw it, once trained",
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
        "PooledDetector.forward": "Like conv4's Detector, with .amax(dim=(2, 3)) over the height and the width.",
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
