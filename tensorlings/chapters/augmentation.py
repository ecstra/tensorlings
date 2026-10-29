"""35_augmentation, flips, padded crops and shifts as tensor operations, and what they buy a small training set."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    check_on_inputs,
    expect_tensor,
    seeded,
    small_tensor,
)
from ..text import prose, source_file

AUGMENTATION = Chapter(
    folder="35_augmentation",
    title="Data augmentation",
    readme=source_file(
        """
        # Data augmentation

        A model learns more from a small set of images when it sees them
        changed in ways that keep their labels, shifted a little or flipped.
        This chapter writes those changes as tensor operations, measures what
        they buy, and sorts the changes that keep a label from the ones that
        break it.

        ## Further reading

        - [Shorten and Khoshgoftaar, a survey on image data augmentation](https://link.springer.com/article/10.1186/s40537-019-0197-0)
        - [PyTorch, torchvision transforms](https://docs.pytorch.org/vision/stable/transforms.html)
        - [CS231n, data augmentation](https://cs231n.github.io/neural-networks-2/#reg)
        """,
    ),
)


def reference_padded_crop(
    image: Tensor,
    pad: int,
    top: int,
    left: int,
) -> Tensor:
    h, w = image.shape[1], image.shape[2]

    return F.pad(image, (pad, pad, pad, pad))[:, top : top + h, left : left + w]


def reference_random_shifts(
    images: Tensor,
    pad: int,
    generator: torch.Generator,
) -> Tensor:
    tops = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator)
    lefts = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator)

    return torch.stack([reference_padded_crop(image, pad, int(top), int(left)) for image, top, left in zip(images, tops, lefts)])


# ---------------------------------------------------------------------------
# augment1
# ---------------------------------------------------------------------------

AUGMENT1_LESSON = prose(
    """
    **Data augmentation** makes new training examples by changing old ones in
    ways that keep their labels. A cat shifted a few pixels is still a cat,
    and so is a cat seen in a mirror. The changes are drawn fresh at every
    step, so the model never sees exactly the same image twice.

    - a flip: the image mirrored left to right. `.flip(-1)` reverses the last dimension, the width.
    - a padded crop: zeros added around, then a crop of the original size taken from a random place in it.

    A padded crop shifts the image by up to `pad` pixels each way. With a
    padding of `pad`, the crop's top row can start anywhere from 0 to
    `2 * pad`, and `pad` gives the image back unshifted. CIFAR-10 models
    usually pad by 4 and crop back to 32 by 32.

    ```python
    image = torch.arange(4.0).reshape(1, 2, 2)    # one channel, 2 by 2
    padded = F.pad(image, (1, 1, 1, 1))           # shape (1, 4, 4)
    padded[:, 0:2, 0:2]                            # shifted down and right by one
    ```

    `torchvision` holds these as ready made transforms. Here they are plain
    tensor operations, so every step is visible.
    """,
)

AUGMENT1_SOURCE = source_file(
    '''
    # augment1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    image = torch.arange(4.0).reshape(1, 2, 2)
    padded = F.pad(image, (1, 1, 1, 1))              # shape (1, 4, 4), a border of zeros
    mirrored = image.flip(-1)                        # tensor([[[1., 0.], [3., 2.]]])


    # TODO: Return images, of shape (batch, channels, h, w), mirrored left to
    #   right.
    def horizontal_flip(images: Tensor) -> Tensor:
        ...


    # TODO: Return image, of shape (channels, h, w), padded by pad zeros on every
    #   side, then cropped back to h by w starting at row top and column left.
    def padded_crop(
        image: Tensor,
        pad: int,
        top: int,
        left: int,
    ) -> Tensor:
        ...


    # TODO: Finish the last line, so every image gets its own padded crop, at the
    #   top and left drawn for it.
    def random_shifts(
        images: Tensor,
        pad: int,
        generator: torch.Generator,
    ) -> Tensor:
        tops = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator)
        lefts = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator)

        return ...
    ''',
)

AUGMENT1_SOLUTION = source_file(
    '''
    # augment1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    image = torch.arange(4.0).reshape(1, 2, 2)
    padded = F.pad(image, (1, 1, 1, 1))
    mirrored = image.flip(-1)


    def horizontal_flip(images: Tensor) -> Tensor:
        return images.flip(-1)


    def padded_crop(
        image: Tensor,
        pad: int,
        top: int,
        left: int,
    ) -> Tensor:
        h, w = image.shape[1], image.shape[2]

        return F.pad(image, (pad, pad, pad, pad))[:, top:top + h, left:left + w]


    def random_shifts(
        images: Tensor,
        pad: int,
        generator: torch.Generator,
    ) -> Tensor:
        tops = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator)
        lefts = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator)

        return torch.stack([padded_crop(image, pad, int(top), int(left)) for image, top, left in zip(images, tops, lefts)])
    ''',
)


def check_augment1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="horizontal_flip",
        description="mirrors every image left to right",
        inputs=(
            (small_tensor(840, 2, 1, 3, 3),),
            (small_tensor(841, 1, 3, 2, 4),),
        ),
        reference=lambda images: images.flip(-1),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="padded_crop",
        description="pads, then crops back to the original size",
        inputs=(
            (small_tensor(842, 1, 3, 3), 1, 0, 2),
            (small_tensor(843, 2, 4, 5), 2, 4, 1),
            (small_tensor(844, 1, 3, 3), 2, 2, 2),
        ),
        reference=reference_padded_crop,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="random_shifts",
        description="crops each image at its own drawn place",
        inputs=(
            (small_tensor(845, 4, 1, 3, 3), 1, seeded(846)),
            (small_tensor(847, 3, 2, 4, 4), 2, seeded(848)),
        ),
        reference=reference_random_shifts,
        expect=expect_tensor,
    )


AUGMENT1 = Exercise(
    name="augment1",
    chapter=AUGMENTATION,
    title="Flips and shifts",
    lesson=AUGMENT1_LESSON,
    source=AUGMENT1_SOURCE,
    solution=AUGMENT1_SOLUTION,
    hints={
        "horizontal_flip": "Flip the last dimension, the width, with .flip(-1).",
        "padded_crop": "Take h and w from image.shape. Pad with F.pad(image, (pad, pad, pad, pad)), then slice rows "
        "top up to top + h and columns left up to left + w.",
        "random_shifts": "torch.stack a list of padded_crop(image, pad, int(top), int(left)), looping over images, "
        "tops and lefts together with zip.",
    },
    introduces=(),
    check=check_augment1,
)
