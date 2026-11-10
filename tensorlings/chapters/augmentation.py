"""35_augmentation, flips, padded crops and shifts as tensor operations, and what they buy a small training set."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    check_on_inputs,
    expect_tensor,
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from ..training import best_device
from .classic_nets import CifarNet, ReferenceLeNet, accuracy_on, cifar_splits, flipped_half, mnist_images, pinned_threads

AUGMENTATION = Chapter(
    folder="35_augmentation",
    title="Data augmentation",
    readme=source_file(
        """
        # Data augmentation

        A model learns more from a small set of images when it also sees
        changed versions of them, such as images shifted a few pixels or
        flipped. A change like this must not change what the image shows, so
        that its label stays correct. This chapter writes those changes as
        tensor operations and measures how much they improve accuracy. It also
        shows which changes keep a label correct and which ones make it wrong.

        ## Further reading

        - [Shorten and Khoshgoftaar, a survey on image data augmentation](https://link.springer.com/article/10.1186/s40537-019-0197-0)
        - [PyTorch, torchvision transforms](https://docs.pytorch.org/vision/stable/transforms.html)
        - [Krizhevsky, Sutskever and Hinton, AlexNet, whose section 4.1 is data augmentation](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
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
    **Data augmentation** makes new training examples by changing existing
    ones in ways that keep their labels correct. A photo of a cat shifted a few
    pixels still shows a cat, and so does its mirror image. The changes are
    chosen at random again at every training step. So the model rarely sees
    the same version of an image twice.

    - a flip: the image mirrored left to right. `.flip(-1)` reverses the last dimension, the width.
    - a padded crop: zeros are added around the image, then a crop of the original size is cut out at a random place.

    A padded crop shifts the image by up to `pad` pixels in each direction.
    With a padding of `pad`, the crop's top row can be any row from 0 to
    `2 * pad` of the padded image. Starting the crop at row `pad` and column
    `pad` gives back the original image, unshifted. CIFAR-10 models usually
    pad by 4 and crop back to 32 by 32.

    ```python
    image = torch.arange(1.0, 5.0).reshape(1, 2, 2)    # one channel, 2 by 2
    padded = F.pad(image, (1, 1, 1, 1))                # shape (1, 4, 4)
    padded[:, 0:2, 0:2]    # tensor([[[0., 0.], [0., 1.]]]), shifted down and right by one
    ```

    The `torchvision` library provides these as ready made transforms. Here
    you write them as plain tensor operations, so you can see every step.
    """,
)

AUGMENT1_SOURCE = source_file(
    '''
    # augment1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    image = torch.arange(1.0, 5.0).reshape(1, 2, 2)
    padded = F.pad(image, (1, 1, 1, 1))              # shape (1, 4, 4), a border of zeros
    mirrored = image.flip(-1)                        # tensor([[[2., 1.], [4., 3.]]])


    # TODO: Return images mirrored left to right. images has shape
    #   (batch, channels, h, w), and so does the result.
    def horizontal_flip(images: Tensor) -> Tensor:
        ...


    # TODO: Pad image with pad zeros on every side, then crop it back to h by w,
    #   starting at row top and column left of the padded image. image has
    #   shape (channels, h, w), and so does the result.
    #   Example: padded_crop(image, 1, 1, 1) returns image unchanged.
    def padded_crop(
        image: Tensor,
        pad: int,
        top: int,
        left: int,
    ) -> Tensor:
        ...


    # TODO: Finish the last line, so every image gets its own padded crop.
    #   Image i uses tops[i] as its top and lefts[i] as its left. images has
    #   shape (batch, channels, h, w), and so does the result.
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

    image = torch.arange(1.0, 5.0).reshape(1, 2, 2)
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
        description="crops each image at its own random place",
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
        "random_shifts": "Build a list of padded_crop(image, pad, int(top), int(left)), looping over images, tops "
        "and lefts together with zip. Then combine the list into one tensor with torch.stack.",
    },
    introduces=(),
    check=check_augment1,
)


# ---------------------------------------------------------------------------
# augment2
# ---------------------------------------------------------------------------

AUGMENT2_LESSON = prose(
    """
    For the table below, two nets were trained on small datasets with Adam,
    on minibatches of 64, with three seeds each. lenet1's LeNet trained for
    600 steps on 500 MNIST digits, with shifts of up to 2 pixels. A small net
    of five convolutions trained for 4,000 steps on 5,000 CIFAR-10 photos,
    with shifts of up to 4 pixels. Its learning rate went down during training
    along a cosine curve. The table gives the accuracy on the training images
    and on the validation images.

    | Augmentation | Digits, training | Digits, validation | Photos, training | Photos, validation |
    |---|---|---|---|---|
    | none | 100% | 89.3% to 90.0% | 100% | 68.2% |
    | shifts | 98.6% to 98.8% | 90.9% to 91.2% | 96.9% | 69.7% to 71.5% |
    | shifts and left to right flips | 93.8% to 96.6% | 83.9% to 85.9% | 90.4% to 90.6% | 71.6% to 72.5% |

    With no augmentation, both nets memorized their training images and got
    100% of them right. With shifts, every step showed slightly different
    images, and validation accuracy went up for both nets.

    Flips helped one net and hurt the other. On photos they raised validation
    accuracy again, on every seed, because a mirrored car or horse still looks
    natural. On digits they lowered validation accuracy by about 6 points. A
    flipped 2 or 7 no longer looks like a 2 or 7, and a flipped 3 is
    backwards. So the model learned from images whose labels were wrong.

    So a change must keep the label correct for the data you have. Rotating a
    6 by half a turn makes a 9. Changing colours works for photos, but a grey
    digit has no colours to change. He and others trained their CIFAR-10
    ResNets with exactly these two changes, shifts of up to 4 pixels and
    flips.

    Augmentation helps most when there is little data and the model is big
    enough to memorize it.

    `torch.where(condition, a, b)` returns a tensor that takes its values from
    `a` where `condition` is `True`, and from `b` everywhere else. The three
    tensors are broadcast to one shape first.
    """,
)

AUGMENT2_SOURCE = source_file(
    '''
    # augment2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    choose = torch.tensor([True, False])
    picked = torch.where(choose, torch.tensor([1, 2]), torch.tensor([9, 9]))    # tensor([1, 9])


    # The shifts from augment1.
    def random_shifts(
        images: Tensor,
        pad: int,
        generator: torch.Generator,
    ) -> Tensor:
        tops = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator)
        lefts = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator)
        h, w = images.shape[2], images.shape[3]
        padded = F.pad(images, (pad, pad, pad, pad))

        return torch.stack([padded[i, :, top:top + h, left:left + w] for i, (top, left) in enumerate(zip(tops, lefts))])


    # TODO: Finish the line that flips. The function already shifts every image
    #   by up to 2 pixels. When flip is True, it should also mirror about half
    #   the images left to right. Draw one random number per image with
    #   torch.rand and generator, in a tensor of shape (batch, 1, 1, 1). Mirror
    #   each image whose number is below 0.5, and leave the others unchanged.
    def augment_batch(
        images: Tensor,
        generator: torch.Generator,
        flip: bool,
    ) -> Tensor:
        images = random_shifts(images, 2, generator)

        if flip:
            images = ...

        return images


    # TODO: Finish the line that sets loss. Augment the 64 images x[batch] with
    #   augment_batch, passing generator and flip. loss is the cross entropy
    #   between the model's scores for the augmented images and the labels
    #   y[batch].
    def train_with_augment(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        flip: bool,
        generator: torch.Generator,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(steps):
            batch = torch.randint(0, len(x), (64,), generator=generator)
            loss = ...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
    ''',
)

AUGMENT2_SOLUTION = source_file(
    '''
    # augment2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    choose = torch.tensor([True, False])
    picked = torch.where(choose, torch.tensor([1, 2]), torch.tensor([9, 9]))


    def random_shifts(
        images: Tensor,
        pad: int,
        generator: torch.Generator,
    ) -> Tensor:
        tops = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator)
        lefts = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator)
        h, w = images.shape[2], images.shape[3]
        padded = F.pad(images, (pad, pad, pad, pad))

        return torch.stack([padded[i, :, top:top + h, left:left + w] for i, (top, left) in enumerate(zip(tops, lefts))])


    def augment_batch(
        images: Tensor,
        generator: torch.Generator,
        flip: bool,
    ) -> Tensor:
        images = random_shifts(images, 2, generator)

        if flip:
            images = torch.where(torch.rand(len(images), 1, 1, 1, generator=generator) < 0.5, images.flip(-1), images)

        return images


    def train_with_augment(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        flip: bool,
        generator: torch.Generator,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(steps):
            batch = torch.randint(0, len(x), (64,), generator=generator)
            loss = F.cross_entropy(model(augment_batch(x[batch], generator, flip)), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
    ''',
)


def reference_augment_batch(
    images: Tensor,
    generator: torch.Generator,
    flip: bool,
) -> Tensor:
    images = reference_random_shifts(images, 2, generator)

    if flip:
        images = torch.where(torch.rand(len(images), 1, 1, 1, generator=generator) < 0.5, images.flip(-1), images)

    return images


def small_net(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)

    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 6 * 6, 3),
    )


def reference_train_with_augment(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    steps: int,
    flip: bool,
    generator: torch.Generator,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(steps):
        batch = torch.randint(0, len(x), (64,), generator=generator)
        loss = F.cross_entropy(model(reference_augment_batch(x[batch], generator, flip)), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def check_train_with_augment(found: object) -> None:
    from ..engine import expect_close

    x = torch.rand(100, 1, 6, 6, generator=torch.Generator().manual_seed(855))
    y = torch.randint(0, 3, (100,), generator=torch.Generator().manual_seed(856))

    for flip in (False, True):
        expect_close(
            call=f"train_with_augment for 10 steps on 100 small images with flip {flip}",
            actual=found(small_net(857), x, y, 10, flip, torch.Generator().manual_seed(858)),
            expected=reference_train_with_augment(small_net(857), x, y, 10, flip, torch.Generator().manual_seed(858)),
            tolerance=1e-5,
        )


def check_augment2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="augment_batch",
        description="shifts every image and flips about half when asked",
        inputs=(
            (small_tensor(850, 6, 1, 4, 4), seeded(851), False),
            (small_tensor(852, 8, 1, 4, 4), seeded(853), True),
        ),
        reference=reference_augment_batch,
        expect=expect_tensor,
    )
    checker.test(
        function="train_with_augment",
        description="trains on augmented minibatches",
        run=check_train_with_augment,
    )


AUGMENT2 = Exercise(
    name="augment2",
    chapter=AUGMENTATION,
    title="How much augmentation helps",
    lesson=AUGMENT2_LESSON,
    source=AUGMENT2_SOURCE,
    solution=AUGMENT2_SOLUTION,
    hints={
        "augment_batch": "Call torch.where with three arguments. The condition is torch.rand(len(images), 1, 1, 1, "
        "generator=generator) < 0.5, and the other two are images.flip(-1) and images.",
        "train_with_augment": "Use F.cross_entropy on model(augment_batch(x[batch], generator, flip)) and y[batch].",
    },
    introduces=(),
    check=check_augment2,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

SETTINGS: tuple[str, ...] = ("none", "shifts", "shifts and flips")


def augment_run(
    kind: str,
    setting: str,
    seed: int,
) -> tuple[float, float]:
    """One row of augment2's tables, trained with Adam at 0.001 on batches of 64. For "digits", lenet1's LeNet learns
    500 MNIST images for 600 steps with shifts of up to 2 pixels. For "photos", the CifarNet learns 5,000 CIFAR-10
    images for 4,000 steps, its rate falling along a cosine, with shifts of up to 4 as CIFAR-10 models usually take.
    Gives the accuracy on the training images and on the validation images, 3,000 for digits and 5,000 for photos."""
    device = best_device()

    if kind == "digits":
        images = mnist_images()
        x, y = images["train_x"][:500], images["train_y"][:500]
        held_x, held_y = images["validation_x"][:3000], images["validation_y"][:3000]
        torch.manual_seed(seed)
        model = ReferenceLeNet().to(device)
        steps, pad = 600, 2
    else:
        sets = cifar_splits()
        x, y = sets["train_x"][:5000], sets["train_y"][:5000]
        held_x, held_y = sets["validation_x"], sets["validation_y"]
        torch.manual_seed(seed)
        model = CifarNet(10).to(device)
        steps, pad = 4000, 4

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # The photos' longer run ends on a falling rate, so the last steps settle and no seed is caught mid swing.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps) if kind == "photos" else None
    generator = torch.Generator().manual_seed(seed + 20)
    train_x, train_y = x.to(device), y.to(device)

    for _ in range(steps):
        batch = torch.randint(0, len(train_x), (64,), generator=generator).to(device)
        batch_x = train_x[batch]

        if setting != "none":
            batch_x = reference_random_shifts(batch_x, pad, generator)

        if setting == "shifts and flips":
            batch_x = flipped_half(batch_x, generator)

        loss = F.cross_entropy(model(batch_x), train_y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if schedule is not None:
            schedule.step()

    return accuracy_on(model, x, y, device), accuracy_on(model, held_x, held_y, device)


def measure_augmentation(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """augment2's two tables, keyed by the kind of image and the setting."""
    results: dict[tuple[str, str], list[tuple[float, float]]] = {}

    with pinned_threads():
        for kind in ("digits", "photos"):
            for setting in SETTINGS:
                for seed in seeds:
                    results.setdefault((kind, setting), []).append(augment_run(kind, setting, seed))

    return results
