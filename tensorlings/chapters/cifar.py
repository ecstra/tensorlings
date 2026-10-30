"""37_cifar, Section 5's project, a small ResNet with augmentation on CIFAR-10, trained for minutes on the best device and measured once on test."""

from __future__ import annotations

import functools
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import load_cifar10
from ..engine import (
    Checker,
    Described,
    DeviceRecorder,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_on_device,
    expect_tensor,
    expect_value,
    fresh_copy,
    seeded,
)
from ..text import prose, source_file
from ..training import MissingCheckpoint, load_checkpoint
from .linear_layers import expect_module

CIFAR = Chapter(
    folder="37_cifar",
    title="Project: CIFAR-10",
    readme=source_file(
        """
        # Project: CIFAR-10

        Section 5's project. You train a small ResNet with augmentation on
        CIFAR-10, 60,000 small color photos of ten kinds of things, past 80
        percent on a validation set, then measure it once on the test set, in
        the style of CS231n's convolutional network assignment. The training
        runs for minutes outside the app, on the fastest device the machine
        has. Download the data once from the project folder.

        ```bash
        .venv/bin/python run.py data cifar10
        ```

        ## Further reading

        - [Krizhevsky, learning multiple layers of features from tiny images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
        - [CS231n, assignment 2](https://cs231n.github.io/assignments2025/assignment2/)
        - [Karpathy, a recipe for training neural networks](https://karpathy.github.io/2019/04/25/recipe/)
        """,
    ),
)

VALIDATION_SIZE = 5000


def images_for_check(
    seed: int,
    count: int,
) -> Tensor:
    return torch.randint(0, 256, (count, 3, 4, 4), generator=torch.Generator().manual_seed(seed), dtype=torch.uint8)


def reference_channel_stats(x: Tensor) -> tuple[Tensor, Tensor]:
    return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))


def reference_prepare(
    images: Tensor,
    labels: Tensor,
    validation_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    x = images.float() / 255
    order = torch.randperm(len(x), generator=generator)
    train, validation = order[:-validation_size], order[-validation_size:]
    mean, spread = reference_channel_stats(x[train])
    x = (x - mean.view(1, -1, 1, 1)) / spread.view(1, -1, 1, 1)

    return x[train], labels[train], x[validation], labels[validation]


@functools.cache
def cifar_sets() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The split every part of the project uses, seed 0 and 5,000 for validation, built once per run of the app."""
    images, labels = load_cifar10("train")

    return reference_prepare(
        images=images,
        labels=labels,
        validation_size=VALIDATION_SIZE,
        generator=torch.Generator().manual_seed(0),
    )


def validation_set() -> tuple[Tensor, Tensor]:
    _, _, validation_x, validation_y = cifar_sets()

    return fresh_copy(validation_x), fresh_copy(validation_y)


# ---------------------------------------------------------------------------
# cifar1
# ---------------------------------------------------------------------------

CIFAR1_LESSON = prose(
    """
    This is Section 5's project. You train a small ResNet on CIFAR-10, 60,000
    color photos of 32 by 32 pixels, each one of ten kinds. Download the data
    once, about 170 MB, from the project folder with
    `.venv/bin/python run.py data cifar10`.

    ```python
    from tensorlings.data import CIFAR10_CLASSES, load_cifar10

    images, labels = load_cifar10("train")
    images.shape                # torch.Size([50000, 3, 32, 32])
    CIFAR10_CLASSES[labels[0]]  # 'frog'
    ```

    The 3 is the color channels, red, green and blue, from 0 to 255. Each
    class has 5,000 training images, and the 10,000 test images stay
    untouched until cifar3. The validation set comes out of the 50,000.

    Look at the data before training on it, the first step of Karpathy's
    recipe. Even the average color of each class says something. Take each
    class's average blue minus its average red, with pixels from 0 to 1.
    Airplanes and ships come out at 0.063 and 0.064, sky and sea. Frogs come
    out lowest, at -0.125.

    Each channel then gets its own normalization. Take the channel's mean away
    and divide by its spread, over every image and every pixel.

    ```python
    x = images.float() / 255
    x.mean(dim=(0, 2, 3))    # tensor([0.4914, 0.4822, 0.4465]), one per channel
    ```

    Those statistics come from the training part only. The validation set
    stands in for photos the model has never seen, so nothing about it may
    shape the model, the scaling included. Here the difference is tiny, 0.4912
    against 0.4914 for red. The rule holds anyway, since on smaller or skewed
    data it is not tiny.
    """,
)

CIFAR1_SOURCE = source_file(
    '''
    # cifar1.py
    #
    # Section 5's project starts with the data. Download it once from the project
    # folder with
    #
    #     .venv/bin/python run.py data cifar10

    import torch
    from torch import Tensor

    from tensorlings.data import CIFAR10_CLASSES, load_cifar10

    images, labels = load_cifar10("train")    # shapes (50000, 3, 32, 32) and (50000,)
    first_class = CIFAR10_CLASSES[labels[0]]  # 'frog'

    x = images.float() / 255
    reds = x[:, 0]                             # every image's red channel, shape (50000, 32, 32)


    # TODO: Return each channel's mean and spread, from torch.std, over every
    #   image and every pixel of x, of shape (count, channels, height, width).
    #   Each has one number per channel.
    def channel_stats(x: Tensor) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return the average color of each class, of shape (classes, 3), one
    #   row per class with its mean red, green and blue over its images.
    def class_colors(
        x: Tensor,
        labels: Tensor,
        classes: int,
    ) -> Tensor:
        ...


    # TODO: Finish the middle lines. This scales the pixels to 0 to 1, shuffles
    #   with generator, and keeps the last validation_size shuffled positions
    #   for validation. Normalize every image with channel_stats of the
    #   training part alone. It returns the training images and labels, then
    #   the validation ones.
    def prepare(
        images: Tensor,
        labels: Tensor,
        validation_size: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = images.float() / 255
        order = torch.randperm(len(x), generator=generator)
        train, validation = order[:-validation_size], order[-validation_size:]
        mean, spread = ...
        x = ...

        return x[train], labels[train], x[validation], labels[validation]
    ''',
)

CIFAR1_SOLUTION = source_file(
    '''
    # cifar1.py, a solution for comparison

    import torch
    from torch import Tensor

    from tensorlings.data import CIFAR10_CLASSES, load_cifar10

    images, labels = load_cifar10("train")
    first_class = CIFAR10_CLASSES[labels[0]]

    x = images.float() / 255
    reds = x[:, 0]


    def channel_stats(x: Tensor) -> tuple[Tensor, Tensor]:
        return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))


    def class_colors(
        x: Tensor,
        labels: Tensor,
        classes: int,
    ) -> Tensor:
        return torch.stack([x[labels == label].mean(dim=(0, 2, 3)) for label in range(classes)])


    def prepare(
        images: Tensor,
        labels: Tensor,
        validation_size: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = images.float() / 255
        order = torch.randperm(len(x), generator=generator)
        train, validation = order[:-validation_size], order[-validation_size:]
        mean, spread = channel_stats(x[train])
        x = (x - mean.view(1, -1, 1, 1)) / spread.view(1, -1, 1, 1)

        return x[train], labels[train], x[validation], labels[validation]
    ''',
)


def expect_pair_of_tensors(
    call: str,
    actual: object,
    expected: tuple[Tensor, ...],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != len(expected):
        raise Mismatch(f"{call} should give {len(expected)} tensors in a tuple, in the order the TODO lists.")

    for position, (part, wanted) in enumerate(zip(actual, expected), start=1):
        expect_tensor(
            call=f"part {position} of {call}",
            actual=part,
            expected=wanted,
        )


def check_cifar1(checker: Checker) -> None:
    floats = images_for_check(370, 6).float() / 255
    check_on_inputs(
        checker=checker,
        function="channel_stats",
        description="gives each channel's mean and spread",
        inputs=(
            (Described(floats, "6 images of 3 channels, 4 by 4"),),
            (Described(floats[:3, :2], "3 images of 2 channels, 4 by 4"),),
        ),
        reference=reference_channel_stats,
        expect=expect_pair_of_tensors,
    )
    labels = torch.tensor([0, 2, 1, 2, 0, 1])
    check_on_inputs(
        checker=checker,
        function="class_colors",
        description="gives each class's average color",
        inputs=(
            (Described(floats, "6 images of 3 channels"), labels, 3),
        ),
        reference=lambda x, labels, classes: torch.stack([x[labels == label].mean(dim=(0, 2, 3)) for label in range(classes)]),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="prepare",
        description="splits, then normalizes with the training part's statistics",
        inputs=(
            (Described(images_for_check(371, 12), "12 images"), torch.arange(12), 4, seeded(372)),
            (Described(images_for_check(373, 9), "9 images"), torch.arange(9) % 3, 3, seeded(374)),
        ),
        reference=reference_prepare,
        expect=expect_pair_of_tensors,
    )


CIFAR1 = Exercise(
    name="cifar1",
    chapter=CIFAR,
    title="Project: the data",
    lesson=CIFAR1_LESSON,
    source=CIFAR1_SOURCE,
    solution=CIFAR1_SOLUTION,
    hints={
        "channel_stats": "x.mean and x.std with dim=(0, 2, 3) leave one number per channel.",
        "class_colors": "x[labels == label] picks one class's images. Take its mean over dim=(0, 2, 3), for each "
        "label in range(classes), and torch.stack the rows.",
        "prepare": "mean, spread = channel_stats(x[train]). Then take mean.view(1, -1, 1, 1) away from x and divide "
        "by spread.view(1, -1, 1, 1).",
    },
    introduces=(),
    check=check_cifar1,
    time_limit=60,
)


# ---------------------------------------------------------------------------
# cifar2
# ---------------------------------------------------------------------------

# What cifar2 and cifar3 are handed, written once so the two files never drift apart.
PREPARE_PART = '''
# prepare from cifar1.
def prepare(
    images: Tensor,
    labels: Tensor,
    validation_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    x = images.float() / 255
    order = torch.randperm(len(x), generator=generator)
    train, validation = order[:-validation_size], order[-validation_size:]
    mean, spread = x[train].mean(dim=(0, 2, 3)), x[train].std(dim=(0, 2, 3))
    x = (x - mean.view(1, -1, 1, 1)) / spread.view(1, -1, 1, 1)

    return x[train], labels[train], x[validation], labels[validation]
'''

BLOCK_PART = '''
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
'''

AUGMENT_PART = '''
# The shifts and flips of augment2, with shifts of up to 4 pixels.
def augment(
    images: Tensor,
    generator: torch.Generator,
) -> Tensor:
    pad = 4
    h, w = images.shape[2], images.shape[3]
    tops = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator).tolist()
    lefts = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator).tolist()
    padded = F.pad(images, (pad, pad, pad, pad))
    shifted = torch.stack([padded[i, :, top : top + h, left : left + w] for i, (top, left) in enumerate(zip(tops, lefts))])
    flips = torch.rand(len(images), 1, 1, 1, generator=generator) < 0.5

    return torch.where(flips, shifted.flip(-1), shifted)
'''

SMALL_RESNET_SOLUTION = '''
class SmallResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            BasicBlock(32, 32, 1),
            BasicBlock(32, 64, 2),
            BasicBlock(64, 128, 2),
        )
        self.head = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        features = self.blocks(self.stem(x))

        return self.head(features.mean(dim=(2, 3)))
'''

EVALUATE_SOLUTION = '''
def evaluate(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    device: torch.device,
    batch_size: int = 500,
) -> float:
    model.eval()
    correct = 0

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            scores = model(x[start : start + batch_size].to(device))
            correct += (scores.argmax(dim=1).cpu() == y[start : start + batch_size]).sum().item()

    model.train()

    return correct / len(x)
'''

# Indented like the files around it, which source_file takes the indentation off.
INDENTED_PREPARE = textwrap.indent(PREPARE_PART.strip("\n"), "    ")
INDENTED_BLOCK = textwrap.indent(BLOCK_PART.strip("\n"), "    ")
INDENTED_AUGMENT = textwrap.indent(AUGMENT_PART.strip("\n"), "    ")
INDENTED_SMALL_RESNET = textwrap.indent(SMALL_RESNET_SOLUTION.strip("\n"), "    ")
INDENTED_EVALUATE = textwrap.indent(EVALUATE_SOLUTION.strip("\n"), "    ")

CIFAR2_MAIN = '''
    # Run by .venv/bin/python run.py train cifar2, never by the app.
    def main() -> None:
        images, labels = load_cifar10("train")
        train_x, train_y, validation_x, validation_y = prepare(images, labels, 5000, torch.Generator().manual_seed(0))
        device = best_device()
        torch.manual_seed(0)
        model = SmallResNet().to(device)
        epochs = 15
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * (len(train_x) // 128))
        generator = torch.Generator().manual_seed(1)
        print(f"Training on {device} for {epochs} epochs.")

        for epoch in range(epochs):
            loss = train_epoch(model, optimizer, scheduler, train_x, train_y, device, generator)
            accuracy = evaluate(model, validation_x, validation_y, device)
            print(f"Epoch {epoch + 1}, loss {loss:.3f}, validation accuracy {accuracy:.1%}")

        save_checkpoint(model, "cifar2")
        print("Saved the model in checkpoints/cifar2.pt. Save cifar2.py in the app to check it.")
'''

CIFAR2_LESSON = prose(
    """
    This step trains for minutes, so it runs outside the app. Once
    `SmallResNet`, `evaluate` and `train_epoch` pass, train from the project
    folder with `.venv/bin/python run.py train cifar2`. That runs the file's
    `main()`, which trains for 15 epochs and saves the model in
    `checkpoints/cifar2.pt`. The last check here reads it back.

    It trains on the fastest device the machine has, which `best_device()`
    picks. That is an NVIDIA GPU through CUDA, Apple silicon's GPU through
    MPS, or the CPU when there is neither. A model and every batch it sees
    must sit on the same device, and `.to(device)` moves either one.

    ```python
    model = model.to(device)            # every weight moves
    scores = model(images.to(device))   # the batch moves to meet it
    scores.argmax(dim=1).cpu()          # and the answer comes back
    ```

    The model is a small ResNet. A stem convolution, three basic blocks that
    halve the image twice while doubling the channels, global average pooling
    and a Linear head. It has 308,650 weights.

    Augmentation runs on the CPU, where the generator lives, then each batch
    moves to the device. It uses augment2's shifts and flips, with shifts of
    up to 4 pixels. The optimizer is SGD with momentum and weight decay. The
    rate follows schedule1's `CosineAnnealingLR`, stepped after every batch.

    Measuring runs in batches with the model in eval mode, so BatchNorm uses
    its running statistics and leaves them alone. `model.train()` switches it
    back afterwards.

    Here is the run behind this project, on MPS.

    | Epoch | 1 | 5 | 10 | 15 |
    |---|---|---|---|---|
    | Validation accuracy | 46.1% | 71.9% | 82.3% | 85.6% |

    It took about 11 minutes here, while other work shared the machine. On a
    CPU alone expect it to take several times longer.
    """,
)

CIFAR2_SOURCE = source_file(
    '''
    # cifar2.py
    #
    # Train once the checks above the last one pass, from the project folder.
    #
    #     .venv/bin/python run.py train cifar2

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_cifar10
    from tensorlings.training import best_device, save_checkpoint

    images = torch.zeros(4, 3, 32, 32)
    moved = images.to(best_device())    # the same numbers, on the fastest device


''' + INDENTED_PREPARE + '''


''' + INDENTED_BLOCK + '''


''' + INDENTED_AUGMENT + '''


    # TODO: Finish SmallResNet. The stem is an nn.Sequential of a 3 by 3
    #   nn.Conv2d from 3 to 32 channels with padding 1 and no bias, an
    #   nn.BatchNorm2d and a ReLU. blocks is an nn.Sequential of
    #   BasicBlock(32, 32, 1), BasicBlock(32, 64, 2) and BasicBlock(64, 128, 2).
    #   The head is a Linear from 128 to 10. forward runs the stem and the
    #   blocks, averages each channel over the image, and runs the head.
    class SmallResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = ...
            self.blocks = ...
            self.head = ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return the share of x that model labels right, as a plain number.
    #   Go through x in batches of batch_size, each moved to device, in eval
    #   mode and without recording for autograd. Then put model back in
    #   training mode.
    def evaluate(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        device: torch.device,
        batch_size: int = 500,
    ) -> float:
        ...


    # TODO: Finish the loop. For each batch, augment its images on the CPU, move
    #   them and their labels to device, and take a step of optimizer, then a
    #   step of scheduler. It returns the last batch's loss.
    def train_epoch(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        x: Tensor,
        y: Tensor,
        device: torch.device,
        generator: torch.Generator,
        batch_size: int = 128,
    ) -> float:
        order = torch.randperm(len(x), generator=generator)

        for start in range(0, len(x) - batch_size + 1, batch_size):
            batch = order[start : start + batch_size]
            ...

        return loss.item()

''' + CIFAR2_MAIN,
)

CIFAR2_SOLUTION = source_file(
    '''
    # cifar2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_cifar10
    from tensorlings.training import best_device, save_checkpoint

    images = torch.zeros(4, 3, 32, 32)
    moved = images.to(best_device())


''' + INDENTED_PREPARE + '''


''' + INDENTED_BLOCK + '''


''' + INDENTED_AUGMENT + '''


''' + INDENTED_SMALL_RESNET + '''


''' + INDENTED_EVALUATE + '''


    def train_epoch(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        x: Tensor,
        y: Tensor,
        device: torch.device,
        generator: torch.Generator,
        batch_size: int = 128,
    ) -> float:
        order = torch.randperm(len(x), generator=generator)

        for start in range(0, len(x) - batch_size + 1, batch_size):
            batch = order[start : start + batch_size]
            inputs = augment(x[batch], generator).to(device)
            loss = F.cross_entropy(model(inputs), y[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return loss.item()

''' + CIFAR2_MAIN,
)


def reference_namespace() -> dict[str, object]:
    """The given parts and the solution's model and evaluate, run once, so the checks compare against
    exactly the code the learner's file holds."""
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(PREPARE_PART + BLOCK_PART + AUGMENT_PART + SMALL_RESNET_SOLUTION + EVALUATE_SOLUTION, namespace)

    return namespace


REFERENCE = reference_namespace()


def seeded_resnet(make: type) -> nn.Module:
    torch.manual_seed(0)

    return make()


def tiny_net() -> nn.Module:
    """A net with BatchNorm, so evaluate's eval mode shows, and small enough to train in a check."""
    torch.manual_seed(375)

    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 10),
    )


def tiny_images(
    seed: int,
    count: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)

    return torch.randn(count, 3, 8, 8, generator=generator), torch.randint(0, 10, (count,), generator=generator)


def check_small_resnet(found: type) -> None:
    model = expect_module("SmallResNet()", seeded_resnet(found))
    reference = seeded_resnet(REFERENCE["SmallResNet"])
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    wanted = {name: tuple(value.shape) for name, value in reference.state_dict().items()}

    if shapes != wanted:
        missing = sorted(set(wanted) - set(shapes))[:3]
        raise Mismatch(
            "SmallResNet() should hold the layers the TODO lists under the names stem, blocks and head. "
            f"It is missing {', '.join(missing) or 'nothing'}, or some of its layers have other shapes.",
        )

    x = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(376))
    model.eval()
    reference.eval()

    with torch.no_grad():
        expected = reference(x)

    expect_tensor(
        call="SmallResNet() on 2 images of 32 by 32",
        actual=model(x),
        expected=expected,
    )


def check_evaluate(found: object) -> None:
    model = tiny_net()
    x, y = tiny_images(377, 60)
    expected = REFERENCE["evaluate"](tiny_net(), x, y, torch.device("cpu"), 25)
    running_mean = model[1].running_mean.clone()
    accuracy = found(model, x, y, torch.device("cpu"), 25)

    if not torch.equal(model[1].running_mean, running_mean):
        raise Mismatch("evaluate changed BatchNorm's running mean, so the model was still in training mode while it measured.")

    if not model.training:
        raise Mismatch("evaluate left the model in eval mode. Call model.train() before returning.")

    expect_close(
        call="evaluate on 60 images in batches of 25",
        actual=accuracy,
        expected=expected,
    )

    recorder = DeviceRecorder(
        names=("the images",),
        output=lambda images: images.flatten(1)[:, :10],
    )
    expect_on_device(
        function="evaluate",
        run=lambda: found(recorder, x, y, torch.device("meta"), 25),
        recorders=[recorder],
        advice="Move each batch of images to device. Then compare on one device, the predictions brought back "
        "with .cpu() or the labels moved to device too.",
    )


def reference_train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    x: Tensor,
    y: Tensor,
    generator: torch.Generator,
    batch_size: int,
) -> float:
    order = torch.randperm(len(x), generator=generator)

    for start in range(0, len(x) - batch_size + 1, batch_size):
        batch = order[start : start + batch_size]
        loss = F.cross_entropy(model(REFERENCE["augment"](x[batch], generator)), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return loss.item()


def check_train_epoch(found: object) -> None:
    x, y = tiny_images(378, 50)
    runs = []

    for train in (found, None):
        model = tiny_net()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        generator = torch.Generator().manual_seed(379)

        if train is None:
            loss = reference_train_epoch(model, optimizer, scheduler, x, y, generator, 16)
        else:
            loss = train(model, optimizer, scheduler, x, y, torch.device("cpu"), generator, 16)

        runs.append((loss, model[-1].weight.detach().clone(), scheduler.last_epoch))

    (loss, weight, steps), (expected_loss, expected_weight, expected_steps) = runs

    if steps != expected_steps:
        raise Mismatch(f"After one epoch of 3 batches the scheduler took {steps} steps, but it should take {expected_steps}, one per batch.")

    expect_close(
        call="train_epoch on 50 images in batches of 16",
        actual=loss,
        expected=expected_loss,
        tolerance=1e-5,
    )
    expect_tensor(
        call="the head's weights after train_epoch",
        actual=weight,
        expected=expected_weight,
    )

    recorder = DeviceRecorder(
        names=("the images",),
        output=lambda images: images.flatten(1)[:, :10],
    )
    optimizer = torch.optim.SGD(recorder.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    expect_on_device(
        function="train_epoch",
        run=lambda: found(recorder, optimizer, scheduler, x, y, torch.device("meta"), torch.Generator().manual_seed(379), 16),
        recorders=[recorder],
        advice="Augment each batch on the CPU, then move its images and its labels to device.",
    )


def check_trained_model(found: type) -> None:
    model = seeded_resnet(found)

    try:
        load_checkpoint(model, "cifar2")
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None
    except RuntimeError:
        raise Mismatch(
            "The saved model doesn't fit SmallResNet as it is now. Train it again from the project folder.\n\n"
            "    .venv/bin/python run.py train cifar2",
        ) from None

    validation_x, validation_y = validation_set()
    accuracy = REFERENCE["evaluate"](model, validation_x, validation_y, torch.device("cpu"))

    if accuracy < 0.8:
        raise Mismatch(f"The trained model labels {accuracy:.1%} of the validation set right, but it should pass 80%.")


def check_cifar2(checker: Checker) -> None:
    checker.test_method(
        class_name="SmallResNet",
        method="forward",
        description="builds the stem, the three blocks and the head",
        run=check_small_resnet,
    )
    checker.test(
        function="evaluate",
        description="measures in eval mode, in batches, then switches back",
        run=check_evaluate,
    )
    checker.test(
        function="train_epoch",
        description="steps the optimizer and the scheduler once per batch",
        run=check_train_epoch,
    )
    checker.test_method(
        class_name="SmallResNet",
        method="forward",
        description="trained, passes 80 percent on the validation set",
        run=check_trained_model,
    )


CIFAR2 = Exercise(
    name="cifar2",
    chapter=CIFAR,
    title="Project: training on the GPU",
    lesson=CIFAR2_LESSON,
    source=CIFAR2_SOURCE,
    solution=CIFAR2_SOLUTION,
    hints={
        "SmallResNet.forward": "Run self.stem, then self.blocks. Average the result over dim=(2, 3), then run "
        "self.head.",
        "evaluate": "Call model.eval(), and inside torch.no_grad() loop over range(0, len(x), batch_size). Compare "
        "each batch's argmax, moved back with .cpu(), to its labels, and count the matches.",
        "train_epoch": "inputs = augment(x[batch], generator).to(device). The loss is F.cross_entropy of model(inputs) "
        "and y[batch].to(device). Then zero the gradients, backward, optimizer.step() and scheduler.step().",
    },
    introduces=(),
    check=check_cifar2,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# cifar3
# ---------------------------------------------------------------------------

CIFAR3_LESSON = prose(
    """
    The test set gets used once, at the very end, as mnist3 did. Every choice
    is made by now, so its number is an honest guess at how the model does on
    photos it has never seen.

    The test images get the training images' treatment, scaled to 0 to 1 and
    normalized with the training part's mean and spread. The model learned on
    inputs scaled exactly that way, so the test images have to match.

    mnist3 built a confusion matrix with a loop. names1's trick builds it in
    one go. Each pair of a true class and a predicted class becomes one number,
    `true * classes + predicted`, and `torch.bincount` counts them all.

    ```python
    pairs = torch.tensor([0, 1, 1]) * 2 + torch.tensor([0, 0, 1])    # tensor([0, 2, 3])
    torch.bincount(pairs, minlength=4).reshape(2, 2)                  # tensor([[1, 0], [1, 1]])
    ```

    `bincount` stops at the largest number it sees. `minlength` keeps a slot
    for every pair, even one that never happened, so the reshape always fits.

    `.topk(k)` gives the k largest entries, largest first, and where they sit.

    The run behind this project labels 85.3% of the test set right, against
    85.6% on validation. Test close to validation means the validation set
    did its job, a fair stand in for photos never seen. Cats are its weakest
    class at 70.8%, then dogs at 78.6%, and ships its best at 92.6%. Here are
    its largest mistakes on the 10,000 test images.

    | True class | Read as | Times |
    |---|---|---|
    | cat | dog | 115 |
    | dog | cat | 108 |
    | bird | deer | 54 |
    | cat | bird | 50 |
    | automobile | truck | 47 |

    Cats and dogs swap the most. At 32 by 32 pixels both are a furry shape
    with four legs, often in the same poses. Your own run will land close to
    these numbers but not on them.
    """,
)

CIFAR3_SOURCE = source_file(
    '''
    # cifar3.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    from tensorlings.data import CIFAR10_CLASSES, load_cifar10
    from tensorlings.training import load_checkpoint

    pairs = torch.tensor([0, 1, 1]) * 2 + torch.tensor([0, 0, 1])
    counts = torch.bincount(pairs, minlength=4).reshape(2, 2)    # tensor([[1, 0], [1, 1]])
    largest = torch.tensor([5, 9, 1]).topk(2)                    # values 9 and 5, at 1 and 0


''' + INDENTED_BLOCK + '''


    # SmallResNet from cifar2.
''' + INDENTED_SMALL_RESNET + '''


    # TODO: Return model's predicted class for every image of x, going through
    #   x in batches of batch_size, in eval mode and without recording for
    #   autograd.
    def predictions(
        model: nn.Module,
        x: Tensor,
        batch_size: int = 500,
    ) -> Tensor:
        ...


    # TODO: Return the confusion matrix of predicted against labels, of shape
    #   (classes, classes), with one row per true class, built with one
    #   torch.bincount and no loop.
    def confusion_matrix(
        predicted: Tensor,
        labels: Tensor,
        classes: int,
    ) -> Tensor:
        ...


    # TODO: Return the share of each true class that matrix shows labelled
    #   right, as floats, one per class.
    def per_class_accuracy(matrix: Tensor) -> Tensor:
        ...


    # TODO: Return the count largest mistakes of matrix, largest first, each as
    #   a (true class, predicted class) pair of ints.
    def top_confusions(
        matrix: Tensor,
        count: int,
    ) -> list[tuple[int, int]]:
        ...


    # The training part's statistics, from the split cifar1 and cifar2 used.
    train_images, _ = load_cifar10("train")
    order = torch.randperm(len(train_images), generator=torch.Generator().manual_seed(0))
    train_part = train_images[order[:-5000]].float() / 255
    mean, spread = train_part.mean(dim=(0, 2, 3)), train_part.std(dim=(0, 2, 3))
    test_images, test_y = load_cifar10("test")

    # TODO: Set test_x to test_images scaled to 0 to 1 and normalized with mean
    #   and spread, the way training saw its images.
    test_x = ...

    model = load_checkpoint(SmallResNet(), "cifar2")
    matrix = confusion_matrix(predictions(model, test_x), test_y, 10)

    # TODO: Set test_accuracy to the share of the test set that model labels
    #   right, as a plain number, from matrix.
    test_accuracy = ...
    ''',
)

CIFAR3_SOLUTION = source_file(
    '''
    # cifar3.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    from tensorlings.data import CIFAR10_CLASSES, load_cifar10
    from tensorlings.training import load_checkpoint

    pairs = torch.tensor([0, 1, 1]) * 2 + torch.tensor([0, 0, 1])
    counts = torch.bincount(pairs, minlength=4).reshape(2, 2)
    largest = torch.tensor([5, 9, 1]).topk(2)


''' + INDENTED_BLOCK + '''


''' + INDENTED_SMALL_RESNET + '''


    def predictions(
        model: nn.Module,
        x: Tensor,
        batch_size: int = 500,
    ) -> Tensor:
        model.eval()

        with torch.no_grad():
            return torch.cat([model(x[start : start + batch_size]).argmax(dim=1) for start in range(0, len(x), batch_size)])


    def confusion_matrix(
        predicted: Tensor,
        labels: Tensor,
        classes: int,
    ) -> Tensor:
        return torch.bincount(labels * classes + predicted, minlength=classes * classes).reshape(classes, classes)


    def per_class_accuracy(matrix: Tensor) -> Tensor:
        return matrix.diagonal().float() / matrix.sum(dim=1).float()


    # The flattened positions of the largest counts, each split back into its row and column.
    def top_confusions(
        matrix: Tensor,
        count: int,
    ) -> list[tuple[int, int]]:
        mistakes = matrix.clone()
        mistakes.fill_diagonal_(0)
        positions = mistakes.flatten().topk(count).indices

        return [divmod(position, matrix.shape[1]) for position in positions.tolist()]


    train_images, _ = load_cifar10("train")
    order = torch.randperm(len(train_images), generator=torch.Generator().manual_seed(0))
    train_part = train_images[order[:-5000]].float() / 255
    mean, spread = train_part.mean(dim=(0, 2, 3)), train_part.std(dim=(0, 2, 3))
    test_images, test_y = load_cifar10("test")

    test_x = (test_images.float() / 255 - mean.view(1, -1, 1, 1)) / spread.view(1, -1, 1, 1)

    model = load_checkpoint(SmallResNet(), "cifar2")
    matrix = confusion_matrix(predictions(model, test_x), test_y, 10)

    test_accuracy = (matrix.diagonal().sum() / matrix.sum()).item()
    ''',
)


def reference_predictions(
    model: nn.Module,
    x: Tensor,
    batch_size: int = 500,
) -> Tensor:
    model.eval()

    with torch.no_grad():
        return torch.cat([model(x[start : start + batch_size]).argmax(dim=1) for start in range(0, len(x), batch_size)])


def reference_confusion(
    predicted: Tensor,
    labels: Tensor,
    classes: int,
) -> Tensor:
    return torch.bincount(labels * classes + predicted, minlength=classes * classes).reshape(classes, classes)


def reference_top_confusions(
    matrix: Tensor,
    count: int,
) -> list[tuple[int, int]]:
    mistakes = matrix.clone()
    mistakes.fill_diagonal_(0)

    return [divmod(position, matrix.shape[1]) for position in mistakes.flatten().topk(count).indices.tolist()]


@functools.cache
def normalized_test_set() -> Tensor:
    train_images, _ = load_cifar10("train")
    order = torch.randperm(len(train_images), generator=torch.Generator().manual_seed(0))
    train_part = train_images[order[:-VALIDATION_SIZE]].float() / 255
    mean, spread = train_part.mean(dim=(0, 2, 3)), train_part.std(dim=(0, 2, 3))
    test_images, _ = load_cifar10("test")

    return (test_images.float() / 255 - mean.view(1, -1, 1, 1)) / spread.view(1, -1, 1, 1)


def check_test_accuracy(value: object) -> None:
    model = seeded_resnet(REFERENCE["SmallResNet"])

    try:
        load_checkpoint(model, "cifar2")
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None

    _, test_y = load_cifar10("test")
    expected = (reference_predictions(model, normalized_test_set()) == test_y).float().mean().item()
    expect_close(
        call="test_accuracy",
        actual=value,
        expected=expected,
        tolerance=1e-6,
    )


# Matrices with no ties off the diagonal, so the order of the largest mistakes is never a coin flip.
CONFUSION_A = torch.tensor([[5, 2, 0], [7, 4, 1], [3, 0, 9]])
CONFUSION_B = torch.tensor([[8, 1, 6, 0], [2, 7, 0, 3], [0, 9, 5, 4], [11, 0, 0, 2]])


def check_cifar3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="predictions",
        description="gives each image's predicted class, in batches",
        inputs=(
            (Described(tiny_net(), "a small net with BatchNorm"), Described(tiny_images(380, 23)[0], "23 images"), 10),
        ),
        reference=lambda model, x, batch_size: reference_predictions(tiny_net(), x, batch_size),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="confusion_matrix",
        description="counts every pair of a true and a predicted class",
        inputs=(
            (torch.tensor([0, 2, 1, 1, 2, 0]), torch.tensor([0, 1, 1, 2, 2, 0]), 3),
            (torch.tensor([3, 0, 1, 1, 2, 3, 0]), torch.tensor([3, 1, 1, 0, 2, 2, 0]), 4),
            (torch.tensor([0, 1, 1, 2]), torch.tensor([0, 1, 2, 1]), 4),
        ),
        reference=reference_confusion,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="per_class_accuracy",
        description="gives the share of each true class labelled right",
        inputs=(
            (CONFUSION_A,),
            (CONFUSION_B,),
        ),
        reference=lambda matrix: matrix.diagonal().float() / matrix.sum(dim=1).float(),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="top_confusions",
        description="gives the largest mistakes, largest first",
        inputs=(
            (CONFUSION_A, 2),
            (CONFUSION_B, 3),
        ),
        reference=reference_top_confusions,
        expect=lambda call, actual, expected: expect_value(
            call=call,
            actual=[tuple(pair) for pair in actual] if isinstance(actual, list) else actual,
            expected=expected,
        ),
    )
    checker.check_name(
        name="test_x",
        description="normalizes the test images with the training part's statistics",
        run=lambda value: expect_tensor(
            call="test_x",
            actual=value,
            expected=normalized_test_set(),
        ),
    )
    checker.check_name(
        name="test_accuracy",
        description="is the share of the test set labelled right",
        run=check_test_accuracy,
    )


CIFAR3 = Exercise(
    name="cifar3",
    chapter=CIFAR,
    title="Project: the test set, once",
    lesson=CIFAR3_LESSON,
    source=CIFAR3_SOURCE,
    solution=CIFAR3_SOLUTION,
    hints={
        "predictions": "model.eval(), then inside torch.no_grad() take model(...).argmax(dim=1) of each batch and "
        "torch.cat them.",
        "confusion_matrix": "torch.bincount of labels * classes + predicted, with minlength=classes * classes, "
        "then .reshape(classes, classes).",
        "per_class_accuracy": "matrix.diagonal() holds the right answers, and matrix.sum(dim=1) each class's total. "
        "Turn both into floats before dividing.",
        "top_confusions": "Clone matrix and .fill_diagonal_(0). Then .flatten().topk(count).indices, and divmod each "
        "position by the number of columns.",
        "test_x": "Divide test_images.float() by 255, take away mean.view(1, -1, 1, 1), divide by "
        "spread.view(1, -1, 1, 1).",
        "test_accuracy": "matrix.diagonal().sum() over matrix.sum(), then .item().",
    },
    introduces=(),
    check=check_cifar3,
    time_limit=180,
)
