"""36_transfer, loading a trained body, freezing it and training a new head, and what that buys a hundred labels."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, QUIZZES, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_tensor,
    expect_value,
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from ..training import best_device
from .classic_nets import CifarNet, accuracy_on, cifar_splits, flipped_half, mnist_images, pinned_threads
from .conv_shapes import reference_output_size
from .linear_layers import expect_module

TRANSFER = Chapter(
    folder="36_transfer",
    title="Transfer learning",
    readme=source_file(
        """
        # Transfer learning

        A model trained on one big task has learned features, like edges and
        strokes, that help with others. Transfer learning reuses its body and
        trains a new head for the new task. This chapter loads weights,
        freezes the body, trains the head, and measures what that buys when
        only a hundred labeled images exist.

        ## Further reading

        - [Yosinski and others, how transferable are features in deep neural networks](https://arxiv.org/abs/1411.1792)
        - [CS231n, transfer learning](https://cs231n.github.io/transfer-learning/)
        - [PyTorch, saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
        """,
    ),
)


class ReferenceNet(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.head = nn.Linear(32 * 7 * 7, classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))


# ---------------------------------------------------------------------------
# transfer1
# ---------------------------------------------------------------------------

TRANSFER1_LESSON = prose(
    """
    The first layers of an image model learn edges and strokes, which most
    image tasks share. **Transfer learning** takes a model trained on one task
    and reuses its body for another, with a new head for the new classes.

    A model's weights are saved and loaded as a state dict, a dict from each
    parameter's name to its tensor. `torch.save(model.state_dict(), path)`
    writes one to a file, and `model.load_state_dict(torch.load(path))` reads
    it back into a model of the same shape. A part of a model has a state dict
    of its own, so `target.body.load_state_dict(source.body.state_dict())`
    copies only the body.

    Freezing a part stops it training. `parameter.requires_grad_(False)` tells
    autograd to leave that parameter out, so it gets no gradient and no step.
    Only the new head learns. Fine tuning goes one step further, training
    everything afterwards at a small learning rate.

    A frozen parameter never gets a `.grad`, so an optimizer leaves it alone
    even when it is handed every parameter. Handing it only the ones that
    train, with a filter on `.requires_grad`, saves it keeping state for
    weights that never move.
    """,
)

TRANSFER1_SOURCE = source_file(
    '''
    # transfer1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    layer = nn.Linear(2, 2)
    names = list(layer.state_dict())                 # ['weight', 'bias']


    class Net(nn.Module):
        def __init__(self, classes: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
            )
            self.head = nn.Linear(32 * 7 * 7, classes)

        def forward(self, x: Tensor) -> Tensor:
            return self.head(self.body(x))


    # TODO: Stop every parameter of module training, in place.
    def freeze_(module: nn.Module) -> None:
        ...


    # TODO: Return how many numbers of model still train, counting only the
    #   parameters that require a gradient.
    def trainable_count(model: nn.Module) -> int:
        ...


    # TODO: Return a new Net for classes classes whose body holds a copy of
    #   source's body weights and is frozen. Its head is new and trains.
    def transfer(
        source: nn.Module,
        classes: int,
    ) -> nn.Module:
        ...
    ''',
)

TRANSFER1_SOLUTION = source_file(
    '''
    # transfer1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    layer = nn.Linear(2, 2)
    names = list(layer.state_dict())


    class Net(nn.Module):
        def __init__(self, classes: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
            )
            self.head = nn.Linear(32 * 7 * 7, classes)

        def forward(self, x: Tensor) -> Tensor:
            return self.head(self.body(x))


    def freeze_(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(False)


    def trainable_count(model: nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


    def transfer(
        source: nn.Module,
        classes: int,
    ) -> nn.Module:
        target = Net(classes)
        target.body.load_state_dict(source.body.state_dict())
        freeze_(target.body)

        return target
    ''',
)


def check_freeze(found: object) -> None:
    model = ReferenceNet(5)
    found(model.body)

    for name, parameter in model.body.named_parameters():
        if parameter.requires_grad:
            raise Mismatch(f"After freeze_ on a body, its {name} still requires a gradient.")

    if not all(parameter.requires_grad for parameter in model.head.parameters()):
        raise Mismatch("freeze_ on the body also froze the head, which it was not given.")


def check_trainable_count(found: object) -> None:
    model = ReferenceNet(5)

    for parameter in model.body.parameters():
        parameter.requires_grad_(False)

    expect_value(
        call="trainable_count of a Net(5) with its body frozen",
        actual=found(model),
        expected=sum(parameter.numel() for parameter in model.head.parameters()),
    )
    expect_value(
        call="trainable_count of a fresh Net(10)",
        actual=found(ReferenceNet(10)),
        expected=sum(parameter.numel() for parameter in ReferenceNet(10).parameters()),
    )


def check_transfer(found: object) -> None:
    torch.manual_seed(850)
    source = ReferenceNet(10)
    call = "transfer(a trained Net(10), 5)"
    target = expect_module(call, found(source, 5))

    if target is source:
        raise Mismatch(f"{call} gave back the source itself. Build a new Net and copy the body's weights into it.")

    source_parameters = {id(parameter) for parameter in source.parameters()}

    if any(id(parameter) in source_parameters for parameter in target.parameters()):
        raise Mismatch(f"{call} shares weights with the source, so training one would change the other. Copy them with load_state_dict.")

    if not all(parameter.requires_grad for parameter in source.parameters()):
        raise Mismatch(f"{call} froze the source's weights too. Freeze only the new model's body.")

    if not isinstance(getattr(target, "head", None), nn.Linear) or target.head.out_features != 5:
        raise Mismatch(f"{call} should have a new head, an nn.Linear to 5 classes.")

    for (name, mine), theirs in zip(target.body.state_dict().items(), source.body.state_dict().values()):
        if not torch.equal(mine, theirs):
            raise Mismatch(f"The body of {call} should hold the source's weights, but its {name} differs.")

    if any(parameter.requires_grad for parameter in target.body.parameters()):
        raise Mismatch(f"The body of {call} should be frozen.")

    if not all(parameter.requires_grad for parameter in target.head.parameters()):
        raise Mismatch(f"The head of {call} should train.")


def check_transfer1(checker: Checker) -> None:
    checker.test(
        function="freeze_",
        description="stops every parameter it is given from training",
        run=check_freeze,
    )
    checker.test(
        function="trainable_count",
        description="counts only what still trains",
        run=check_trainable_count,
    )
    checker.test(
        function="transfer",
        description="copies and freezes the body, with a new head",
        run=check_transfer,
    )


TRANSFER1 = Exercise(
    name="transfer1",
    chapter=TRANSFER,
    title="A frozen body and a new head",
    lesson=TRANSFER1_LESSON,
    source=TRANSFER1_SOURCE,
    solution=TRANSFER1_SOLUTION,
    hints={
        "freeze_": "Loop over module.parameters() and call .requires_grad_(False) on each.",
        "trainable_count": "Add up .numel() of the parameters whose .requires_grad is True.",
        "transfer": "Make Net(classes), load source.body.state_dict() into its body, freeze_ the body, and return "
        "it.",
    },
    introduces=(),
    check=check_transfer1,
)


# ---------------------------------------------------------------------------
# transfer2
# ---------------------------------------------------------------------------

TRANSFER2_LESSON = prose(
    """
    Here a small net learned digits 0 to 4 from 20,000 MNIST images and
    reached 99.3% on them. The new task was digits 5 to 9, from only 100
    labeled images. The same test ran on photos too. CIFAR-10's first five
    classes, planes to deer, trained a body of convolutions with BatchNorm to
    85.6%. The new task was its other five, dogs to trucks.

    Each start then trained 300 steps on the few labels, over three draws of
    them, and was scored on the new classes' validation images. Photos were
    flipped at random, as in augment2, here and while the body trained.

    | Start | 100 digits | 100 photos | 1,000 photos |
    |---|---|---|---|
    | from scratch | 87.1% to 88.4% | 63.5% to 66.7% | 64.5% to 75.0% |
    | the trained body, frozen, with a new head | 89.6% to 92.5% | 72.3% to 73.3% | 74.7% to 75.3% |
    | the trained body, everything at one rate | 88.2% to 91.2% | 73.8% to 76.6% | 78.4% to 84.0% |
    | the trained body at a tenth of the head's rate | 89.4% to 92.3% | 72.9% to 76.9% | 81.4% to 83.4% |

    The trained body helped every time, and most where learning from scratch
    was hardest. Frozen, it added 2.6 to 4.2 points on digits, whose strokes
    are easy to learn from a few. On photos it added 5.8 to 9.8, since 100
    photos are far too few to learn edges and textures. That is why most image
    work starts from a body trained on ImageNet.

    A frozen body runs in eval mode while the head trains, so its BatchNorm
    averages stay the ones it learned.

    With 1,000 photos the frozen body fell behind, since features for planes
    and deer don't fit trucks exactly. Training everything let them adapt.
    `fine_tune_optimizer`'s smaller body rate scored highest on average, 82.6%
    against 80.7%, and varied least. Yosinski and others found the same boost,
    one that lingers even after fine tuning. Howard and Ruder's ULMFiT, a
    recipe for fine tuning language models, goes further. It divides the rate
    by 2.6 for each layer down.

    An optimizer takes one rate per group, as in adamw2.
    """,
)

TRANSFER2_SOURCE = source_file(
    '''
    # transfer2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    y = torch.tensor([3, 7, 5, 0])
    below = y < 5                                    # tensor([ True, False, False,  True])
    rest = ~below                                    # tensor([False,  True,  True, False]), ~ flips each one


    # TODO: Split x and y into the examples whose label is below low and the
    #   rest. Return x and y of the first, then x and y of the second with low
    #   taken off its labels, so they count from 0.
    def split_by_label(
        x: Tensor,
        y: Tensor,
        low: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ...


    # TODO: Return count examples of x and y picked at random without repeats,
    #   with the first count positions of torch.randperm from generator.
    def few_labels(
        x: Tensor,
        y: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return Adam with two groups, model.body's parameters at body_lr and
    #   model.head's at head_lr.
    def fine_tune_optimizer(
        model: nn.Module,
        body_lr: float,
        head_lr: float,
    ) -> torch.optim.Optimizer:
        ...
    ''',
)

TRANSFER2_SOLUTION = source_file(
    '''
    # transfer2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    y = torch.tensor([3, 7, 5, 0])
    below = y < 5
    rest = ~below


    def split_by_label(
        x: Tensor,
        y: Tensor,
        low: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        below = y < low

        return x[below], y[below], x[~below], y[~below] - low


    def few_labels(
        x: Tensor,
        y: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        picked = torch.randperm(len(x), generator=generator)[:count]

        return x[picked], y[picked]


    def fine_tune_optimizer(
        model: nn.Module,
        body_lr: float,
        head_lr: float,
    ) -> torch.optim.Optimizer:
        groups = [{"params": model.body.parameters(), "lr": body_lr}, {"params": model.head.parameters(), "lr": head_lr}]

        return torch.optim.Adam(groups)
    ''',
)


def check_fine_tune_optimizer(found: object) -> None:
    model = ReferenceNet(5)
    call = "fine_tune_optimizer(a Net(5), 0.0001, 0.001)"
    optimizer = found(model, 0.0001, 0.001)

    if type(optimizer) is not torch.optim.Adam:
        raise Mismatch(f"{call} is {type(optimizer).__name__}, but it should be torch.optim.Adam.")

    groups = optimizer.param_groups

    if len(groups) != 2:
        raise Mismatch(f"{call} should have two groups, but it has {len(groups)}.")

    for group, part, rate in ((groups[0], model.body, 0.0001), (groups[1], model.head, 0.001)):
        if [id(parameter) for parameter in group["params"]] != [id(parameter) for parameter in part.parameters()] or group["lr"] != rate:
            name = "body" if part is model.body else "head"
            raise Mismatch(f"In {call}, the {name}'s group should hold the {name}'s parameters at a rate of {rate}.")


def check_transfer2(checker: Checker) -> None:
    from ..engine import check_on_inputs, expect_parts, expect_tensor, seeded, small_tensor

    labels = torch.tensor([3, 7, 5, 0, 9, 4, 6, 1])
    check_on_inputs(
        checker=checker,
        function="split_by_label",
        description="splits by label and counts the second part from 0",
        inputs=(
            (small_tensor(860, 8, 2), labels, 5),
            (small_tensor(861, 8, 3), labels, 3),
        ),
        reference=lambda x, y, low: (x[y < low], y[y < low], x[y >= low], y[y >= low] - low),
        expect=expect_parts("x of the first part", "y of the first part", "x of the second part", "y of the second part"),
    )
    check_on_inputs(
        checker=checker,
        function="few_labels",
        description="picks count examples at random without repeats",
        inputs=(
            (small_tensor(862, 10, 2), torch.arange(10), 4, seeded(863)),
            (small_tensor(864, 6, 3), torch.arange(6), 6, seeded(865)),
        ),
        reference=lambda x, y, count, generator: (lambda picked: (x[picked], y[picked]))(torch.randperm(len(x), generator=generator)[:count]),
        expect=expect_parts("the examples", "their labels"),
    )
    checker.test(
        function="fine_tune_optimizer",
        description="gives the body and the head rates of their own",
        run=check_fine_tune_optimizer,
    )


TRANSFER2 = Exercise(
    name="transfer2",
    chapter=TRANSFER,
    title="What a trained body buys",
    lesson=TRANSFER2_LESSON,
    source=TRANSFER2_SOURCE,
    solution=TRANSFER2_SOLUTION,
    hints={
        "split_by_label": "below = y < low picks the first part, and ~below the rest. Take low off the second "
        "part's labels.",
        "few_labels": "torch.randperm(len(x), generator=generator)[:count] gives the positions. Index x and y with "
        "them.",
        "fine_tune_optimizer": "torch.optim.Adam of a list of two dicts, each with \"params\" and \"lr\".",
    },
    introduces=(),
    check=check_transfer2,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# quiz7
# ---------------------------------------------------------------------------

QUIZ7_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines moves
    from Section 5. If a step is unclear, the exercise in the table shows it.

    | Function | Moves | Shown in |
    |---|---|---|
    | `image_size_through` | the output size formula, followed through layers, for height and width | stride1, conv5 |
    | `conv_bn_relu` | a 3 by 3 convolution with a stride and padding 1, no bias, BatchNorm and ReLU | stride1, resnet1 |
    | `pooled_scores` | global average pooling of feature maps, then a Linear head | pooling1, lenet1 |
    | `flip_and_normalize` | a random flip per image, then each channel's mean taken away and divided by its spread | augment2, resnet1, broadcasting1 |
    """,
)

QUIZ7_SOURCE = source_file(
    '''
    # quiz7.py
    #
    # No examples and no new tools. Each function combines moves from Section 5.

    import torch
    import torch.nn as nn
    from torch import Tensor


    # TODO: Return the height and width of an image of h by w after each layer of
    #   layers in turn, as one (height, width) pair at the end. Each layer is a
    #   (k, s, p) triple for a square kernel, stride and padding.
    def image_size_through(
        h: int,
        w: int,
        layers: list[tuple[int, int, int]],
    ) -> tuple[int, int]:
        ...


    # TODO: Return an nn.Sequential of a 3 by 3 nn.Conv2d from in_channels to
    #   out_channels with the given stride, padding 1 and no bias, then an
    #   nn.BatchNorm2d and a ReLU.
    def conv_bn_relu(
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> nn.Sequential:
        ...


    # TODO: Return head's scores for features, of shape (batch, channels, h, w),
    #   after global average pooling.
    def pooled_scores(
        features: Tensor,
        head: nn.Linear,
    ) -> Tensor:
        ...


    # TODO: Mirror each image of images, of shape (batch, channels, h, w), whose
    #   draw from torch.rand with generator is below 0.5, one draw per image.
    #   Then take each channel's mean away and divide by its spread, mean and
    #   spread holding one number per channel.
    def flip_and_normalize(
        images: Tensor,
        mean: Tensor,
        spread: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        ...
    ''',
)

QUIZ7_SOLUTION = source_file(
    '''
    # quiz7.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor


    def image_size_through(
        h: int,
        w: int,
        layers: list[tuple[int, int, int]],
    ) -> tuple[int, int]:
        for k, s, p in layers:
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1

        return h, w


    def conv_bn_relu(
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


    def pooled_scores(
        features: Tensor,
        head: nn.Linear,
    ) -> Tensor:
        return head(features.mean(dim=(2, 3)))


    def flip_and_normalize(
        images: Tensor,
        mean: Tensor,
        spread: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        draws = torch.rand(len(images), 1, 1, 1, generator=generator)
        flipped = torch.where(draws < 0.5, images.flip(-1), images)

        return (flipped - mean.view(1, -1, 1, 1)) / spread.view(1, -1, 1, 1)
    ''',
)


def reference_image_size_through(
    h: int,
    w: int,
    layers: list[tuple[int, int, int]],
) -> tuple[int, int]:
    for k, s, p in layers:
        h = reference_output_size(h, k, p, s)
        w = reference_output_size(w, k, p, s)

    return h, w


def reference_flip_and_normalize(
    images: Tensor,
    mean: Tensor,
    spread: Tensor,
    generator: torch.Generator,
) -> Tensor:
    draws = torch.rand(len(images), 1, 1, 1, generator=generator)
    flipped = torch.where(draws < 0.5, images.flip(-1), images)

    return (flipped - mean.view(1, -1, 1, 1)) / spread.view(1, -1, 1, 1)


def check_conv_bn_relu(found: object) -> None:
    for in_channels, out_channels, stride in ((3, 16, 1), (16, 32, 2)):
        call = f"conv_bn_relu({in_channels}, {out_channels}, {stride})"
        block = expect_module(call, found(in_channels, out_channels, stride))
        layer_names = [type(layer).__name__ for layer in block] if isinstance(block, nn.Sequential) else []

        if layer_names != ["Conv2d", "BatchNorm2d", "ReLU"]:
            raise Mismatch(f"{call} should be an nn.Sequential of Conv2d, BatchNorm2d and ReLU.")

        conv, norm = block[0], block[1]
        settings = (
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.bias is None,
            norm.num_features,
        )
        wanted = (
            in_channels,
            out_channels,
            (3, 3),
            (stride, stride),
            (1, 1),
            True,
            out_channels,
        )

        if settings != wanted:
            raise Mismatch(
                f"{call} should hold a 3 by 3 convolution from {in_channels} to {out_channels} channels "
                f"with stride {stride}, padding 1 and no bias, then BatchNorm2d({out_channels}).",
            )


def check_quiz7(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="image_size_through",
        description="follows the height and width through the layers",
        inputs=(
            (224, 224, [(7, 2, 3), (3, 2, 1)]),
            (32, 20, [(3, 1, 1), (3, 2, 1), (3, 2, 0)]),
        ),
        reference=reference_image_size_through,
        expect=lambda call, actual, expected: expect_value(
            call=call,
            actual=tuple(actual) if isinstance(actual, (tuple, list)) else actual,
            expected=expected,
        ),
    )
    checker.test(
        function="conv_bn_relu",
        description="builds convolution, BatchNorm and ReLU",
        run=check_conv_bn_relu,
    )

    torch.manual_seed(870)
    head = nn.Linear(4, 3)
    check_on_inputs(
        checker=checker,
        function="pooled_scores",
        description="pools each channel, then scores",
        inputs=(
            (small_tensor(871, 2, 4, 3, 3), Described(head, "a Linear head from 4 to 3")),
        ),
        reference=lambda features, head: head(features.mean(dim=(2, 3))),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="flip_and_normalize",
        description="flips about half, then normalizes each channel",
        inputs=(
            (
                small_tensor(872, 4, 3, 2, 3),
                torch.tensor([0.5, -1.0, 0.0]),
                torch.tensor([2.0, 1.0, 0.5]),
                seeded(873),
            ),
        ),
        reference=reference_flip_and_normalize,
        expect=expect_tensor,
    )


QUIZ7 = Exercise(
    name="quiz7",
    chapter=QUIZZES,
    title="Convolutional nets, start to finish",
    lesson=QUIZ7_LESSON,
    source=QUIZ7_SOURCE,
    solution=QUIZ7_SOLUTION,
    hints={
        "image_size_through": "For each (k, s, p), update h and w with stride1's formula.",
        "conv_bn_relu": "nn.Sequential of nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, "
        "bias=False), nn.BatchNorm2d(out_channels) and nn.ReLU().",
        "pooled_scores": "Average over dim=(2, 3), then pass the result to head.",
        "flip_and_normalize": "Flip as augment2's augment_batch does. Then give mean and spread the shape "
        "(1, channels, 1, 1) with .view so they line up with the channels.",
    },
    introduces=(),
    check=check_quiz7,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

STARTS: tuple[str, ...] = ("from scratch", "frozen", "same rate", "body at a tenth")


def reference_fine_tune_optimizer(
    model: nn.Module,
    body_lr: float,
    head_lr: float,
) -> torch.optim.Optimizer:
    groups = [{"params": model.body.parameters(), "lr": body_lr}, {"params": model.head.parameters(), "lr": head_lr}]

    return torch.optim.Adam(groups)


def train_for(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    steps: int,
    batch: int,
    generator: torch.Generator,
    flips: bool,
    frozen_body: bool,
) -> None:
    model.train()

    for _ in range(steps):
        picked = torch.randint(0, len(x), (min(batch, len(x)),), generator=generator).to(x.device)
        images = flipped_half(x[picked], generator) if flips else x[picked]

        # A frozen body keeps the BatchNorm averages it learned, so it runs in eval mode while the head trains.
        if frozen_body:
            model.body.eval()

        loss = F.cross_entropy(model(images), y[picked])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def transfer_source(
    kind: str,
    device: torch.device,
) -> tuple[nn.Module, float]:
    """The model trained on classes 0 to 4: for "digits" transfer1's Net on 20,000 MNIST images, 1,500 Adam steps of
    64, and for "photos" the CifarNet on CIFAR-10's 22,500 training images of those classes, 3,000 Adam steps of 128
    with flips. Gives the model and its validation accuracy on 0 to 4."""
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)

    if kind == "digits":
        images = mnist_images()
        below = images["train_y"] < 5
        x, y = images["train_x"][below][:20000].to(device), images["train_y"][below][:20000].to(device)
        model = ReferenceNet(5).to(device)
        train_for(model, torch.optim.Adam(model.parameters(), lr=0.001), x, y, 1500, 64, generator, False, False)
        held = images["validation_y"] < 5

        return model, accuracy_on(model, images["validation_x"][held], images["validation_y"][held], device)

    sets = cifar_splits()
    below = sets["train_y"] < 5
    x, y = sets["train_x"][below].to(device), sets["train_y"][below].to(device)
    model = CifarNet(5).to(device)
    train_for(model, torch.optim.Adam(model.parameters(), lr=0.001), x, y, 3000, 128, generator, True, False)
    held = sets["validation_y"] < 5

    return model, accuracy_on(model, sets["validation_x"][held], sets["validation_y"][held], device)


def transfer_scores(
    kind: str,
    source: nn.Module,
    count: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """transfer2's rows for one draw of count labelled images of classes 5 to 9, each start trained 300 Adam steps of
    64 at 0.001 and scored on the validation images of 5 to 9. "body at a tenth" is fine_tune_optimizer's, with the
    body at 0.0001."""
    if kind == "digits":
        images = mnist_images()
        train_x, train_y, held_x, held_y = images["train_x"], images["train_y"], images["validation_x"], images["validation_y"]
        make_model: Callable[[int], nn.Module] = ReferenceNet
    else:
        sets = cifar_splits()
        train_x, train_y, held_x, held_y = sets["train_x"], sets["train_y"], sets["validation_x"], sets["validation_y"]
        make_model = CifarNet

    upper = train_y >= 5
    picked = torch.randperm(int(upper.sum()), generator=torch.Generator().manual_seed(100 + seed))[:count]
    few_x = train_x[upper][picked].to(device)
    few_y = (train_y[upper][picked] - 5).to(device)
    held = held_y >= 5
    scores = {}

    for start in STARTS:
        torch.manual_seed(seed)
        model = make_model(5).to(device)

        if start != "from scratch":
            model.body.load_state_dict(source.body.state_dict())

        if start == "frozen":
            for parameter in model.body.parameters():
                parameter.requires_grad_(False)

            optimizer = torch.optim.Adam(model.head.parameters(), lr=0.001)
        elif start == "body at a tenth":
            optimizer = reference_fine_tune_optimizer(model, 0.0001, 0.001)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        generator = torch.Generator().manual_seed(seed + 7)
        train_for(model, optimizer, few_x, few_y, 300, 64, generator, kind == "photos", start == "frozen")
        scores[start] = accuracy_on(model, held_x[held], held_y[held] - 5, device)

    return scores


def measure_transfer(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[tuple[str, int, str], list[float]]:
    """transfer2's table, keyed by the kind of image, the count of labels and the start."""
    device = best_device()
    results: dict[tuple[str, int, str], list[float]] = {}

    with pinned_threads():
        for kind, counts in (("digits", (100,)), ("photos", (100, 1000))):
            source, _ = transfer_source(kind, device)

            for count in counts:
                for seed in seeds:
                    for start, score in transfer_scores(kind, source, count, seed, device).items():
                        results.setdefault((kind, count, start), []).append(score)

    return results
