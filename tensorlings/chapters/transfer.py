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

        A model trained on one big task has learned features, such as edges
        and strokes, that are useful for other tasks too. Transfer learning
        reuses the model's body, which is every layer except the last. It
        trains a new head, a new last layer, for the new task. This chapter
        loads saved weights, freezes the body and trains the head. Then it
        measures how much this improves accuracy when only a hundred labeled
        images are available.

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
    The first layers of an image model learn to detect edges and strokes,
    which are useful for most image tasks. **Transfer learning** takes a model
    trained on one task and reuses most of it for another task. A model's
    **body** is every layer except the last. Its **head** is the last layer,
    which gives the class scores. Transfer learning keeps the trained body and
    gives it a new head for the new classes.

    A model's weights are saved and loaded as a state dict. A state dict is a
    Python dict that maps each parameter's name to its tensor.
    `torch.save(model.state_dict(), path)` writes a state dict to a file.
    `model.load_state_dict(torch.load(path))` reads it back into a model with
    the same layers and shapes. Each part of a model has its own state dict
    too. So `target.body.load_state_dict(source.body.state_dict())` copies
    only the body's weights from `source` into `target`.

    Freezing a part of a model stops training from changing it.
    `parameter.requires_grad_(False)` tells autograd not to compute a gradient
    for that parameter, so the optimizer never changes it. When the body is
    frozen, only the new head learns. Fine tuning goes one step further. After
    the head has trained, fine tuning trains all the layers, usually at a
    small learning rate.

    A frozen parameter never gets a `.grad`, so an optimizer skips it even if
    you pass it every parameter. You can still pass only the parameters that
    train, by filtering on `.requires_grad`. Then the optimizer does not store
    state, like Adam's running averages, for weights that never change.
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


    # TODO: Freeze every parameter of module in place, so that none of them gets
    #   a gradient or changes during training. Return nothing.
    def freeze_(module: nn.Module) -> None:
        ...


    # TODO: Return the total number of values in the parameters of model that
    #   still train. Count only the parameters whose requires_grad is True.
    #   Example: for a fresh Net(10), trainable_count returns 20490.
    def trainable_count(model: nn.Module) -> int:
        ...


    # TODO: Return a new Net with classes outputs. Copy the weights of source's
    #   body into the new Net's body, then freeze that body. The new head keeps
    #   its fresh random weights and still trains. Leave source unchanged.
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
        description="counts only the parameters that still train",
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
        "transfer": "Make a Net(classes) and load source.body.state_dict() into its body. Then call freeze_ on that "
        "body and return the new Net.",
    },
    introduces=(),
    check=check_transfer1,
)


# ---------------------------------------------------------------------------
# transfer2
# ---------------------------------------------------------------------------

TRANSFER2_LESSON = prose(
    """
    For the table below, a small net first learned to classify the digits 0
    to 4 from 20,000 MNIST images. It reached 99.3% accuracy on validation
    images of those digits. The new task was to classify the digits 5 to 9,
    with only 100 labeled images. The same experiment ran on photos too. A net
    of convolutions with BatchNorm learned CIFAR-10's first five classes,
    planes to deer, and reached 85.6%. Its new task was the other five
    classes, dogs to trucks.

    Each starting point in the table then trained for 300 steps on the few
    labeled images. This was repeated for three different random draws of
    those images. Each run was scored on the validation images of the new
    classes. Photos were flipped at random, as in augment2, both here and when
    the body was first trained.

    | Start | 100 digits | 100 photos | 1,000 photos |
    |---|---|---|---|
    | from scratch | 87.1% to 88.4% | 63.5% to 66.7% | 64.5% to 75.0% |
    | the trained body, frozen, with a new head | 89.6% to 92.5% | 72.3% to 73.3% | 74.7% to 75.3% |
    | the trained body, all layers at one rate | 88.2% to 91.2% | 73.8% to 76.6% | 78.4% to 84.0% |
    | the trained body at a tenth of the head's rate | 89.4% to 92.3% | 72.9% to 76.9% | 81.4% to 83.4% |

    Starting from the trained body helped every time. It helped most where
    learning from scratch was hardest. The frozen body added 2.6 to 4.2 points
    on digits, because strokes can be learned from a few examples anyway. On
    photos it added 5.8 to 9.8 points, because 100 photos are far too few to
    learn edges and textures from. That is why most work on images starts from
    a body trained on ImageNet.

    While the head trains, the frozen body runs in eval mode. That way its
    BatchNorm layers keep the running averages they learned on the first task.

    With 1,000 photos, the frozen body fell behind, because features learned
    for planes and deer do not fit trucks exactly. Training all the layers let
    those features adapt to the new classes. The best setup gave the body a
    tenth of the head's learning rate, as `fine_tune_optimizer` does. It
    scored 82.6% on average, against 80.7% for one shared rate. Its scores
    also varied less from draw to draw. Yosinski and others also found that a
    trained body helps, and that the gain remains even after fine tuning.
    Howard and Ruder's ULMFiT, a recipe for fine tuning language models, goes
    further. It divides the learning rate by 2.6 for each layer closer to the
    input.

    An optimizer can take groups of parameters, each with its own learning
    rate, as in adamw2.
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
    rest = ~below                                    # tensor([False,  True,  True, False]), ~ negates each entry


    # TODO: Split x and y into two parts. The first part has the examples whose
    #   label is below low, and the second part has the rest. Return four
    #   tensors, x and y of the first part, then x and y of the second part.
    #   Subtract low from the second part's labels, so they count from 0.
    #   Example: for y = [3, 7, 5, 0] and low = 5, the two label tensors
    #   returned are tensor([3, 0]) and tensor([2, 0]).
    def split_by_label(
        x: Tensor,
        y: Tensor,
        low: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ...


    # TODO: Return count examples of x and their labels in y, picked at random
    #   without repeats. Shuffle the indices of x with torch.randperm and
    #   generator, and keep the first count of them.
    def few_labels(
        x: Tensor,
        y: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return an Adam optimizer with two parameter groups. The first group
    #   holds model.body's parameters with learning rate body_lr. The second
    #   holds model.head's parameters with learning rate head_lr.
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
        description="gives the body and the head their own learning rates",
        run=check_fine_tune_optimizer,
    )


TRANSFER2 = Exercise(
    name="transfer2",
    chapter=TRANSFER,
    title="How much a trained body helps",
    lesson=TRANSFER2_LESSON,
    source=TRANSFER2_SOURCE,
    solution=TRANSFER2_SOLUTION,
    hints={
        "split_by_label": "The mask below = y < low selects the first part, and ~below selects the rest. Subtract "
        "low from the second part's labels.",
        "few_labels": "torch.randperm(len(x), generator=generator)[:count] gives the indices to keep. Index x and y "
        "with them.",
        "fine_tune_optimizer": "Pass torch.optim.Adam a list of two dicts, each with a \"params\" key and an \"lr\" "
        "key.",
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
    This quiz has no examples and teaches no new tools. Each function combines
    steps you learned in Section 5. If a step is unclear, the exercises listed
    in the table show how it works.

    | Function | Steps | Shown in |
    |---|---|---|
    | `head_inputs` | the output size formula applied layer by layer to height and width, then the count after flattening | stride1, conv5, pooling2, lenet1 |
    | `halving_stack` | 3 by 3 convolutions at stride 2 without bias, then BatchNorm and ReLU, each taking the previous layer's channels | stride1, resnet2, vgg1 |
    | `pooled_scores` | global average pooling of feature maps, then a Linear head | pooling1, lenet1 |
    | `flip_and_normalize` | a random flip for each image, then subtracting each channel's mean and dividing by its standard deviation | augment2, resnet1, broadcasting1 |
    """,
)

QUIZ7_SOURCE = source_file(
    '''
    # quiz7.py
    #
    # This quiz has no examples and no new tools. Each function combines steps
    # from Section 5.

    import torch
    import torch.nn as nn
    from torch import Tensor


    # TODO: Return the in_features of the nn.Linear that follows nn.Flatten at
    #   the end of a stack of convolutions, for input images of h by w. Each
    #   entry of layers is a tuple (out_channels, k, s, p). It gives one layer's
    #   output channels, square kernel size, stride and padding, and the layers
    #   are applied in order. The count is the last layer's channels times the
    #   final height times the final width.
    #   Example: head_inputs(32, 32, [(16, 3, 1, 1), (32, 3, 2, 1)]) returns
    #   8192.
    def head_inputs(
        h: int,
        w: int,
        layers: list[tuple[int, int, int, int]],
    ) -> int:
        ...


    # TODO: Return an nn.Sequential with one layer for each entry of widths,
    #   where each layer halves the height and width. A layer is a 3 by 3
    #   nn.Conv2d at stride 2 with padding 1 and no bias, whose output channels
    #   are that entry of widths, followed by an nn.BatchNorm2d and an nn.ReLU.
    #   The first layer takes in_channels as input, and each later layer takes
    #   the previous entry of widths. Make every module a new object.
    def halving_stack(
        in_channels: int,
        widths: list[int],
    ) -> nn.Sequential:
        ...


    # TODO: Apply global average pooling to features, then return head's scores
    #   for the pooled result. features has shape (batch, channels, h, w).
    #   Pooling averages over h and w, which gives shape (batch, channels).
    def pooled_scores(
        features: Tensor,
        head: nn.Linear,
    ) -> Tensor:
        ...


    # TODO: Draw one random number per image with torch.rand and generator, and
    #   mirror each image whose number is below 0.5. images has shape
    #   (batch, channels, h, w). Then subtract each channel's mean and divide
    #   by that channel's standard deviation. mean and spread each hold one
    #   number per channel, and spread is the standard deviation.
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


    def head_inputs(
        h: int,
        w: int,
        layers: list[tuple[int, int, int, int]],
    ) -> int:
        for out_channels, k, s, p in layers:
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1

        return layers[-1][0] * h * w


    def halving_stack(
        in_channels: int,
        widths: list[int],
    ) -> nn.Sequential:
        layers = []

        for width in widths:
            layers.extend([nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU()])
            in_channels = width

        return nn.Sequential(*layers)


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


def reference_head_inputs(
    h: int,
    w: int,
    layers: list[tuple[int, int, int, int]],
) -> int:
    for _, k, s, p in layers:
        h = reference_output_size(h, k, p, s)
        w = reference_output_size(w, k, p, s)

    channels = layers[-1][0]

    return channels * h * w


def reference_flip_and_normalize(
    images: Tensor,
    mean: Tensor,
    spread: Tensor,
    generator: torch.Generator,
) -> Tensor:
    draws = torch.rand(len(images), 1, 1, 1, generator=generator)
    flipped = torch.where(draws < 0.5, images.flip(-1), images)

    return (flipped - mean.view(1, -1, 1, 1)) / spread.view(1, -1, 1, 1)


def stack_layers(stack: nn.Sequential) -> list[nn.Module]:
    """The layers in order, any nn.Sequential inside opened up. Looping over a Sequential keeps a layer that sits in
    it twice, where .modules() would list it once."""
    layers: list[nn.Module] = []

    for layer in stack:
        if isinstance(layer, nn.Sequential):
            layers.extend(stack_layers(layer))
            continue

        layers.append(layer)

    return layers


def check_halving_stack(found: object) -> None:
    # Equal widths let a stack that repeats one layer object get every setting right.
    for in_channels, widths in ((3, [8, 16, 32]), (4, [4, 4])):
        call = f"halving_stack({in_channels}, {widths})"
        stack = expect_module(call, found(in_channels, list(widths)))

        if not isinstance(stack, nn.Sequential):
            raise Mismatch(f"{call} is {type(stack).__name__}, but it should be an nn.Sequential.")

        layers = stack_layers(stack)
        kinds = [type(layer).__name__ for layer in layers]
        wanted_kinds = ["Conv2d", "BatchNorm2d", "ReLU"] * len(widths)

        if kinds != wanted_kinds:
            raise Mismatch(f"{call} holds {', '.join(kinds) or 'nothing'}, but it should hold {', '.join(wanted_kinds)}.")

        if len({id(layer) for layer in layers}) != len(layers):
            raise Mismatch(f"{call} holds one layer object in more than one place. Make a new Conv2d, BatchNorm2d and ReLU for every width.")

        reads = in_channels

        for position, width in enumerate(widths):
            conv, norm = layers[3 * position], layers[3 * position + 1]
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
                reads,
                width,
                (3, 3),
                (2, 2),
                (1, 1),
                True,
                width,
            )

            if settings != wanted:
                raise Mismatch(
                    f"Layer {position + 1} of {call} should be a 3 by 3 convolution from {reads} to {width} channels "
                    f"with stride 2, padding 1 and no bias, then BatchNorm2d({width}).",
                )

            reads = width


def check_quiz7(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="head_inputs",
        description="tracks the height and width through the layers and counts the values nn.Flatten outputs",
        inputs=(
            (32, 32, [(16, 3, 1, 1), (32, 3, 2, 1), (64, 3, 2, 1)]),
            (28, 20, [(6, 5, 1, 2), (16, 5, 2, 0), (8, 3, 2, 1)]),
            (15, 9, [(4, 3, 2, 1), (10, 3, 2, 0)]),
        ),
        reference=reference_head_inputs,
        expect=expect_value,
    )
    checker.test(
        function="halving_stack",
        description="chains halving convolutions with BatchNorm and ReLU",
        run=check_halving_stack,
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
        "head_inputs": "Update h and w for each layer with stride1's output size formula, as conv5's sizes_through "
        "does. nn.Flatten puts each image's channels, rows and columns into one row, so multiply the three counts.",
        "halving_stack": "Build a list, adding a convolution, a BatchNorm2d and a ReLU for each width, as resnet2's "
        "plain_stack does. Each convolution after the first takes the previous width as its input channels, as in "
        "vgg1's blocks. Unpack the list into nn.Sequential with *.",
        "pooled_scores": "Average over dim=(2, 3), then pass the result to head.",
        "flip_and_normalize": "Flip the images the way augment2's augment_batch does. Then reshape mean and spread "
        "to (1, channels, 1, 1) with .view, so they broadcast along the channel dimension.",
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
