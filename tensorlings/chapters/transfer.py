"""36_transfer, loading a trained body, freezing it and training a new head, and what that buys a hundred labels."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Mismatch,
    expect_value,
)
from ..text import prose, source_file
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

    An optimizer should get only the parameters that train, which a filter on
    `.requires_grad` gives.
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
    Here a small convolutional net learned digits 0 to 4 from 20,000 MNIST
    images, and reached 98.8% on them. The new task was digits 5 to 9, with
    only 100 labeled images. Each model trained 300 steps on those 100 and was
    measured on 5 to 9's validation images.

    | Start | Validation on 5 to 9 |
    |---|---|
    | from scratch | 90.7% |
    | the 0 to 4 body, frozen, with a new head | 93.9% |
    | the 0 to 4 body, everything trained at the same rate | 91.7% |

    The frozen body did best. It learned strokes and loops from thousands of
    other digits, and those serve 5 to 9 as well. Only the small head had to
    learn from 100 images. Training everything at the same rate let the 100 images
    pull the body away from what it knew.

    MNIST is easy, so even 100 images get far from scratch. On photos, where
    100 images are far too few to learn edges and textures, the gap is much
    larger. That is why most image work starts from a body trained on
    ImageNet.

    When the new task has more data, train everything, with the body at a
    smaller rate than the head. An optimizer takes one rate per group, as in
    adamw2.
    """,
)

TRANSFER2_SOURCE = source_file(
    '''
    # transfer2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    y = torch.tensor([3, 7, 5, 0])
    low = y < 5                                      # tensor([ True, False, False,  True])


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
    low = y < 5


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
