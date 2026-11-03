"""20_dropout, zeroing numbers at random in training, why it helps a model that overfits, and where it goes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_parts,
    expect_shape,
    expect_tensor,
    expect_value,
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from .initialization import expect_numbers
from .linear_layers import expect_module
from .mnist import mnist_sets

DROPOUT = Chapter(
    folder="20_dropout",
    title="Dropout",
    readme=source_file(
        """
        # Dropout

        Dropout zeroes numbers inside a model at random while it trains, so no
        neuron can lean on any other. This chapter builds it, measures what it
        does for a model that overfits, and sets out where it goes and when to
        leave it out.

        ## Further reading

        - [Srivastava and others, dropout, the paper](https://jmlr.org/papers/v15/srivastava14a.html)
        - [PyTorch, nn.Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html)
        - [Dive into Deep Learning, dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html)
        """,
    ),
)


def labelled_points(
    seed: int,
    count: int,
    features: int,
    classes: int,
) -> tuple[Tensor, Tensor]:
    """Points whose class follows one fixed rule plus noise, so a model has something to learn and to overfit."""
    generator = torch.Generator().manual_seed(seed)
    rule = torch.randn(features, classes, generator=torch.Generator().manual_seed(0))
    x = torch.randn(count, features, generator=generator)
    y = (x @ rule + torch.randn(count, classes, generator=generator)).argmax(dim=1)

    return x, y


def layer_summary(model: nn.Module) -> list[tuple[str, object]]:
    summary: list[tuple[str, object]] = []

    for layer in model:
        if isinstance(layer, nn.Linear):
            summary.append(("Linear", (layer.in_features, layer.out_features)))
            continue

        if isinstance(layer, nn.Dropout):
            summary.append(("Dropout", layer.p))
            continue

        summary.append((type(layer).__name__, None))

    return summary


def summary_text(summary: list[tuple[str, object]]) -> str:
    parts = []

    for name, detail in summary:
        if name == "Linear":
            parts.append(f"Linear{detail}")
            continue

        if name == "Dropout":
            parts.append(f"Dropout({detail})")
            continue

        parts.append(name)

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# dropout1
# ---------------------------------------------------------------------------

DROPOUT1_LESSON = prose(
    """
    The generalization chapter fought overfitting with weight decay and early
    stopping. **Dropout** is another way, from Hinton and colleagues in 2012.
    While the model trains, each number coming out of a layer is set to 0 with
    probability `p`, picked again at random every step.

    No neuron can then count on any other being there, so each has to be
    useful on its own. The paper's way to see it is that every step trains a
    different thinned model, and the full model acts like their average.

    Zeroing a fraction `p` of the numbers makes their average `1 - p` times
    smaller. So the numbers that survive are divided by `1 - p`, which keeps
    the average where it was.

    `torch.rand(shape, generator=generator)` draws evenly between 0 and 1, so
    each number is below `p` with probability `p`. A tensor of `True` and
    `False` multiplies like 1 and 0.

    ```python
    x = torch.ones(8)
    keep = torch.rand(8, generator=generator) >= 0.5    # True where a number survives
    x * keep / (1 - 0.5)    # like tensor([2., 0., 2., 2., 0., 0., 2., 0.]), mean about 1
    ```

    When the model is used for real, nothing is dropped. A module knows which
    it is doing from `self.training`. `model.train()` sets it to `True` in the
    module and every module inside it, and `model.eval()` sets it to `False`.
    A new module starts in training mode. PyTorch's `nn.Dropout(p)` works this
    way.
    """,
)

DROPOUT1_SOURCE = source_file(
    '''
    # dropout1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    generator = torch.Generator().manual_seed(0)
    draws = torch.rand(8, generator=generator)       # 8 numbers drawn evenly between 0 and 1
    below = draws < 0.5                              # True for about half of them


    # TODO: Return a tensor of the given shape for dropout that drops each
    #   number with probability p. Draw it with torch.rand from generator, so it
    #   is True where a draw is at or above p and the number survives.
    def keep_mask(
        shape: tuple[int, ...],
        p: float,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Finish the last line, so this zeroes the numbers keep_mask drops and
    #   scales the survivors to keep the average where it was.
    def dropout(
        x: Tensor,
        p: float,
        generator: torch.Generator,
    ) -> Tensor:
        keep = keep_mask(x.shape, p, generator)

        return ...


    layer = nn.Dropout(0.5)
    layer.eval()
    training = layer.training                        # False, so this layer drops nothing


    # TODO: Finish forward, so in training mode it drops each number with
    #   probability self.p, drawn with torch.rand, and scales the rest, and in
    #   eval mode it returns x unchanged.
    class MyDropout(nn.Module):
        def __init__(self, p: float) -> None:
            super().__init__()
            self.p = p

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

DROPOUT1_SOLUTION = source_file(
    '''
    # dropout1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    generator = torch.Generator().manual_seed(0)
    draws = torch.rand(8, generator=generator)
    below = draws < 0.5


    def keep_mask(
        shape: tuple[int, ...],
        p: float,
        generator: torch.Generator,
    ) -> Tensor:
        return torch.rand(shape, generator=generator) >= p


    def dropout(
        x: Tensor,
        p: float,
        generator: torch.Generator,
    ) -> Tensor:
        keep = keep_mask(x.shape, p, generator)

        return x * keep / (1 - p)


    layer = nn.Dropout(0.5)
    layer.eval()
    training = layer.training


    class MyDropout(nn.Module):
        def __init__(self, p: float) -> None:
            super().__init__()
            self.p = p

        def forward(self, x: Tensor) -> Tensor:
            if not self.training:
                return x

            keep = torch.rand(x.shape) >= self.p

            return x * keep / (1 - self.p)
    ''',
)


def check_my_dropout(found: type) -> None:
    for p in (0.5, 0.2):
        call = f"MyDropout({p})"
        module = expect_module(call, found(p))
        x = torch.rand(400, 250, generator=torch.Generator().manual_seed(260)) + 1

        module.eval()
        evaluated = module(x.clone())
        expect_shape(
            call=f"{call} in eval mode on x of shape (400, 250)",
            actual=evaluated,
            shape=tuple(x.shape),
        )

        if not torch.equal(evaluated, x):
            raise Mismatch(f"{call} in eval mode changes x, but it should return x unchanged. Check self.training first.")

        module.train()
        call = f"{call} in training mode on x of shape (400, 250), every number between 1 and 2"
        first = module(x.clone())
        expect_shape(
            call=call,
            actual=first,
            shape=tuple(x.shape),
        )
        dropped = (first == 0).float().mean().item()

        if abs(dropped - p) > 0.01:
            raise Mismatch(f"{call} zeroes {dropped:.1%} of the numbers, but it should zero about {p:.0%}.")

        survivors = first != 0

        if not torch.allclose(first[survivors], x[survivors] / (1 - p)):
            raise Mismatch(f"{call} should divide the numbers that survive by 1 - p, {1 - p:g}, so their average stays where it was.")

        if torch.equal(module(x.clone()) == 0, first == 0):
            raise Mismatch(f"{call} drops the same numbers on every call. Draw a new pick each time with torch.rand.")


def check_dropout1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="keep_mask",
        description="is True where a number survives, drawn with torch.rand",
        inputs=(
            ((2, 5), 0.5, seeded(261)),
            ((3, 4), 0.25, seeded(262)),
        ),
        reference=lambda shape, p, generator: torch.rand(shape, generator=generator) >= p,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="dropout",
        description="zeroes the dropped numbers and scales the survivors",
        inputs=(
            (small_tensor(263, 2, 5), 0.5, seeded(264)),
            (small_tensor(265, 3, 4), 0.25, seeded(266)),
        ),
        reference=lambda x, p, generator: x * (torch.rand(x.shape, generator=generator) >= p) / (1 - p),
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="MyDropout",
        method="forward",
        description="drops and scales in training mode and passes x through in eval mode",
        run=check_my_dropout,
    )


DROPOUT1 = Exercise(
    name="dropout1",
    chapter=DROPOUT,
    title="Building dropout",
    lesson=DROPOUT1_LESSON,
    source=DROPOUT1_SOURCE,
    solution=DROPOUT1_SOLUTION,
    hints={
        "keep_mask": "Draw torch.rand(shape, generator=generator) and compare it with p, so a number at or above p "
        "survives.",
        "dropout": "Multiply x by keep, then divide by 1 - p.",
        "MyDropout.forward": "If not self.training, return x. Otherwise build keep with torch.rand(x.shape) and "
        "self.p, as keep_mask does, and scale as dropout does.",
    },
    introduces=("dropout",),
    check=check_dropout1,
)


# ---------------------------------------------------------------------------
# dropout2
# ---------------------------------------------------------------------------

DROPOUT2_LESSON = prose(
    """
    Here an MLP with two hidden layers of 512 learned from only 1000 MNIST
    images. It trained for 300 epochs on minibatches of 100, at a learning rate
    of 0.1, with dropout after each hidden ReLU. The last row also drops 20% of
    the input pixels, as Hinton and colleagues did. Each number is the average
    of 3 seeds, and the training loss was measured with dropout off.

    | Dropout | Training loss | Validation loss | Validation accuracy |
    |---|---|---|---|
    | none | 0.0018 | 0.624 | 88.0% |
    | hidden 0.2 | 0.0009 | 0.553 | 89.0% |
    | hidden 0.5 | 0.0005 | 0.498 | 89.7% |
    | inputs 0.2, hidden 0.5 | 0.0005 | 0.373 | 91.4% |

    By the end all four fit the training images perfectly, so dropout didn't
    stop that. It changed where the validation loss went. Without dropout it
    rose from 0.41 at epoch 30 to 0.62. With dropout on the inputs too, it
    kept falling to 0.37, and the accuracy rose 3.4 points on every seed.

    The paper saw the same steps on all of MNIST. Its plain network made 160
    errors on the test set. 50% dropout in the hidden layers brought that to
    about 130, and dropping 20% of the pixels too brought it to about 110.

    Dropout has a cost. Each step trains a thinned model, so training takes
    longer to get as far, two to three times as long by the paper's count. At
    epoch 10 here, the model without dropout had a training loss of 0.62, and
    the one with hidden 0.5 had 0.72. Both were measured with dropout off.
    Their validation accuracy was 80.6% against 77.4%.

    A model with plenty of data for its size doesn't overfit, and there dropout
    only slows it down. Karpathy's nanoGPT turns it off for training from
    scratch, and suggests 0.1 or more for further training on a small set.

    The trap is the mode. Measuring needs `model.eval()`, and the model stays
    that way until `model.train()`. Forget it, and every epoch after the first
    trains with dropout off, and nothing warns you.
    """,
)

DROPOUT2_SOURCE = source_file(
    '''
    # dropout2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 3))
    model.eval()
    dropping = model.training                        # False until model.train()


    # TODO: Return the model's cross entropy and accuracy on x and y, measured in
    #   eval mode without recording anything for autograd, as two plain numbers.
    #   The accuracy is the share of right answers, between 0 and 1.
    def evaluate(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> tuple[float, float]:
        ...


    # TODO: This trains with dropout in the first epoch only, since evaluate
    #   leaves the model in eval mode. Fix it, so every epoch trains with dropout.
    def train_epochs(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        epochs: int,
        generator: torch.Generator,
    ) -> list[float]:
        validation_losses = []

        for _ in range(epochs):
            order = torch.randperm(len(x), generator=generator)

            for start in range(0, len(x), 100):
                batch = order[start:start + 100]
                loss = F.cross_entropy(model(x[batch]), y[batch])
                loss.backward()

                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter -= 0.1 * parameter.grad
                        parameter.grad = None

            validation_losses.append(evaluate(model, validation_x, validation_y)[0])

        return validation_losses


    position = [0.9, 0.5, 0.7].index(0.5)            # 1, counted from 0


    # TODO: Return the epoch, counted from 1, with the lowest validation loss,
    #   the one early stopping would keep.
    def best_epoch(validation_losses: list[float]) -> int:
        ...
    ''',
)

DROPOUT2_SOLUTION = source_file(
    '''
    # dropout2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 3))
    model.eval()
    dropping = model.training


    def evaluate(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> tuple[float, float]:
        model.eval()

        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, y).item()
            accuracy = (logits.argmax(dim=1) == y).float().mean().item()

        return loss, accuracy


    # model.train() at the start of every epoch turns dropout back on after evaluate.
    def train_epochs(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        epochs: int,
        generator: torch.Generator,
    ) -> list[float]:
        validation_losses = []

        for _ in range(epochs):
            model.train()
            order = torch.randperm(len(x), generator=generator)

            for start in range(0, len(x), 100):
                batch = order[start:start + 100]
                loss = F.cross_entropy(model(x[batch]), y[batch])
                loss.backward()

                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter -= 0.1 * parameter.grad
                        parameter.grad = None

            validation_losses.append(evaluate(model, validation_x, validation_y)[0])

        return validation_losses


    position = [0.9, 0.5, 0.7].index(0.5)


    def best_epoch(validation_losses: list[float]) -> int:
        return validation_losses.index(min(validation_losses)) + 1
    ''',
)


def dropout_model(
    seed: int,
    features: int,
    width: int,
    classes: int,
    p: float,
) -> nn.Sequential:
    torch.manual_seed(seed)

    return nn.Sequential(
        nn.Linear(features, width),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(width, classes),
    )


def reference_evaluate(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> tuple[float, float]:
    model.eval()

    with torch.no_grad():
        logits = model(x)

        return F.cross_entropy(logits, y).item(), (logits.argmax(dim=1) == y).float().mean().item()


def reference_train_epochs(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    validation_x: Tensor,
    validation_y: Tensor,
    epochs: int,
    generator: torch.Generator,
) -> list[float]:
    validation_losses = []

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator)

        for start in range(0, len(x), 100):
            batch = order[start:start + 100]
            F.cross_entropy(model(x[batch]), y[batch]).backward()

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= 0.1 * parameter.grad
                    parameter.grad = None

        validation_losses.append(reference_evaluate(model, validation_x, validation_y)[0])

    return validation_losses


def check_evaluate(found: object) -> None:
    for seed, p in ((270, 0.5), (271, 0.3)):
        x, y = labelled_points(seed, 50, 6, 3)
        call = f"evaluate on a model with dropout {p}, passed in still in training mode, and 50 labelled points"
        model = dropout_model(seed, 6, 16, 3, p)
        actual = found(model, x, y)
        expected = reference_evaluate(dropout_model(seed, 6, 16, 3, p), x, y)

        try:
            expect_parts("the loss", "the accuracy")(
                call=call,
                actual=actual,
                expected=expected,
            )
        except Mismatch as mismatch:
            if model.training:
                raise Mismatch(f"{call} measures with dropout still on, so its numbers change from run to run. Call model.eval() first.") from mismatch

            raise


def check_train_epochs(found: object) -> None:
    x, y = labelled_points(272, 300, 20, 4)
    validation_x, validation_y = labelled_points(273, 200, 20, 4)
    call = "train_epochs for 4 epochs on 300 labelled points, with dropout 0.5"

    torch.manual_seed(5)
    actual = found(dropout_model(274, 20, 64, 4, 0.5), x, y, validation_x, validation_y, 4, torch.Generator().manual_seed(6))
    torch.manual_seed(5)
    expected = reference_train_epochs(dropout_model(274, 20, 64, 4, 0.5), x, y, validation_x, validation_y, 4, torch.Generator().manual_seed(6))

    try:
        expect_numbers(
            call=call,
            actual=actual,
            expected=expected,
        )
    except Mismatch as mismatch:
        first_epoch_right = isinstance(actual, list) and len(actual) == len(expected) and abs(actual[0] - expected[0]) < 1e-5

        if first_epoch_right:
            raise Mismatch(f"{call} gives the right validation loss after epoch 1 but not after epoch 2, so the later epochs trained differently. Every epoch should train with dropout on.") from mismatch

        raise


def check_dropout2(checker: Checker) -> None:
    checker.test(
        function="evaluate",
        description="measures the loss and the accuracy in eval mode",
        run=check_evaluate,
    )
    checker.test(
        function="train_epochs",
        description="trains every epoch with dropout on",
        run=check_train_epochs,
    )
    check_on_inputs(
        checker=checker,
        function="best_epoch",
        description="gives the epoch with the lowest validation loss, counted from 1",
        inputs=(
            ([0.9, 0.5, 0.7],),
            ([0.41, 0.38, 0.37, 0.39, 0.45],),
            ([0.3, 0.4, 0.5],),
        ),
        reference=lambda losses: losses.index(min(losses)) + 1,
        expect=expect_value,
    )


DROPOUT2 = Exercise(
    name="dropout2",
    chapter=DROPOUT,
    title="What dropout buys",
    lesson=DROPOUT2_LESSON,
    source=DROPOUT2_SOURCE,
    solution=DROPOUT2_SOLUTION,
    hints={
        "evaluate": "Call model.eval() first, then work inside a with torch.no_grad() block. Take the logits once. "
        "F.cross_entropy of them is the loss, and the share of rows whose .argmax(dim=1) equals y is the accuracy.",
        "train_epochs": "Add model.train() as the first line inside the loop over epochs.",
        "best_epoch": "Find the position of the smallest loss with .index(min(...)), then add 1.",
    },
    introduces=(),
    check=check_dropout2,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# dropout3
# ---------------------------------------------------------------------------

DROPOUT3_LESSON = prose(
    """
    The 2014 dropout paper by Srivastava and others did not scale while
    training. It dropped numbers, and at test time multiplied the weights after
    them by the keep probability `1 - p` to make up for it. PyTorch scales
    during training instead, which is called inverted dropout. Each gives the
    next layer the same average in training and at test time. Inverted
    dropout leaves the model at test time with nothing to change.

    Each draw of dropout gives different numbers, but their average over many
    draws is the input itself. That is what dividing by `1 - p` buys.

    Where dropout goes in an MLP, from the paper.

    - after each hidden layer's activation: a neuron's output is what gets dropped, usually with p 0.5.
    - on the inputs: with a smaller p, 0.2 in the paper's image models.
    - never after the last layer: its output is the answer.

    Models in later chapters put dropout in more places, and those chapters
    say where.

    ```python
    nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10),
    )
    ```
    """,
)

DROPOUT3_SOURCE = source_file(
    '''
    # dropout3.py

    import torch
    import torch.nn as nn
    from torch import Tensor


    # keep_mask and dropout from dropout1.
    def keep_mask(
        shape: tuple[int, ...],
        p: float,
        generator: torch.Generator,
    ) -> Tensor:
        return torch.rand(shape, generator=generator) >= p


    def dropout(
        x: Tensor,
        p: float,
        generator: torch.Generator,
    ) -> Tensor:
        return x * keep_mask(x.shape, p, generator) / (1 - p)


    keep_probability = 1 - 0.5                       # the paper's scale at test time for p 0.5


    # TODO: Return the paper's dropout. In training, zero the numbers keep_mask
    #   drops, with no scaling. At test time, scale x by the keep probability.
    def paper_dropout(
        x: Tensor,
        p: float,
        generator: torch.Generator,
        training: bool,
    ) -> Tensor:
        ...


    # torch.stack piles tensors of the same shape along a new first dimension.
    piled = torch.stack([torch.ones(3), torch.zeros(3)])    # shape (2, 3)


    # TODO: Run dropout on x as many times as draws, each run from generator,
    #   and return the average of the runs.
    def average_of_draws(
        x: Tensor,
        p: float,
        generator: torch.Generator,
        draws: int,
    ) -> Tensor:
        ...


    # TODO: Finish the line in the loop, so this returns an MLP with dropout of
    #   p_input on the inputs, then each hidden layer as Linear, ReLU and dropout
    #   of p_hidden, then a last Linear. sizes holds every layer's size in order.
    def dropout_mlp(
        sizes: list[int],
        p_input: float,
        p_hidden: float,
    ) -> nn.Sequential:
        layers = [nn.Dropout(p_input)]

        for inputs, outputs in zip(sizes[:-2], sizes[1:-1]):
            layers.extend(...)

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        return nn.Sequential(*layers)
    ''',
)

DROPOUT3_SOLUTION = source_file(
    '''
    # dropout3.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor


    def keep_mask(
        shape: tuple[int, ...],
        p: float,
        generator: torch.Generator,
    ) -> Tensor:
        return torch.rand(shape, generator=generator) >= p


    def dropout(
        x: Tensor,
        p: float,
        generator: torch.Generator,
    ) -> Tensor:
        return x * keep_mask(x.shape, p, generator) / (1 - p)


    keep_probability = 1 - 0.5


    def paper_dropout(
        x: Tensor,
        p: float,
        generator: torch.Generator,
        training: bool,
    ) -> Tensor:
        if training:
            return x * keep_mask(x.shape, p, generator)

        return x * (1 - p)


    def average_of_draws(
        x: Tensor,
        p: float,
        generator: torch.Generator,
        draws: int,
    ) -> Tensor:
        return torch.stack([dropout(x, p, generator) for _ in range(draws)]).mean(dim=0)


    def dropout_mlp(
        sizes: list[int],
        p_input: float,
        p_hidden: float,
    ) -> nn.Sequential:
        layers = [nn.Dropout(p_input)]

        for inputs, outputs in zip(sizes[:-2], sizes[1:-1]):
            layers.extend([nn.Linear(inputs, outputs), nn.ReLU(), nn.Dropout(p_hidden)])

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        return nn.Sequential(*layers)
    ''',
)


def reference_dropout_mlp(
    sizes: list[int],
    p_input: float,
    p_hidden: float,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Dropout(p_input)]

    for inputs, outputs in zip(sizes[:-2], sizes[1:-1]):
        layers.extend([nn.Linear(inputs, outputs), nn.ReLU(), nn.Dropout(p_hidden)])

    layers.append(nn.Linear(sizes[-2], sizes[-1]))

    return nn.Sequential(*layers)


def expect_summary(
    call: str,
    actual: object,
    expected: nn.Sequential,
) -> None:
    model = expect_module(call, actual)

    if not isinstance(model, nn.Sequential):
        raise Mismatch(f"{call} is a {type(model).__name__}. It should be an nn.Sequential.")

    got = layer_summary(model)
    wanted = layer_summary(expected)

    if got != wanted:
        raise Mismatch(f"{call} should hold {summary_text(wanted)}, but it holds {summary_text(got)}.")


def reference_average(
    x: Tensor,
    p: float,
    generator: torch.Generator,
    draws: int,
) -> Tensor:
    runs = [x * (torch.rand(x.shape, generator=generator) >= p) / (1 - p) for _ in range(draws)]

    return torch.stack(runs).mean(dim=0)


def check_dropout3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="paper_dropout",
        description="zeroes without scaling in training and scales at test time",
        inputs=(
            (small_tensor(275, 2, 5), 0.5, seeded(276), True),
            (small_tensor(277, 3, 4), 0.25, seeded(278), True),
            (small_tensor(279, 2, 3), 0.25, seeded(280), False),
        ),
        reference=lambda x, p, generator, training: x * (torch.rand(x.shape, generator=generator) >= p) if training else x * (1 - p),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="average_of_draws",
        description="averages many runs of dropout",
        inputs=(
            (small_tensor(281, 2, 3), 0.5, seeded(282), 4),
            (Described(torch.full((2, 2), 3.0), "x of shape (2, 2) full of 3"), 0.2, seeded(283), 1000),
        ),
        reference=reference_average,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="dropout_mlp",
        description="puts dropout on the inputs and after each hidden ReLU",
        inputs=(
            ([784, 512, 10], 0.2, 0.5),
            ([20, 64, 64, 4], 0.1, 0.3),
        ),
        reference=reference_dropout_mlp,
        expect=expect_summary,
    )


DROPOUT3 = Exercise(
    name="dropout3",
    chapter=DROPOUT,
    title="Where dropout goes",
    lesson=DROPOUT3_LESSON,
    source=DROPOUT3_SOURCE,
    solution=DROPOUT3_SOLUTION,
    hints={
        "paper_dropout": "When training, multiply x by keep_mask(x.shape, p, generator). Otherwise multiply x by 1 - p.",
        "average_of_draws": "Make a list of as many runs of dropout(x, p, generator) as draws, torch.stack them, and "
        "take the mean along dim=0.",
        "dropout_mlp": "Each hidden layer adds three modules, a Linear from inputs to outputs, a ReLU and an "
        "nn.Dropout of p_hidden.",
    },
    introduces=(),
    check=check_dropout3,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

# dropout2's rows: the dropout on the inputs, then after each hidden ReLU.
DROPOUT_SETUPS: dict[str, tuple[float, float]] = {
    "none": (0.0, 0.0),
    "hidden 0.2": (0.0, 0.2),
    "hidden 0.5": (0.0, 0.5),
    "inputs 0.2, hidden 0.5": (0.2, 0.5),
}


def measure_dropout(
    setup: str,
    seed: int,
) -> dict[str, float]:
    """dropout2's numbers for one setup and seed. An MLP with two hidden layers of 512 trains on mnist_sets' 1000 small
    images for 300 epochs of minibatches of 100 at a rate of 0.1, and is measured with dropout off at epochs 10, 30
    and 300 on its training images and on the 10,000 validation images."""
    sets = mnist_sets()
    x, y = sets["small_x"], sets["small_y"]
    validation_x, validation_y = sets["validation_x"], sets["validation_y"]
    p_inputs, p_hidden = DROPOUT_SETUPS[setup]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    try:
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Dropout(p_inputs),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(p_hidden),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p_hidden),
            nn.Linear(512, 10),
        )
        generator = torch.Generator().manual_seed(1000 + seed)
        results: dict[str, float] = {}

        for epoch in range(1, 301):
            model.train()
            order = torch.randperm(len(x), generator=generator)

            for start in range(0, len(x), 100):
                batch = order[start:start + 100]
                F.cross_entropy(model(x[batch]), y[batch]).backward()

                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter -= 0.1 * parameter.grad
                        parameter.grad = None

            if epoch not in (10, 30, 300):
                continue

            training_loss, _ = reference_evaluate(model, x, y)
            validation_loss, validation_accuracy = reference_evaluate(model, validation_x, validation_y)
            results[f"training loss {epoch}"] = training_loss
            results[f"validation loss {epoch}"] = validation_loss
            results[f"validation accuracy {epoch}"] = validation_accuracy
    finally:
        torch.set_num_threads(threads)

    return results


def measure_dropout_table(seeds: range = range(3)) -> dict[str, list[dict[str, float]]]:
    """Every row of dropout2's table, on every seed. It takes many minutes, so runs of one setup and seed can also be
    spread over several processes with measure_dropout."""
    return {
        setup: [
            measure_dropout(
                setup=setup,
                seed=seed,
            )
            for seed in seeds
        ]
        for setup in DROPOUT_SETUPS
    }
