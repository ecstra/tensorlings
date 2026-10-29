"""29_broken_training, Section 4's project, the names model with planted slips, found with the recipe and then tuned."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import Checker, Mismatch, expect_close, expect_parts
from ..text import prose, source_file
from .linear_layers import expect_module
from .names import CHARACTERS, ReferenceNamesMLP, context_sets, reference_contexts

BROKEN_TRAINING = Chapter(
    folder="29_broken_training",
    title="Project, a broken training run",
    readme=source_file(
        """
        # Project, a broken training run

        Section 4's project is the names model of Section 3 with slips planted
        in it. Nothing crashes, it only trains worse. You find each slip with a
        check from Karpathy's recipe, then tune the fixed run with AdamW,
        warmup, cosine decay and clipping until it beats a target. It needs the
        names data, which Section 3's project downloads.

        ```bash
        .venv/bin/python run.py data names
        ```

        ## Further reading

        - [Karpathy, a recipe for training neural networks](https://karpathy.github.io/2019/04/25/recipe/)
        - [CS231n, babysitting the learning process](https://cs231n.github.io/neural-networks-3/#baby)
        """,
    ),
)

NAMES_MLP_CLASS = '''
class NamesMLP(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        embedding_size: int,
        context: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, embedding_size)
        self.hidden = nn.Linear(context * embedding_size, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocabulary)

        with torch.no_grad():
            self.head.weight *= 0.1
            self.head.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        vectors = self.embedding(x).reshape(len(x), -1)

        return self.head(torch.tanh(self.norm(self.hidden(vectors))))
'''


def names_model(
    seed: int,
    make: type = ReferenceNamesMLP,
) -> nn.Module:
    torch.manual_seed(seed)

    return make(27, 10, 3, 200)


def validation_loss_of(model: nn.Module) -> float:
    validation_x, validation_y = context_sets()["validation"]
    model.eval()

    with torch.no_grad():
        return F.cross_entropy(model(validation_x), validation_y).item()


def reference_train_names(
    model: nn.Module,
    X: Tensor,
    Y: Tensor,
    steps: int,
    generator: torch.Generator,
) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.train()

    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = 0.1 if step < steps * 3 // 4 else 0.01

        batch = torch.randint(0, len(X), (32,), generator=generator)
        loss = F.cross_entropy(model(X[batch]), Y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# broken1
# ---------------------------------------------------------------------------

BROKEN1_LESSON = prose(
    """
    Section 4's project starts with a broken run. The file holds the names
    model of Section 3 with one slip planted in each of its four parts.
    Nothing crashes. It trains, and comes out worse than names3, and the job
    is to find out why.

    Guessing wastes hours. Karpathy's recipe goes through cheap checks in
    order, and each one here points at one part.

    - look at the data: a few contexts with the character after each, which should read like `.em` then `m`.
    - the loss at the start: it should be near `log(27)`, 3.30.
    - every layer's gradient: after one backward pass, no parameter should be left without one.
    - the training loop: a sound loop trains the model down, and the same loop with its gradients piling up does not.

    Run each check in your own scratch file or in a Python prompt, with the
    tools of debug1 to debug5, then fix what it finds. The checks of this
    exercise say which part is still off and what they saw, so you can work
    one at a time.

    The fixed model, trained like names3, reaches about 2.22 on the validation
    names in 10,000 steps.
    """,
)

BROKEN1_SOURCE = source_file(
    '''
    # broken1.py
    #
    # The names model of Section 3, with a slip planted in each part. Find each
    # one with a check from Karpathy's recipe and fix it.

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Look at the contexts this gives with the next id of each. Something
    #   is off. Fix it.
    def build_contexts(
        names: list[str],
        chars: list[str],
        context: int,
    ) -> tuple[Tensor, Tensor]:
        ids = {char: position for position, char in enumerate(chars)}
        X, Y = [], []

        for name in names:
            window = [0] * context

            for char in name + ".":
                Y.append(ids[char])
                window = window[1:] + [ids[char]]
                X.append(window)

        return torch.tensor(X), torch.tensor(Y)


    # TODO: Two slips, one in each method. The loss at the start is far from
    #   log(27), and one layer never gets a gradient. Fix both.
    class NamesMLP(nn.Module):
        def __init__(
            self,
            vocabulary: int,
            embedding_size: int,
            context: int,
            hidden: int,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocabulary, embedding_size)
            self.hidden = nn.Linear(context * embedding_size, hidden)
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, vocabulary)

            with torch.no_grad():
                self.head.weight *= 10
                self.head.bias.zero_()

        def forward(self, x: Tensor) -> Tensor:
            vectors = self.embedding(x).reshape(len(x), -1)

            return self.head(torch.tanh(self.hidden(vectors)))


    # TODO: This loop trains, but worse than it should. Find the slip and fix it.
    def train_names(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model.train()

        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = 0.1 if step < steps * 3 // 4 else 0.01

            batch = torch.randint(0, len(X), (32,), generator=generator)
            loss = F.cross_entropy(model(X[batch]), Y[batch])
            loss.backward()
            optimizer.step()
    ''',
)

BROKEN1_SOLUTION = source_file(
    '''
    # broken1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # The context went into X after it had moved on, so it already held the answer.
    def build_contexts(
        names: list[str],
        chars: list[str],
        context: int,
    ) -> tuple[Tensor, Tensor]:
        ids = {char: position for position, char in enumerate(chars)}
        X, Y = [], []

        for name in names:
            window = [0] * context

            for char in name + ".":
                X.append(window)
                Y.append(ids[char])
                window = window[1:] + [ids[char]]

        return torch.tensor(X), torch.tensor(Y)


    # The head was scaled up instead of down, and forward skipped the norm.
    class NamesMLP(nn.Module):
        def __init__(
            self,
            vocabulary: int,
            embedding_size: int,
            context: int,
            hidden: int,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocabulary, embedding_size)
            self.hidden = nn.Linear(context * embedding_size, hidden)
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, vocabulary)

            with torch.no_grad():
                self.head.weight *= 0.1
                self.head.bias.zero_()

        def forward(self, x: Tensor) -> Tensor:
            vectors = self.embedding(x).reshape(len(x), -1)

            return self.head(torch.tanh(self.norm(self.hidden(vectors))))


    # Without zero_grad every step's gradient piled onto the ones before.
    def train_names(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model.train()

        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = 0.1 if step < steps * 3 // 4 else 0.01

            batch = torch.randint(0, len(X), (32,), generator=generator)
            loss = F.cross_entropy(model(X[batch]), Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    ''',
)


def check_build_contexts(found: object) -> None:
    actual = found(["emma"], CHARACTERS, 3)
    expected = reference_contexts(["emma"], CHARACTERS, 3)

    if isinstance(actual, tuple) and len(actual) == 2 and isinstance(actual[0], Tensor) and actual[0].shape == expected[0].shape:
        contexts = ["".join(CHARACTERS[index] for index in row) for row in actual[0].tolist()]
        following = [CHARACTERS[index] for index in actual[1].tolist()]

        if contexts != ["".join(CHARACTERS[index] for index in row) for row in expected[0].tolist()]:
            pairs = ", ".join(f"{context} then {after}" for context, after in zip(contexts, following))
            raise Mismatch(f"build_contexts on emma gives {pairs}. Each context should end before the character that follows it.")

    expect_parts("the contexts", "the next ids")(
        call="build_contexts(['emma'], the 27 characters, 3)",
        actual=actual,
        expected=expected,
    )


def check_start(found: type) -> None:
    model = expect_module("NamesMLP(27, 10, 3, 200)", names_model(660, found))
    start = validation_loss_of(model)

    if abs(start - math.log(27)) > 0.1:
        raise Mismatch(f"NamesMLP starts at a validation loss of {start:.2f}, but a fresh model should start near log(27), 3.30.")

    theirs = names_model(660)

    if not torch.allclose(model.head.weight, theirs.head.weight):
        raise Mismatch("NamesMLP's head should start at a tenth of what nn.Linear drew.")


def check_every_gradient(found: type) -> None:
    model = expect_module("NamesMLP(27, 10, 3, 200)", names_model(661, found))
    X, Y = context_sets()["train"]
    F.cross_entropy(model(X[:64]), Y[:64]).backward()
    silent = [name for name, parameter in model.named_parameters() if parameter.grad is None]

    if silent:
        listed = silent[0] if len(silent) == 1 else ", ".join(silent[:-1]) + " and " + silent[-1]
        raise Mismatch(f"After one backward pass, {listed} got no gradient, so that layer never trains.")

    theirs = names_model(661)
    x = X[:8]

    with torch.no_grad():
        expected = theirs(x)

    if not torch.allclose(model(x), expected, atol=1e-5):
        raise Mismatch("NamesMLP's forward should run hidden, norm, tanh and head in that order.")


def check_train_names(found: object) -> None:
    X, Y = context_sets()["train"]
    mine = names_model(662)
    found(mine, X, Y, 400, torch.Generator().manual_seed(663))
    theirs = names_model(662)
    reference_train_names(theirs, X, Y, 400, torch.Generator().manual_seed(663))
    actual, expected = validation_loss_of(mine), validation_loss_of(theirs)

    if abs(actual - expected) > 1e-3:
        raise Mismatch(f"After 400 steps of train_names the validation loss is {actual:.3f}, where a sound loop reaches {expected:.3f}. Check what each step starts from.")


def check_broken1(checker: Checker) -> None:
    checker.test(
        function="build_contexts",
        description="pairs each context with the character after it",
        run=check_build_contexts,
    )
    checker.test_method(
        class_name="NamesMLP",
        method="__init__",
        description="starts near a loss of log(27)",
        run=check_start,
    )
    checker.test_method(
        class_name="NamesMLP",
        method="forward",
        description="gives every layer a gradient",
        run=check_every_gradient,
    )
    checker.test(
        function="train_names",
        description="trains like a sound loop",
        run=check_train_names,
    )


BROKEN1 = Exercise(
    name="broken1",
    chapter=BROKEN_TRAINING,
    title="Finding the slips",
    lesson=BROKEN1_LESSON,
    source=BROKEN1_SOURCE,
    solution=BROKEN1_SOLUTION,
    hints={
        "build_contexts": "Look at the order of the three lines in the inner loop. Which context does X get, the one "
        "before the character or the one after?",
        "NamesMLP.__init__": "The head should start small, so the scores start nearly equal.",
        "NamesMLP.forward": "One module made in __init__ is never called here.",
        "train_names": "Compare the loop with sgd1's. One of its three lines is missing.",
    },
    introduces=(),
    check=check_broken1,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)
