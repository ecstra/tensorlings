"""29_broken_training, Section 4's project, the names model with planted slips, found with the recipe and then tuned."""

from __future__ import annotations

import math
import textwrap

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

# Indented like the files around it, which source_file takes the indentation off.
INDENTED_NAMES_MLP = textwrap.indent(NAMES_MLP_CLASS.strip("\n"), "    ")


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
    model of Section 3 with four slips planted in it, one for each check
    below. Nothing crashes. It trains, and comes out worse than names3, and
    the job is to find out why.

    Guessing wastes hours. Karpathy's recipe goes through cheap checks in
    order, and each one here points at one part.

    - look at the data: a few contexts with the character after each, which should read like `.em` then `m`.
    - the loss at the start: it should be near `log(27)`, 3.30.
    - every layer's gradient: after one backward pass, no parameter should be left without one.
    - the training loop: a sound loop trains the model down, and the same loop with its gradients piling up does not.

    Run each check in a Python prompt with the tools of debug1 to debug5,
    then fix what it finds. From the project folder, this opens a prompt that
    has run your file, and the lines after it look at the data.

    ```bash
    PYTHONPATH=. .venv/bin/python -i exercises/29_broken_training/broken1.py
    ```

    ```python
    from tensorlings.data import load_names
    names = load_names()
    chars = ["."] + sorted(set("".join(names)))
    build_contexts(["emma"], chars, 3)
    ```

    The checks of this exercise say which part is still off and what they
    saw, so you can work one at a time.

    The fixed model, trained like names3, reaches about 2.22 on the validation
    names in 10,000 steps.
    """,
)

BROKEN1_SOURCE = source_file(
    '''
    # broken1.py
    #
    # The names model of Section 3, with four slips planted in it. Find each
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
    #   log(27), and one layer never gets a gradient. Fix both, so the model
    #   matches names2's NamesMLP.
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
    theirs.load_state_dict(model.state_dict())
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
        "NamesMLP.__init__": "The head should start small, so the scores start nearly equal. Shrink its weights to a "
        "tenth of what nn.Linear drew, as names2 did.",
        "NamesMLP.forward": "One module made in __init__ is never called here. Run hidden, norm, tanh and head in that "
        "order, as names2 did.",
        "train_names": "Compare the loop with sgd1's. One of its three lines is missing.",
    },
    introduces=(),
    check=check_broken1,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# broken2
# ---------------------------------------------------------------------------

TUNED_TARGET = 2.20

BROKEN2_LESSON = prose(
    """
    With the slips fixed, tune the run. names3 trained with plain SGD and one
    drop of the rate. Swap in what Section 4 built.

    - AdamW at 0.003, with weight decay 0.01 on the weight matrices and embeddings only.
    - 200 steps of warmup, then cosine decay to a tenth of the base rate.
    - the global gradient norm clipped at 1.0.
    - minibatches of 64, which halves the variance of each step's gradient, at twice the work per step.

    | Run | Batch | Steps | Validation loss |
    |---|---|---|---|
    | names3, SGD with one drop | 32 | 10,000 | 2.217 |
    | tuned, AdamW | 32 | 10,000 | 2.202 |
    | tuned, AdamW | 64 | 10,000 | 2.170 |

    The tuned run beats names3 in the same number of steps. The larger batch
    helps most, though each of its steps sees twice the examples. The gains
    are small next to a slip, since names3 was already sound. That is why the
    recipe's checks come before any tuning.

    The check trains your tuned run for 10,000 steps and needs it below 2.20.
    It takes a minute or two, so let it finish.
    """,
)

BROKEN2_SOURCE = source_file(
    '''
    # broken2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # The fixed model from broken1.
''' + INDENTED_NAMES_MLP + '''


    # TODO: Return AdamW for model at lr, with weight decay 0.01 on the
    #   parameters of two or more dimensions and none on the rest.
    def make_optimizer(
        model: nn.Module,
        lr: float,
    ) -> torch.optim.Optimizer:
        ...


    # TODO: Return the rate at step. It warms up over warmup steps to base, as
    #   warmup1 does, then falls along a cosine to base / 10 at step total.
    def rate_at(
        step: int,
        base: float,
        warmup: int,
        total: int,
    ) -> float:
        ...


    # TODO: Finish the two lines with ..., so each step sets the rate from
    #   rate_at, with base 0.003 and 200 warmup steps, and clips the global
    #   norm at 1.0 between backward and step.
    def train_tuned(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> None:
        optimizer = make_optimizer(model, 0.003)
        model.train()

        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = ...

            batch = torch.randint(0, len(X), (64,), generator=generator)
            loss = F.cross_entropy(model(X[batch]), Y[batch])
            optimizer.zero_grad()
            loss.backward()
            ...
            optimizer.step()
    ''',
)

BROKEN2_SOLUTION = source_file(
    '''
    # broken2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


''' + INDENTED_NAMES_MLP + '''


    def make_optimizer(
        model: nn.Module,
        lr: float,
    ) -> torch.optim.Optimizer:
        decayed = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
        kept = [parameter for parameter in model.parameters() if parameter.dim() < 2]

        return torch.optim.AdamW([{"params": decayed, "weight_decay": 0.01}, {"params": kept, "weight_decay": 0.0}], lr=lr)


    def rate_at(
        step: int,
        base: float,
        warmup: int,
        total: int,
    ) -> float:
        if step < warmup:
            return base * (step + 1) / warmup

        progress = (step - warmup) / (total - warmup)

        return base / 10 + (base - base / 10) * (1 + math.cos(math.pi * progress)) / 2


    def train_tuned(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> None:
        optimizer = make_optimizer(model, 0.003)
        model.train()

        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = rate_at(step, 0.003, 200, steps)

            batch = torch.randint(0, len(X), (64,), generator=generator)
            loss = F.cross_entropy(model(X[batch]), Y[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    ''',
)


def reference_rate_at(
    step: int,
    base: float,
    warmup: int,
    total: int,
) -> float:
    if step < warmup:
        return base * (step + 1) / warmup

    progress = (step - warmup) / (total - warmup)

    return base / 10 + (base - base / 10) * (1 + math.cos(math.pi * progress)) / 2


def reference_train_tuned(
    model: nn.Module,
    X: Tensor,
    Y: Tensor,
    steps: int,
    generator: torch.Generator,
) -> None:
    decayed = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    kept = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decayed, "weight_decay": 0.01}, {"params": kept, "weight_decay": 0.0}], lr=0.003)
    model.train()

    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = reference_rate_at(step, 0.003, 200, steps)

        batch = torch.randint(0, len(X), (64,), generator=generator)
        loss = F.cross_entropy(model(X[batch]), Y[batch])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def check_make_optimizer(found: object) -> None:
    model = names_model(670)
    call = "make_optimizer for the names model at 0.003"
    optimizer = found(model, 0.003)

    if type(optimizer) is not torch.optim.AdamW:
        raise Mismatch(f"{call} is {type(optimizer).__name__}, but it should be torch.optim.AdamW.")

    names = {id(parameter): name for name, parameter in model.named_parameters()}
    wanted = (
        ([name for name, parameter in model.named_parameters() if parameter.dim() >= 2], 0.01),
        ([name for name, parameter in model.named_parameters() if parameter.dim() < 2], 0.0),
    )
    decays = {}

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            decays[names.get(id(parameter), "something else")] = group["weight_decay"]

    for expected, decay in wanted:
        for name in expected:
            if decays.get(name) != decay:
                raise Mismatch(f"In {call}, {name} should have a weight decay of {decay}, but it has {decays.get(name, 'no group')}.")


def check_rate_at(found: object) -> None:
    for step in (0, 199, 200, 5100, 9999):
        expected = reference_rate_at(step, 0.003, 200, 10000)
        expect_close(
            call=f"rate_at({step}, 0.003, 200, 10000)",
            actual=found(step, 0.003, 200, 10000),
            expected=expected,
            tolerance=1e-12 + 1e-9 * expected,
        )


def check_train_tuned_steps(found: object) -> None:
    X, Y = context_sets()["train"]
    mine = names_model(671)
    found(mine, X, Y, 300, torch.Generator().manual_seed(672))
    theirs = names_model(671)
    reference_train_tuned(theirs, X, Y, 300, torch.Generator().manual_seed(672))
    expect_close(
        call="the validation loss after 300 steps of train_tuned",
        actual=validation_loss_of(mine),
        expected=validation_loss_of(theirs),
        tolerance=1e-3,
    )


def check_beats_names3(found: object) -> None:
    X, Y = context_sets()["train"]
    model = names_model(0)
    found(model, X, Y, 10000, torch.Generator().manual_seed(1))
    loss = validation_loss_of(model)

    if loss >= TUNED_TARGET:
        raise Mismatch(f"After 10,000 steps of train_tuned the validation loss is {loss:.4f}, which isn't below the target of {TUNED_TARGET}.")


def check_broken2(checker: Checker) -> None:
    checker.test(
        function="make_optimizer",
        description="decays the weight matrices and embeddings only",
        run=check_make_optimizer,
    )
    checker.test(
        function="rate_at",
        description="warms up, then decays to a tenth along a cosine",
        run=check_rate_at,
    )
    checker.test(
        function="train_tuned",
        description="steps like the tuned reference",
        run=check_train_tuned_steps,
    )
    checker.test(
        function="train_tuned",
        description="gets the validation loss below 2.20 in 10,000 steps",
        run=check_beats_names3,
    )


BROKEN2 = Exercise(
    name="broken2",
    chapter=BROKEN_TRAINING,
    title="Tuning the fixed run",
    lesson=BROKEN2_LESSON,
    source=BROKEN2_SOURCE,
    solution=BROKEN2_SOLUTION,
    hints={
        "make_optimizer": "Make the two groups as quiz6's adamw_for does, with 0.01 as the decay.",
        "rate_at": "quiz6's set_rate works the rate out the same way, without setting it.",
        "train_tuned": "The rate is rate_at(step, 0.003, 200, steps). The missing line is "
        "torch.nn.utils.clip_grad_norm_ on model.parameters() at 1.0.",
    },
    introduces=(),
    check=check_broken2,
    time_limit=300,
    imports_first=OPTIMIZER_IMPORTS,
)
