"""29_broken_training, Section 4's project, the names model with planted slips, found with the recipe and then tuned."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import Checker, Mismatch, expect_close, expect_parts
from ..text import prose, source_file
from .adaptive import pinned_threads
from .linear_layers import expect_module
from .names import CHARACTERS, ReferenceNamesMLP, context_sets, name_splits, reference_contexts

BROKEN_TRAINING = Chapter(
    folder="29_broken_training",
    title="Project, a broken training run",
    readme=source_file(
        """
        # Project, a broken training run

        Section 4's project is the names model of Section 3 with slips planted
        in it. Nothing crashes, but its numbers make no sense. You find each
        slip with a check from Karpathy's recipe or the CS231n notes, then
        tune the fixed run with Adam, warmup, cosine decay and clipping until it
        beats a target. It needs the names data, which Section 3's project
        downloads.

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


def validation_loss_of(
    model: nn.Module,
    validation: tuple[Tensor, Tensor] | None = None,
) -> float:
    validation_x, validation_y = context_sets()["validation"] if validation is None else validation
    model.eval()

    with torch.no_grad():
        return F.cross_entropy(model(validation_x), validation_y).item()


def reference_train_names(
    model: nn.Module,
    X: Tensor,
    Y: Tensor,
    steps: int,
    generator: torch.Generator,
    batch_size: int = 32,
    clears_gradients: bool = True,
    after_step: Callable[[int], None] | None = None,
) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.train()

    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = 0.1 if step < steps * 3 // 4 else 0.01

        batch = torch.randint(0, len(X), (batch_size,), generator=generator)
        loss = F.cross_entropy(model(X[batch]), Y[batch])

        if clears_gradients:
            optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if after_step is not None:
            after_step(step + 1)
            model.train()


def leaky_contexts(
    names: list[str],
    chars: list[str],
    context: int,
) -> tuple[Tensor, Tensor]:
    """broken1's planted data slip, each context taken after the window has moved on to the character it predicts."""
    ids = {char: position for position, char in enumerate(chars)}
    X, Y = [], []

    for name in names:
        window = [0] * context

        for char in name + ".":
            Y.append(ids[char])
            window = window[1:] + [ids[char]]
            X.append(window)

    return torch.tensor(X), torch.tensor(Y)


class SlippedNamesMLP(ReferenceNamesMLP):
    """broken1's starting model, its head 10 times nn.Linear's draw where names2 shrinks it, and its norm skipped."""

    def __init__(
        self,
        vocabulary: int,
        embedding_size: int,
        context: int,
        hidden: int,
    ) -> None:
        super().__init__(vocabulary, embedding_size, context, hidden)

        # The parent already shrank the head to a tenth, so 100 times that is 10 times nn.Linear's draw.
        with torch.no_grad():
            self.head.weight *= 100

    def forward(self, x: Tensor) -> Tensor:
        vectors = self.embedding(x).reshape(len(x), -1)

        return self.head(torch.tanh(self.hidden(vectors)))


# ---------------------------------------------------------------------------
# broken1
# ---------------------------------------------------------------------------

BROKEN1_LESSON = prose(
    """
    Section 4's project starts with a broken run. The file holds the names
    model of Section 3 with four slips planted in it, one for each check
    below. Nothing crashes, but the numbers make no sense.

    Here the file trained like names3 for 10,000 steps on three seeds, on
    contexts from its own `build_contexts`. Its validation loss started at
    6.1 to 6.9, where a fresh model gives 3.30. After 100 steps it read 0.19
    to 0.54, far below the 2.10 of Karpathy's long runs. At the end it was
    116 to 1,429. The fixed file started at 3.29 to 3.30 and ended at 2.217
    to 2.219.

    Guessing wastes hours. Karpathy's recipe and the CS231n notes go through
    cheap checks in order, and each one here points at one part.

    - look at the data: each context with the character after it, which should start with `...` then `e`, as in names2's table.
    - the loss at the start: it should be near `log(27)`, 3.30.
    - every layer's gradient: after one backward pass, no parameter should be left without one.
    - the training loop: a sound loop trains the model down, and the same loop with its gradients piling up sends the loss up.

    Run each check in a Python prompt with the tools of debug1 to debug5,
    then fix what it finds. From the project folder, this opens a prompt that
    has run your file. The lines after it turn the ids of `emma`'s contexts
    back into characters, each context next to the character after it.

    ```bash
    PYTHONPATH=. .venv/bin/python -i exercises/29_broken_training/broken1.py
    ```

    ```python
    from tensorlings.data import load_names
    names = load_names()
    chars = ["."] + sorted(set("".join(names)))
    X, Y = build_contexts(["emma"], chars, 3)
    contexts = ["".join(chars[i] for i in row) for row in X.tolist()]
    list(zip(contexts, [chars[i] for i in Y.tolist()]))
    ```

    The checks of this exercise say which part is still off and what they
    saw, so you can work one at a time.
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


    # TODO: Trained for long, this loop's loss climbs instead of falling. Find
    #   the slip and fix it.
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

    - Adam at 0.003.
    - 200 steps of warmup, then cosine decay to a tenth of the base rate.
    - minibatches of 64, which halves the variance of each step's gradient, at twice the work per step.
    - the global gradient norm clipped at 1.3.

    The cap comes from the norms, as in clipping2. Unclipped, their 90th
    percentile came out at 1.29 to 1.31, so 1.3 clips about one step in ten.
    Weight decay is left out. It pays off on a model with room to memorize,
    like adamw2's wide one.

    Each row trained for 10,000 steps on three seeds. The last column says
    how much worse each row ended than the tuned run, seed by seed.

    | Run | Validation loss | Worse than tuned by |
    |---|---|---|
    | names3's loop, batches of 32 | 2.217 to 2.219 | 0.048 to 0.053 |
    | names3's loop, batches of 64 | 2.194 to 2.197 | 0.026 to 0.031 |
    | tuned | 2.166 to 2.169 | |
    | tuned, batches of 32 | 2.199 to 2.203 | 0.029 to 0.037 |
    | tuned, no cosine decay | 2.219 to 2.226 | 0.051 to 0.061 |
    | tuned, no warmup | 2.167 to 2.172 | 0.001 to 0.003 |
    | tuned, one loss 1000 times, not clipped | 2.194 to 2.197 | 0.025 to 0.030 |
    | tuned, one loss 1000 times, clipped | 2.166 to 2.169 | 0.000 |

    The cosine decay buys the most. Without it, Adam ends behind names3's
    loop on every seed. Batches of 64 buy the next most, and warmup a little
    on every seed.

    The last two rows scaled the loss at step 5,000 by 1000, as clipping2
    did. Unclipped, that one step cost about as much as halving the batch.
    Clipped, it changed nothing.

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
    #   norm at 1.3 between backward and step.
    def train_tuned(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        model.train()

        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = rate_at(step, 0.003, 200, steps)

            batch = torch.randint(0, len(X), (64,), generator=generator)
            loss = F.cross_entropy(model(X[batch]), Y[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.3)
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


TUNED_RATE = 0.003
TUNED_WARMUP = 200

# The 90th percentile of the global norms of the tuned run without clipping, as clipping2 sets a cap. measure_tuned_norms
# gives it per seed.
TUNED_CAP = 1.3

# measure_tuned scales one loss at this step by 1000, as clipping2's spike.
SPIKE_STEP = 5000


def reference_train_tuned(
    model: nn.Module,
    X: Tensor,
    Y: Tensor,
    steps: int,
    generator: torch.Generator,
    batch_size: int = 64,
    warmup: int = TUNED_WARMUP,
    cosine: bool = True,
    max_norm: float | None = TUNED_CAP,
    spike_step: int | None = None,
    norms: list[float] | None = None,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=TUNED_RATE)
    model.train()

    for step in range(steps):
        if cosine:
            rate = reference_rate_at(step, TUNED_RATE, warmup, steps)
        else:
            rate = TUNED_RATE * min(1.0, (step + 1) / warmup)

        for group in optimizer.param_groups:
            group["lr"] = rate

        batch = torch.randint(0, len(X), (batch_size,), generator=generator)
        loss = F.cross_entropy(model(X[batch]), Y[batch])

        if step == spike_step:
            loss = loss * 1000

        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf") if max_norm is None else max_norm)

        if norms is not None:
            norms.append(norm.item())

        optimizer.step()


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

    # A right loop gives the reference's weights exactly. A cap of 0.5, 1.0 or none moves some weight by 1e-3 or more,
    # while the validation loss can stay within 1e-4, so the weights are compared.
    gap = max((actual - expected).abs().max().item() for actual, expected in zip(mine.parameters(), theirs.parameters()))

    if gap > 1e-5:
        raise Mismatch(
            f"After 300 steps of train_tuned the weights differ from the tuned reference's by up to {gap:.2g}, and the "
            f"validation loss is {validation_loss_of(mine):.4f} where the reference reaches {validation_loss_of(theirs):.4f}. "
            "Check the rate each step and the clip at 1.3 between backward and step.",
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
        "rate_at": "quiz6's set_rate works the rate out the same way, without setting it.",
        "train_tuned": "Take the rate from rate_at, with base 0.003, 200 warmup steps and steps as the total. The "
        "missing line clips with torch.nn.utils.clip_grad_norm_ on model.parameters() at 1.3.",
    },
    introduces=(),
    check=check_broken2,
    time_limit=300,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

def measure_broken_run(
    seeds: tuple[int, ...] = (0, 1, 2),
    marks: tuple[int, ...] = (0, 100, 10000),
) -> dict[str, list[dict[int, float]]]:
    """broken1's lesson: the starting file and the fixed one trained like names3 for 10,000 steps. Each file builds its
    own training and validation contexts, and the result is the validation loss at each mark, per seed."""
    splits = name_splits()
    files = {
        "broken": (leaky_contexts, SlippedNamesMLP, False),
        "fixed": (reference_contexts, ReferenceNamesMLP, True),
    }
    results: dict[str, list[dict[int, float]]] = {}

    with pinned_threads():
        for label, (build, make, clears_gradients) in files.items():
            X, Y = build(list(splits["train"]), CHARACTERS, 3)
            validation = build(list(splits["validation"]), CHARACTERS, 3)

            for seed in seeds:
                model = names_model(seed, make)
                losses = {0: validation_loss_of(model, validation)}

                def record(step: int) -> None:
                    if step in marks:
                        losses[step] = validation_loss_of(model, validation)

                reference_train_names(
                    model,
                    X,
                    Y,
                    max(marks),
                    torch.Generator().manual_seed(seed + 1),
                    clears_gradients=clears_gradients,
                    after_step=record,
                )
                results.setdefault(label, []).append(losses)

    return results


# Each run is (its row in broken2's table, which loop, what it changes). names3's loop is SGD at 0.1 with one drop to
# 0.01 for the last quarter. The tuned loop is reference_train_tuned, and every run trains 10,000 steps.
TUNED_RUNS: tuple[tuple[str, str, dict[str, object]], ...] = (
    ("names3's loop, batches of 32", "names3", {"batch_size": 32}),
    ("names3's loop, batches of 64", "names3", {"batch_size": 64}),
    ("tuned", "tuned", {}),
    ("tuned, batches of 32", "tuned", {"batch_size": 32}),
    ("tuned, no warmup", "tuned", {"warmup": 0}),
    ("tuned, no cosine decay", "tuned", {"cosine": False}),
    ("tuned, one loss 1000 times, not clipped", "tuned", {"spike_step": SPIKE_STEP, "max_norm": None}),
    ("tuned, one loss 1000 times, clipped", "tuned", {"spike_step": SPIKE_STEP}),
)


def measure_tuned(
    runs: tuple[tuple[str, str, dict[str, object]], ...] = TUNED_RUNS,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, list[float]]:
    """broken2's table: what each piece of the tuned run buys. The validation loss per seed, each run starting from
    names_model(seed) with minibatches drawn from a generator seeded seed + 1, as the check's."""
    X, Y = context_sets()["train"]
    results: dict[str, list[float]] = {}

    with pinned_threads():
        for label, loop, changes in runs:
            train = reference_train_names if loop == "names3" else reference_train_tuned

            for seed in seeds:
                model = names_model(seed)
                train(
                    model,
                    X,
                    Y,
                    10000,
                    torch.Generator().manual_seed(seed + 1),
                    **changes,
                )
                results.setdefault(label, []).append(validation_loss_of(model))

    return results


def measure_tuned_norms(seeds: tuple[int, ...] = (0, 1, 2)) -> list[dict[str, float]]:
    """Where broken2's cap comes from: the global norms of the tuned run with no clipping, their 90th percentile, and
    the share of steps above TUNED_CAP, per seed."""
    X, Y = context_sets()["train"]
    results: list[dict[str, float]] = []

    with pinned_threads():
        for seed in seeds:
            norms: list[float] = []
            reference_train_tuned(
                names_model(seed),
                X,
                Y,
                10000,
                torch.Generator().manual_seed(seed + 1),
                max_norm=None,
                norms=norms,
            )
            logged = torch.tensor(norms)
            results.append(
                {
                    "median": logged.median().item(),
                    "90th percentile": logged.quantile(0.9).item(),
                    "share above the cap": (logged > TUNED_CAP).float().mean().item(),
                },
            )

    return results
