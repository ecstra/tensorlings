"""23_names, Section 3's project, a model that makes up names, from a counting bigram to Bengio's MLP."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..data import load_names
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
)
from ..text import prose, source_file
from .linear_layers import expect_module

NAMES = Chapter(
    folder="23_names",
    title="Project, making up names",
    readme=source_file(
        """
        # Project, making up names

        Section 3's project is a model that makes up names, as the first models
        of Karpathy's makemore do. It starts with a model that only counts
        which letter follows which, then builds the MLP language model of
        Bengio and colleagues from this section's blocks, trains it past the
        counting model, and samples new names from it. Download the data once
        from the project folder.

        ```bash
        .venv/bin/python run.py data names
        ```

        ## Further reading

        - [Karpathy, building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)
        - [Karpathy, building makemore part 2, the MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
        - [Bengio and others, a neural probabilistic language model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
        - [Karpathy, the makemore code](https://github.com/karpathy/makemore)
        """,
    ),
)

CHARACTERS = ["."] + [chr(code) for code in range(ord("a"), ord("z") + 1)]
BIGRAM_VALIDATION_LOSS = 2.465


@functools.cache
def name_splits() -> dict[str, tuple[str, ...]]:
    """The names shuffled once and split 80, 10 and 10 percent, the splits the lessons quote numbers from."""
    names = load_names()
    order = torch.randperm(len(names), generator=torch.Generator().manual_seed(42)).tolist()
    first, second = int(0.8 * len(names)), int(0.9 * len(names))

    return {
        "train": tuple(names[index] for index in order[:first]),
        "validation": tuple(names[index] for index in order[first:second]),
        "test": tuple(names[index] for index in order[second:]),
    }


def reference_pairs(
    names: list[str],
    chars: list[str],
) -> tuple[Tensor, Tensor]:
    ids = {char: position for position, char in enumerate(chars)}
    xs, ys = [], []

    for name in names:
        padded = "." + name + "."

        for first, second in zip(padded, padded[1:]):
            xs.append(ids[first])
            ys.append(ids[second])

    return torch.tensor(xs), torch.tensor(ys)


def reference_contexts(
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


class ReferenceNamesMLP(nn.Module):
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


# ---------------------------------------------------------------------------
# names1
# ---------------------------------------------------------------------------

NAMES1_LESSON = prose(
    """
    Section 3's project is a model that makes up names, like the first models
    of Karpathy's makemore. It reads a name one character at a time and
    learns to predict the next character. Download the data once, 32,033 names
    in about 230 KB, with `.venv/bin/python run.py data names`.

    ```python
    from tensorlings.data import load_names

    names = load_names()
    names[:3]    # ['emma', 'olivia', 'ava']
    ```

    Each character needs an id. The vocabulary is the 26 letters plus `.`,
    which marks where a name starts and ends, and takes id 0. A name then
    gives one pair per step. `emma` gives `(., e)`, `(e, m)`, `(m, m)`,
    `(m, a)` and `(a, .)`.

    The plainest model counts. Over the training names, count how often each
    character follows each other one. A row of counts, divided by its total,
    is a probability distribution over the next character. That is a
    **bigram** model, with one character of context. Its loss is the cross
    entropy, minus the log of the probability it gave each real next
    character, averaged.

    A pair that never shows up in training gets probability 0, and minus the
    log of 0 is infinite. Some validation names hold such a pair. So 1 is
    added to every count first, which is called smoothing.

    The names are shuffled and split 80, 10 and 10 percent. The smoothed
    bigram's loss is 2.454 on the training names and 2.465 on the validation
    names. That is the number to beat.

    `sorted(set(text))` gives the different characters of a string, in order.
    A pair of ids `x` and `y`, each below 27, can be numbered `x * 27 + y`,
    so one `torch.bincount` counts every pair at once.
    """,
)

NAMES1_SOURCE = source_file(
    '''
    # names1.py
    #
    # Section 3's project starts with the data. Download it once from the project
    # folder with
    #
    #     .venv/bin/python run.py data names

    import torch
    from torch import Tensor

    from tensorlings.data import load_names

    names = load_names()                             # 32,033 names, like 'emma'
    letters = sorted(set("banana"))                  # ['a', 'b', 'n']


    # TODO: Return the vocabulary, "." first and then every character that shows
    #   up in names, in order.
    def vocabulary(names: list[str]) -> list[str]:
        ...


    steps = list(zip(".ab.", "ab."))                 # [('.', 'a'), ('a', 'b'), ('b', '.')]


    # TODO: Finish the loop, so this returns the id of each character and the id
    #   of the one after it, for every step of every name with "." at each end.
    def bigram_pairs(
        names: list[str],
        chars: list[str],
    ) -> tuple[Tensor, Tensor]:
        ids = {char: position for position, char in enumerate(chars)}
        xs, ys = [], []

        for name in names:
            padded = "." + name + "."

            for first, second in zip(padded, padded[1:]):
                ...

        return torch.tensor(xs), torch.tensor(ys)


    counted = torch.bincount(torch.tensor([0, 2, 2]), minlength=4)    # tensor([1, 0, 2, 0])


    # TODO: Return a float tensor of shape (size, size), where row x, column y
    #   counts how often the pair x, y shows up in xs and ys.
    def bigram_counts(
        xs: Tensor,
        ys: Tensor,
        size: int,
    ) -> Tensor:
        ...


    # TODO: Return the bigram model's loss on the pairs xs and ys, as a plain
    #   number, with 1 added to every count before the rows become probabilities.
    def bigram_loss(
        counts: Tensor,
        xs: Tensor,
        ys: Tensor,
    ) -> float:
        ...
    ''',
)

NAMES1_SOLUTION = source_file(
    '''
    # names1.py, a solution for comparison

    import torch
    from torch import Tensor

    from tensorlings.data import load_names

    names = load_names()
    letters = sorted(set("banana"))


    def vocabulary(names: list[str]) -> list[str]:
        return ["."] + sorted(set("".join(names)))


    steps = list(zip(".ab.", "ab."))


    def bigram_pairs(
        names: list[str],
        chars: list[str],
    ) -> tuple[Tensor, Tensor]:
        ids = {char: position for position, char in enumerate(chars)}
        xs, ys = [], []

        for name in names:
            padded = "." + name + "."

            for first, second in zip(padded, padded[1:]):
                xs.append(ids[first])
                ys.append(ids[second])

        return torch.tensor(xs), torch.tensor(ys)


    counted = torch.bincount(torch.tensor([0, 2, 2]), minlength=4)


    def bigram_counts(
        xs: Tensor,
        ys: Tensor,
        size: int,
    ) -> Tensor:
        return torch.bincount(xs * size + ys, minlength=size * size).reshape(size, size).float()


    def bigram_loss(
        counts: Tensor,
        xs: Tensor,
        ys: Tensor,
    ) -> float:
        smoothed = counts + 1
        probabilities = smoothed / smoothed.sum(dim=1, keepdim=True)

        return -probabilities[xs, ys].log().mean().item()
    ''',
)


def reference_bigram_loss(
    counts: Tensor,
    xs: Tensor,
    ys: Tensor,
) -> float:
    smoothed = counts + 1
    probabilities = smoothed / smoothed.sum(dim=1, keepdim=True)

    return -probabilities[xs, ys].log().mean().item()


def check_names1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="vocabulary",
        description="gives . first, then every character in order",
        inputs=(
            (["emma", "ava", "bob"],),
            (["zoe", "max"],),
        ),
        reference=lambda names: ["."] + sorted(set("".join(names))),
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="bigram_pairs",
        description="gives every step of every name, with . at each end",
        inputs=(
            (["ab", "ba"], [".", "a", "b"]),
            (["emma"], [".", "a", "e", "m"]),
        ),
        reference=reference_pairs,
        expect=expect_parts("the first ids", "the next ids"),
    )
    check_on_inputs(
        checker=checker,
        function="bigram_counts",
        description="counts every pair into a square",
        inputs=(
            (torch.tensor([0, 1, 1, 2]), torch.tensor([1, 2, 2, 0]), 3),
            (torch.tensor([3, 0, 3, 3, 1]), torch.tensor([3, 2, 3, 0, 1]), 4),
        ),
        reference=lambda xs, ys, size: torch.bincount(xs * size + ys, minlength=size * size).reshape(size, size).float(),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="bigram_loss",
        description="gives the smoothed bigram's cross entropy",
        inputs=(
            (torch.tensor([[0.0, 3.0], [2.0, 1.0]]), torch.tensor([0, 1, 1]), torch.tensor([1, 0, 1])),
            (torch.tensor([[5.0, 0.0, 1.0], [0.0, 0.0, 4.0], [2.0, 2.0, 0.0]]), torch.tensor([0, 1, 2, 2]), torch.tensor([1, 2, 0, 2])),
        ),
        reference=reference_bigram_loss,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )


NAMES1 = Exercise(
    name="names1",
    chapter=NAMES,
    title="Counting pairs",
    lesson=NAMES1_LESSON,
    source=NAMES1_SOURCE,
    solution=NAMES1_SOLUTION,
    hints={
        "vocabulary": "Join the names into one string, take sorted(set(...)) of it, and put \".\" in a list in "
        "front.",
        "bigram_pairs": "Append ids[first] to xs and ids[second] to ys.",
        "bigram_counts": "Number each pair xs * size + ys, count them with torch.bincount and "
        "minlength=size * size, then .reshape(size, size) and .float().",
        "bigram_loss": "Add 1 to counts, divide each row by its sum with keepdim=True, pick probabilities[xs, ys], "
        "then take minus the mean of their logs.",
    },
    introduces=(),
    check=check_names1,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# names2
# ---------------------------------------------------------------------------

NAMES2_LESSON = prose(
    """
    A bigram sees one character. The language model of Bengio and colleagues,
    from 2003, reads the last few. Each goes through an embedding, the
    embedding vectors are laid end to end, and an MLP scores every possible
    next character. Karpathy builds it for names in makemore part 2, and adds
    normalization in part 3.

    With a context of 3, `emma` gives five examples, each three ids in and one
    out. A name starts with a context of `...`.

    | Context | Next |
    |---|---|
    | `...` | e |
    | `..e` | m |
    | `.em` | m |
    | `emm` | a |
    | `mma` | . |

    The model here starts with an embedding of size 10. A hidden Linear layer
    of 200 with LayerNorm and tanh follows, then a head scoring all 27
    characters. tanh is what the paper used. The LayerNorm keeps the hidden spread steady, as
    the normalization chapter showed.

    The head starts small, its weights shrunk to a tenth and its bias at 0.
    The scores then start nearly equal, so the starting loss is near
    `log(27)`, 3.30, as init3 asked.

    `window[1:] + [new]` drops a list's first item and adds a new one at the
    end. `*=` multiplies a tensor in place.
    """,
)

NAMES2_SOURCE = source_file(
    '''
    # names2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    window = [0, 0, 5]
    shifted = window[1:] + [13]                      # [0, 5, 13]


    # TODO: Finish the line that moves window, so this returns the context of
    #   every step of every name and the id that follows it. Each name starts
    #   from a context of all 0, the id of ".", and ends with ".".
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
                window = ...

        return torch.tensor(X), torch.tensor(Y)


    # TODO: Two things to finish. In __init__, shrink head's weights to a tenth
    #   and set its bias to 0. forward looks up the embedding of each id in x,
    #   lays each example's vectors end to end, then applies hidden, norm, tanh
    #   and head.
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
                ...

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

NAMES2_SOLUTION = source_file(
    '''
    # names2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    window = [0, 0, 5]
    shifted = window[1:] + [13]


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
    ''',
)


def check_names_start(found: type) -> None:
    for vocabulary, size, context, hidden in ((27, 10, 3, 20), (5, 2, 4, 6)):
        call = f"NamesMLP({vocabulary}, {size}, {context}, {hidden})"
        torch.manual_seed(450 + context)
        model = expect_module(call, found(vocabulary, size, context, hidden))
        torch.manual_seed(450 + context)
        reference = ReferenceNamesMLP(vocabulary, size, context, hidden)

        if model.head.bias.detach().abs().max().item() > 0:
            raise Mismatch(f"The bias of head in {call} should start at 0.")

        if not torch.allclose(model.head.weight, reference.head.weight):
            ratio = (model.head.weight.detach().std() / reference.head.weight.detach().std() / 10).item()
            raise Mismatch(f"The weights of head in {call} should be a tenth of what nn.Linear drew, but they are {ratio:.3g} times what it drew.")


def check_names_forward(found: type) -> None:
    for vocabulary, size, context, hidden in ((27, 10, 3, 20), (5, 2, 4, 6)):
        call = f"NamesMLP({vocabulary}, {size}, {context}, {hidden})"
        torch.manual_seed(460 + context)
        model = expect_module(call, found(vocabulary, size, context, hidden))
        x = torch.randint(0, vocabulary, (4, context), generator=torch.Generator().manual_seed(462 + context))

        with torch.no_grad():
            vectors = model.embedding(x).reshape(len(x), -1)
            expected = model.head(torch.tanh(model.norm(model.hidden(vectors))))

        expect_tensor(
            call=f"{call} on ids of shape (4, {context})",
            actual=model(x),
            expected=expected,
        )


def check_names2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="build_contexts",
        description="gives every context and the id after it",
        inputs=(
            (["emma"], Described(CHARACTERS, "the 27 characters, . then a to z"), 3),
            (["ab", "b"], [".", "a", "b"], 2),
        ),
        reference=reference_contexts,
        expect=expect_parts("the contexts", "the next ids"),
    )
    checker.test_method(
        class_name="NamesMLP",
        method="__init__",
        description="starts the head small, with its bias at 0",
        run=check_names_start,
    )
    checker.test_method(
        class_name="NamesMLP",
        method="forward",
        description="looks up, lays end to end, and applies hidden, norm, tanh and head",
        run=check_names_forward,
    )


NAMES2 = Exercise(
    name="names2",
    chapter=NAMES,
    title="Bengio's MLP",
    lesson=NAMES2_LESSON,
    source=NAMES2_SOURCE,
    solution=NAMES2_SOLUTION,
    hints={
        "build_contexts": "Drop the window's first id and add ids[char] at the end, as the line that sets shifted "
        "does.",
        "NamesMLP.__init__": "Multiply self.head.weight by 0.1 in place with *=, and call .zero_() on "
        "self.head.bias.",
        "NamesMLP.forward": "self.embedding(x) has shape (examples, context, size). Reshape it to len(x) rows, then "
        "pass it through self.hidden, self.norm, torch.tanh and self.head in that order.",
    },
    introduces=(),
    check=check_names2,
)


# ---------------------------------------------------------------------------
# names3
# ---------------------------------------------------------------------------

NAMES3_LESSON = prose(
    """
    Training is the minibatch loop of Section 2. Each step draws 32 random
    examples, and the learning rate drops from 0.1 to 0.01 for the last
    quarter of the steps. Section 4 explains why dropping it helps.

    | Model | Steps | Validation loss | Time on a laptop |
    |---|---|---|---|
    | bigram | none, it counts | 2.465 | |
    | MLP | 5,000 | 2.258 | 6 s |
    | MLP | 10,000 | 2.217 | 10 s |
    | MLP | 20,000 | 2.181 | 30 s |

    Karpathy's runs of the same shape reach about 2.10 after 200,000 steps.

    To make up a name, start from the context `...` and turn the model's
    scores into probabilities with softmax. Draw the next character from them,
    shift it into the context, and repeat until the model draws `.`.

    ```python
    probabilities = torch.tensor([[0.1, 0.0, 0.9]])
    torch.multinomial(probabilities, 1, generator=generator)    # tensor([[2]]) nine times in ten
    ```

    `torch.multinomial(probabilities, 1)` draws one position from each row,
    each position with its probability. The model takes a batch, so one
    context goes in as `torch.tensor([window])`, a batch of 1.

    Names from the 20,000 step model came out like kilyn, collie, caimen and
    amonia, and some like fameellieshael. The bigram's came out like zan,
    ayenn and deveerhacarolyme.
    """,
)

NAMES3_SOURCE = source_file(
    '''
    # names3.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # The model from names2.
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


    # TODO: Finish the two lines with ..., so each step trains on the 32 random
    #   examples in batch, at a learning rate of 0.1 for the first three quarters
    #   of the steps and 0.01 after.
    def train_names(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> None:
        model.train()

        for step in range(steps):
            batch = torch.randint(0, len(X), (32,), generator=generator)
            loss = ...
            loss.backward()
            rate = ...

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= rate * parameter.grad
                    parameter.grad = None


    # TODO: Return the model's cross entropy on X against Y, as a plain number,
    #   in eval mode and without recording anything for autograd.
    def validation_loss(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
    ) -> float:
        ...


    generator = torch.Generator().manual_seed(0)
    drawn = torch.multinomial(torch.tensor([[0.1, 0.0, 0.9]]), 1, generator=generator)    # tensor([[2]])


    # TODO: Finish the line that sets probabilities, the softmax of the model's
    #   scores for the context in window, so this makes up one name.
    def sample_name(
        model: nn.Module,
        chars: list[str],
        context: int,
        generator: torch.Generator,
    ) -> str:
        model.eval()
        window = [0] * context
        name = ""

        with torch.no_grad():
            while True:
                probabilities = ...
                next_id = torch.multinomial(probabilities, 1, generator=generator).item()

                if next_id == 0:
                    return name

                name += chars[next_id]
                window = window[1:] + [next_id]
    ''',
)

NAMES3_SOLUTION = source_file(
    '''
    # names3.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


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


    def train_names(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> None:
        model.train()

        for step in range(steps):
            batch = torch.randint(0, len(X), (32,), generator=generator)
            loss = F.cross_entropy(model(X[batch]), Y[batch])
            loss.backward()
            rate = 0.1 if step < steps * 3 // 4 else 0.01

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= rate * parameter.grad
                    parameter.grad = None


    def validation_loss(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
    ) -> float:
        model.eval()

        with torch.no_grad():
            return F.cross_entropy(model(X), Y).item()


    generator = torch.Generator().manual_seed(0)
    drawn = torch.multinomial(torch.tensor([[0.1, 0.0, 0.9]]), 1, generator=generator)


    def sample_name(
        model: nn.Module,
        chars: list[str],
        context: int,
        generator: torch.Generator,
    ) -> str:
        model.eval()
        window = [0] * context
        name = ""

        with torch.no_grad():
            while True:
                probabilities = torch.softmax(model(torch.tensor([window])), dim=1)
                next_id = torch.multinomial(probabilities, 1, generator=generator).item()

                if next_id == 0:
                    return name

                name += chars[next_id]
                window = window[1:] + [next_id]
    ''',
)


def reference_train_names(
    model: nn.Module,
    X: Tensor,
    Y: Tensor,
    steps: int,
    generator: torch.Generator,
) -> None:
    model.train()

    for step in range(steps):
        batch = torch.randint(0, len(X), (32,), generator=generator)
        F.cross_entropy(model(X[batch]), Y[batch]).backward()
        rate = 0.1 if step < steps * 3 // 4 else 0.01

        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= rate * parameter.grad
                parameter.grad = None


def reference_validation_loss(
    model: nn.Module,
    X: Tensor,
    Y: Tensor,
) -> float:
    model.eval()

    with torch.no_grad():
        return F.cross_entropy(model(X), Y).item()


def names_model(seed: int) -> ReferenceNamesMLP:
    torch.manual_seed(seed)

    return ReferenceNamesMLP(27, 10, 3, 200)


@functools.cache
def context_sets() -> dict[str, tuple[Tensor, Tensor]]:
    splits = name_splits()

    return {name: reference_contexts(list(names), CHARACTERS, 3) for name, names in splits.items()}


def check_train_steps(found: object) -> None:
    X, Y = context_sets()["train"]
    validation_x, validation_y = context_sets()["validation"]
    mine = names_model(470)
    found(mine, X, Y, 400, torch.Generator().manual_seed(471))
    theirs = names_model(470)
    reference_train_names(theirs, X, Y, 400, torch.Generator().manual_seed(471))

    expect_close(
        call="the validation loss after train_names for 400 steps",
        actual=reference_validation_loss(mine, validation_x, validation_y),
        expected=reference_validation_loss(theirs, validation_x, validation_y),
        tolerance=1e-4,
    )


def check_beats_bigram(found: object) -> None:
    X, Y = context_sets()["train"]
    validation_x, validation_y = context_sets()["validation"]
    model = names_model(0)
    found(model, X, Y, 10000, torch.Generator().manual_seed(1))
    loss = reference_validation_loss(model, validation_x, validation_y)

    if loss >= BIGRAM_VALIDATION_LOSS:
        raise Mismatch(f"After 10,000 steps of train_names the validation loss is {loss:.4f}, which doesn't beat the bigram's {BIGRAM_VALIDATION_LOSS}.")


def trained_names_model(seed: int) -> ReferenceNamesMLP:
    model = names_model(seed)
    X, Y = context_sets()["train"]
    reference_train_names(model, X, Y, 300, torch.Generator().manual_seed(seed + 1))

    return model


def check_validation_loss(found: object) -> None:
    validation_x, validation_y = context_sets()["validation"]
    model = trained_names_model(480)
    model.train()
    expect_close(
        call="validation_loss on a names model trained for 300 steps and the validation contexts",
        actual=found(model, validation_x, validation_y),
        expected=reference_validation_loss(trained_names_model(480), validation_x, validation_y),
        tolerance=1e-5,
    )


def check_sample_name(found: object) -> None:
    model = trained_names_model(482)

    for seed in (483, 484, 485):
        expect_value(
            call=f"sample_name on a names model trained for 300 steps, with a generator seeded with {seed}",
            actual=found(model, CHARACTERS, 3, torch.Generator().manual_seed(seed)),
            expected=reference_sample(model, CHARACTERS, 3, torch.Generator().manual_seed(seed)),
        )


def reference_sample(
    model: nn.Module,
    chars: list[str],
    context: int,
    generator: torch.Generator,
) -> str:
    model.eval()
    window = [0] * context
    name = ""

    with torch.no_grad():
        while True:
            probabilities = torch.softmax(model(torch.tensor([window])), dim=1)
            next_id = torch.multinomial(probabilities, 1, generator=generator).item()

            if next_id == 0:
                return name

            name += chars[next_id]
            window = window[1:] + [next_id]


def check_names3(checker: Checker) -> None:
    checker.test(
        function="train_names",
        description="steps like the reference, 0.1 then 0.01",
        run=check_train_steps,
    )
    checker.test(
        function="train_names",
        description="trains below the bigram's validation loss of 2.465 in 10,000 steps",
        run=check_beats_bigram,
    )
    checker.test(
        function="validation_loss",
        description="gives the cross entropy in eval mode",
        run=check_validation_loss,
    )
    checker.test(
        function="sample_name",
        description="makes up a name one character at a time",
        run=check_sample_name,
    )


NAMES3 = Exercise(
    name="names3",
    chapter=NAMES,
    title="Training and making up names",
    lesson=NAMES3_LESSON,
    source=NAMES3_SOURCE,
    solution=NAMES3_SOLUTION,
    hints={
        "train_names": "The loss is F.cross_entropy of model(X[batch]) and Y[batch]. The rate is 0.1 while step is "
        "below steps * 3 // 4, and 0.01 after.",
        "validation_loss": "Call model.eval(), then inside with torch.no_grad(): return F.cross_entropy of model(X) "
        "and Y, as a plain number.",
        "sample_name": "Pass torch.tensor([window]) to the model and take torch.softmax of the scores along dim=1.",
    },
    introduces=(),
    check=check_names3,
    time_limit=120,
)
