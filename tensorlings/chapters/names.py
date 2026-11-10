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

        Section 3's project is a model that makes up names, like the first
        models in Karpathy's makemore. It starts with a model that only counts
        which letter follows which. Then it builds the MLP language model of
        Bengio and colleagues from this section's building blocks, and trains
        it until it beats the counting model. Finally it samples new names
        from the trained model. Download the data once, from the project
        folder.

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
    in Karpathy's makemore. It reads a name one character at a time and
    learns to predict the next character. Download the data once with
    `.venv/bin/python run.py data names`. It is 32,033 names in about 230 KB.

    ```python
    from tensorlings.data import load_names

    names = load_names()
    names[:3]    # ['emma', 'olivia', 'ava']
    ```

    Each character needs an id, a whole number that stands for it. The
    vocabulary is the 26 letters plus `.`, which marks where a name starts and
    ends. The `.` gets id 0. With a `.` added at both ends, each name gives
    one pair for each character and the character after it. `emma` gives
    `(., e)`, `(e, m)`, `(m, m)`, `(m, a)` and `(a, .)`.

    The simplest model only counts. Over the training names, count how often
    each character follows each other character. Put the counts in a table
    with one row for each first character. Divide a row by its total, and you
    get a probability distribution over the next character. That is a
    **bigram** model, which uses one character of context. Its loss is the
    cross entropy. That is the average, over all pairs, of minus the log of
    the probability the model gave the real next character.

    A pair that never appears in the training names gets probability 0, and
    minus the log of 0 is infinite. Some validation names contain such a pair.
    So 1 is added to every count before dividing, which is called smoothing.

    The names are shuffled and split into 80 percent for training, 10 percent
    for validation and 10 percent for testing. The smoothed bigram's loss is
    2.454 on the training names and 2.465 on the validation names. The later
    models have to get below 2.465.

    `sorted(set(text))` gives the distinct characters of a string, in sorted
    order. `torch.bincount` counts how often each whole number appears in a
    tensor. A pair of ids `x` and `y`, each below 27, can be turned into the
    single number `x * 27 + y`. Each pair gets its own number, so one call to
    `torch.bincount` counts every pair at once.
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


    # TODO: Return the vocabulary as a list, "." first and then every character
    #   that appears in names, in sorted order.
    #   Example: vocabulary(["emma", "ava"]) returns [".", "a", "e", "m", "v"].
    def vocabulary(names: list[str]) -> list[str]:
        ...


    steps = list(zip(".ab.", "ab."))                 # [('.', 'a'), ('a', 'b'), ('b', '.')]


    # TODO: Finish the loop. bigram_pairs adds "." to both ends of every name,
    #   then returns two tensors of ids. xs holds the id of each character, and
    #   ys holds the id of the character after it.
    #   Example: bigram_pairs(["ab"], [".", "a", "b"]) returns
    #   (tensor([0, 1, 2]), tensor([1, 2, 0])).
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


    # TODO: Return a float tensor of shape (size, size). The entry at row x,
    #   column y is the number of positions i where xs[i] is x and ys[i] is y.
    #   Example: bigram_counts(torch.tensor([0, 1, 1, 2]),
    #   torch.tensor([1, 2, 2, 0]), 3) returns
    #   tensor([[0., 1., 0.],
    #           [0., 0., 2.],
    #           [1., 0., 0.]]).
    def bigram_counts(
        xs: Tensor,
        ys: Tensor,
        size: int,
    ) -> Tensor:
        ...


    # TODO: Return the bigram model's cross entropy loss on the pairs xs and ys,
    #   as a float. Add 1 to every count in counts, then divide each row by its
    #   total to get probabilities. The loss is the mean, over the pairs, of
    #   minus the log of the probability of ys[i] in row xs[i].
    #   Example: bigram_loss(torch.tensor([[0.0, 3.0], [2.0, 1.0]]),
    #   torch.tensor([0, 1, 1]), torch.tensor([1, 0, 1])) returns about 0.5501.
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


def expect_float_counts(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    if isinstance(actual, Tensor) and not actual.is_floating_point():
        raise Mismatch(f"{call} holds whole numbers of type {actual.dtype}, but it should be a float tensor. Add .float() at the end.")

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


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
        description="gives the id pairs of every name, with . at each end",
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
        description="counts every pair into a size by size table",
        inputs=(
            (torch.tensor([0, 1, 1, 2]), torch.tensor([1, 2, 2, 0]), 3),
            (torch.tensor([3, 0, 3, 3, 1]), torch.tensor([3, 2, 3, 0, 1]), 4),
        ),
        reference=lambda xs, ys, size: torch.bincount(xs * size + ys, minlength=size * size).reshape(size, size).float(),
        expect=expect_float_counts,
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
        "bigram_counts": "Turn each pair into one number with xs * size + ys. Count those numbers with "
        "torch.bincount and minlength=size * size, then use .reshape(size, size) and .float().",
        "bigram_loss": "Add 1 to counts and divide each row by its sum, using keepdim=True. Pick "
        "probabilities[xs, ys], then take minus the mean of their logs.",
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
    A bigram model sees only one character of context. The language model of
    Bengio and colleagues, from 2003, reads the last few words. It looks up an
    embedding vector for each of those words and joins the vectors end to end
    into one long vector. Then an MLP gives a score to every possible next
    word. Karpathy applies the same model to the characters of names in
    makemore part 2.

    With a context of 3, `emma` gives five examples. Each example has three
    ids as input and the id of the next character as its target. Every name
    starts with the context `...`.

    | Context | Next |
    |---|---|
    | `...` | e |
    | `..e` | m |
    | `.em` | m |
    | `emm` | a |
    | `mma` | . |

    The model here starts with an embedding of size 10. Next comes a hidden
    Linear layer with 200 outputs, followed by LayerNorm and tanh. Last comes
    the head, a Linear layer that gives a score to each of the 27 characters.
    The paper used tanh too. The LayerNorm keeps the mean and standard
    deviation of the hidden values steady, as the normalization chapter
    showed. Karpathy adds BatchNorm in makemore part 3. LayerNorm does the
    same job without depending on the other examples in the batch. That helps
    when sampling one name at a time.

    The head starts small. Its weights are shrunk to a tenth of their random
    starting values, and its bias is set to 0. So at the start, the scores for
    all 27 characters are nearly equal. That makes the starting loss close to
    `log(27)`, which is 3.30, as init3 showed.

    `window[1:] + [new]` makes a new list without the first item of `window`
    and with `new` added at the end. `*=` multiplies a tensor in place.
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


    # TODO: Finish the line that updates window. build_contexts returns X, the
    #   context at every step of every name, and Y, the id of the character
    #   that follows each context. Each name starts from a context of all 0s,
    #   the id of ".", and its last target is ".".
    #   Example: build_contexts(["ab"], [".", "a", "b"], 2) returns
    #   (tensor([[0, 0], [0, 1], [1, 2]]), tensor([1, 2, 0])).
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


    # TODO: Finish two things. In __init__, multiply head's weights by 0.1 and
    #   set its bias to 0. forward takes ids x of shape (examples, context). It
    #   looks up the embedding of each id, joins each example's vectors end to
    #   end into one row, then applies hidden, norm, tanh and head in that
    #   order. It returns scores of shape (examples, vocabulary).
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
            raise Mismatch(f"The weights of head in {call} should be a tenth of the random values nn.Linear started them at, but they are {ratio:.3g} times those values.")


def check_names_forward(found: type) -> None:
    for vocabulary, size, context, hidden in ((27, 10, 3, 20), (5, 2, 4, 6)):
        call = f"NamesMLP({vocabulary}, {size}, {context}, {hidden})"
        torch.manual_seed(460 + context)
        model = expect_module(call, found(vocabulary, size, context, hidden))
        x = torch.randint(0, vocabulary, (4, context), generator=torch.Generator().manual_seed(462 + context))
        generator = torch.Generator().manual_seed(464 + context)

        # A fresh LayerNorm scales by 1 and shifts by 0, so forward could skip self.norm's own weights unnoticed.
        with torch.no_grad():
            model.norm.weight.copy_(torch.rand(hidden, generator=generator) + 0.5)
            model.norm.bias.copy_(torch.randn(hidden, generator=generator) * 0.5)

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
        description="looks up the embeddings, joins them end to end, and applies hidden, norm, tanh and head",
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
    Training uses the minibatch loop of Section 2. Each step picks 32 random
    examples. The learning rate is 0.1, then drops to 0.01 for the last
    quarter of the steps. Section 4 explains why lowering it helps. The
    examples come from `torch.randint(0, len(X), (32,))`, which gives 32
    random positions. The same position can come up more than once, unlike in
    the shuffled epochs of Section 2.

    | Model | Steps | Validation loss |
    |---|---|---|
    | bigram | none, it counts | 2.465 |
    | MLP | 5,000 | 2.258 |
    | MLP | 10,000 | 2.217 |
    | MLP | 20,000 | 2.181 |

    Karpathy's runs of a model with the same shape reach about 2.10 after
    200,000 steps.

    To make up a name, start from the context `...` and turn the model's
    scores into probabilities with softmax. Pick the next character at random
    using those probabilities. Add it to the end of the context and drop the
    oldest character. Repeat until the model picks `.`, which ends the name.

    ```python
    probabilities = torch.tensor([[0.1, 0.0, 0.9]])
    torch.multinomial(probabilities, 1, generator=generator)    # tensor([[2]]) nine times in ten
    ```

    `torch.multinomial(probabilities, 1)` picks one position from each row at
    random, and each position's chance of being picked is its probability.
    The model expects a batch, so a single context goes in as
    `torch.tensor([window])`, a batch of 1.

    The 20,000 step model made up names like kilyn, collie, caimen and amonia,
    and a few odd ones like fameellieshael. The bigram model made up names
    like zan, ayenn and deveerhacarolyme.
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


    # TODO: Finish the lines that set loss and rate. loss is the cross entropy
    #   of the model's scores for the 32 random examples in batch. rate is the
    #   learning rate, 0.1 for the first three quarters of the steps and 0.01
    #   after that.
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


    # TODO: Return the model's cross entropy loss on X against Y, as a float.
    #   Put the model in eval mode first, and run it inside torch.no_grad() so
    #   autograd records nothing.
    def validation_loss(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
    ) -> float:
        ...


    generator = torch.Generator().manual_seed(0)
    drawn = torch.multinomial(torch.tensor([[0.1, 0.0, 0.9]]), 1, generator=generator)    # tensor([[2]])


    # TODO: Finish the line that sets probabilities. It is the softmax of the
    #   model's scores for the context in window, a tensor of shape
    #   (1, len(chars)). With it, sample_name makes up one name.
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
    modes: list[tuple[bool, bool]] = []
    forward = model.forward

    def recording_forward(x: Tensor) -> Tensor:
        modes.append((model.training, torch.is_grad_enabled()))

        return forward(x)

    # This model has no dropout or BatchNorm, so its numbers can't show eval mode, and the check watches each call.
    model.forward = recording_forward
    expect_close(
        call="validation_loss on a names model trained for 300 steps and the validation contexts",
        actual=found(model, validation_x, validation_y),
        expected=reference_validation_loss(trained_names_model(480), validation_x, validation_y),
        tolerance=1e-5,
    )

    if any(training for training, _ in modes):
        raise Mismatch("validation_loss ran the model in training mode. Call model.eval() before the forward pass, so layers like dropout switch off.")

    if any(recording for _, recording in modes):
        raise Mismatch("validation_loss ran the model while autograd was recording. Put the forward pass inside a with torch.no_grad() block.")


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
        description="trains like the reference, at 0.1 then 0.01",
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
        "validation_loss": "Call model.eval(). Then, inside a with torch.no_grad() block, return F.cross_entropy "
        "of model(X) and Y as a float with .item().",
        "sample_name": "Pass torch.tensor([window]) to the model and take torch.softmax of the scores along dim=1.",
    },
    introduces=(),
    check=check_names3,
    time_limit=120,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_names(seed: int = 0) -> dict[str, object]:
    """names1 and names3's numbers: the smoothed bigram's training and validation loss, the names model's validation
    loss after 5,000, 10,000 and 20,000 steps of train_names, and 12 names from the 20,000 step model and 12 from the
    bigram, each drawn with a generator seeded with 2."""
    splits = name_splits()
    train_x, train_y = reference_pairs(list(splits["train"]), CHARACTERS)
    validation_x, validation_y = reference_pairs(list(splits["validation"]), CHARACTERS)
    counts = torch.bincount(train_x * 27 + train_y, minlength=27 * 27).reshape(27, 27).float()
    X, Y = context_sets()["train"]
    contexts_x, contexts_y = context_sets()["validation"]
    results: dict[str, object] = {
        "bigram train": reference_bigram_loss(counts, train_x, train_y),
        "bigram validation": reference_bigram_loss(counts, validation_x, validation_y),
    }

    # CPU results move in the third decimal with the thread count, so this runs on 4 like the other measurements.
    before = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        for steps in (5000, 10000, 20000):
            model = names_model(seed)
            reference_train_names(model, X, Y, steps, torch.Generator().manual_seed(seed + 1))
            results[f"mlp {steps}"] = reference_validation_loss(model, contexts_x, contexts_y)
    finally:
        torch.set_num_threads(before)

    generator = torch.Generator().manual_seed(2)
    results["mlp names"] = [reference_sample(model, CHARACTERS, 3, generator) for _ in range(12)]

    smoothed = counts + 1
    probabilities = smoothed / smoothed.sum(dim=1, keepdim=True)
    generator = torch.Generator().manual_seed(2)
    bigram_names = []

    for _ in range(12):
        current, made = 0, ""

        while True:
            current = torch.multinomial(probabilities[current], 1, generator=generator).item()

            if current == 0:
                break

            made += CHARACTERS[current]

        bigram_names.append(made)

    results["bigram names"] = bigram_names

    return results
