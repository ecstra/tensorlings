"""18_embeddings, a learned row of numbers per id, built and matched, and why ids can't go in as plain numbers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import Checker, Described, Mismatch, check_on_inputs, expect_close, expect_tensor, small_tensor, value_text
from ..text import prose, source_file
from .linear_layers import expect_module, expect_parameter

EMBEDDINGS = Chapter(
    folder="18_embeddings",
    title="Embeddings",
    readme=source_file(
        """
        # Embeddings

        Characters, words and product codes come in as ids, whole numbers that
        name things. An embedding gives each id a learned row of numbers. This
        chapter shows it is a Linear layer on a one hot input, builds it, shows
        why ids can't go in as plain numbers, and puts it at the front of a
        model with a Linear head at the back.

        ## Further reading

        - [PyTorch, nn.Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)
        - [Karpathy, building makemore part 2, the MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
        """,
    ),
)

TABLE = torch.tensor([[0.1, 0.2], [1.0, -1.0], [0.5, 0.5]])
CATEGORY_TARGETS = torch.tensor([3.0, -1.0, 2.0, 0.5, -2.0, 1.0, 4.0, -0.5, 2.5, 0.0])


def id_tensor(
    seed: int,
    count: int,
    vocabulary: int,
    *shape: int,
) -> Tensor:
    return torch.randint(0, vocabulary, (count, *shape), generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# embedding1
# ---------------------------------------------------------------------------

EMBEDDING1_LESSON = prose(
    """
    Many inputs are picks from a fixed set, like characters, words or product
    codes. Each gets an id, a whole number from 0 up. The set of every
    possible id is the vocabulary.

    The one hot rows of likelihood4 can carry an id into a model. A one hot
    row times a matrix picks out one row of the matrix, the row at the id.

    ```python
    table = torch.tensor([[0.1, 0.2],
                          [1.0, -1.0],
                          [0.5, 0.5]])         # 3 ids, 2 numbers each
    ids = torch.tensor([2, 0, 2])
    F.one_hot(ids, 3).float() @ table     # tensor([[0.5, 0.5], [0.1, 0.2], [0.5, 0.5]])
    table[ids]                            # the same rows, picked directly
    ```

    So a Linear layer on one hot inputs is a table lookup, and building the
    one hot rows is wasted work. An **embedding** is the lookup done directly.
    Its table has shape `(vocabulary size, size)`, and its numbers are
    parameters, learned like any weight. A flat tensor of numbers that stands
    for something is a **vector**, so each id's row is its embedding vector.

    Indexing with a tensor of ids of any shape gives one row per id, in the
    same arrangement. Ids of shape `(examples, positions)` give a result of
    shape `(examples, positions, size)`. `.reshape(len(x), -1)` then lays each
    example's rows end to end, one long row per example, which a Linear layer
    can take.
    """,
)

EMBEDDING1_SOURCE = source_file(
    '''
    # embedding1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    table = torch.tensor([[0.1, 0.2],
                          [1.0, -1.0],
                          [0.5, 0.5]])                 # 3 ids, 2 numbers each
    ids = torch.tensor([2, 0, 2])

    # Indexing the table with ids picks one row per id.
    rows = table[ids]                                 # tensor([[0.5, 0.5], [0.1, 0.2], [0.5, 0.5]])


    # TODO: Return the same rows the slow way, a one hot row per id times the
    #   table. It should give what table[ids] gives.
    def one_hot_lookup(
        table: Tensor,
        ids: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the rows of table for ids of shape (examples, positions), so
    #   the result has shape (examples, positions, size).
    def embed_sequences(
        table: Tensor,
        ids: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish the last line, so each example's rows are laid end to end, one
    #   row of positions * size numbers per example.
    def flat_context(
        table: Tensor,
        ids: Tensor,
    ) -> Tensor:
        vectors = embed_sequences(table, ids)

        return ...
    ''',
)

EMBEDDING1_SOLUTION = source_file(
    '''
    # embedding1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    table = torch.tensor([[0.1, 0.2],
                          [1.0, -1.0],
                          [0.5, 0.5]])
    ids = torch.tensor([2, 0, 2])

    rows = table[ids]


    def one_hot_lookup(
        table: Tensor,
        ids: Tensor,
    ) -> Tensor:
        return F.one_hot(ids, table.shape[0]).float() @ table


    # Indexing with ids of any shape gives one row per id, in the same arrangement.
    def embed_sequences(
        table: Tensor,
        ids: Tensor,
    ) -> Tensor:
        return table[ids]


    def flat_context(
        table: Tensor,
        ids: Tensor,
    ) -> Tensor:
        vectors = embed_sequences(table, ids)

        return vectors.reshape(len(vectors), -1)
    ''',
)


def check_embedding1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="one_hot_lookup",
        description="picks the rows with one hot rows times the table",
        inputs=(
            (TABLE, torch.tensor([2, 0, 2])),
            (small_tensor(222, 5, 3), torch.tensor([4, 1, 1, 0])),
        ),
        reference=lambda table, ids: table[ids],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="embed_sequences",
        description="gives one row per id, shaped (examples, positions, size)",
        inputs=(
            (small_tensor(223, 6, 2), id_tensor(224, 3, 6, 4)),
            (small_tensor(225, 4, 3), id_tensor(226, 2, 4, 2)),
        ),
        reference=lambda table, ids: table[ids],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="flat_context",
        description="lays each example's rows end to end",
        inputs=(
            (small_tensor(227, 6, 2), id_tensor(228, 3, 6, 4)),
            (small_tensor(229, 4, 3), id_tensor(230, 2, 4, 2)),
        ),
        reference=lambda table, ids: table[ids].reshape(len(ids), -1),
        expect=expect_tensor,
    )


EMBEDDING1 = Exercise(
    name="embedding1",
    chapter=EMBEDDINGS,
    title="A lookup table",
    lesson=EMBEDDING1_LESSON,
    source=EMBEDDING1_SOURCE,
    solution=EMBEDDING1_SOLUTION,
    hints={
        "one_hot_lookup": "F.one_hot(ids, number of rows of table) gives the one hot rows. Turn them into floats and "
        "multiply by the table with @.",
        "embed_sequences": "Index the table with ids, as the line that sets rows does. It works for ids of any shape.",
        "flat_context": "Reshape vectors to len(vectors) rows, with -1 for the rest.",
    },
    introduces=("embedding", "vector"),
    check=check_embedding1,
)


# ---------------------------------------------------------------------------
# embedding2
# ---------------------------------------------------------------------------

EMBEDDING2_LESSON = prose(
    """
    PyTorch's `nn.Embedding(count, size)` holds a table of shape
    `(count, size)` as its `weight`, drawn from the bell curve with spread 1.
    Calling it with ids looks up their rows. Building one is a module with one
    parameter and a `forward` that indexes it.

    `torch.randn(3, 2)` draws a tensor of shape `(3, 2)` from the bell curve
    with mean 0 and spread 1.

    The backward of a lookup is worth knowing. A row that no id picked took no
    part in the output, so its gradient is 0, and a step leaves it alone. A row
    picked twice gets the gradient of both uses added up, as chain2 said. So a
    rare id learns slowly, since few examples ever send it a gradient.

    ```python
    table = nn.Embedding(4, 2)
    table(torch.tensor([1, 1, 3])).sum().backward()
    table.weight.grad    # rows 0 and 2 all zero, row 1 all 2, row 3 all 1
    ```

    `(grad != 0).any(dim=1)` says which rows have any number that isn't 0, one
    `True` or `False` per row. `.nonzero()` on a flat tensor of `True` and
    `False` gives the positions of the `True` ones, as a column, and
    `.flatten()` makes that column flat.
    """,
)

EMBEDDING2_SOURCE = source_file(
    '''
    # embedding2.py

    import torch
    import torch.nn as nn
    from torch import Tensor


    # TODO: Finish the line that sets self.weight and the forward method, so
    #   MyEmbedding works like nn.Embedding. The table starts drawn from the bell
    #   curve, one row of size numbers per id.
    class MyEmbedding(nn.Module):
        def __init__(
            self,
            count: int,
            size: int,
        ) -> None:
            super().__init__()
            self.weight = ...

        def forward(self, ids: Tensor) -> Tensor:
            ...


    # TODO: Return the positions of the rows of the table's gradient that got any
    #   number other than 0, as a flat tensor, the ids that were looked up.
    def used_rows(grad: Tensor) -> Tensor:
        ...
    ''',
)

EMBEDDING2_SOLUTION = source_file(
    '''
    # embedding2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor


    class MyEmbedding(nn.Module):
        def __init__(
            self,
            count: int,
            size: int,
        ) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(count, size))

        def forward(self, ids: Tensor) -> Tensor:
            return self.weight[ids]


    def used_rows(grad: Tensor) -> Tensor:
        return (grad != 0).any(dim=1).nonzero().flatten()
    ''',
)


def check_my_embedding(found: type) -> None:
    for count, size in ((5, 3), (10, 2)):
        call = f"MyEmbedding({count}, {size})"
        module = expect_module(call, found(count, size))
        weight = expect_parameter(module, call, "weight", (count, size))

        if weight.detach().std() < 0.3:
            raise Mismatch(f"The weight of {call} should start drawn from the bell curve with torch.randn, but its numbers are near 0.")

        ids = id_tensor(231 + count, 3, count, 2)
        expect_tensor(
            call=f"{call} on ids {value_text(ids)}",
            actual=module(ids),
            expected=weight.detach()[ids],
        )


def check_embedding2(checker: Checker) -> None:
    checker.test(
        function="MyEmbedding",
        description="holds a random table and looks up a row per id",
        run=check_my_embedding,
    )
    check_on_inputs(
        checker=checker,
        function="used_rows",
        description="gives the positions of the rows with any gradient",
        inputs=(
            (torch.tensor([[0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]),),
            (torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),),
        ),
        reference=lambda grad: (grad != 0).any(dim=1).nonzero().flatten(),
        expect=expect_tensor,
    )


EMBEDDING2 = Exercise(
    name="embedding2",
    chapter=EMBEDDINGS,
    title="Building Embedding",
    lesson=EMBEDDING2_LESSON,
    source=EMBEDDING2_SOURCE,
    solution=EMBEDDING2_SOLUTION,
    hints={
        "MyEmbedding": "The weight is an nn.Parameter of torch.randn(count, size). forward indexes self.weight with "
        "ids.",
        "used_rows": "Ask which rows have any entry other than 0. Then turn the True ones into a flat tensor of "
        "positions, as the lesson's last paragraph shows.",
    },
    introduces=(),
    check=check_embedding2,
)


# ---------------------------------------------------------------------------
# embedding3
# ---------------------------------------------------------------------------

EMBEDDING3_LESSON = prose(
    """
    Why not feed an id in as a plain number? Because a Linear layer then
    treats it as an amount. To it, id 9 is nine times id 1, and id 5 sits
    halfway between 1 and 9. Ids that name things carry no such order, and the
    model wastes itself fitting one that isn't there.

    Here 10 categories, ids 0 to 9, each have a target with no order to it.

    ```python
    targets = torch.tensor([3.0, -1.0, 2.0, 0.5, -2.0, 1.0, 4.0, -0.5, 2.5, 0.0])
    ```

    A Linear layer on the raw id can only draw a straight line through the
    ten. After 2000 steps its mean squared error is 3.27, as bad as always
    answering the mean.

    An embedding of size 4 followed by a Linear layer gives every id its own
    learned row, so each can land on its own target. After 500 steps the error
    is about 0.

    The rule is general. A discrete input, one picked from a set, enters a
    model through an embedding. Numbers that measure an amount, like a price
    or a length, enter as they are.
    """,
)

EMBEDDING3_SOURCE = source_file(
    '''
    # embedding3.py

    import torch
    import torch.nn as nn
    from torch import Tensor


    # The training loop of mlp3, giving the final loss.
    def train(
        model: nn.Module,
        inputs: Tensor,
        targets: Tensor,
        steps: int,
        learning_rate: float,
    ) -> float:
        for _ in range(steps):
            loss = ((model(inputs).squeeze(1) - targets) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= learning_rate * parameter.grad
                    parameter.grad = None

        return loss.item()


    # TODO: Finish the line that sets inputs, the ids as a column of floats of
    #   shape (count, 1), so this trains one Linear layer on the raw ids.
    def fit_raw_ids(
        ids: Tensor,
        targets: Tensor,
        steps: int,
    ) -> float:
        torch.manual_seed(0)
        model = nn.Linear(1, 1)
        inputs = ...

        return train(model, inputs, targets, steps, 0.01)


    # TODO: Finish the line that sets model, an nn.Embedding with a row of size 4
    #   for each of the count ids, then an nn.Linear from 4 to 1, chained by
    #   nn.Sequential.
    def fit_embedded_ids(
        ids: Tensor,
        targets: Tensor,
        count: int,
        steps: int,
    ) -> float:
        torch.manual_seed(0)
        model = ...

        return train(model, ids, targets, steps, 0.1)
    ''',
)

EMBEDDING3_SOLUTION = source_file(
    '''
    # embedding3.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor


    def train(
        model: nn.Module,
        inputs: Tensor,
        targets: Tensor,
        steps: int,
        learning_rate: float,
    ) -> float:
        for _ in range(steps):
            loss = ((model(inputs).squeeze(1) - targets) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= learning_rate * parameter.grad
                    parameter.grad = None

        return loss.item()


    def fit_raw_ids(
        ids: Tensor,
        targets: Tensor,
        steps: int,
    ) -> float:
        torch.manual_seed(0)
        model = nn.Linear(1, 1)
        inputs = ids.float().unsqueeze(1)

        return train(model, inputs, targets, steps, 0.01)


    def fit_embedded_ids(
        ids: Tensor,
        targets: Tensor,
        count: int,
        steps: int,
    ) -> float:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Embedding(count, 4), nn.Linear(4, 1))

        return train(model, ids, targets, steps, 0.1)
    ''',
)


def train_categories(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    steps: int,
    learning_rate: float,
) -> float:
    for _ in range(steps):
        loss = ((model(inputs).squeeze(1) - targets) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= learning_rate * parameter.grad
                parameter.grad = None

    return loss.item()


def reference_raw(
    ids: Tensor,
    targets: Tensor,
    steps: int,
) -> float:
    torch.manual_seed(0)

    return train_categories(nn.Linear(1, 1), ids.float().unsqueeze(1), targets, steps, 0.01)


def reference_embedded(
    ids: Tensor,
    targets: Tensor,
    count: int,
    steps: int,
) -> float:
    torch.manual_seed(0)

    return train_categories(nn.Sequential(nn.Embedding(count, 4), nn.Linear(4, 1)), ids, targets, steps, 0.1)


CATEGORIES = (Described(torch.arange(10), "ids 0 to 9"), Described(CATEGORY_TARGETS, "their targets"))
OTHER_CATEGORIES = (Described(torch.tensor([0, 1, 2, 3, 4, 5]), "ids 0 to 5"), Described(torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]), "targets that rise with the id"))


def check_embedding3(checker: Checker) -> None:
    close = lambda call, actual, expected: expect_close(
        call=call,
        actual=actual,
        expected=expected,
        tolerance=1e-4,
    )
    check_on_inputs(
        checker=checker,
        function="fit_raw_ids",
        description="trains one Linear layer on the ids as numbers",
        inputs=(
            (*CATEGORIES, 2000),
            (*OTHER_CATEGORIES, 500),
        ),
        reference=reference_raw,
        expect=close,
    )
    check_on_inputs(
        checker=checker,
        function="fit_embedded_ids",
        description="trains an embedding and a Linear layer on the ids",
        inputs=(
            (*CATEGORIES, 10, 500),
            (*OTHER_CATEGORIES, 6, 200),
        ),
        reference=reference_embedded,
        expect=close,
    )


EMBEDDING3 = Exercise(
    name="embedding3",
    chapter=EMBEDDINGS,
    title="Why an embedding",
    lesson=EMBEDDING3_LESSON,
    source=EMBEDDING3_SOURCE,
    solution=EMBEDDING3_SOLUTION,
    hints={
        "fit_raw_ids": "Turn the ids into floats with .float(), then give them a column shape with .unsqueeze(1).",
        "fit_embedded_ids": "nn.Sequential takes the modules in order, the embedding first, then the Linear layer "
        "that turns each row of 4 into one number.",
    },
    introduces=(),
    check=check_embedding3,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# embedding4
# ---------------------------------------------------------------------------

EMBEDDING4_LESSON = prose(
    """
    A model that takes ids in and picks a class out has a usual shape. An
    embedding at the front turns each id into a row of numbers. A Linear layer
    at the back, the head, turns that row into one score per class, the
    logits. The loss is cross entropy from the logits, as likelihood2 showed.

    ```python
    class IdClassifier(nn.Module):
        def __init__(self, count, size, classes):
            super().__init__()
            self.embedding = nn.Embedding(count, size)
            self.head = nn.Linear(size, classes)
    ```

    Anything can sit between the two, and later models put most of their
    layers there. Language models are this shape, with ids for pieces of text
    going in and scores for the next piece coming out.

    Training it is the loop of mlp3 with cross entropy in place of the squared
    error. `F.cross_entropy(logits, labels)` takes logits of shape
    `(examples, classes)` and labels of shape `(examples,)`.

    Here 12 ids each belong to one of 3 classes, with no pattern a raw number
    could follow. An embedding of size 4 and a Linear head learn every id's
    class within 300 steps.
    """,
)

EMBEDDING4_SOURCE = source_file(
    '''
    # embedding4.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Finish forward, so it looks each id up in the embedding, then passes
    #   the rows through the head, giving one row of class scores per id.
    class IdClassifier(nn.Module):
        def __init__(
            self,
            count: int,
            size: int,
            classes: int,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(count, size)
            self.head = nn.Linear(size, classes)

        def forward(self, ids: Tensor) -> Tensor:
            ...


    # TODO: Finish the line that sets loss, the cross entropy of the model's
    #   scores for ids against labels, so this trains the model and returns the
    #   final loss.
    def train_classifier(
        model: nn.Module,
        ids: Tensor,
        labels: Tensor,
        steps: int,
        learning_rate: float,
    ) -> float:
        for _ in range(steps):
            loss = ...
            loss.backward()

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= learning_rate * parameter.grad
                    parameter.grad = None

        return loss.item()
    ''',
)

EMBEDDING4_SOLUTION = source_file(
    '''
    # embedding4.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    class IdClassifier(nn.Module):
        def __init__(
            self,
            count: int,
            size: int,
            classes: int,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(count, size)
            self.head = nn.Linear(size, classes)

        def forward(self, ids: Tensor) -> Tensor:
            return self.head(self.embedding(ids))


    def train_classifier(
        model: nn.Module,
        ids: Tensor,
        labels: Tensor,
        steps: int,
        learning_rate: float,
    ) -> float:
        for _ in range(steps):
            loss = F.cross_entropy(model(ids), labels)
            loss.backward()

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= learning_rate * parameter.grad
                    parameter.grad = None

        return loss.item()
    ''',
)

ID_LABELS = torch.tensor([2, 0, 1, 1, 0, 2, 0, 2, 1, 0, 1, 2])


class ReferenceClassifier(nn.Module):
    def __init__(
        self,
        count: int,
        size: int,
        classes: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(count, size)
        self.head = nn.Linear(size, classes)

    def forward(self, ids: Tensor) -> Tensor:
        return self.head(self.embedding(ids))


def check_id_classifier(found: type) -> None:
    torch.manual_seed(0)
    call = "IdClassifier(12, 4, 3)"
    model = expect_module(call, found(12, 4, 3))
    ids = torch.tensor([0, 5, 11, 5])
    expect_tensor(
        call=f"{call} on ids {value_text(ids)}",
        actual=model(ids),
        expected=model.head(model.embedding(ids)).detach(),
    )


def check_train_classifier(found: object) -> None:
    torch.manual_seed(0)
    model = ReferenceClassifier(12, 4, 3)
    final = found(model, torch.arange(12), ID_LABELS.clone(), 300, 0.5)
    torch.manual_seed(0)
    reference = ReferenceClassifier(12, 4, 3)

    for _ in range(300):
        loss = F.cross_entropy(reference(torch.arange(12)), ID_LABELS)
        loss.backward()

        with torch.no_grad():
            for parameter in reference.parameters():
                parameter -= 0.5 * parameter.grad
                parameter.grad = None

    expect_close(
        call="train_classifier on 12 ids in 3 classes, for 300 steps with learning rate 0.5",
        actual=final,
        expected=loss.item(),
        tolerance=1e-5,
    )


def check_embedding4(checker: Checker) -> None:
    checker.test(
        function="IdClassifier",
        description="gives class scores from the embedding and the head",
        run=check_id_classifier,
    )
    checker.test(
        function="train_classifier",
        description="trains on the cross entropy of the scores",
        run=check_train_classifier,
    )


EMBEDDING4 = Exercise(
    name="embedding4",
    chapter=EMBEDDINGS,
    title="Ids in, classes out",
    lesson=EMBEDDING4_LESSON,
    source=EMBEDDING4_SOURCE,
    solution=EMBEDDING4_SOLUTION,
    hints={
        "IdClassifier": "Call self.embedding on ids, then self.head on what it gives.",
        "train_classifier": "The model's scores are model(ids). Pass them and labels to F.cross_entropy.",
    },
    introduces=(),
    check=check_embedding4,
    time_limit=30,
)
