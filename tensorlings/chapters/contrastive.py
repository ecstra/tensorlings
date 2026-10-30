"""61_contrastive, two views of one image pulled together and every other image pushed apart, InfoNCE and NT-Xent, the temperature, and the collapse when nothing pushes apart."""

from __future__ import annotations

import functools
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    seeded,
)
from ..text import prose, source_file
from .classic_nets import mnist_images

CONTRASTIVE = Chapter(
    folder="61_contrastive",
    title="Contrastive learning",
    readme=source_file(
        """
        # Contrastive learning

        Two changed views of one image should get features close together,
        and views of different images features far apart. That one rule,
        with no labels at all, teaches a network what to ignore. This chapter
        builds the InfoNCE loss and SimCLR's version of it, trains an encoder
        on MNIST with it, and shows what goes wrong when nothing pushes apart.

        ## Further reading

        - [van den Oord, Li and Vinyals, contrastive predictive coding](https://arxiv.org/abs/1807.03748)
        - [Chen and colleagues, SimCLR](https://arxiv.org/abs/2002.05709)
        - [Wang and Isola, alignment and uniformity on the hypersphere](https://arxiv.org/abs/2005.10242)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


SIMILARITY_PART = '''
def similarity_matrix(
    z1: Tensor,
    z2: Tensor,
) -> Tensor:
    return F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T


def info_nce(
    z1: Tensor,
    z2: Tensor,
    temperature: float,
) -> Tensor:
    scores = similarity_matrix(z1, z2) / temperature

    return F.cross_entropy(scores, torch.arange(len(z1)))
'''

NT_XENT_PART = '''
def nt_xent(
    z1: Tensor,
    z2: Tensor,
    temperature: float,
) -> Tensor:
    both = torch.cat([z1, z2])
    scores = similarity_matrix(both, both) / temperature
    scores.fill_diagonal_(float("-inf"))
    count = len(z1)
    partners = torch.cat([torch.arange(count, 2 * count), torch.arange(count)])

    return F.cross_entropy(scores, partners)
'''

SHIFT_PART = '''
def random_shifts(
    images: Tensor,
    pad: int,
    generator: torch.Generator,
) -> Tensor:
    tops = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator).tolist()
    lefts = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator).tolist()
    h, w = images.shape[2], images.shape[3]
    padded = F.pad(images, (pad, pad, pad, pad))

    return torch.stack([padded[i, :, top : top + h, left : left + w] for i, (top, left) in enumerate(zip(tops, lefts))])
'''

ENCODER_PART = '''
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.body(images)
'''

TWO_VIEWS_PART = '''
def two_views(
    images: Tensor,
    generator: torch.Generator,
    noise: float,
) -> tuple[Tensor, Tensor]:
    first = random_shifts(images, 4, generator) + noise * torch.randn(images.shape, generator=generator)
    second = random_shifts(images, 4, generator) + noise * torch.randn(images.shape, generator=generator)

    return first, second
'''

TRAIN_PART = '''
def train_contrastive(
    model: nn.Module,
    images: Tensor,
    steps: int,
    temperature: float,
    generator: torch.Generator,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(steps):
        batch = images[torch.randint(0, len(images), (256,), generator=generator)]
        first, second = two_views(batch, generator, 0.3)
        loss = info_nce(model(first), model(second), temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
'''

KNN_PART = '''
def knn_accuracy(
    bank: Tensor,
    bank_labels: Tensor,
    queries: Tensor,
    query_labels: Tensor,
    k: int,
) -> float:
    nearest = similarity_matrix(queries, bank).topk(k, dim=1).indices
    votes = torch.mode(bank_labels[nearest], dim=1).values

    return (votes == query_labels).float().mean().item()


def feature_spread(features: Tensor) -> float:
    return F.normalize(features, dim=1).std(dim=0).mean().item()


def alignment_loss(
    z1: Tensor,
    z2: Tensor,
) -> Tensor:
    return (F.normalize(z1, dim=1) - F.normalize(z2, dim=1)).pow(2).sum(dim=1).mean()
'''


def reference_namespace() -> dict[str, object]:
    """Every solution piece of the chapter run once, so the checks compare against exactly the code the files hold."""
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(SIMILARITY_PART + NT_XENT_PART + SHIFT_PART + ENCODER_PART + TWO_VIEWS_PART + TRAIN_PART + KNN_PART, namespace)

    return namespace


REFERENCE = reference_namespace()
reference_similarity_matrix = REFERENCE["similarity_matrix"]
reference_info_nce = REFERENCE["info_nce"]
reference_nt_xent = REFERENCE["nt_xent"]


@functools.cache
def contrastive_sets() -> dict[str, Tensor]:
    """The MNIST images the lessons measure on, 20,000 to train on without labels, 5,000 with labels to look
    neighbors up in, and 2,000 validation images, centered and moved by up to 4 pixels."""
    sets = mnist_images()
    test = sets["validation_x"][:2000]

    return {
        "train_x": sets["train_x"][:20000],
        "bank_x": sets["train_x"][20000:25000],
        "bank_y": sets["train_y"][20000:25000],
        "test_x": test,
        "moved_x": REFERENCE["random_shifts"](test, 4, torch.Generator().manual_seed(5)),
        "test_y": sets["validation_y"][:2000],
    }


def seeded_encoder() -> nn.Module:
    torch.manual_seed(0)

    return REFERENCE["Encoder"]()


def neighbor_scores(model: nn.Module | None) -> tuple[float, float]:
    """kNN accuracy on centered and on moved validation digits, of model's features or of raw pixels when None."""
    sets = contrastive_sets()
    knn = REFERENCE["knn_accuracy"]

    def features(images: Tensor) -> Tensor:
        if model is None:
            return images.flatten(1)

        with torch.no_grad():
            return model(images)

    bank = features(sets["bank_x"])

    return (
        knn(bank, sets["bank_y"], features(sets["test_x"]), sets["test_y"], 10),
        knn(bank, sets["bank_y"], features(sets["moved_x"]), sets["test_y"], 10),
    )


# ---------------------------------------------------------------------------
# contrastive1
# ---------------------------------------------------------------------------

CONTRASTIVE1_LESSON = prose(
    """
    word2vec learned from text by telling real pairs from made up ones.
    Contrastive learning does the same with images. Change one image two ways,
    a shift here and some noise there, and the two views are a real pair. Any
    view of another image in the batch makes a made up pair.

    An encoder turns each view into features. For a batch of N images, `z1`
    holds the features of the first views and `z2` of the second, each of
    shape `(N, size)`. The **similarity matrix** holds the cosine of every
    first view with every second view, so row i column j compares image i
    with image j.

    The real pairs sit on the diagonal. So each row is a classification with
    N choices, whose right answer is its own column. That makes the loss a
    cross entropy over each row with target i, which is **InfoNCE**, named by
    van den Oord and colleagues in 2018.

    ```python
    scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    F.cross_entropy(scores, torch.arange(2))    # tensor(0.1269), both rows pick their own column
    ```

    A cosine lies between -1 and 1, too narrow a range for a sharp softmax.
    So the cosines are divided by a temperature, as in soft1, before the
    cross entropy. contrastive2 measures what it does.

    Features that know nothing give every column the same chance, a loss of
    `log(N)`, 5.55 for a batch of 256.

    SimCLR, from Chen and colleagues, uses every view in the batch. Each of
    the 2N views has its partner as the right answer. The other 2N - 2 are
    made up pairs, and a view never counts against itself. The paper calls
    this NT-Xent.
    """,
)

CONTRASTIVE1_SOURCE = source_file(
    '''
    # contrastive1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    loss = F.cross_entropy(scores, torch.arange(2))      # tensor(0.1269)
    square = torch.ones(3, 3)
    square.fill_diagonal_(0.0)                           # zeros on the diagonal, in place


    # TODO: Return the cosine similarity of every row of z1 with every row of z2,
    #   of shape (N, N), row i of z1 down the side and row j of z2 across.
    def similarity_matrix(
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return InfoNCE, the cross entropy of each row of similarity_matrix
    #   divided by temperature, where row i's right answer is column i.
    def info_nce(
        z1: Tensor,
        z2: Tensor,
        temperature: float,
    ) -> Tensor:
        ...


    # TODO: Return SimCLR's NT-Xent. Stack z1 above z2 into 2N rows, take the
    #   similarity of every row with every row divided by temperature, and set
    #   the diagonal to minus infinity. Row i's right answer is i + N for the
    #   first N rows and i - N for the rest.
    def nt_xent(
        z1: Tensor,
        z2: Tensor,
        temperature: float,
    ) -> Tensor:
        ...
    ''',
)

CONTRASTIVE1_SOLUTION = source_file(
    '''
    # contrastive1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    loss = F.cross_entropy(scores, torch.arange(2))
    square = torch.ones(3, 3)
    square.fill_diagonal_(0.0)


''' + indented(SIMILARITY_PART) + '''


''' + indented(NT_XENT_PART) + '''
    ''',
)


def feature_pair(
    seed: int,
    count: int,
    size: int,
) -> tuple[Tensor, Tensor]:
    """Two views' features that roughly agree, image by image, as a trained encoder's would."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(count, size, generator=generator)

    return base + 0.5 * torch.randn(count, size, generator=generator), 2 * base + 0.5 * torch.randn(count, size, generator=generator)


def check_contrastive1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="similarity_matrix",
        description="gives the cosine of every pair of rows",
        inputs=(
            (torch.tensor([[3.0, 4.0], [1.0, 0.0]]), torch.tensor([[6.0, 8.0], [0.0, -2.0]])),
            feature_pair(620, 4, 3),
        ),
        reference=reference_similarity_matrix,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="info_nce",
        description="takes each row's own column as its right answer",
        inputs=(
            (*feature_pair(621, 5, 4), 0.1),
            (*feature_pair(622, 3, 6), 0.5),
        ),
        reference=reference_info_nce,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="nt_xent",
        description="pairs each view with its partner among all 2N",
        inputs=(
            (*feature_pair(623, 4, 3), 0.2),
            (*feature_pair(624, 3, 5), 1.0),
        ),
        reference=reference_nt_xent,
        expect=expect_tensor,
    )


CONTRASTIVE1 = Exercise(
    name="contrastive1",
    chapter=CONTRASTIVE,
    title="InfoNCE, a classification over the batch",
    lesson=CONTRASTIVE1_LESSON,
    source=CONTRASTIVE1_SOURCE,
    solution=CONTRASTIVE1_SOLUTION,
    hints={
        "similarity_matrix": "Scale every row of both to length 1 with F.normalize and dim=1, then multiply the first "
        "by the transpose of the second.",
        "info_nce": "Divide similarity_matrix(z1, z2) by the temperature. The targets are torch.arange(len(z1)).",
        "nt_xent": "torch.cat([z1, z2]) stacks the rows. After .fill_diagonal_(float(\"-inf\")) on the scores, the "
        "targets are torch.arange(N, 2 * N) followed by torch.arange(N).",
    },
    introduces=(),
    check=check_contrastive1,
)


# ---------------------------------------------------------------------------
# contrastive2
# ---------------------------------------------------------------------------

CONTRASTIVE2_LESSON = prose(
    """
    The views decide what the features learn to ignore. Here each view of an
    MNIST digit is shifted by up to 4 pixels, as in augment1, with some noise
    added. Features that pair two views of one digit have to ignore where the
    digit sits and the noise on it. Nothing else tells them what matters.

    The encoder is an MLP from 784 pixels to 64 features, trained with
    contrastive1's InfoNCE on 20,000 digits and no labels. To see what it
    learned, each validation digit gets a kNN guess. It takes the label most
    common among its 10 nearest training digits, by cosine of their features. The digits
    are tested centered, and moved by up to 4 pixels. Here are 500 steps of
    256 digits at four temperatures.

    | Temperature | Centered digits right | Moved digits right |
    |---|---|---|
    | 0.05 | 86.3% | 85.7% |
    | 0.1 | 87.1% | 85.9% |
    | 0.5 | 78.1% | 76.7% |
    | 1 | 73.8% | 72.8% |

    A low temperature sharpens the softmax, so most of the loss comes from the
    made up pairs that look most like the real one. A high one spreads the
    loss over every made up pair, most of them easy. SimCLR found the same on
    ImageNet. With rows scaled to length 1, a temperature of 0.1 gave 64.4%,
    against 59.7% at 0.05, 60.7% at 0.5 and 58.0% at 1.

    `torch.randn` takes a generator as `randint` does, so each view's noise is
    drawn the same way on every run.
    """,
)

CONTRASTIVE2_SOURCE = source_file(
    '''
    # contrastive2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    noise = torch.randn(2, 3, generator=torch.Generator().manual_seed(0))    # the same numbers every run


    # similarity_matrix and info_nce from contrastive1.
''' + indented(SIMILARITY_PART) + '''


    # augment1's random shifts, for images of shape (batch, channels, height, width).
''' + indented(SHIFT_PART) + '''


    # The encoder, an MLP from 784 pixels to 64 features.
''' + indented(ENCODER_PART) + '''


    # TODO: Return two views of images. Each is random_shifts of images by up to
    #   4 pixels, plus noise times torch.randn of their shape, drawn with
    #   generator. Draw the whole first view, then the whole second.
    def two_views(
        images: Tensor,
        generator: torch.Generator,
        noise: float,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Finish the loop. Each step draws a batch of 256 images with
    #   torch.randint and generator, makes two_views of it with a noise of 0.3,
    #   and takes one step of optimizer on info_nce of model's features of the
    #   two. Return the last loss as a plain number.
    def train_contrastive(
        model: nn.Module,
        images: Tensor,
        steps: int,
        temperature: float,
        generator: torch.Generator,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(steps):
            ...

        return loss.item()
    ''',
)

CONTRASTIVE2_SOLUTION = source_file(
    '''
    # contrastive2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    noise = torch.randn(2, 3, generator=torch.Generator().manual_seed(0))


''' + indented(SIMILARITY_PART) + '''


''' + indented(SHIFT_PART) + '''


''' + indented(ENCODER_PART) + '''


''' + indented(TWO_VIEWS_PART) + '''


''' + indented(TRAIN_PART) + '''
    ''',
)


def small_images(
    seed: int,
    count: int,
) -> Tensor:
    return torch.rand(count, 1, 28, 28, generator=torch.Generator().manual_seed(seed))


def expect_two_views(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should give the two views as two tensors in a tuple.")

    for name, part, wanted in zip(("first", "second"), actual, expected):
        expect_tensor(
            call=f"the {name} view of {call}",
            actual=part,
            expected=wanted,
        )


def check_train_steps(found: object) -> None:
    images = small_images(630, 300)
    weights = []

    for train in (found, REFERENCE["train_contrastive"]):
        model = seeded_encoder()
        loss = train(model, images, 3, 0.2, torch.Generator().manual_seed(631))
        weights.append((loss, model.body[1].weight.detach().clone()))

    (loss, weight), (expected_loss, expected_weight) = weights
    expect_close(
        call="train_contrastive for 3 steps on 300 images",
        actual=loss,
        expected=expected_loss,
        tolerance=1e-5,
    )
    expect_tensor(
        call="the encoder's first weights after 3 steps",
        actual=weight,
        expected=expected_weight,
    )


def check_learns_shifts(found: object) -> None:
    model = seeded_encoder()
    found(model, contrastive_sets()["train_x"], 150, 0.1, torch.Generator().manual_seed(0))
    _, moved = neighbor_scores(model)

    if moved < 0.65:
        raise Mismatch(
            f"After 150 steps of train_contrastive, the kNN guess gets {moved:.1%} of the moved digits right, but it "
            "should pass 65%. Raw pixels get under 40%.",
        )


def check_contrastive2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="two_views",
        description="shifts and adds noise twice, the first view first",
        inputs=(
            (Described(small_images(632, 3), "3 images of 28 by 28"), seeded(633), 0.3),
            (Described(small_images(634, 2), "2 images of 28 by 28"), seeded(635), 0.0),
        ),
        reference=REFERENCE["two_views"],
        expect=expect_two_views,
    )
    checker.test(
        function="train_contrastive",
        description="steps on InfoNCE of two views of each batch",
        run=check_train_steps,
    )
    checker.test(
        function="train_contrastive",
        description="learns features that don't care where a digit sits",
        run=check_learns_shifts,
    )


CONTRASTIVE2 = Exercise(
    name="contrastive2",
    chapter=CONTRASTIVE,
    title="Views and the temperature",
    lesson=CONTRASTIVE2_LESSON,
    source=CONTRASTIVE2_SOURCE,
    solution=CONTRASTIVE2_SOLUTION,
    hints={
        "two_views": "random_shifts(images, 4, generator), plus noise * torch.randn(images.shape, generator=generator). "
        "Build the first view this way, then the second.",
        "train_contrastive": "batch = images[torch.randint(0, len(images), (256,), generator=generator)]. Then "
        "two_views of it with 0.3, info_nce of model of each with the temperature, zero_grad, backward and step.",
    },
    introduces=(),
    check=check_contrastive2,
    time_limit=90,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# contrastive3
# ---------------------------------------------------------------------------

CONTRASTIVE3_LESSON = prose(
    """
    Here is what the encoder of contrastive2 learned with no labels at all, at
    a temperature of 0.1, next to what came before it. Each row is a kNN guess
    from the 10 nearest of 5,000 labeled training digits.

    | Features | Centered digits right | Moved digits right |
    |---|---|---|
    | raw pixels | 93.8% | 37.0% |
    | the encoder, untrained | 86.1% | 28.9% |
    | trained with InfoNCE | 87.1% | 85.9% |
    | trained with the pulling alone | 70.6% | 31.5% |

    Raw pixels do well on centered digits, since MNIST centers every digit.
    Shift the test digits and raw pixels fall apart, because a moved stroke
    lands on other pixels. The contrastive features hold up, since the views
    taught them to ignore the shift. Labels never came into it.

    InfoNCE does two jobs. It pulls each real pair together, and it pushes
    every made up pair apart. Wang and Isola call these alignment and
    uniformity. Train with the pulling alone, `alignment_loss` below, and the
    easiest answer wins, every image mapped to the same features. That is a
    **collapse**. The loss reaches 0 and the features stop telling digits
    apart.

    `feature_spread` measures it, the spread of each feature across a batch
    after scaling every row to length 1, averaged over the features. It is
    0.123 for the InfoNCE features and 0.0008 for the pulled ones. The tiny
    differences left still sort neighbors a little on centered digits, but the
    moved ones fall below raw pixels.
    """,
)

CONTRASTIVE3_SOURCE = source_file(
    '''
    # contrastive3.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    votes = torch.mode(torch.tensor([[3, 1, 3], [2, 2, 5]]), dim=1).values    # tensor([3, 2]), the most common of each row
    spread = torch.tensor([[1.0, 2.0], [3.0, 6.0]]).std(dim=0)                # tensor([1.4142, 2.8284])


    # similarity_matrix from contrastive1.
    def similarity_matrix(
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        return F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T


    # TODO: Return the share of queries whose kNN guess is right. A query's k
    #   nearest rows of bank are the ones with the largest cosine, and its guess
    #   is the label most of them carry, from torch.mode.
    def knn_accuracy(
        bank: Tensor,
        bank_labels: Tensor,
        queries: Tensor,
        query_labels: Tensor,
        k: int,
    ) -> float:
        ...


    # TODO: Return the spread of features, of shape (batch, size), as a plain
    #   number. Scale every row to length 1, take the spread of each feature
    #   down the batch, and average those.
    def feature_spread(features: Tensor) -> float:
        ...


    # TODO: Return the pulling half of InfoNCE alone, the mean over the batch of
    #   the squared distance between each row of z1 and its row of z2, both
    #   scaled to length 1.
    def alignment_loss(
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        ...
    ''',
)

CONTRASTIVE3_SOLUTION = source_file(
    '''
    # contrastive3.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    votes = torch.mode(torch.tensor([[3, 1, 3], [2, 2, 5]]), dim=1).values
    spread = torch.tensor([[1.0, 2.0], [3.0, 6.0]]).std(dim=0)


    def similarity_matrix(
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        return F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T


''' + indented(KNN_PART) + '''
    ''',
)


def train_alignment_only(
    model: nn.Module,
    images: Tensor,
    steps: int,
    generator: torch.Generator,
) -> float:
    """contrastive2's loop with the pulling half alone, the run behind the collapse row of the lesson."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(steps):
        batch = images[torch.randint(0, len(images), (256,), generator=generator)]
        first, second = REFERENCE["two_views"](batch, generator, 0.3)
        loss = REFERENCE["alignment_loss"](model(first), model(second))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def labeled_features(
    seed: int,
    count: int,
) -> tuple[Tensor, Tensor]:
    """Features that cluster by their label, so the nearest rows mostly share it."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 3, (count,), generator=generator)
    centers = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])

    return centers[labels] + 1.5 * torch.randn(count, 3, generator=generator), labels


def check_contrastive3(checker: Checker) -> None:
    bank, bank_labels = labeled_features(640, 40)
    queries, query_labels = labeled_features(641, 25)
    check_on_inputs(
        checker=checker,
        function="knn_accuracy",
        description="gives the share of queries whose neighbors vote right",
        inputs=(
            (Described(bank, "40 rows of features"), Described(bank_labels, "their labels"), Described(queries, "25 queries"), Described(query_labels, "their labels"), 5),
            (Described(bank, "40 rows of features"), Described(bank_labels, "their labels"), Described(queries, "25 queries"), Described(query_labels, "their labels"), 1),
            (
                torch.tensor([[1.0, 0.05], [1.0, 0.1], [1.0, 0.12], [0.0, 1.0]]),
                torch.tensor([1, 0, 0, 1]),
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([0]),
                3,
            ),
        ),
        reference=REFERENCE["knn_accuracy"],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="feature_spread",
        description="averages each feature's spread after scaling the rows",
        inputs=(
            (Described(bank, "40 rows of features"),),
            (torch.tensor([[1.0, 0.0], [2.0, 0.1], [3.0, -0.1]]),),
        ),
        reference=REFERENCE["feature_spread"],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="alignment_loss",
        description="pulls each pair together and pushes nothing apart",
        inputs=(
            feature_pair(642, 5, 4),
            feature_pair(643, 3, 2),
        ),
        reference=REFERENCE["alignment_loss"],
        expect=expect_tensor,
    )


CONTRASTIVE3 = Exercise(
    name="contrastive3",
    chapter=CONTRASTIVE,
    title="Why push apart",
    lesson=CONTRASTIVE3_LESSON,
    source=CONTRASTIVE3_SOURCE,
    solution=CONTRASTIVE3_SOLUTION,
    hints={
        "knn_accuracy": "similarity_matrix(queries, bank).topk(k, dim=1).indices gives each query's neighbors. Index "
        "bank_labels with them, take torch.mode over dim=1, and compare with query_labels.",
        "feature_spread": "F.normalize the rows, take .std(dim=0), then .mean() and .item().",
        "alignment_loss": "Normalize both, subtract, square, sum over dim=1, then take the mean.",
    },
    introduces=(),
    check=check_contrastive3,
)
