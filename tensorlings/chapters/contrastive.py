"""61_contrastive, two views of one image pulled together and every other image pushed apart, InfoNCE and NT-Xent, the temperature, and the collapse when nothing pushes apart."""

from __future__ import annotations

import contextlib
import functools
import math
import textwrap
from collections.abc import Iterator

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
from ..training import best_device
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

AFFINE_PART = '''
def random_affine(
    images: Tensor,
    generator: torch.Generator,
) -> Tensor:
    count = len(images)
    angle = (torch.rand(count, generator=generator) * 2 - 1) * math.radians(15)
    scale = 0.8 + 0.4 * torch.rand(count, generator=generator)
    # The grid runs from -1 to 1 across 28 pixels, so 4 pixels is 4 / 14 of it.
    shift = (torch.rand(count, 2, generator=generator) * 2 - 1) * (4 / 14)
    theta = torch.zeros(count, 2, 3)
    theta[:, 0, 0] = angle.cos() / scale
    theta[:, 0, 1] = -angle.sin() / scale
    theta[:, 1, 0] = angle.sin() / scale
    theta[:, 1, 1] = angle.cos() / scale
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)

    return F.grid_sample(images, grid, align_corners=False)
'''

ENCODER_PART = '''
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.body(images)


class WithHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.encoder(images))
'''

TWO_VIEWS_PART = '''
def two_views(
    images: Tensor,
    generator: torch.Generator,
    noise: float,
) -> tuple[Tensor, Tensor]:
    first = random_affine(images, generator) + noise * torch.randn(images.shape, generator=generator)
    second = random_affine(images, generator) + noise * torch.randn(images.shape, generator=generator)

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
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(SIMILARITY_PART + NT_XENT_PART + SHIFT_PART + AFFINE_PART + ENCODER_PART + TWO_VIEWS_PART + TRAIN_PART + KNN_PART, namespace)

    return namespace


REFERENCE = reference_namespace()
reference_similarity_matrix = REFERENCE["similarity_matrix"]
reference_info_nce = REFERENCE["info_nce"]
reference_nt_xent = REFERENCE["nt_xent"]


@functools.cache
def contrastive_sets() -> dict[str, Tensor]:
    """The MNIST images the lessons measure on, 20,000 to train on without labels, 5,000 with labels to look
    neighbours up in, and 2,000 validation images, centered and moved by up to 4 pixels."""
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


def seeded_model() -> nn.Module:
    torch.manual_seed(0)

    return REFERENCE["WithHead"]()


class LightModel(nn.Module):
    """An MLP encoder with the same head, light enough to train 150 steps inside a check on any laptop."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.encoder(images))


# The share of moved digits the light model's features must get right after 150 steps. Raw pixels get 37%.
MOVED_BAR = 0.5


def neighbour_scores(model: nn.Module | None) -> tuple[float, float]:
    """kNN accuracy on centered and on moved validation digits, of model's features or of raw pixels when None."""
    sets = contrastive_sets()
    knn = REFERENCE["knn_accuracy"]

    def features(images: Tensor) -> Tensor:
        if model is None:
            return images.flatten(1)

        with torch.no_grad():
            return torch.cat([model(part) for part in images.split(500)])

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
    MNIST digit is turned by up to 15 degrees and scaled by 0.8 to 1.2. It is
    moved by up to 4 pixels, then gets some noise. Features that pair two views of
    one digit have to ignore all of that. SimCLR found that composing several
    changes is what makes the task worth learning, crops with color changes
    on its photos.

    `random_affine` makes the whole change in one step. `F.affine_grid` works
    out, for every pixel of the output, the place in the input to read, and
    `F.grid_sample` reads it.

    The encoder is a small CNN like Section 5's, from a digit to 128
    features. The loss reads a **projection head** on top, an MLP from those
    128 features to 64. SimCLR found a head like this beat no head by more
    than 10%, and that the features before the head serve best afterwards.

    To see what the encoder learned, each validation digit gets a kNN guess.
    That is the label most common among its 10 nearest labelled digits, by
    cosine of their features before the head. The digits are tested
    centered, and moved by up to 4 pixels.

    The loss pushes each made up pair away in proportion to its weight in the
    softmax. The temperature decides how that push is shared out. A low one
    sharpens the softmax onto the made up pairs that look most like the real
    one. A high one spreads the push evenly over all 255.

    Here are 1,500 steps of 256 digits at four temperatures, each on 3 seeds.
    The share is the hardest made up pair's part of the push, where an even
    share would be 0.39%.

    | Temperature | Hardest pair's share | Centered digits right | Moved digits right |
    |---|---|---|---|
    | 0.01 | 68% to 76% | 95.3% to 95.8% | 93.8% to 94.6% |
    | 0.1 | 30% to 32% | 94.9% to 95.4% | 94.2% to 95.4% |
    | 1 | 0.93% | 95.2% to 95.7% | 94.0% to 95.2% |
    | 10 | 0.43% | 92.5% to 93.8% | 90.8% to 91.3% |

    Leaning the push on the hard pairs is what makes the features good, and
    anywhere from 0.01 to 1 does it here. At 10 the push is nearly even. On
    each seed it scores 1.4 to 3.3 points below the other three on centered
    digits, and 3.0 to 4.1 below on moved ones. Centered, it even falls below
    the 93.9% of raw pixels that contrastive3 measures. SimCLR explains its
    temperature the same way, as what lets the model learn from the hard made
    up pairs.

    `torch.randn` takes a generator as `randint` does, so each view's noise is
    drawn the same way on every run.
    """,
)

CONTRASTIVE2_SOURCE = source_file(
    '''
    # contrastive2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    noise = torch.randn(2, 3, generator=torch.Generator().manual_seed(0))    # the same numbers every run


    # similarity_matrix and info_nce from contrastive1.
''' + indented(SIMILARITY_PART) + '''


    # Each image turned by up to 15 degrees, scaled by 0.8 to 1.2 and moved by up
    #   to 4 pixels, for images of shape (batch, channels, height, width).
''' + indented(AFFINE_PART) + '''


    # The encoder, a small CNN from a digit to 128 features, and the projection
    #   head on top of it that the loss reads.
''' + indented(ENCODER_PART) + '''


    # TODO: Return two views of images. For each view, first call random_affine
    #   on images, then add noise times torch.randn of their shape, drawn with
    #   generator. Make the whole first view, then the whole second.
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

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    noise = torch.randn(2, 3, generator=torch.Generator().manual_seed(0))


''' + indented(SIMILARITY_PART) + '''


''' + indented(AFFINE_PART) + '''


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
        try:
            expect_tensor(
                call=f"the {name} view of {call}",
                actual=part,
                expected=wanted,
            )
        except Mismatch as mismatch:
            if not isinstance(part, Tensor) or part.shape != wanted.shape:
                raise

            # The same sums drawn in another order give other numbers, which the plain message never says.
            raise Mismatch(
                f"{mismatch}\nEvery draw comes from one generator, so the order counts. Call random_affine first, then "
                "torch.randn, and make the whole first view before the second.",
            ) from None


def check_train_steps(found: object) -> None:
    images = small_images(630, 300)
    weights = []

    for train in (found, REFERENCE["train_contrastive"]):
        model = seeded_model()
        loss = train(model, images, 3, 0.2, torch.Generator().manual_seed(631))
        weights.append((loss, model.encoder.body[0].weight.detach().clone()))

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
    # The CNN takes seconds a step on a laptop's CPU, so this run trains a light MLP with the same head.
    torch.manual_seed(0)
    model = LightModel()
    found(model, contrastive_sets()["train_x"], 150, 0.1, torch.Generator().manual_seed(0))
    _, moved = neighbour_scores(model.encoder)

    if moved < MOVED_BAR:
        raise Mismatch(
            f"After 150 steps of train_contrastive on a light MLP with the head, the kNN guess gets {moved:.1%} of the "
            f"moved digits right, but it should pass {MOVED_BAR:.0%}. Raw pixels get under 40%.",
        )


def check_contrastive2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="two_views",
        description="turns, scales, moves and adds noise twice, the first view first",
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
        "two_views": "random_affine(images, generator), plus noise * torch.randn(images.shape, generator=generator). "
        "Build the first view this way, then the second.",
        "train_contrastive": "Pick a batch of 256 images at places drawn by torch.randint below len(images), using "
        "generator. Then two_views of it with 0.3, info_nce of model of each with the temperature, zero_grad, "
        "backward and step.",
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
    Here is what contrastive2's encoder learned with no labels at all, at a
    temperature of 0.1, next to what came before it. Each row is a kNN guess
    from the 10 nearest of 5,000 labelled training digits, read from the
    features before the head.

    | Features | Centered digits right | Moved digits right |
    |---|---|---|
    | raw pixels | 93.9% | 37.0% |
    | the CNN, untrained | 82.4% | 51.3% |
    | trained with InfoNCE, on three seeds | 94.9% to 95.4% | 94.2% to 95.4% |
    | trained with the pulling alone | 60.6% | 29.8% |

    The trained features beat raw pixels on both. Raw pixels do well on
    centered digits, since MNIST centers every digit. They fall apart once a
    digit moves, because a moved stroke lands on other pixels. The
    contrastive features hold up, since the views taught them to ignore
    turns, sizes and shifts. Labels never came into it.

    InfoNCE does two jobs. It pulls each real pair together, and it pushes
    every made up pair apart. Wang and Isola call these alignment and
    uniformity. Train with the pulling alone, `alignment_loss` below, and the
    easiest answer wins, every image mapped to the same features. That is a
    **collapse**. The loss reaches 0 and the features stop telling digits
    apart.

    `feature_spread` measures it, the spread of each feature across a batch
    after scaling every row to length 1, averaged over the features. On the
    head's output, which the loss reads, it is 0.122 for the InfoNCE run and
    under 0.0001 for the pulled one. The tiny differences left before the
    head still sort neighbours a little on centered digits, but the moved
    ones fall below raw pixels.
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


def labelled_features(
    seed: int,
    count: int,
) -> tuple[Tensor, Tensor]:
    """Features that cluster by their label, so the nearest rows mostly share it."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 3, (count,), generator=generator)
    centers = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])

    return centers[labels] + 1.5 * torch.randn(count, 3, generator=generator), labels


def check_contrastive3(checker: Checker) -> None:
    bank, bank_labels = labelled_features(640, 40)
    queries, query_labels = labelled_features(641, 25)
    check_on_inputs(
        checker=checker,
        function="knn_accuracy",
        description="gives the share of queries whose neighbours vote right",
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
        "knn_accuracy": "similarity_matrix(queries, bank).topk(k, dim=1).indices gives each query's neighbours. Index "
        "bank_labels with them, take torch.mode over dim=1, and compare with query_labels.",
        "feature_spread": "F.normalize the rows, take .std(dim=0), then .mean() and .item().",
        "alignment_loss": "Normalize both, subtract, square, sum over dim=1, then take the mean.",
    },
    introduces=(),
    check=check_contrastive3,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

# contrastive2's rows. From 0.05 to 1 the scores tie, so the table spans 0.01 to 10, where an even push shows its cost.
TEMPERATURES = (0.01, 0.1, 1.0, 10.0)


@contextlib.contextmanager
def pinned_threads(count: int) -> Iterator[None]:
    before = torch.get_num_threads()
    torch.set_num_threads(count)

    try:
        yield
    finally:
        torch.set_num_threads(before)


def train_on_device(
    model: nn.Module,
    steps: int,
    temperature: float,
    seed: int,
    pull_only: bool,
) -> float:
    """contrastive2's loop on the best device, batches of 256 with two views at a noise of 0.3 and Adam at 1e-3, on
    InfoNCE or, when pull_only, on alignment_loss alone. The views are drawn on the CPU, so a seed draws the same
    ones on any machine."""
    device = best_device()
    model.to(device)
    images = contrastive_sets()["train_x"]
    generator = torch.Generator().manual_seed(seed + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(steps):
        batch = images[torch.randint(0, len(images), (256,), generator=generator)]
        first, second = REFERENCE["two_views"](batch, generator, 0.3)
        z1, z2 = model(first.to(device)), model(second.to(device))

        if pull_only:
            loss = REFERENCE["alignment_loss"](z1, z2)
        else:
            scores = REFERENCE["similarity_matrix"](z1, z2) / temperature
            loss = F.cross_entropy(scores, torch.arange(len(z1), device=device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.cpu()

    return loss.item()


def measure_run(
    temperature: float,
    seed: int,
    pull_only: bool = False,
    steps: int = 1500,
) -> dict[str, float]:
    """One row of the lessons: the CNN with its head trained steps steps, then the kNN guess on its encoder's
    features, centered and moved, and the spread of those features and of the head's."""
    with pinned_threads(4):
        torch.manual_seed(seed)
        model = REFERENCE["WithHead"]()
        loss = train_on_device(model, steps, temperature, seed, pull_only)
        centered, moved = neighbour_scores(model.encoder)
        sample = contrastive_sets()["test_x"][:1000]

        with torch.no_grad():
            features = model.encoder(sample)
            spread = REFERENCE["feature_spread"](features)
            head_spread = REFERENCE["feature_spread"](model.head(features))
            first, second = REFERENCE["two_views"](sample[:256], torch.Generator().manual_seed(seed + 7), 0.3)
            scores = REFERENCE["similarity_matrix"](model(first), model(second)) / temperature
            # The real pair leaves the softmax, so what is left shares out the loss's push among the made up pairs.
            scores.fill_diagonal_(float("-inf"))
            hardest = scores.softmax(dim=1).max(dim=1).values.mean().item()

    return {
        "centered": centered,
        "moved": moved,
        "spread": spread,
        "head spread": head_spread,
        "hardest share": hardest,
        "loss": loss,
    }


def measure_contrastive() -> dict[str, object]:
    """Every number in contrastive2's and contrastive3's tables: raw pixels, the untrained CNN, InfoNCE at the
    temperatures 0.01, 0.1, 1 and 10 on seeds 0 to 2, and the pulling alone on seed 0. Each run takes about 3
    minutes on MPS."""
    with pinned_threads(4):
        results: dict[str, object] = {
            "pixels": neighbour_scores(None),
            "untrained": neighbour_scores(seeded_model().encoder),
        }

    for temperature in TEMPERATURES:
        for seed in range(3):
            results[f"{temperature}, seed {seed}"] = measure_run(temperature, seed)

    results["pulling alone"] = measure_run(0.1, 0, pull_only=True)

    return results
