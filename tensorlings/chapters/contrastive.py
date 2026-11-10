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

        Make two randomly changed copies of an image, called views. The
        features of two views of the same image should be close together, and
        the features of views of different images should be far apart. This
        one rule needs no labels, and it teaches a network which changes to
        ignore. This chapter builds the InfoNCE loss and SimCLR's version of
        it, and trains an encoder on MNIST with it. Then it shows what goes
        wrong when the loss only pulls pairs together and never pushes others
        apart.

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
    # Grid positions run from -1 to 1 across 28 pixels, so one pixel is 1 / 14
    # of a grid unit, and 4 pixels is 4 / 14.
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
    word2vec learned from text by telling real pairs apart from made up ones.
    Contrastive learning does the same with images. Change one image in two
    random ways, for example by shifting it and by adding noise. The two
    changed copies are called views, and together they form a real pair. A
    view of one image and a view of a different image in the batch form a
    made up pair.

    An encoder is a network that turns each view into a vector of features.
    For a batch of N images, `z1` holds the features of the first views and
    `z2` holds the features of the second views. Each has shape `(N, size)`.
    The **similarity matrix** holds the cosine similarity of every first view
    with every second view. So entry `[i, j]` compares the first view of
    image i with the second view of image j.

    The real pairs are on the diagonal, at `[i, i]`. So each row can be read
    as a classification with N choices, where the right answer for row i is
    column i. The loss is then the cross entropy of each row with target i.
    This loss is called **InfoNCE**, a name from van den Oord and colleagues
    in 2018.

    ```python
    scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    F.cross_entropy(scores, torch.arange(2))    # tensor(0.1269), low because each row scores its own column highest
    ```

    A cosine similarity lies between -1 and 1. That range is too narrow for
    the softmax inside the cross entropy to give one column most of the
    probability. So the cosines are first divided by a temperature, as in
    soft1. A small temperature makes the scores larger and the softmax
    sharper. contrastive2 measures the effect of the temperature.

    If the features carry no information, the softmax gives every column the
    same probability, `1 / N`. The loss is then `log(N)`, which is 5.55 for a
    batch of 256.

    SimCLR, from Chen and colleagues, compares every view in the batch with
    every other view. There are 2N views, and for each one the right answer
    is its partner, the other view of the same image. The other 2N - 2 views
    form made up pairs with it. A view is never compared with itself. The
    paper calls this loss NT-Xent.
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
    square.fill_diagonal_(0.0)                           # sets the diagonal to 0, changing square in place


    # TODO: Return the cosine similarity of every row of z1 with every row of
    #   z2. z1 and z2 have shape (N, size). The result has shape (N, N), and
    #   entry [i, j] is the cosine similarity of z1[i] and z2[j].
    #   Example: for z1 = [[3.0, 4.0], [1.0, 0.0]] and z2 = [[6.0, 8.0],
    #   [0.0, -2.0]], it returns tensor([[1.0000, -0.8000], [0.6000, 0.0000]]).
    def similarity_matrix(
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the InfoNCE loss. Divide similarity_matrix(z1, z2) by
    #   temperature to get the scores. Then return the cross entropy of the
    #   scores, where the target of row i is column i.
    #   Example: info_nce(torch.eye(2), torch.eye(2), 0.5) returns
    #   tensor(0.1269), the same scores and loss as the example above.
    def info_nce(
        z1: Tensor,
        z2: Tensor,
        temperature: float,
    ) -> Tensor:
        ...


    # TODO: Return SimCLR's NT-Xent loss. Stack z1 above z2 into one tensor of
    #   2N rows. Compute the similarity matrix of those rows with themselves,
    #   and divide it by temperature. Set its diagonal to minus infinity, so no
    #   row is compared with itself. Return the cross entropy, where the target
    #   of row i is i + N for the first N rows and i - N for the last N rows.
    #   Example: nt_xent(torch.eye(2), torch.eye(2), 0.5) returns tensor(0.2395).
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
        description="returns the cosine similarity of every pair of rows",
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
        description="uses column i as the target of row i",
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
        description="uses each view's partner as its target among all 2N views",
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
        "similarity_matrix": "Scale every row of z1 and of z2 to length 1 with F.normalize and dim=1. Then matrix "
        "multiply the scaled z1 by the transpose of the scaled z2.",
        "info_nce": "Divide similarity_matrix(z1, z2) by the temperature. The targets are torch.arange(len(z1)).",
        "nt_xent": "torch.cat([z1, z2]) stacks the rows. Call .fill_diagonal_(float(\"-inf\")) on the scores. The "
        "targets are torch.arange(N, 2 * N) followed by torch.arange(N), joined with torch.cat.",
    },
    introduces=(),
    check=check_contrastive1,
)


# ---------------------------------------------------------------------------
# contrastive2
# ---------------------------------------------------------------------------

CONTRASTIVE2_LESSON = prose(
    """
    The changes used to make the views decide what the features learn to
    ignore. Here each view of an MNIST digit is rotated by up to 15 degrees
    and scaled by a factor from 0.8 to 1.2. It is also moved by up to 4
    pixels, and then some noise is added. To match two views of the same digit, the features have
    to ignore all of these changes. SimCLR found that combining several
    changes is what makes the task useful to learn. On its photos, it
    combined random crops with color changes.

    `random_affine` applies the rotation, scaling and move in one step.
    `F.affine_grid` computes, for every pixel of the output, which position in
    the input to read from. Then `F.grid_sample` reads the input at those
    positions.

    The encoder is a small CNN like the ones in Section 5. It turns a digit
    into 128 features. On top of it sits a **projection head**, an MLP that
    turns those 128 features into 64. The loss is computed on the head's
    output. SimCLR found that a head like this did more than 10% better than
    no head. It also found that the features before the head are the best
    ones to use afterwards, for other tasks.

    To test what the encoder learned, each validation digit gets a kNN guess,
    where kNN stands for k nearest neighbours. Find the 10 labelled digits
    whose features before the head have the highest cosine similarity to the
    digit's features. The guess is the most common label among those 10. The
    validation digits are tested twice, once centered and once moved by up to
    4 pixels.

    The loss pushes the features of each made up pair apart. How hard it
    pushes a pair is proportional to that pair's probability in the softmax.
    The temperature decides how the push is divided among the 255 made up
    pairs in a row. A low temperature makes the softmax sharp, so most of the
    push goes to the made up pairs that look most like the real one. These are
    called hard pairs. A high temperature spreads the push evenly over all
    255.

    The table shows training for 1,500 steps on batches of 256 digits, at
    four temperatures, each with 3 seeds. The second column is the share of
    the push that goes to the hardest made up pair. If the push were spread
    evenly, that share would be `1 / 255`, about 0.39%.

    | Temperature | Hardest pair's share | Centered digits right | Moved digits right |
    |---|---|---|---|
    | 0.01 | 68% to 76% | 95.3% to 95.8% | 93.8% to 94.6% |
    | 0.1 | 30% to 32% | 94.9% to 95.4% | 94.2% to 95.4% |
    | 1 | 0.93% | 95.2% to 95.7% | 94.0% to 95.2% |
    | 10 | 0.43% | 92.5% to 93.8% | 90.8% to 91.3% |

    Putting most of the push on the hard pairs is what makes the features
    good, and any temperature from 0.01 to 1 does that here. At 10 the push is
    nearly even. On each seed, temperature 10 scores 1.4 to 3.3 points below
    the other three on centered digits. On moved digits it scores 3.0 to 4.1
    points below. On centered digits it even falls below the 93.9% that raw pixels
    get in contrastive3. SimCLR explains its temperature the same way, as
    what lets the model learn from the hard made up pairs.

    `torch.randn` takes a `generator` argument, as `torch.randint` does. With
    a seeded generator, each view's noise is the same on every run.
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


    # Rotates each image by up to 15 degrees, scales it by 0.8 to 1.2 and moves
    #   it by up to 4 pixels. images has shape (batch, channels, height, width).
''' + indented(AFFINE_PART) + '''


    # The encoder, a small CNN that turns a digit into 128 features, and the
    #   projection head on top of it. The loss is computed on the head's output.
''' + indented(ENCODER_PART) + '''


    # TODO: Return two views of images, as a tuple (first, second). To make one
    #   view, call random_affine on images with generator. Then add noise times
    #   a tensor from torch.randn with the shape of images, drawn with
    #   generator. Make the whole first view before the second, because both
    #   draw from the same generator and the order changes the numbers.
    def two_views(
        images: Tensor,
        generator: torch.Generator,
        noise: float,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Finish the loop. In each step, pick a batch of 256 images at random
    #   indices from torch.randint, drawn with generator. Make two_views of the
    #   batch with a noise of 0.3. Pass each view through model, compute
    #   info_nce of the two outputs at temperature, and take one optimizer step
    #   on it. Store the loss in a variable named loss, because the last line
    #   returns loss.item().
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
        raise Mismatch(f"{call} should return the two views as a tuple of two tensors.")

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
                f"{mismatch}\nEvery random draw comes from the same generator, so the order of the calls changes the "
                "numbers. Call random_affine first, then torch.randn, and make the whole first view before the second.",
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
        call="the weights of the encoder's first layer after 3 steps",
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
            f"After 150 steps of train_contrastive on a small MLP encoder with the projection head, the kNN guess gets "
            f"{moved:.1%} of the moved digits right. It should get more than {MOVED_BAR:.0%}. Raw pixels get under 40%.",
        )


def check_contrastive2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="two_views",
        description="rotates, scales, moves and adds noise for each view, the first view first",
        inputs=(
            (Described(small_images(632, 3), "3 images of 28 by 28"), seeded(633), 0.3),
            (Described(small_images(634, 2), "2 images of 28 by 28"), seeded(635), 0.0),
        ),
        reference=REFERENCE["two_views"],
        expect=expect_two_views,
    )
    checker.test(
        function="train_contrastive",
        description="takes optimizer steps on the InfoNCE loss of two views of each batch",
        run=check_train_steps,
    )
    checker.test(
        function="train_contrastive",
        description="learns features that stay similar when a digit moves",
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
        "two_views": "Each view is random_affine(images, generator), plus noise * torch.randn(images.shape, "
        "generator=generator). Build the first view this way, then the second.",
        "train_contrastive": "Index images with 256 random indices from torch.randint below len(images), using "
        "generator. Call two_views on that batch with 0.3, and compute info_nce of model applied to each view, with "
        "the temperature. Then call zero_grad, backward and step.",
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
    The table compares contrastive2's encoder, trained at a temperature of
    0.1 with no labels at all, against simpler features. Each row uses the
    kNN guess from contrastive2. The guess comes from the 10 nearest of 5,000
    labelled training digits, using the features before the head.

    | Features | Centered digits right | Moved digits right |
    |---|---|---|
    | raw pixels | 93.9% | 37.0% |
    | the CNN, untrained | 82.4% | 51.3% |
    | trained with InfoNCE, on three seeds | 94.9% to 95.4% | 94.2% to 95.4% |
    | trained with the pulling alone | 60.6% | 29.8% |

    The InfoNCE features beat raw pixels on both kinds of digit. Raw pixels
    do well on centered digits, because MNIST centers every digit. They do
    badly once a digit moves, because its strokes then cover different
    pixels. The contrastive features stay accurate, because the views taught
    them to ignore rotation, size and position. No labels were used to train
    them.

    InfoNCE does two jobs. It pulls the features of each real pair together,
    and it pushes the features of every made up pair apart. Wang and Isola
    call these two jobs alignment and uniformity. Suppose you train with only
    the pulling part, which is `alignment_loss` below. Then the easiest
    solution is to map every image to the same features. This is called a
    **collapse**. The loss reaches 0, and the features can no longer tell
    digits apart.

    `feature_spread` detects a collapse. It scales every row of features to
    length 1. Then it takes the standard deviation of each feature across
    the batch, which measures how much that feature varies from image to
    image. Finally it averages those standard deviations. On the head's
    output, where the loss is computed, this is 0.122 for the InfoNCE run.
    For the run trained only on pulling, it is under 0.0001. The features
    before the head still differ a tiny bit, so they sort neighbours a little
    on centered digits. But on moved digits they do worse than raw pixels.
    """,
)

CONTRASTIVE3_SOURCE = source_file(
    '''
    # contrastive3.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    votes = torch.mode(torch.tensor([[3, 1, 3], [2, 2, 5]]), dim=1).values    # tensor([3, 2]), the most common value in each row
    spread = torch.tensor([[1.0, 2.0], [3.0, 6.0]]).std(dim=0)                # tensor([1.4142, 2.8284]), the standard deviation of each column


    # similarity_matrix from contrastive1.
    def similarity_matrix(
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        return F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T


    # TODO: Return the fraction of queries whose kNN guess is right, as a Python
    #   float. bank has shape (B, size), with labels bank_labels of shape (B,).
    #   queries has shape (Q, size), with labels query_labels of shape (Q,). A
    #   query's k nearest rows of bank are the k with the highest cosine
    #   similarity to it. Its guess is the most common label among those k
    #   rows, found with torch.mode.
    def knn_accuracy(
        bank: Tensor,
        bank_labels: Tensor,
        queries: Tensor,
        query_labels: Tensor,
        k: int,
    ) -> float:
        ...


    # TODO: Return feature_spread from the lesson as a Python float. features
    #   has shape (batch, size). Scale every row to length 1, take the standard
    #   deviation of each column over the batch, then average those numbers.
    #   Example: feature_spread(torch.tensor([[1.0, 0.0], [0.0, 1.0]])) returns
    #   about 0.7071.
    def feature_spread(features: Tensor) -> float:
        ...


    # TODO: Return alignment_loss, only the pulling part of InfoNCE. Scale
    #   every row of z1 and of z2 to length 1. For every i, compute the squared
    #   distance between row i of z1 and row i of z2, which is the sum of the
    #   squared differences. Return the mean of those distances over the batch.
    #   Example: alignment_loss(torch.tensor([[1.0, 0.0]]),
    #   torch.tensor([[0.0, 2.0]])) returns tensor(2.).
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
        description="returns the fraction of queries whose neighbours' most common label is right",
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
        description="averages each feature's standard deviation after scaling the rows",
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
        description="returns the mean squared distance between the scaled rows of each pair",
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
        "bank_labels with those indices, take torch.mode over dim=1, and compare the votes with query_labels. The "
        "mean of the matches, as floats, is the accuracy.",
        "feature_spread": "Normalize the rows with F.normalize, take .std(dim=0), then call .mean() and .item().",
        "alignment_loss": "Normalize the rows of both inputs and subtract them. Square the differences, sum over "
        "dim=1, then take the mean.",
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
