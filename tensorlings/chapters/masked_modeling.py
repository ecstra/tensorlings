"""62_masked_modeling, BERT's masking of text with the loss on masked tokens alone, and MAE's masking of most image patches, and why images need so much more hidden."""

from __future__ import annotations

import textwrap

import torch
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..data import load_cifar10, load_shakespeare
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    seeded,
    small_tensor,
)
from ..text import prose, source_file

MASKED_MODELING = Chapter(
    folder="62_masked_modeling",
    title="Masked modeling",
    readme=source_file(
        """
        # Masked modeling

        Hide part of the input and train the model to fill it in. The hidden
        part is the answer, so any pile of text or images becomes training
        data with no labels. This chapter builds BERT's masking for text and
        MAE's for images, and measures why images need far more hidden.

        ## Further reading

        - [Devlin and colleagues, BERT](https://arxiv.org/abs/1810.04805)
        - [He and colleagues, masked autoencoders are scalable vision learners](https://arxiv.org/abs/2111.06377)
        - [Weng, self supervised representation learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)
        """,
    ),
)

# The id of [MASK], one past the 65 characters of tiny Shakespeare.
MASK = 65


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


# ---------------------------------------------------------------------------
# masked1
# ---------------------------------------------------------------------------

BERT_PARTS = '''
def choose(
    ids: Tensor,
    rate: float,
    generator: torch.Generator,
) -> Tensor:
    return torch.rand(ids.shape, generator=generator) < rate


def corrupt(
    ids: Tensor,
    chosen: Tensor,
    vocabulary: int,
    generator: torch.Generator,
) -> Tensor:
    roll = torch.rand(ids.shape, generator=generator)
    random_ids = torch.randint(0, vocabulary, ids.shape, generator=generator)
    corrupted = torch.where(chosen & (roll < 0.8), MASK, ids)

    return torch.where(chosen & (roll >= 0.8) & (roll < 0.9), random_ids, corrupted)


def masked_loss(
    scores: Tensor,
    ids: Tensor,
    chosen: Tensor,
) -> Tensor:
    return F.cross_entropy(scores[chosen], ids[chosen])


def copy_scores(
    corrupted: Tensor,
    ids: Tensor,
    chosen: Tensor,
) -> tuple[float, float]:
    right = corrupted == ids

    return right.float().mean().item(), right[chosen].float().mean().item()
'''

BERT_REFERENCE: dict[str, object] = {
    "torch": torch,
    "F": F,
    "Tensor": Tensor,
    "MASK": MASK,
}
exec(BERT_PARTS, BERT_REFERENCE)


def copy_table() -> tuple[float, float]:
    """The run behind masked1's numbers: 100,000 characters of tiny Shakespeare as 1,000 rows of 100,
    15% chosen, and what a model that copies its input scores."""
    text = load_shakespeare()
    characters = sorted(set(text))
    positions = {character: position for position, character in enumerate(characters)}
    ids = torch.tensor([positions[character] for character in text[:100000]]).view(-1, 100)
    generator = torch.Generator().manual_seed(0)
    chosen = BERT_REFERENCE["choose"](ids, 0.15, generator)
    corrupted = BERT_REFERENCE["corrupt"](ids, chosen, len(characters), generator)

    return BERT_REFERENCE["copy_scores"](corrupted, ids, chosen)


MASKED1_LESSON = prose(
    """
    Section 8 learns from data with no labels. One way is to hide part of the
    input and ask the model to fill it in. The hidden part is the answer, so
    any pile of text becomes training data.

    BERT does this to text. In each sequence it chooses 15% of the tokens at
    random. A chosen token becomes a special `[MASK]` token 80% of the time,
    and a random token 10% of the time. The last 10% it stays as it was. The
    model reads the whole corrupted sequence, both ways at once, and predicts
    the original token at every chosen position.

    The loss counts the chosen positions alone. Everywhere else the input
    already holds the answer. Here is what that means, on 100,000 characters
    of Shakespeare with characters as tokens and 15% chosen.

    | A model that copies its input gets right | |
    |---|---|
    | of all positions | 86.6% |
    | of the chosen positions | 10.7% |

    The copier learns nothing, yet a loss over every position would rate it
    well. Counted on the chosen positions alone, the only way to score is to
    read the context.

    Why not turn every chosen token into `[MASK]`? `[MASK]` never shows up in
    real text, so a model that only learned to fill it in would meet
    something new once used. The random and unchanged tokens mean any token
    might be the one to predict, so the model keeps track of all of them.

    A tensor of True and False picks entries of a tensor, and from a tensor of
    scores, it picks whole rows of scores.

    ```python
    scores = torch.zeros(2, 3, 5)
    scores[torch.tensor([[True, False, True], [False, False, True]])].shape    # torch.Size([3, 5])
    ```
    """,
)

MASKED1_SOURCE = source_file(
    '''
    # masked1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    MASK = 65    # the id of [MASK], one past the 65 characters

    ids = torch.tensor([7, 3, 9])
    hidden = torch.where(torch.tensor([False, True, True]), MASK, ids)    # tensor([ 7, 65, 65])


    # TODO: Return a tensor of True and False the shape of ids, True where a
    #   token is chosen. Draw torch.rand(ids.shape, generator=generator) and
    #   choose where the draw is below rate.
    def choose(
        ids: Tensor,
        rate: float,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return ids with BERT's corruption of the chosen tokens. First draw
    #   roll with torch.rand(ids.shape, generator=generator), then random_ids
    #   with torch.randint(0, vocabulary, ids.shape, generator=generator). A
    #   chosen token becomes MASK where roll is below 0.8, its random id where
    #   roll is from 0.8 up to 0.9, and stays where roll is 0.9 or more.
    def corrupt(
        ids: Tensor,
        chosen: Tensor,
        vocabulary: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the cross entropy of scores, of shape (batch, steps,
    #   vocabulary), against ids at the chosen positions alone.
    def masked_loss(
        scores: Tensor,
        ids: Tensor,
        chosen: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return what a model that copies its input scores, as two plain
    #   numbers: the share of all positions where corrupted equals ids, then
    #   the same share among the chosen positions.
    def copy_scores(
        corrupted: Tensor,
        ids: Tensor,
        chosen: Tensor,
    ) -> tuple[float, float]:
        ...
    ''',
)

MASKED1_SOLUTION = source_file(
    '''
    # masked1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    MASK = 65

    ids = torch.tensor([7, 3, 9])
    hidden = torch.where(torch.tensor([False, True, True]), MASK, ids)


''' + indented(BERT_PARTS) + '''
    ''',
)


def token_rows(
    seed: int,
    rows: int,
    steps: int,
    vocabulary: int,
) -> Tensor:
    return torch.randint(0, vocabulary, (rows, steps), generator=torch.Generator().manual_seed(seed))


def chosen_rows(
    seed: int,
    rows: int,
    steps: int,
) -> Tensor:
    return torch.rand(rows, steps, generator=torch.Generator().manual_seed(seed)) < 0.4


def copy_inputs(
    seed: int,
    rows: int,
    steps: int,
    vocabulary: int,
) -> tuple[Tensor, Tensor, Tensor]:
    ids = token_rows(seed, rows, steps, vocabulary)
    chosen = chosen_rows(seed + 1, rows, steps)
    corrupted = BERT_REFERENCE["corrupt"](ids, chosen, vocabulary, torch.Generator().manual_seed(seed + 2))

    return corrupted, ids, chosen


def expect_chosen(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )

    if actual.dtype != torch.bool:
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but scores[chosen] picks positions only with True and False. Return the comparison as it is.")


def expect_ids(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )

    if actual.dtype.is_floating_point:
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but token ids are whole numbers. Keep the dtype of ids.")


def expect_pair_close(
    call: str,
    actual: object,
    expected: tuple[float, float],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should give two numbers in a tuple, all positions first.")

    for name, part, wanted in zip(("all positions", "the chosen positions"), actual, expected):
        expect_close(
            call=f"the share of {name} from {call}",
            actual=part,
            expected=wanted,
        )


def check_masked1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="choose",
        description="chooses where the draw is below the rate",
        inputs=(
            (token_rows(620, 3, 8, 20), 0.15, seeded(621)),
            (token_rows(622, 2, 6, 10), 0.5, seeded(623)),
        ),
        reference=BERT_REFERENCE["choose"],
        expect=expect_chosen,
    )
    check_on_inputs(
        checker=checker,
        function="corrupt",
        description="masks 80%, swaps 10% and keeps 10% of the chosen tokens",
        inputs=(
            (token_rows(624, 4, 10, 20), chosen_rows(625, 4, 10), 20, seeded(626)),
            (token_rows(627, 3, 12, 65), chosen_rows(628, 3, 12), 65, seeded(629)),
        ),
        reference=BERT_REFERENCE["corrupt"],
        expect=expect_ids,
    )
    check_on_inputs(
        checker=checker,
        function="masked_loss",
        description="takes the cross entropy at the chosen positions alone",
        inputs=(
            (small_tensor(630, 2, 5, 7) / 2, token_rows(631, 2, 5, 7), chosen_rows(632, 2, 5)),
            (small_tensor(633, 3, 4, 6) / 2, token_rows(634, 3, 4, 6), chosen_rows(635, 3, 4)),
        ),
        reference=BERT_REFERENCE["masked_loss"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="copy_scores",
        description="gives what copying the input scores, everywhere and where chosen",
        inputs=(
            copy_inputs(636, 5, 20, 12),
            copy_inputs(656, 4, 25, 30),
        ),
        reference=BERT_REFERENCE["copy_scores"],
        expect=expect_pair_close,
    )


MASKED1 = Exercise(
    name="masked1",
    chapter=MASKED_MODELING,
    title="BERT's masking",
    lesson=MASKED1_LESSON,
    source=MASKED1_SOURCE,
    solution=MASKED1_SOLUTION,
    hints={
        "choose": "Compare torch.rand of ids.shape, drawn with the generator, with rate.",
        "corrupt": "torch.where(chosen & (roll < 0.8), MASK, ids) handles the masked ones. A second torch.where puts "
        "random_ids where chosen and roll is from 0.8 up to 0.9.",
        "masked_loss": "scores[chosen] gives one row of scores per chosen position, and ids[chosen] the answers. Pass "
        "both to F.cross_entropy.",
        "copy_scores": "right = corrupted == ids. Its mean as floats is the first share, and right[chosen]'s mean the "
        "second. .item() each one.",
    },
    introduces=(),
    check=check_masked1,
)


# ---------------------------------------------------------------------------
# masked2
# ---------------------------------------------------------------------------

MAE_PARTS = '''
def random_masking(
    patches: Tensor,
    ratio: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    batch, count, _ = patches.shape
    kept = int(count * (1 - ratio))
    noise = torch.rand(batch, count, generator=generator)
    keep = noise.argsort(dim=1)[:, :kept]
    rows = torch.arange(batch).unsqueeze(1)
    hidden = torch.ones(batch, count, dtype=torch.bool)
    hidden[rows, keep] = False

    return patches[rows, keep], keep, hidden


def fill_in(
    visible: Tensor,
    keep: Tensor,
    count: int,
    mask_token: Tensor,
) -> Tensor:
    batch, _, size = visible.shape
    full = mask_token.expand(batch, count, size).clone()
    full[torch.arange(batch).unsqueeze(1), keep] = visible

    return full


def masked_mse(
    predicted: Tensor,
    patches: Tensor,
    hidden: Tensor,
) -> Tensor:
    return ((predicted - patches) ** 2).mean(dim=-1)[hidden].mean()
'''

MAE_REFERENCE: dict[str, object] = {
    "torch": torch,
    "Tensor": Tensor,
}
exec(MAE_PARTS, MAE_REFERENCE)


def neighbour_fill_table() -> dict[float, tuple[float, float]]:
    """The run behind masked2's table: 1,000 CIFAR-10 test images in patches of 4 by 4. Each hidden patch is
    filled with the average of its visible neighbours above, below, left and right, or with the image's
    average visible patch when it has none. The result is that error over the error of filling every
    hidden patch with the average, and the share of hidden patches with no visible neighbour."""
    images, _ = load_cifar10("test")
    grid = (images[:1000].float() / 255).view(-1, 3, 8, 4, 8, 4).permute(0, 2, 4, 1, 3, 5)
    batch = len(grid)
    table = {}

    for ratio in (0.15, 0.5, 0.75):
        hidden = torch.rand(batch, 8, 8, generator=torch.Generator().manual_seed(1)) < ratio
        visible = (~hidden).float()
        shape = (batch, 8, 8, 1, 1, 1)
        shown = grid * visible.view(shape)
        padded = F.pad(shown.permute(0, 3, 4, 5, 1, 2), (1, 1, 1, 1))
        padded_visible = F.pad(visible, (1, 1, 1, 1))
        total = torch.zeros_like(grid)
        count = torch.zeros(batch, 8, 8)

        for row, column in ((0, 1), (2, 1), (1, 0), (1, 2)):
            total += padded[..., row : row + 8, column : column + 8].permute(0, 4, 5, 1, 2, 3)
            count += padded_visible[:, row : row + 8, column : column + 8]

        average = shown.sum(dim=(1, 2), keepdim=True) / visible.sum(dim=(1, 2)).clamp(min=1).view(batch, 1, 1, 1, 1, 1)
        fill = torch.where(count.view(shape) > 0, total / count.clamp(min=1).view(shape), average)
        neighbour_error = ((fill - grid) ** 2).mean(dim=(3, 4, 5))[hidden].mean()
        average_error = ((average.expand_as(grid) - grid) ** 2).mean(dim=(3, 4, 5))[hidden].mean()
        alone = ((count == 0) & hidden).float().sum() / hidden.float().sum()
        table[ratio] = ((neighbour_error / average_error).item(), alone.item())

    return table


MASKED2_LESSON = prose(
    """
    MAE, the masked autoencoder, does the same to images. It cuts an image into
    patches, as vit1 did, and hides 75% of them at random. The encoder reads
    only the visible quarter. A light decoder puts a learned mask token where
    each hidden patch was and rebuilds every patch. The loss counts only the
    hidden ones, as in BERT.

    Why 75% when BERT hides 15%? Neighbouring patches of a photo look alike,
    which the MAE paper calls spatial redundancy. Hide a few, and averaging
    their neighbours fills them in, with no idea of what the image shows.
    Here is that averaging on 1,000 CIFAR-10 test images in patches of 4 by
    4. Its error is shown as a share of the error of filling every hidden
    patch with the average of the visible patches.

    | Hidden | 15% | 50% | 75% |
    |---|---|---|---|
    | Error left by averaging neighbours | 49% | 68% | 87% |
    | Hidden patches with no visible neighbour | 0.4% | 9.5% | 37% |

    At 15%, averaging halves the error, so a model could score well by
    learning to blur. At 75% it barely helps, and the model has to know what
    images look like. A **linear probe** trains one Linear layer on the
    frozen encoder's outputs, to test how much those outputs already hold.
    The paper found 75% best for a linear probe. Fine tuning the whole
    encoder worked well anywhere from 40% to 80%. Hiding most patches also
    makes the encoder cheap, since it reads a quarter of them.

    Random masking shuffles the patches. Draw a random number per patch, sort
    them, and keep the patches that come first.

    ```python
    torch.tensor([[0.7, 0.1, 0.4]]).argsort(dim=1)    # tensor([[1, 2, 0]]), patch 1 first
    ```

    Indexing with a column of row numbers and a matrix of positions picks
    different positions from each row.

    ```python
    x = torch.tensor([[10, 11, 12], [20, 21, 22]])
    x[torch.arange(2).unsqueeze(1), torch.tensor([[2, 0], [1, 2]])]    # tensor([[12, 10], [21, 22]])
    ```
    """,
)

MASKED2_SOURCE = source_file(
    '''
    # masked2.py

    import torch
    from torch import Tensor

    order = torch.tensor([[0.7, 0.1, 0.4]]).argsort(dim=1)    # tensor([[1, 2, 0]])
    x = torch.tensor([[10, 11, 12], [20, 21, 22]])
    picked = x[torch.arange(2).unsqueeze(1), torch.tensor([[2, 0], [1, 2]])]    # tensor([[12, 10], [21, 22]])


    # TODO: Hide ratio of the patches of each image, at random. patches has shape
    #   (batch, count, size). Draw torch.rand(batch, count, generator=generator),
    #   and keep the int(count * (1 - ratio)) patches whose draws sort first, in
    #   that sorted order. Return the visible patches, of shape (batch, kept,
    #   size), their positions, of shape (batch, kept), and a tensor of True
    #   and False of shape (batch, count), True where a patch is hidden.
    def random_masking(
        patches: Tensor,
        ratio: float,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ...


    # TODO: Return the decoder's input, of shape (batch, count, size), the visible
    #   patches put back at their positions and mask_token, of shape (size,), at
    #   every hidden one.
    def fill_in(
        visible: Tensor,
        keep: Tensor,
        count: int,
        mask_token: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the mean squared error over the hidden patches alone: each
    #   patch's mean squared error over its numbers, averaged over the patches
    #   where hidden is True.
    def masked_mse(
        predicted: Tensor,
        patches: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        ...
    ''',
)

MASKED2_SOLUTION = source_file(
    '''
    # masked2.py, a solution for comparison

    import torch
    from torch import Tensor

    order = torch.tensor([[0.7, 0.1, 0.4]]).argsort(dim=1)
    x = torch.tensor([[10, 11, 12], [20, 21, 22]])
    picked = x[torch.arange(2).unsqueeze(1), torch.tensor([[2, 0], [1, 2]])]


''' + indented(MAE_PARTS) + '''
    ''',
)


def expect_masking(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor, Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 3:
        raise Mismatch(f"{call} should give the visible patches, their positions and the hidden mask, in a tuple.")

    for name, part, wanted in zip(("visible patches", "positions", "hidden mask"), actual, expected):
        expect_tensor(
            call=f"the {name} of {call}",
            actual=part,
            expected=wanted,
        )

    if actual[1].dtype.is_floating_point:
        raise Mismatch(f"The positions of {call} hold {actual[1].dtype} numbers, but positions pick patches and must be whole numbers.")

    if actual[2].dtype != torch.bool:
        raise Mismatch(f"The hidden mask of {call} should hold True and False, not {actual[2].dtype}.")


def visible_patches(
    seed: int,
    batch: int,
    count: int,
    size: int,
    ratio: float,
) -> tuple[Tensor, Tensor]:
    visible, keep, _ = MAE_REFERENCE["random_masking"](small_tensor(seed, batch, count, size), ratio, torch.Generator().manual_seed(seed + 1))

    return visible, keep


def hidden_patches(
    seed: int,
    batch: int,
    count: int,
    size: int,
    ratio: float,
) -> Tensor:
    _, _, hidden = MAE_REFERENCE["random_masking"](small_tensor(seed, batch, count, size), ratio, torch.Generator().manual_seed(seed + 1))

    return hidden


def check_masked2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="random_masking",
        description="keeps the patches whose draws sort first",
        inputs=(
            (small_tensor(640, 2, 8, 3), 0.75, seeded(641)),
            (small_tensor(642, 3, 10, 2), 0.4, seeded(643)),
        ),
        reference=MAE_REFERENCE["random_masking"],
        expect=expect_masking,
    )
    check_on_inputs(
        checker=checker,
        function="fill_in",
        description="puts the visible patches back and the mask token everywhere else",
        inputs=(
            (*visible_patches(644, 2, 8, 3, 0.75), 8, torch.tensor([9.0, -9.0, 0.5])),
            (*visible_patches(650, 3, 10, 2, 0.4), 10, torch.tensor([-2.0, 7.0])),
        ),
        reference=MAE_REFERENCE["fill_in"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="masked_mse",
        description="averages the error over the hidden patches alone",
        inputs=(
            (small_tensor(648, 3, 6, 4), small_tensor(649, 3, 6, 4), Described(hidden_patches(646, 3, 6, 4, 0.5), "half of each image's patches hidden")),
            (small_tensor(654, 2, 8, 3), small_tensor(655, 2, 8, 3), Described(hidden_patches(652, 2, 8, 3, 0.75), "three quarters of each image's patches hidden")),
        ),
        reference=MAE_REFERENCE["masked_mse"],
        expect=expect_tensor,
    )


MASKED2 = Exercise(
    name="masked2",
    chapter=MASKED_MODELING,
    title="MAE's masking, and why so much",
    lesson=MASKED2_LESSON,
    source=MASKED2_SOURCE,
    solution=MASKED2_SOLUTION,
    hints={
        "random_masking": "keep = the draws' argsort over dim=1, cut to the first kept. rows = torch.arange(batch)."
        "unsqueeze(1) picks each row's patches with patches[rows, keep]. Start hidden all True and set it False at "
        "[rows, keep].",
        "fill_in": "mask_token.expand(batch, count, size).clone() gives a tensor you can write into. Put visible at "
        "[rows, keep] the same way random_masking picked them.",
        "masked_mse": "((predicted - patches) ** 2).mean(dim=-1) is one error per patch. Pick the hidden ones with "
        "[hidden] and take their mean.",
    },
    introduces=("linear probe",),
    check=check_masked2,
)
