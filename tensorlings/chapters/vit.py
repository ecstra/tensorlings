"""57_vit, an image read as a sequence of patches with a class token, and how much data a ViT needs next to a CNN."""

from __future__ import annotations

import textwrap
from collections.abc import Callable

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
    expect_value,
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from .classic_nets import mnist_images
from .linear_layers import expect_module

VIT = Chapter(
    folder="57_vit",
    title="Vision transformers",
    readme=source_file(
        """
        # Vision transformers

        A transformer reads a sequence of tokens. The Vision Transformer, ViT,
        reads an image the same way, by cutting it into patches and treating
        each patch as a token. This chapter builds the patch embedding and the
        class token, then trains a small ViT on MNIST next to LeNet to see how
        much data each needs.

        ## Further reading

        - [Dosovitskiy and colleagues, an image is worth 16x16 words](https://arxiv.org/abs/2010.11929)
        - [Steiner and colleagues, how to train your ViT](https://arxiv.org/abs/2106.10270)
        - [Beyer, Zhai and Kolesnikov, better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


# vit1's pieces, which vit2 is handed, written once so the two files never drift apart.
PATCHIFY_SOLUTION = '''
def patchify(
    images: Tensor,
    patch: int,
) -> Tensor:
    batch, channels, height, width = images.shape
    cells = images.reshape(batch, channels, height // patch, patch, width // patch, patch)

    return cells.permute(0, 2, 4, 1, 3, 5).reshape(batch, -1, channels * patch * patch)
'''

PATCH_EMBEDDING_INIT = '''
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        channels: int,
        patch: int,
        size: int,
        patches: int,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.project = nn.Linear(channels * patch * patch, size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, size))
        self.positions = nn.Parameter(torch.randn(1, patches + 1, size) * 0.02)
'''

PATCH_EMBEDDING_FORWARD = '''
    def forward(self, images: Tensor) -> Tensor:
        tokens = self.project(patchify(images, self.patch))
        first = self.class_token.expand(len(images), -1, -1)

        return torch.cat([first, tokens], dim=1) + self.positions
'''


def reference_patchify(
    images: Tensor,
    patch: int,
) -> Tensor:
    """F.unfold cuts out every patch as a column, channel by channel and row by row, which is the order
    patchify is asked for, so the check holds it to PyTorch's own cutting."""
    return F.unfold(images, kernel_size=patch, stride=patch).transpose(1, 2)


def reference_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "Tensor": Tensor,
    }
    exec(PATCHIFY_SOLUTION + PATCH_EMBEDDING_INIT + PATCH_EMBEDDING_FORWARD, namespace)

    return namespace


REFERENCE = reference_namespace()


# ---------------------------------------------------------------------------
# vit1
# ---------------------------------------------------------------------------

VIT1_LESSON = prose(
    """
    A transformer reads a sequence of tokens. The Vision Transformer of
    Dosovitskiy and colleagues reads an image the same way, by cutting it into
    patches and treating each patch as a token. Its paper's title says so, an
    image is worth 16x16 words.

    A 28 by 28 MNIST digit cut into 7 by 7 patches gives 16 patches, 4 across
    and 4 down. Each patch flattens to 49 numbers, one per pixel, row by row.
    Here is a 4 by 4 image cut into patches of 2.

    ```python
    grid = torch.arange(16.0).reshape(1, 1, 4, 4)
    cells = grid.reshape(1, 1, 2, 2, 2, 2)
    cells.permute(0, 2, 4, 1, 3, 5).reshape(1, 4, 4)[0, 1]    # tensor([2., 3., 6., 7.]), the top right patch
    ```

    The first `.reshape` splits each side into which patch and where inside
    it. `.permute` moves the two patch dimensions in front, so the last
    `.reshape` gives one row per patch.

    One Linear embeds every patch, the same Linear for all of them. A Conv2d
    whose kernel and stride both equal the patch size does exactly the same
    sums. Real code uses it, since one call cuts and embeds at once.

    Two more pieces finish the input.

    - The **class token** is a learned vector put in front of the patches. It belongs to no patch, so the head reads its output alone.
    - A learned position vector is added to every token, the class token too, since attention ignores order, as position1 showed.

    After the blocks, the class token has gathered from every patch what the
    head needs. The paper took the idea from BERT's class token. It also found
    that positions aware of the 2D grid did no better than a plain list of 17.
    """,
)

VIT1_SOURCE = source_file(
    '''
    # vit1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    grid = torch.arange(16.0).reshape(1, 1, 4, 4)
    cells = grid.reshape(1, 1, 2, 2, 2, 2)
    top_right = cells.permute(0, 2, 4, 1, 3, 5).reshape(1, 4, 4)[0, 1]    # tensor([2., 3., 6., 7.])

    first = torch.zeros(1, 1, 3).expand(2, -1, -1)    # shape (2, 1, 3), one copy per image


    # TODO: Return the patches of images, of shape (batch, channels, height,
    #   width), as rows of shape (batch, patches, channels * patch * patch).
    #   The patches go row by row, and each one flattens channel by channel,
    #   then row by row, as in the lesson.
    def patchify(
        images: Tensor,
        patch: int,
    ) -> Tensor:
        ...


    # TODO: Finish forward. Embed the patches of images with self.project, put
    #   the class token in front of each image's patches, and add the
    #   positions. The result has shape (batch, patches + 1, size).
''' + indented(PATCH_EMBEDDING_INIT) + '''

        def forward(self, images: Tensor) -> Tensor:
            ...


    # TODO: Return an nn.Conv2d whose kernel and stride are both patch, holding
    #   the weights and bias of project, so it embeds every patch as project
    #   does. project is an nn.Linear from channels * patch * patch to size.
    def conv_embedding(
        project: nn.Linear,
        channels: int,
        patch: int,
    ) -> nn.Conv2d:
        ...
    ''',
)

VIT1_SOLUTION = source_file(
    '''
    # vit1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    grid = torch.arange(16.0).reshape(1, 1, 4, 4)
    cells = grid.reshape(1, 1, 2, 2, 2, 2)
    top_right = cells.permute(0, 2, 4, 1, 3, 5).reshape(1, 4, 4)[0, 1]

    first = torch.zeros(1, 1, 3).expand(2, -1, -1)


''' + indented(PATCHIFY_SOLUTION) + '''


''' + indented(PATCH_EMBEDDING_INIT) + '''

''' + indented(PATCH_EMBEDDING_FORWARD) + '''


    # The Linear's weight row for one output is the patch's 49 numbers in order, so it folds back into a kernel.
    def conv_embedding(
        project: nn.Linear,
        channels: int,
        patch: int,
    ) -> nn.Conv2d:
        size = project.out_features
        conv = nn.Conv2d(channels, size, patch, stride=patch)

        with torch.no_grad():
            conv.weight.copy_(project.weight.reshape(size, channels, patch, patch))
            conv.bias.copy_(project.bias)

        return conv
    ''',
)


def seeded_embedding(make: type) -> nn.Module:
    """A patch embedding of 3 channels, patches of 2 and size 5 for images of 4 by 6, with a class token
    that isn't zero, so leaving it out shows."""
    torch.manual_seed(570)
    embedding = make(3, 2, 5, 6)

    with torch.no_grad():
        embedding.class_token.copy_(torch.randn(1, 1, 5))

    return embedding


def check_patch_embedding(found: type) -> None:
    model = expect_module("PatchEmbedding(3, 2, 5, 6)", seeded_embedding(found))
    images = small_tensor(571, 2, 3, 4, 6)

    with torch.no_grad():
        expected = seeded_embedding(REFERENCE["PatchEmbedding"])(images)
        actual = model(images)

    expect_tensor(
        call="PatchEmbedding(3, 2, 5, 6) on 2 images of 3 channels, 4 by 6",
        actual=actual,
        expected=expected,
    )


def check_conv_embedding(found: object) -> None:
    for channels, patch, size in ((1, 7, 4), (3, 2, 5)):
        torch.manual_seed(572 + patch)
        project = nn.Linear(channels * patch * patch, size)
        call = f"conv_embedding(a Linear from {channels * patch * patch} to {size}, {channels}, {patch})"
        conv = found(project, channels, patch)

        if not isinstance(conv, nn.Conv2d):
            raise Mismatch(f"{call} should be an nn.Conv2d.")

        if conv.kernel_size != (patch, patch) or conv.stride != (patch, patch):
            raise Mismatch(f"{call} should have a kernel and a stride of {patch}.")

        images = torch.randn(2, channels, 4 * patch, 2 * patch, generator=torch.Generator().manual_seed(574))

        with torch.no_grad():
            expected = project(reference_patchify(images, patch))
            actual = conv(images).flatten(2).transpose(1, 2)

        expect_tensor(
            call=f"{call} on 2 images, read back one row per patch",
            actual=actual,
            expected=expected,
        )


def check_vit1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="patchify",
        description="cuts images into rows of patches",
        inputs=(
            (Described(torch.arange(16.0).reshape(1, 1, 4, 4), "one 4 by 4 image of 0 to 15"), 2),
            (small_tensor(575, 2, 3, 4, 6), 2),
            (small_tensor(576, 1, 2, 6, 3), 3),
        ),
        reference=reference_patchify,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="PatchEmbedding",
        method="forward",
        description="embeds the patches, puts the class token first and adds the positions",
        run=check_patch_embedding,
    )
    checker.test(
        function="conv_embedding",
        description="embeds every patch as the Linear does, in one convolution",
        run=check_conv_embedding,
    )


VIT1 = Exercise(
    name="vit1",
    chapter=VIT,
    title="An image as a sequence of patches",
    lesson=VIT1_LESSON,
    source=VIT1_SOURCE,
    solution=VIT1_SOLUTION,
    hints={
        "patchify": "Reshape to (batch, channels, height // patch, patch, width // patch, patch). Then "
        ".permute(0, 2, 4, 1, 3, 5), and reshape to (batch, -1, channels * patch * patch).",
        "PatchEmbedding.forward": "self.project(patchify(images, self.patch)) embeds the patches. "
        "self.class_token.expand(len(images), -1, -1) gives one class token per image to torch.cat in front on "
        "dim=1. Then add self.positions.",
        "conv_embedding": "nn.Conv2d(channels, size, patch, stride=patch), with size from project.out_features. "
        "Inside torch.no_grad(), copy project.weight reshaped to (size, channels, patch, patch) into conv.weight, "
        "and project.bias into conv.bias.",
    },
    introduces=("class token",),
    check=check_vit1,
)


# ---------------------------------------------------------------------------
# vit2
# ---------------------------------------------------------------------------

SMALL_VIT_INIT = '''
class SmallViT(nn.Module):
    def __init__(
        self,
        size: int = 64,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.embed = PatchEmbedding(1, 7, size, 16)
        self.blocks = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    size,
                    heads,
                    2 * size,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ],
        )
        self.norm = nn.LayerNorm(size)
        self.head = nn.Linear(size, 10)
'''

SMALL_VIT_FORWARD = '''
    def forward(self, images: Tensor) -> Tensor:
        outputs = self.norm(self.blocks(self.embed(images)))

        return self.head(outputs[:, 0])
'''

ACCURACY_PART = '''
def accuracy(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> float:
    model.eval()

    with torch.no_grad():
        right = (model(x).argmax(dim=1) == y).float().mean().item()

    model.train()

    return right
'''

TRAIN_STEPS_SOLUTION = '''
def train_steps(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    steps: int,
    generator: torch.Generator,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(steps):
        batch = torch.randint(0, len(x), (128,), generator=generator)
        loss = F.cross_entropy(model(x[batch]), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
'''

DATA_CURVE_SOLUTION = '''
def data_curve(
    make_model: Callable[[], nn.Module],
    x: Tensor,
    y: Tensor,
    validation_x: Tensor,
    validation_y: Tensor,
    sizes: list[int],
    steps: int,
    generator: torch.Generator,
) -> list[float]:
    scores = []

    for size in sizes:
        model = make_model()
        train_steps(model, x[:size], y[:size], steps, generator)
        scores.append(accuracy(model, validation_x, validation_y))

    return scores
'''

VIT2_LESSON = prose(
    """
    A ViT is a stack of transformer blocks over patches, read out at the class
    token. The one here is small, of size 64 with 2 blocks of 4 heads, 72,074
    weights in all. Next to it is LeNet from lenet1, with 61,706.

    PyTorch's `nn.TransformerEncoderLayer` is block2's block without the mask.
    `norm_first=True` gives the pre norm order, and dropout is off here.

    ```python
    block = nn.TransformerEncoderLayer(16, 2, 32, batch_first=True, norm_first=True)
    block(torch.zeros(3, 5, 16)).shape    # torch.Size([3, 5, 16]), the shape it was given
    ```

    Both models trained the same way, 3,000 steps of AdamW on batches of 128,
    on the first N training images. Here is their accuracy on the 10,000
    validation images.

    | Training images | 500 | 2,000 | 10,000 | 50,000 |
    |---|---|---|---|---|
    | LeNet | 89.9% | 94.1% | 97.0% | 98.4% |
    | ViT | 75.7% | 87.9% | 95.5% | 96.9% |

    Both fit their own training images, 98% or more at every size. The
    difference is what each makes of images it never saw. The gap shrinks from
    14.2 points at 500 images to 1.5 at 10,000, and stays there at 50,000. At
    MNIST's size LeNet never loses.

    A convolution builds in two assumptions. Nearby pixels go together, and a
    pattern means the same thing anywhere in the image. A ViT has neither,
    beyond cutting patches. Attention lets any patch use any other from the
    first block on. So the ViT has to learn from examples what a convolution
    is given, and that takes examples.

    The ViT paper found the same at scale. Trained on ImageNet's 1.3 million
    images, ViTs came out a few points below ResNets of their size. Then they
    were pre-trained on JFT's 303 million images. There its largest ViT,
    ViT-H/14, beat a ResNet given the same images on every task of Table 2.
    """,
)

VIT2_SOURCE = source_file(
    '''
    # vit2.py

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    block = nn.TransformerEncoderLayer(16, 2, 32, batch_first=True, norm_first=True)
    same_shape = block(torch.zeros(3, 5, 16)).shape    # torch.Size([3, 5, 16])


    # patchify and PatchEmbedding from vit1.
''' + indented(PATCHIFY_SOLUTION) + '''


''' + indented(PATCH_EMBEDDING_INIT) + '''

''' + indented(PATCH_EMBEDDING_FORWARD) + '''


    # accuracy, as lenet1 measured it.
''' + indented(ACCURACY_PART) + '''


    # TODO: Finish forward. Embed images, run the blocks, take the final
    #   LayerNorm, and return the head's scores from the class token's output
    #   alone, of shape (batch, 10).
''' + indented(SMALL_VIT_INIT) + '''

        def forward(self, images: Tensor) -> Tensor:
            ...


    # TODO: Finish the line that sets loss, the cross entropy of model's scores
    #   on the batch of x against its labels, so this trains with AdamW.
    def train_steps(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> float:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for _ in range(steps):
            batch = torch.randint(0, len(x), (128,), generator=generator)
            loss = ...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()


    # TODO: Return the validation accuracy of a fresh model from make_model for
    #   each size in sizes, in order. Train each one with train_steps for steps
    #   on the first size images of x, all with the one generator.
    def data_curve(
        make_model: Callable[[], nn.Module],
        x: Tensor,
        y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        sizes: list[int],
        steps: int,
        generator: torch.Generator,
    ) -> list[float]:
        ...
    ''',
)

VIT2_SOLUTION = source_file(
    '''
    # vit2.py, a solution for comparison

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    block = nn.TransformerEncoderLayer(16, 2, 32, batch_first=True, norm_first=True)
    same_shape = block(torch.zeros(3, 5, 16)).shape


''' + indented(PATCHIFY_SOLUTION) + '''


''' + indented(PATCH_EMBEDDING_INIT) + '''

''' + indented(PATCH_EMBEDDING_FORWARD) + '''


''' + indented(ACCURACY_PART) + '''


''' + indented(SMALL_VIT_INIT) + '''

''' + indented(SMALL_VIT_FORWARD) + '''


''' + indented(TRAIN_STEPS_SOLUTION) + '''


''' + indented(DATA_CURVE_SOLUTION) + '''
    ''',
)


def vit2_namespace() -> dict[str, object]:
    namespace = dict(REFERENCE)
    namespace["F"] = F
    namespace["Callable"] = Callable
    exec(ACCURACY_PART + SMALL_VIT_INIT + SMALL_VIT_FORWARD + TRAIN_STEPS_SOLUTION + DATA_CURVE_SOLUTION, namespace)

    return namespace


VIT2_REFERENCE = vit2_namespace()


def seeded_vit(make: type) -> nn.Module:
    torch.manual_seed(577)

    return make(32, 1, 2)


def tiny_classifier() -> nn.Module:
    torch.manual_seed(578)

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 10),
    )


def mnist_slice(
    name: str,
    count: int,
) -> Tensor:
    return mnist_images()[name][:count].clone()


def check_small_vit(found: type) -> None:
    model = expect_module("SmallViT(32, 1, 2)", seeded_vit(found))
    images = mnist_slice("validation_x", 4)
    model.eval()
    reference = seeded_vit(VIT2_REFERENCE["SmallViT"]).eval()

    with torch.no_grad():
        expected = reference(images)
        actual = model(images)

    expect_tensor(
        call="SmallViT(32, 1, 2) on 4 MNIST images",
        actual=actual,
        expected=expected,
    )


def check_train_steps(found: object) -> None:
    x, y = mnist_slice("train_x", 300), mnist_slice("train_y", 300)
    model = tiny_classifier()
    reference = tiny_classifier()
    loss = found(model, x, y, 5, torch.Generator().manual_seed(579))
    expected_loss = VIT2_REFERENCE["train_steps"](reference, x, y, 5, torch.Generator().manual_seed(579))

    expect_close(
        call="train_steps for 5 steps on 300 images",
        actual=loss,
        expected=expected_loss,
        tolerance=1e-5,
    )
    expect_tensor(
        call="the weights after train_steps",
        actual=model[1].weight.detach(),
        expected=reference[1].weight.detach(),
    )


def check_data_curve(found: object) -> None:
    x, y = mnist_slice("train_x", 400), mnist_slice("train_y", 400)
    validation_x, validation_y = mnist_slice("validation_x", 200), mnist_slice("validation_y", 200)
    arguments = (tiny_classifier, x, y, validation_x, validation_y, [50, 400], 4)
    actual = found(*arguments, torch.Generator().manual_seed(580))
    expected = VIT2_REFERENCE["data_curve"](*arguments, torch.Generator().manual_seed(580))

    if not isinstance(actual, list):
        raise Mismatch(f"data_curve should give a list of accuracies, one per size, but it gave {type(actual).__name__}.")

    expect_value(
        call="data_curve of a small classifier at 50 and 400 images, 4 steps each",
        actual=[round(score, 6) for score in actual],
        expected=[round(score, 6) for score in expected],
    )


def check_vit_learns(found: type) -> None:
    model = seeded_vit(found)
    VIT2_REFERENCE["train_steps"](
        model,
        mnist_slice("train_x", 2000),
        mnist_slice("train_y", 2000),
        150,
        torch.Generator().manual_seed(1),
    )
    right = VIT2_REFERENCE["accuracy"](model, mnist_slice("validation_x", 2000), mnist_slice("validation_y", 2000))

    if right < 0.55:
        raise Mismatch(
            f"SmallViT(32, 1, 2), trained for 150 steps on 2,000 images, labels {right:.1%} of 2,000 validation images "
            "right, but it should pass 55%.",
        )


def check_vit2(checker: Checker) -> None:
    checker.test_method(
        class_name="SmallViT",
        method="forward",
        description="embeds, runs the blocks and scores the class token",
        run=check_small_vit,
    )
    checker.test(
        function="train_steps",
        description="trains with AdamW on the cross entropy",
        run=check_train_steps,
    )
    checker.test(
        function="data_curve",
        description="measures a fresh model at each training size",
        run=check_data_curve,
    )
    checker.test_method(
        class_name="SmallViT",
        method="forward",
        description="learns MNIST digits from 2,000 images",
        run=check_vit_learns,
    )


VIT2 = Exercise(
    name="vit2",
    chapter=VIT,
    title="How much data a ViT needs",
    lesson=VIT2_LESSON,
    source=VIT2_SOURCE,
    solution=VIT2_SOLUTION,
    hints={
        "SmallViT.forward": "self.blocks(self.embed(images)), then self.norm. Its output at position 0 is the class "
        "token's, which goes to self.head.",
        "train_steps": "F.cross_entropy(model(x[batch]), y[batch]).",
        "data_curve": "For each size, model = make_model(), then train_steps(model, x[:size], y[:size], steps, "
        "generator), then append accuracy(model, validation_x, validation_y).",
    },
    introduces=(),
    check=check_vit2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)
