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
from ..training import best_device
from .classic_nets import ReferenceLeNet, mnist_images
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
        class token. Then it trains a small ViT and LeNet on MNIST to compare
        how much data each one needs.

        ## Further reading

        - [Dosovitskiy and colleagues, an image is worth 16x16 words](https://arxiv.org/abs/2010.11929)
        - [Steiner and colleagues, how to train your ViT](https://arxiv.org/abs/2106.10270)
        - [Beyer, Zhai and Kolesnikov, better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580)
        - [Tolstikhin and colleagues, MLP-Mixer, whose section 3.4 shuffles the patches and pixels](https://arxiv.org/abs/2105.01601)
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
    patches and treating each patch as a token. The paper's title puts it this
    way, an image is worth 16x16 words.

    Cutting a 28 by 28 MNIST digit into 7 by 7 patches gives 16 patches, 4
    across and 4 down. Each patch is flattened into 49 numbers, one per
    pixel, row by row. Here is a 4 by 4 image cut into 2 by 2 patches.

    ```python
    grid = torch.arange(16.0).reshape(1, 1, 4, 4)
    cells = grid.reshape(1, 1, 2, 2, 2, 2)
    cells.permute(0, 2, 4, 1, 3, 5).reshape(1, 4, 4)[0, 1]    # tensor([2., 3., 6., 7.]), the top right patch
    ```

    The first `.reshape` splits the height into two dimensions, which row of
    patches and which row inside the patch. It splits the width the same way.
    `.permute` moves the two dimensions that pick the patch to the front. So
    the last `.reshape` gives one row per patch.

    One Linear layer embeds every patch, and all the patches share that same
    Linear. A Conv2d whose kernel size and stride both equal the patch size
    computes exactly the same sums. Real code uses the Conv2d, because one
    call cuts out the patches and embeds them at once.

    The Linear's weight has one row per output feature. Each row holds one
    weight per number of the patch, in patchify's order, which is by channel,
    then row, then column. So `.reshape(size, channels, patch, patch)` turns
    each row into one kernel of that Conv2d.

    Two more pieces complete the input to the transformer blocks.

    - The **class token** is a learned vector in front of the patch tokens. It belongs to no patch, and the head reads only its output.
    - A learned position vector is added to every token, including the class token. Attention ignores order, as position1 showed.

    Every image uses the same class token, so the token is repeated along the
    batch dimension. `.expand(2, -1, -1)` turns a dimension of size 1 into 2
    copies without copying any memory. Each -1 keeps that dimension's size
    unchanged.

    After the transformer blocks, the output at the class token holds what
    the head needs, gathered from every patch. The paper took this idea from
    BERT's class token. It also found that position vectors that know the 2D
    grid did no better than a plain numbered list, like the 17 positions here.
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


    # TODO: Cut images, of shape (batch, channels, height, width), into
    #   squares of patch by patch pixels. Return them as a tensor of shape
    #   (batch, patches, channels * patch * patch), with one row per patch.
    #   The patches go in order row by row across the image. Each patch is
    #   flattened by channel, then row, then column, as in the lesson.
    #   Example: patchify(grid, 2)[0, 1] for the grid above is
    #   tensor([2., 3., 6., 7.]).
    def patchify(
        images: Tensor,
        patch: int,
    ) -> Tensor:
        ...


    # TODO: Finish forward. Cut images into patches and embed them with
    #   self.project. Put the class token in front of each image's patch
    #   tokens, then add self.positions. The result has shape
    #   (batch, patches + 1, size).
''' + indented(PATCH_EMBEDDING_INIT) + '''

        def forward(self, images: Tensor) -> Tensor:
            ...


    # TODO: Return an nn.Conv2d that embeds every patch exactly as project
    #   does. project is an nn.Linear from channels * patch * patch to size.
    #   The Conv2d's kernel size and stride are both patch, and it holds the
    #   weights and bias of project.
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


    # Each row of the Linear's weight holds the patch's numbers in patchify's order, so it reshapes into a kernel.
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
            raise Mismatch(f"{call} should return an nn.Conv2d.")

        if conv.kernel_size != (patch, patch) or conv.stride != (patch, patch):
            raise Mismatch(f"{call} should return a Conv2d with a kernel size and a stride of {patch}.")

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
        description="cuts images into patches, one row per patch",
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
        "PatchEmbedding.forward": "Call patchify with self.patch and pass the result through self.project. "
        "self.class_token.expand(len(images), -1, -1) gives one class token per image. Put it in front with "
        "torch.cat on dim=1, then add self.positions.",
        "conv_embedding": "Build nn.Conv2d(channels, size, patch, stride=patch), taking size from "
        "project.out_features. Inside torch.no_grad(), copy project.weight reshaped to (size, channels, patch, "
        "patch) into conv.weight, and copy project.bias into conv.bias.",
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
    A ViT is a stack of transformer blocks that runs over the patch tokens.
    Its prediction is read from the output at the class token. The ViT here
    is small, with size 64 and 2 blocks of 4 heads, which is 72,074 weights in
    total. It is compared with LeNet from lenet1, which has 61,706 weights.

    PyTorch's `nn.TransformerEncoderLayer` is the block from block2, without
    the causal mask. `norm_first=True` gives the pre norm order. By default
    the layer uses a ReLU and a dropout of 0.1, so the ViT here sets GELU and
    no dropout instead.

    ```python
    block = nn.TransformerEncoderLayer(16, 2, 32, batch_first=True, norm_first=True)
    block(torch.zeros(3, 5, 16)).shape    # torch.Size([3, 5, 16]), the same shape as the input
    ```

    A convolution builds in two assumptions. The first is that nearby pixels
    belong together, which is called **locality**. The second is that a
    pattern means the same thing anywhere in the image, which is called
    **translation equivariance**. The assumptions built into a layer are its
    **inductive bias**. A ViT has almost none. Only the cutting into patches
    keeps neighbouring pixels together. Attention lets any patch use any other
    patch, starting from the first block.

    Both models were trained for 3,000 steps of AdamW on batches of 128,
    using the first N training images. Then each model was scored on the
    10,000 validation images. On normal digits, where LeNet's assumptions
    hold, LeNet had the higher accuracy at every training size, on one seed.
    At 500 images it scored 89.8% against the ViT's 76.2%. At 50,000 it
    scored 98.6% against 97.2%.

    The authors of MLP-Mixer tested these assumptions by breaking them.
    Shuffle every image in the same fixed way, both the order of the 16
    patches and the order of the 49 pixels inside each patch. For the ViT, a
    fixed shuffle of a patch's pixels only reorders the weights its patch
    Linear needs. A fixed shuffle of the patches only reorders the learned
    positions. So the ViT faces the same task as before. For LeNet, pixels
    that were neighbours are no longer neighbours. Here are three seeds of
    each model, all run on this Mac's GPU.

    | Shuffled, training images | 500 | 2,000 | 10,000 | 50,000 |
    |---|---|---|---|---|
    | LeNet | 79.8% to 81.1% | 89.4% to 90.2% | 94.1% to 94.3% | 95.6% to 96.0% |
    | ViT | 73.5% to 77.8% | 87.3% to 89.4% | 94.3% to 96.0% | 96.4% to 97.1% |

    LeNet still leads at 2,000 images and below, on every seed. The ViT
    scores about what it scored on the normal digits. From 10,000 images up,
    the ViT won on every seed. It won by 0.1 to 1.7 points at 10,000 images,
    and by 0.6 to 1.3 points at 50,000.

    This is the trade the ViT paper measured. A model with little inductive
    bias needs more data, and with enough data "large scale training trumps
    inductive bias". The paper worked on photos, where the CNN's assumptions
    held. There, ViT-B/32 needed 90 million JFT images to beat ResNet50.
    MNIST has only 50,000 training images, so here the shuffle makes LeNet's
    assumptions wrong instead.
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


    # accuracy, measured as in lenet1.
''' + indented(ACCURACY_PART) + '''


    # TODO: Finish forward. Embed images with self.embed, run the result
    #   through self.blocks, then apply the final LayerNorm, self.norm. Pass
    #   only the class token's output, at position 0, to self.head. Return the
    #   scores, of shape (batch, 10).
''' + indented(SMALL_VIT_INIT) + '''

        def forward(self, images: Tensor) -> Tensor:
            ...


    # TODO: Finish the line that sets loss, so this function trains the model
    #   with AdamW. The loss is the cross entropy between the model's scores on
    #   the batch of x and the matching labels in y.
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


    # TODO: Return a list with one validation accuracy for each size in sizes,
    #   in order. For each size, build a fresh model with make_model. Train it
    #   with train_steps on the first size images of x and their labels in y,
    #   passing steps and the same generator every time. Then measure its
    #   accuracy on validation_x and validation_y.
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
        raise Mismatch(f"data_curve should return a list of accuracies, one per size, but it returned {type(actual).__name__}.")

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
            "correctly, but it should get more than 55% right.",
        )


def check_vit2(checker: Checker) -> None:
    checker.test_method(
        class_name="SmallViT",
        method="forward",
        description="embeds the images, runs the blocks and scores the class token",
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
        "SmallViT.forward": "Run self.embed, then self.blocks, then self.norm. The output at position 0 belongs to "
        "the class token, and only that output goes to self.head.",
        "train_steps": "Call F.cross_entropy with the model's scores on x[batch] and the labels y[batch].",
        "data_curve": "For each size, build a fresh model with make_model. Train it with train_steps on the first size "
        "rows of x and y, for steps, with generator. Then append accuracy(model, validation_x, validation_y).",
    },
    introduces=(),
    check=check_vit2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind vit2's tables
# ---------------------------------------------------------------------------


def shuffled(
    images: Tensor,
    seed: int = 123,
) -> Tensor:
    """One fixed shuffle of the 16 patches of 7 by 7 and of the 49 pixels inside each, the same for every image,
    as MLP-Mixer's section 3.4 shuffled its inputs."""
    generator = torch.Generator().manual_seed(seed)
    patch_order = torch.randperm(16, generator=generator)
    pixel_order = torch.randperm(49, generator=generator)
    count = images.shape[0]
    cells = images.reshape(count, 1, 4, 7, 4, 7).permute(0, 2, 4, 1, 3, 5).reshape(count, 16, 49)
    cells = cells[:, patch_order][:, :, pixel_order]

    return cells.reshape(count, 4, 4, 1, 7, 7).permute(0, 3, 1, 4, 2, 5).reshape(count, 1, 28, 28)


def measure_data_appetite(
    shuffle: bool,
    sizes: tuple[int, ...] = (500, 2000, 10000, 50000),
    seeds: tuple[int, ...] = (0, 1, 2),
    device: torch.device | None = None,
) -> dict[tuple[str, int], list[float]]:
    """vit2's numbers: LeNet and the SmallViT trained 3,000 steps of AdamW at 1e-3 on batches of 128, on the
    first size training images, scored on the 10,000 validation images. With shuffle, every image goes through
    the same shuffle first. The lesson's table is shuffle=True on seeds 0 to 2, and its plain line is
    shuffle=False on seed 0, both run on this Mac's GPU through MPS."""
    device = device or best_device()
    images = mnist_images()
    x, y = images["train_x"], images["train_y"]
    validation_x, validation_y = images["validation_x"], images["validation_y"]

    if shuffle:
        x, validation_x = shuffled(x), shuffled(validation_x)

    x, y = x.to(device), y.to(device)
    validation_x, validation_y = validation_x.to(device), validation_y.to(device)
    makers: dict[str, Callable[[], nn.Module]] = {
        "LeNet": ReferenceLeNet,
        "ViT": lambda: VIT2_REFERENCE["SmallViT"](),
    }
    results: dict[tuple[str, int], list[float]] = {}

    for seed in seeds:
        for size in sizes:
            for name, make in makers.items():
                torch.manual_seed(seed)
                model = make().to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                generator = torch.Generator().manual_seed(seed + 10)

                for _ in range(3000):
                    batch = torch.randint(0, size, (128,), generator=generator).to(device)
                    loss = F.cross_entropy(model(x[batch]), y[batch])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()

                with torch.no_grad():
                    right = sum(
                        (model(validation_x[start : start + 2000]).argmax(dim=1) == validation_y[start : start + 2000]).sum().item()
                        for start in range(0, len(validation_x), 2000)
                    )

                results.setdefault((name, size), []).append(right / len(validation_x))

    return results
