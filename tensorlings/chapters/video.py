"""97_video, frames over time turned into tokens: ViViT's tubelets and the 3D convolution they match, then a task
only motion answers, where a model of single frames stays at chance and one that reads time solves it."""

from __future__ import annotations

import contextlib
import functools
import math
import textwrap
from collections.abc import Callable, Iterator

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
    reject_missing_return,
    seeded,
    small_tensor,
    value_text,
)
from ..text import prose, source_file

VIDEO = Chapter(
    folder="97_video",
    title="Video as tokens",
    readme=source_file(
        """
        # Video as tokens

        A video is a sequence of frames over time. This chapter cuts a video
        into tubelets, as ViViT does. A tubelet is a small box of pixels that
        spans a few frames. The chapter shows that embedding tubelets is the
        same as a 3D convolution. Then a task that only motion can answer
        shows why a token has to see more than one frame.

        ## Further reading

        - [Arnab and colleagues, ViViT, a video vision transformer](https://arxiv.org/abs/2103.15691)
        - [Dosovitskiy and colleagues, an image is worth 16x16 words](https://arxiv.org/abs/2010.11929)
        - [PyTorch, Conv3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)
        """,
    ),
)

MEASURE_THREADS = 4

TUBELET_PART = '''
def tubelets(
    video: Tensor,
    t: int,
    h: int,
    w: int,
) -> Tensor:
    channels, frames, height, width = video.shape
    boxes = video.reshape(channels, frames // t, t, height // h, h, width // w, w)

    return boxes.permute(1, 3, 5, 0, 2, 4, 6).reshape(-1, channels * t * h * w)


def tubelet_embedding(
    videos: Tensor,
    conv: nn.Conv3d,
) -> Tensor:
    return conv(videos).flatten(2).transpose(1, 2)


def as_linear(
    video: Tensor,
    conv: nn.Conv3d,
) -> Tensor:
    return tubelets(video, *conv.kernel_size) @ conv.weight.flatten(1).T + conv.bias


def token_count(
    frames: int,
    height: int,
    width: int,
    t: int,
    h: int,
    w: int,
) -> int:
    return (frames // t) * (height // h) * (width // w)
'''

DOTS_PART = '''
FRAMES = 8
SIZE = 16


def moving_dots(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Clips of one dot moving a pixel a frame along its row, wrapping around. Label 1 moves right, 0 moves left."""
    videos = torch.zeros(count, 1, FRAMES, SIZE, SIZE)
    rows = torch.randint(0, SIZE, (count,), generator=generator)
    starts = torch.randint(0, SIZE, (count,), generator=generator)
    labels = torch.randint(0, 2, (count,), generator=generator)
    steps = 2 * labels - 1

    for frame in range(FRAMES):
        columns = (starts + steps * frame) % SIZE
        videos[torch.arange(count), 0, frame, rows, columns] = 1.0

    return videos, labels
'''

FRAME_INIT = '''
class FrameModel(nn.Module):
    def __init__(self, size: int = 16) -> None:
        super().__init__()
        self.embed = nn.Conv2d(1, size, 4, stride=4)
        self.head = nn.Linear(size, 2)
'''

FRAME_FORWARD = '''
    def forward(self, videos: Tensor) -> Tensor:
        batch, channels, frames, height, width = videos.shape
        images = videos.transpose(1, 2).reshape(batch * frames, channels, height, width)
        tokens = F.relu(self.embed(images)).mean(dim=(2, 3))

        return self.head(tokens.reshape(batch, frames, -1).mean(dim=1))
'''

TUBELET_INIT = '''
class TubeletModel(nn.Module):
    def __init__(self, size: int = 16) -> None:
        super().__init__()
        self.embed = nn.Conv3d(1, size, (2, 4, 4), stride=(2, 4, 4))
        self.head = nn.Linear(size, 2)
'''

TUBELET_FORWARD = '''
    def forward(self, videos: Tensor) -> Tensor:
        return self.head(F.relu(self.embed(videos)).mean(dim=(2, 3, 4)))
'''

MOTION_PART = '''
def reversal_gap(
    model: nn.Module,
    videos: Tensor,
) -> float:
    model.eval()

    with torch.no_grad():
        return (model(videos) - model(videos.flip(2))).abs().max().item()


def train(
    model: nn.Module,
    steps: int,
    lr: float,
    generator: torch.Generator,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        videos, labels = moving_dots(32, generator)
        loss = F.cross_entropy(model(videos), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def accuracy(
    model: nn.Module,
    videos: Tensor,
    labels: Tensor,
) -> float:
    model.eval()

    with torch.no_grad():
        return (model(videos).argmax(dim=1) == labels).float().mean().item()
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(code, namespace)

    return namespace


REFERENCE = namespace_of(TUBELET_PART + DOTS_PART + FRAME_INIT + FRAME_FORWARD + TUBELET_INIT + TUBELET_FORWARD + MOTION_PART)


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(MEASURE_THREADS)

    try:
        yield
    finally:
        torch.set_num_threads(threads)


def seeded_conv(
    seed: int,
    channels: int,
    size: int,
    kernel: tuple[int, int, int],
) -> nn.Conv3d:
    torch.manual_seed(seed)

    return nn.Conv3d(channels, size, kernel, stride=kernel)


# ---------------------------------------------------------------------------
# video1
# ---------------------------------------------------------------------------

VIDEO1_LESSON = prose(
    """
    A video is a sequence of frames over time, stored as a tensor of shape
    `(channels, frames, height, width)`. vit1 cut an image into patches and
    turned each patch into a token. ViViT, the video vision transformer,
    turns a video into tokens in two ways.

    - Uniform frame sampling cuts each frame into patches, as vit1 did. Then only the later layers can compare frames.
    - Tubelet embedding cuts **tubelets**, boxes t frames long, h pixels high and w pixels wide. So each token can contain motion.

    A video of T frames, each H by W pixels, holds
    `(T / t) * (H / h) * (W / w)` tubelets. To cut them out, a reshape
    splits each axis into two, the number of boxes along that axis and the
    size of one box. Then a permute moves the three box counts to the front.

    ```python
    x = torch.arange(8.0).reshape(1, 2, 2, 2)    # (channels, frames, height, width)
    x.reshape(1, 1, 2, 2, 1, 2, 1).shape         # torch.Size([1, 1, 2, 2, 1, 2, 1])
    ```

    Inside each tubelet, the numbers go in the order channel, frame, row,
    column. That is the same order as a 3D convolution's weights. So applying
    a Linear layer to each tubelet is the same as a 3D convolution whose
    kernel size and stride are both `(t, h, w)`. The ViViT paper points this
    out too, as vit1 showed that patches match a 2D convolution.

    ```python
    conv = nn.Conv3d(3, 8, kernel_size=(2, 4, 4), stride=(2, 4, 4))
    conv.weight.shape                  # torch.Size([8, 3, 2, 4, 4])
    conv.weight.flatten(1).shape       # torch.Size([8, 96]), one row per output channel
    ```

    How many tokens does a video make? ViViT's tubelets are 16 by 16 pixels
    and 2 frames long. Take two seconds of video at 25 frames a second, with
    frames of 224 by 224 pixels.

    | Tokens from | Count |
    |---|---|
    | every frame, patches of 16 by 16 | 9,800 |
    | tubelets of 2 frames | 4,900 |

    Putting two frames in each token halves the number of tokens. Attention
    scores every pair of tokens, so it has a quarter as many pairs to score.
    """,
)

VIDEO1_SOURCE = source_file(
    '''
    # video1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.arange(8.0).reshape(1, 2, 2, 2)
    split = x.reshape(1, 1, 2, 2, 1, 2, 1)                     # each axis split into a number of boxes and a box size
    moved = torch.zeros(2, 3, 4).permute(2, 0, 1).shape         # torch.Size([4, 2, 3])
    conv = nn.Conv3d(3, 8, kernel_size=(2, 4, 4), stride=(2, 4, 4))
    rows = conv.weight.flatten(1).shape                         # torch.Size([8, 96])


    # TODO: video has shape (channels, frames, height, width). Cut it into
    #   tubelets of t frames by h rows by w columns, and return one tubelet per
    #   row, of shape (number of tubelets, channels * t * h * w). Order the
    #   tubelets by their position in time, then down, then across. Inside a
    #   row, order the numbers by channel, then frame, then row, then column.
    #   Example: tubelets(torch.arange(8.0).reshape(1, 2, 2, 2), 2, 1, 1)
    #   returns tensor([[0., 4.], [1., 5.], [2., 6.], [3., 7.]]).
    def tubelets(
        video: Tensor,
        t: int,
        h: int,
        w: int,
    ) -> Tensor:
        ...


    # TODO: videos has shape (batch, channels, frames, height, width). Run conv
    #   on videos and return its output as tokens, of shape
    #   (batch, tokens, conv's output channels), in the same order as the
    #   tubelets.
    def tubelet_embedding(
        videos: Tensor,
        conv: nn.Conv3d,
    ) -> Tensor:
        ...


    # TODO: Return the same tokens for one video, of shape
    #   (channels, frames, height, width), without calling conv. Cut the
    #   tubelets with conv.kernel_size as t, h and w. Flatten conv.weight to one
    #   row per output channel, multiply the tubelets by its transpose, and add
    #   conv.bias. The result has shape (tokens, output channels).
    def as_linear(
        video: Tensor,
        conv: nn.Conv3d,
    ) -> Tensor:
        ...


    # TODO: Return how many whole tubelets of t frames by h by w pixels fit in
    #   a video of `frames` frames, each `height` by `width` pixels, as a plain
    #   int.
    #   Example: token_count(50, 224, 224, 2, 16, 16) returns 4900.
    def token_count(
        frames: int,
        height: int,
        width: int,
        t: int,
        h: int,
        w: int,
    ) -> int:
        ...


    # Two seconds at 25 frames a second, 224 by 224 pixels, cut into patches of single frames and then into
    # tubelets of 2 frames.
    counts = [token_count(50, 224, 224, 1, 16, 16), token_count(50, 224, 224, 2, 16, 16)]
    ''',
)

VIDEO1_SOLUTION = source_file(
    '''
    # video1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.arange(8.0).reshape(1, 2, 2, 2)
    split = x.reshape(1, 1, 2, 2, 1, 2, 1)
    moved = torch.zeros(2, 3, 4).permute(2, 0, 1).shape
    conv = nn.Conv3d(3, 8, kernel_size=(2, 4, 4), stride=(2, 4, 4))
    rows = conv.weight.flatten(1).shape


''' + indented(TUBELET_PART) + '''


    counts = [token_count(50, 224, 224, 1, 16, 16), token_count(50, 224, 224, 2, 16, 16)]
    ''',
)


def check_counts(value: object) -> None:
    expected = [
        REFERENCE["token_count"](50, 224, 224, 1, 16, 16),
        REFERENCE["token_count"](50, 224, 224, 2, 16, 16),
    ]

    if value != expected:
        raise Mismatch(f"counts is {value_text(value)}, but it should be {expected}.")


def check_video1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="tubelets",
        description="cuts boxes of t frames by h by w, one per row",
        inputs=(
            (torch.arange(32.0).reshape(1, 2, 4, 4), 2, 2, 2),
            (small_tensor(970, 2, 4, 4, 6), 2, 2, 3),
            (small_tensor(971, 3, 6, 4, 2), 3, 4, 1),
        ),
        reference=REFERENCE["tubelets"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="tubelet_embedding",
        description="returns one token per tubelet, as (batch, tokens, size)",
        inputs=(
            (small_tensor(972, 2, 1, 4, 4, 4), Described(seeded_conv(973, 1, 5, (2, 2, 2)), "nn.Conv3d(1, 5, (2, 2, 2), stride=(2, 2, 2))")),
            (small_tensor(974, 3, 2, 6, 4, 6), Described(seeded_conv(975, 2, 3, (3, 2, 3)), "nn.Conv3d(2, 3, (3, 2, 3), stride=(3, 2, 3))")),
        ),
        reference=REFERENCE["tubelet_embedding"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="as_linear",
        description="computes conv's tokens from the tubelets and conv's weights",
        inputs=(
            (small_tensor(976, 1, 4, 4, 4), Described(seeded_conv(977, 1, 5, (2, 2, 2)), "nn.Conv3d(1, 5, (2, 2, 2), stride=(2, 2, 2))")),
            (small_tensor(978, 2, 6, 4, 6), Described(seeded_conv(979, 2, 3, (3, 2, 3)), "nn.Conv3d(2, 3, (3, 2, 3), stride=(3, 2, 3))")),
        ),
        reference=lambda video, conv: REFERENCE["tubelet_embedding"](video.unsqueeze(0), conv)[0].detach(),
        expect=lambda call, actual, expected: expect_tensor(
            call=call,
            actual=actual.detach() if isinstance(actual, Tensor) else actual,
            expected=expected,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="token_count",
        description="counts the tubelets",
        inputs=(
            (50, 224, 224, 2, 16, 16),
            (16, 32, 48, 4, 8, 8),
            (10, 11, 13, 3, 5, 4),
        ),
        reference=REFERENCE["token_count"],
        expect=expect_value,
    )
    checker.check_name(
        name="counts",
        description="is the token counts for two seconds of video, frame by frame and in tubelets",
        run=check_counts,
    )


VIDEO1 = Exercise(
    name="video1",
    chapter=VIDEO,
    title="Tubelets",
    lesson=VIDEO1_LESSON,
    source=VIDEO1_SOURCE,
    solution=VIDEO1_SOLUTION,
    hints={
        "tubelets": "Reshape to (channels, frames // t, t, height // h, h, width // w, w). Then permute with "
        "(1, 3, 5, 0, 2, 4, 6), which puts the three box counts first, then channels and the three box sizes. Reshape "
        "to one row per tubelet.",
        "tubelet_embedding": "conv(videos) has shape (batch, size, frames // t, height // h, width // w). Flatten from "
        "dim 2 and swap the last two dimensions.",
        "as_linear": "conv.weight.flatten(1) has one row per output channel. Multiply the tubelets by its transpose, "
        "then add conv.bias.",
        "token_count": "Count the whole boxes along each axis, frames // t and so on, and multiply the three counts "
        "together.",
    },
    introduces=(),
    check=check_video1,
)


# ---------------------------------------------------------------------------
# video2
# ---------------------------------------------------------------------------

VIDEO2_LESSON = prose(
    """
    Here is a task that only motion can answer. A dot moves one pixel per
    frame along a row of a 16 by 16 grid, for 8 frames. When it reaches one
    edge, it wraps around to the other. Is it moving left or right? The given
    function `moving_dots` makes these clips.

    Any single frame shows one dot and nothing else. A clip played backwards
    has the same frames as a clip of the dot moving the other way. So a model
    that reads each frame on its own and then averages over time cannot tell
    the two directions apart.

    - `FrameModel` embeds 4 by 4 patches of each frame with a Conv2d, then averages all the tokens over every frame.
    - `TubeletModel` embeds tubelets of 2 frames by 4 by 4 pixels with a Conv3d, as in video1, then averages the same way.

    equivariance1's `invariance_gap` was the largest absolute difference
    between `f(g(x))` and `f(x)`. Here `g` plays a clip backwards with
    `video.flip(2)`, which reverses the frames dimension. With that `g`, the
    gap is called the reversal gap.

    ```python
    torch.tensor([[1, 2, 3]]).flip(1)    # tensor([[3, 2, 1]])
    ```

    Both models trained for 150 steps of Adam with a learning rate of 0.01.
    Each step used a batch of 32 new clips. Each model was then scored on
    1,000 other clips, with 5 seeds.

    | Model | Accuracy on new clips | Largest change when played backwards |
    |---|---|---|
    | FrameModel | 47.6% to 50.6% | under 4e-8 |
    | TubeletModel | 100% | 0.68 to 0.91 |

    FrameModel's gap is 0 apart from rounding, so it gives a clip and its
    reverse the same answer. That means it can do no better than a coin
    flip, however long it trains. Each of TubeletModel's tokens covers two
    frames, so the direction of motion is visible inside a single token.

    A model can only learn what its tokens and layers let it see. To detect
    motion, some layer has to combine information from different frames.
    That can happen inside the tokens, as here, or in attention across frames
    in later layers.
    """,
)

VIDEO2_SOURCE = source_file(
    '''
    # video2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    backwards = torch.tensor([[1, 2, 3]]).flip(1)               # tensor([[3, 2, 1]])
    averaged = torch.ones(2, 3, 4, 5).mean(dim=(2, 3)).shape    # torch.Size([2, 3])
    frames_first = torch.zeros(2, 1, 8, 16, 16).transpose(1, 2).shape    # torch.Size([2, 8, 1, 16, 16])


    # Clips of a dot moving left, label 0, or right, label 1.
''' + indented(DOTS_PART) + '''


    # TODO: Finish forward. videos has shape (batch, 1, frames, height, width).
    #   Run every frame through self.embed as a separate image, apply ReLU, and
    #   average over each frame's patches to get one vector per frame. Then
    #   average those vectors over the frames, and return self.head applied to
    #   the result, of shape (batch, 2).
''' + indented(FRAME_INIT) + '''

        def forward(self, videos: Tensor) -> Tensor:
            ...


    # TODO: Finish forward. Embed the tubelets of videos with self.embed, apply
    #   ReLU, and average over all the tubelets to get one vector per clip.
    #   Return self.head applied to that vector, of shape (batch, 2).
''' + indented(TUBELET_INIT) + '''

        def forward(self, videos: Tensor) -> Tensor:
            ...


    # TODO: Return the largest absolute change in model's output when the
    #   videos play backwards, as a plain Python float. Playing backwards means
    #   flipping dim 2, the frames. Put model in eval mode and run it inside
    #   torch.no_grad().
    def reversal_gap(
        model: nn.Module,
        videos: Tensor,
    ) -> float:
        ...


    # TODO: Train model with torch.optim.Adam at learning rate lr for `steps`
    #   steps. Each step draws a new batch with moving_dots(32, generator) and
    #   uses the F.cross_entropy loss. Return the last step's loss as a plain
    #   Python float.
    def train(
        model: nn.Module,
        steps: int,
        lr: float,
        generator: torch.Generator,
    ) -> float:
        ...


    # TODO: Return the fraction of videos that model labels correctly, as a
    #   plain Python float. The predicted label is the index of the larger of
    #   the two scores. Put model in eval mode and run it inside
    #   torch.no_grad().
    def accuracy(
        model: nn.Module,
        videos: Tensor,
        labels: Tensor,
    ) -> float:
        ...


    # Trains both models the same way, then scores them on 1,000 new clips and measures their reversal gaps.
    test_videos, test_labels = moving_dots(1000, torch.Generator().manual_seed(1000))
    torch.manual_seed(0)
    frame_model = FrameModel()
    train(frame_model, 150, 0.01, torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    tubelet_model = TubeletModel()
    train(tubelet_model, 150, 0.01, torch.Generator().manual_seed(0))
    frame_score = accuracy(frame_model, test_videos, test_labels)
    tubelet_score = accuracy(tubelet_model, test_videos, test_labels)
    frame_gap = reversal_gap(frame_model, test_videos)
    tubelet_gap = reversal_gap(tubelet_model, test_videos)
    ''',
)

VIDEO2_SOLUTION = source_file(
    '''
    # video2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    backwards = torch.tensor([[1, 2, 3]]).flip(1)
    averaged = torch.ones(2, 3, 4, 5).mean(dim=(2, 3)).shape
    frames_first = torch.zeros(2, 1, 8, 16, 16).transpose(1, 2).shape


''' + indented(DOTS_PART) + '''


''' + indented(FRAME_INIT) + '''
''' + indented(FRAME_FORWARD) + '''


''' + indented(TUBELET_INIT) + '''
''' + indented(TUBELET_FORWARD) + '''


''' + indented(MOTION_PART) + '''


    test_videos, test_labels = moving_dots(1000, torch.Generator().manual_seed(1000))
    torch.manual_seed(0)
    frame_model = FrameModel()
    train(frame_model, 150, 0.01, torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    tubelet_model = TubeletModel()
    train(tubelet_model, 150, 0.01, torch.Generator().manual_seed(0))
    frame_score = accuracy(frame_model, test_videos, test_labels)
    tubelet_score = accuracy(tubelet_model, test_videos, test_labels)
    frame_gap = reversal_gap(frame_model, test_videos)
    tubelet_gap = reversal_gap(tubelet_model, test_videos)
    ''',
)


def dot_clips(
    seed: int,
    count: int,
) -> tuple[Tensor, Tensor]:
    return REFERENCE["moving_dots"](count, torch.Generator().manual_seed(seed))


def seeded_model(
    make: type,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)

    return make()


def check_forward(
    class_name: str,
) -> Callable[[type], None]:
    def check(found: type) -> None:
        for seed in (980, 981):
            videos, _ = dot_clips(seed, 6)
            model = seeded_model(found, seed)
            reference = seeded_model(REFERENCE[class_name], seed)
            scores = model(videos)
            expect_tensor(
                call=f"{class_name}() seeded with {seed}, on 6 clips of moving dots",
                actual=scores.detach() if isinstance(scores, Tensor) else scores,
                expected=reference(videos).detach(),
            )

    return check


def recorded_modes(model: nn.Module) -> list[tuple[bool, bool]]:
    """Every forward pass's training flag and whether autograd was recording, through a forward pre hook."""
    modes: list[tuple[bool, bool]] = []
    model.register_forward_pre_hook(lambda module, inputs: modes.append((module.training, torch.is_grad_enabled())))

    return modes


def check_quiet_scoring(
    name: str,
    run: Callable[[nn.Module], object],
) -> None:
    model = seeded_model(REFERENCE["TubeletModel"], 982)
    model.train()
    modes = recorded_modes(model)
    run(model)

    if any(training for training, _ in modes):
        raise Mismatch(f"{name} ran the model in training mode. Call model.eval() first.")

    if any(recording for _, recording in modes):
        raise Mismatch(f"{name} recorded for autograd. Run the model inside torch.no_grad().")


def check_reversal_gap(found: object) -> None:
    for make, seed in ((REFERENCE["TubeletModel"], 983), (REFERENCE["FrameModel"], 984), (REFERENCE["TubeletModel"], 985)):
        videos, _ = dot_clips(seed, 20)
        expect_close(
            call=f"reversal_gap({make.__name__}() seeded with {seed}, 20 clips)",
            actual=found(seeded_model(make, seed), videos),
            expected=REFERENCE["reversal_gap"](seeded_model(make, seed), videos),
            tolerance=1e-5,
        )

    check_quiet_scoring(
        name="reversal_gap",
        run=lambda model: found(model, dot_clips(986, 4)[0]),
    )


def check_accuracy(found: object) -> None:
    for make, seed in ((REFERENCE["TubeletModel"], 987), (REFERENCE["FrameModel"], 988)):
        videos, labels = dot_clips(seed, 50)
        expect_close(
            call=f"accuracy({make.__name__}() seeded with {seed}, 50 clips and their labels)",
            actual=found(seeded_model(make, seed), videos, labels),
            expected=REFERENCE["accuracy"](seeded_model(make, seed), videos, labels),
        )

    check_quiet_scoring(
        name="accuracy",
        run=lambda model: found(model, *dot_clips(989, 4)),
    )


def check_train(found: object) -> None:
    for make, seed, steps in ((REFERENCE["TubeletModel"], 990, 2), (REFERENCE["FrameModel"], 991, 3)):
        model = seeded_model(make, seed)
        reference = seeded_model(make, seed)
        call = f"train({make.__name__}() seeded with {seed}, {steps}, 0.01, a generator seeded with {seed})"
        loss = found(model, steps, 0.01, torch.Generator().manual_seed(seed))
        expected_loss = REFERENCE["train"](reference, steps, 0.01, torch.Generator().manual_seed(seed))
        expect_close(
            call=f"the loss from {call}",
            actual=loss,
            expected=expected_loss,
            tolerance=1e-5,
        )

        for (name, parameter), wanted in zip(model.named_parameters(), reference.parameters()):
            if not torch.allclose(parameter, wanted, atol=1e-5):
                raise Mismatch(
                    f"After {call}, the model's {name} differs from the reference's. Zero the gradients each step, "
                    "and draw each batch from moving_dots(32, generator).",
                )


def check_range(
    name: str,
    low: float,
    high: float,
    advice: str,
) -> Callable[[object], None]:
    def check(value: object) -> None:
        reject_missing_return(
            call=name,
            actual=value,
        )

        if isinstance(value, Tensor) or not isinstance(value, (int, float)):
            raise Mismatch(f"{name} is {value_text(value)}, but it should be a plain number.")

        if not low <= value <= high:
            raise Mismatch(f"{name} is {value:.4g}, but it should be between {low:g} and {high:g}. {advice}")

    return check


def check_video2(checker: Checker) -> None:
    for class_name in ("FrameModel", "TubeletModel"):
        checker.test_method(
            class_name=class_name,
            method="forward",
            description="returns two scores per clip",
            run=check_forward(class_name),
        )

    checker.test(
        function="reversal_gap",
        description="returns the largest change when the clips play backwards",
        run=check_reversal_gap,
    )
    checker.test(
        function="train",
        description="trains with Adam on new clips each step",
        run=check_train,
    )
    checker.test(
        function="accuracy",
        description="returns the fraction labelled correctly",
        run=check_accuracy,
    )

    for name, low, high, advice in (
        ("frame_score", 0.0, 0.6, "Check FrameModel.forward against its TODO."),
        ("tubelet_score", 0.95, 1.0, "Check TubeletModel.forward and train."),
        ("frame_gap", 0.0, 1e-4, "FrameModel should average over the frames, so their order can't matter."),
        ("tubelet_gap", 0.1, math.inf, "TubeletModel's tokens should span two frames."),
    ):
        checker.check_name(
            name=name,
            description="comes from the two trained models",
            run=check_range(
                name=name,
                low=low,
                high=high,
                advice=advice,
            ),
        )


VIDEO2 = Exercise(
    name="video2",
    chapter=VIDEO,
    title="A task only motion answers",
    lesson=VIDEO2_LESSON,
    source=VIDEO2_SOURCE,
    solution=VIDEO2_SOLUTION,
    hints={
        "FrameModel.forward": "transpose(1, 2) puts the frames dimension next to batch, so a reshape can make one image "
        "per frame. After self.embed and ReLU, the mean over dims 2 and 3 averages the patches. Reshape back to "
        "(batch, frames, size), average over dim 1, then apply self.head.",
        "TubeletModel.forward": "self.embed(videos) has shape (batch, size, then one dimension each for the tubelets "
        "along time, down and across). Apply ReLU, take the mean over dims 2, 3 and 4, then apply self.head.",
        "reversal_gap": "Take the largest absolute difference between model(videos) and the model's output on videos "
        "flipped along dim 2. Turn it into a plain number with .item().",
        "train": "Make the optimizer once, before the loop. Each step draws a batch, computes the loss, zeroes the "
        "gradients, calls backward and calls the optimizer's step.",
        "accuracy": "Compare the argmax over dim 1 with labels, turn the result into floats, and take the mean.",
    },
    introduces=(),
    check=check_video2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_tokens() -> dict[str, object]:
    """video1's numbers: the token counts of two seconds at 25 frames a second, and the largest gap between the
    convolution's tokens and the tubelets times its weights."""
    conv = seeded_conv(0, 3, 8, (2, 4, 4))
    video = torch.randn(3, 8, 16, 16, generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        gap = (REFERENCE["as_linear"](video, conv) - REFERENCE["tubelet_embedding"](video.unsqueeze(0), conv)[0]).abs().max().item()

    return {
        "frame by frame": REFERENCE["token_count"](50, 224, 224, 1, 16, 16),
        "tubelets": REFERENCE["token_count"](50, 224, 224, 2, 16, 16),
        "largest gap": gap,
    }


def measure_motion(seeds: range = range(5)) -> dict[str, list[float]]:
    """video2's table: each model trained 150 steps at 0.01 on batches of 32, scored on 1,000 other clips, with the
    largest change in its output when the clips play backwards, for each seed, on 4 threads."""
    results: dict[str, list[float]] = {}

    with pinned_threads():
        for seed in seeds:
            videos, labels = REFERENCE["moving_dots"](1000, torch.Generator().manual_seed(1000 + seed))

            for name in ("FrameModel", "TubeletModel"):
                model = seeded_model(REFERENCE[name], seed)
                REFERENCE["train"](model, 150, 0.01, torch.Generator().manual_seed(seed))
                results.setdefault(f"{name} score", []).append(REFERENCE["accuracy"](model, videos, labels))
                results.setdefault(f"{name} gap", []).append(REFERENCE["reversal_gap"](model, videos))

    return results
