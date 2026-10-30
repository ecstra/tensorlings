"""71_digit_diffusion, Section 9's project, a U-Net denoiser for MNIST conditioned on the digit, trained for minutes and sampled with guidance."""

from __future__ import annotations

import functools
import math
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
    seeded,
)
from ..text import prose, source_file
from ..training import MissingCheckpoint, load_checkpoint
from .classic_nets import ReferenceLeNet, mnist_images, reference_train_lenet
from .diffusion import GUIDANCE_PART, SAMPLING_PART, SCHEDULE_PART, TIME_PART
from .linear_layers import expect_module

DIGIT_DIFFUSION = Chapter(
    folder="71_digit_diffusion",
    title="Project: digits from noise",
    readme=source_file(
        """
        # Project: digits from noise

        Section 9's project. You build a small U-Net that predicts the noise
        in an MNIST digit and takes the digit it should be as a condition,
        train it for minutes outside the app, and sample new digits with
        classifier free guidance. A LeNet from lenet1 then scores whether
        each sample is the digit that was asked for.

        ## Further reading

        - [Ho, Jain and Abbeel, denoising diffusion probabilistic models](https://arxiv.org/abs/2006.11239)
        - [Song, Meng and Ermon, denoising diffusion implicit models](https://arxiv.org/abs/2010.02502)
        - [Ho and Salimans, classifier free diffusion guidance](https://arxiv.org/abs/2207.12598)
        """,
    ),
)

STEPS = 1000
NULL = 10

RANGE_SOLUTION = '''
def to_model_range(images: Tensor) -> Tensor:
    x = images.float().unsqueeze(1) / 127.5 - 1

    return F.pad(x, (2, 2, 2, 2), value=-1.0)


def to_picture(x: Tensor) -> Tensor:
    return ((x[:, :, 2:30, 2:30] + 1) / 2).clamp(0, 1)
'''

BLOCK_PART = '''
# Two convolutions, each with GroupNorm and SiLU, with the condition added to every channel between them.
class ConditionedBlock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        condition: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels_out)
        self.condition = nn.Linear(condition, channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels_out)

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
    ) -> Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.condition(condition)[:, :, None, None]

        return F.silu(self.norm2(self.conv2(h)))
'''

UNET_INIT = '''
class DigitUNet(nn.Module):
    def __init__(
        self,
        steps: int = 1000,
        width: int = 32,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.time = nn.Sequential(nn.Linear(16, 4 * width), nn.SiLU(), nn.Linear(4 * width, 4 * width))
        # Ten digits, and one row more for the null label that means no digit at all.
        self.labels = nn.Embedding(11, 4 * width)
        self.down1 = ConditionedBlock(1, width, 4 * width)
        self.down2 = ConditionedBlock(width, 2 * width, 4 * width)
        self.middle = ConditionedBlock(2 * width, 2 * width, 4 * width)
        self.up2 = ConditionedBlock(4 * width, width, 4 * width)
        self.up1 = ConditionedBlock(2 * width, width, 4 * width)
        self.out = nn.Conv2d(width, 1, 1)
'''

UNET_FORWARD = '''
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        labels: Tensor,
    ) -> Tensor:
        condition = self.time(time_features(t, self.steps)) + self.labels(labels)
        first = self.down1(x, condition)
        second = self.down2(F.avg_pool2d(first, 2), condition)
        middle = self.middle(F.avg_pool2d(second, 2), condition)
        up = self.up2(torch.cat([F.interpolate(middle, scale_factor=2), second], dim=1), condition)
        up = self.up1(torch.cat([F.interpolate(up, scale_factor=2), first], dim=1), condition)

        return self.out(up)
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
        "Callable": Callable,
        "NULL": NULL,
    }
    exec(code, namespace)

    return namespace


DIGITS_REFERENCE = namespace_of(SCHEDULE_PART + TIME_PART + SAMPLING_PART + GUIDANCE_PART + RANGE_SOLUTION + BLOCK_PART + UNET_INIT + UNET_FORWARD)


# ---------------------------------------------------------------------------
# digits1
# ---------------------------------------------------------------------------

DIGITS1_LESSON = prose(
    """
    This is Section 9's project. diffusion2 trained a small MLP to predict
    the noise in 2D points. Here a U-Net, as unet1 built it, predicts the
    noise in MNIST digits. It takes the digit it should be as a condition,
    as diffusion4 did for points.

    Diffusion works on numbers around 0, so pixels from 0 to 255 move to the
    range -1 to 1. The U-Net halves its image twice. So each 28 by 28 digit
    gets a border of 2 pixels of black, -1, to make 32 by 32.

    The condition is one vector per image, the step's time features after a
    small MLP plus the digit's row from an `nn.Embedding`. Row 10 is the null
    label, which diffusion4 used for the unconditional pass of guidance.
    Every block adds the condition to each of its channels, after its first
    convolution.

    ```python
    condition = torch.ones(2, 4)
    condition[:, :, None, None].shape    # torch.Size([2, 4, 1, 1]), one number per channel
    ```

    The way down keeps the output of each level for the way back up, joined
    on the channels, as unet1's skips did. GroupNorm in place of BatchNorm
    normalizes each image on its own, so a sample doesn't depend on the
    others in its batch.
    """,
)

DIGITS1_SOURCE = source_file(
    '''
    # digits1.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    padded = F.pad(torch.zeros(1, 1, 2, 2), (1, 1, 1, 1), value=-1.0)    # shape (1, 1, 4, 4), a border of -1


    # time_features from diffusion2.
''' + indented(TIME_PART) + '''


''' + indented(BLOCK_PART) + '''


    # TODO: Return images, uint8 digits of shape (n, 28, 28) from 0 to 255, as
    #   floats from -1 to 1 of shape (n, 1, 32, 32), with a border of 2 pixels
    #   of -1 on every side.
    def to_model_range(images: Tensor) -> Tensor:
        ...


    # TODO: Return x, of shape (n, 1, 32, 32), as pictures from 0 to 1 of shape
    #   (n, 1, 28, 28), the border cut off and every pixel clamped to 0 to 1.
    def to_picture(x: Tensor) -> Tensor:
        ...


    # TODO: Finish forward. The condition is self.time of the time features of
    #   t, plus self.labels of labels. down1 reads x, down2 reads it pooled to
    #   half with F.avg_pool2d, and middle reads that pooled again. On the way
    #   up, F.interpolate doubles each result, which is joined with the output
    #   of the same size from the way down, the upsampled one first, before
    #   up2 and then up1. self.out gives the predicted noise.
''' + indented(UNET_INIT) + '''

        def forward(
            self,
            x: Tensor,
            t: Tensor,
            labels: Tensor,
        ) -> Tensor:
            ...
    ''',
)

DIGITS1_SOLUTION = source_file(
    '''
    # digits1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    padded = F.pad(torch.zeros(1, 1, 2, 2), (1, 1, 1, 1), value=-1.0)


''' + indented(TIME_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(RANGE_SOLUTION) + '''


''' + indented(UNET_INIT) + '''

''' + indented(UNET_FORWARD) + '''
    ''',
)


def seeded_unet(
    make: type,
    width: int = 8,
) -> nn.Module:
    torch.manual_seed(710)

    return make(STEPS, width)


def check_unet_forward(found: type) -> None:
    model = expect_module("DigitUNet(1000, 8)", seeded_unet(found))
    generator = torch.Generator().manual_seed(711)
    x = torch.randn(3, 1, 32, 32, generator=generator)
    t = torch.tensor([0, 500, 999])
    labels = torch.tensor([3, 10, 7])

    with torch.no_grad():
        expected = seeded_unet(DIGITS_REFERENCE["DigitUNet"])(x, t, labels)

    expect_tensor(
        call="DigitUNet(1000, 8) on 3 noisy images at steps 0, 500 and 999, labels 3, null and 7",
        actual=model(x, t, labels),
        expected=expected,
    )


def digit_images(
    seed: int,
    count: int,
) -> Tensor:
    return torch.randint(0, 256, (count, 28, 28), generator=torch.Generator().manual_seed(seed), dtype=torch.uint8)


def check_digits1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="to_model_range",
        description="scales to -1 to 1 and adds a border of -1",
        inputs=(
            (Described(digit_images(712, 3), "3 digits of 28 by 28"),),
        ),
        reference=DIGITS_REFERENCE["to_model_range"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="to_picture",
        description="cuts the border and scales back to 0 to 1",
        inputs=(
            (Described(torch.randn(2, 1, 32, 32, generator=torch.Generator().manual_seed(713)) * 1.5, "2 images of 32 by 32"),),
        ),
        reference=DIGITS_REFERENCE["to_picture"],
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="DigitUNet",
        method="forward",
        description="conditions every block and joins the skips",
        run=check_unet_forward,
    )


DIGITS1 = Exercise(
    name="digits1",
    chapter=DIGIT_DIFFUSION,
    title="Project: a U-Net that knows the digit",
    lesson=DIGITS1_LESSON,
    source=DIGITS1_SOURCE,
    solution=DIGITS1_SOLUTION,
    hints={
        "to_model_range": "images.float().unsqueeze(1) / 127.5 - 1, then F.pad with (2, 2, 2, 2) and value=-1.0.",
        "to_picture": "Slice [:, :, 2:30, 2:30], add 1, halve, and .clamp(0, 1).",
        "DigitUNet.forward": "Each torch.cat puts the F.interpolate(..., scale_factor=2) result first and the matching "
        "output from the way down second, on dim=1.",
    },
    introduces=(),
    check=check_digits1,
)


# ---------------------------------------------------------------------------
# digits2
# ---------------------------------------------------------------------------

LOSS_SOLUTION = '''
def digit_loss(
    model: nn.Module,
    x0: Tensor,
    labels: Tensor,
    alpha_bars: Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    labels = drop_labels(labels, 0.1, NULL, generator)
    t = torch.randint(0, len(alpha_bars), (len(x0),), generator=generator)
    noise = torch.randn(x0.shape, generator=generator)
    noisy = noise_to(x0, t, alpha_bars, noise)

    return F.mse_loss(model(noisy.to(device), t.to(device), labels.to(device)), noise.to(device))
'''

DIGITS2_MAIN = '''
    # Run by .venv/bin/python run.py train digits2, never by the app.
    def main() -> None:
        images, labels = load_mnist("train")
        x = to_model_range(images)
        alpha_bars = alpha_bars_of(linear_betas(1000, 1e-4, 0.02))
        device = best_device()
        torch.manual_seed(0)
        model = DigitUNet().to(device)
        steps = 4000
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        generator = torch.Generator().manual_seed(0)
        print(f"Training on {device} for {steps} steps.")

        for step in range(steps):
            batch = torch.randint(0, len(x), (128,), generator=generator)
            loss = digit_loss(model, x[batch], labels[batch], alpha_bars, generator, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (step + 1) % 500 == 0:
                print(f"Step {step + 1}, loss {loss.item():.4f}")

        save_checkpoint(model, "digits2")
        print("Saved the model in checkpoints/digits2.pt. Save digits2.py in the app to check it.")
'''

DIGITS_REFERENCE.update(namespace_of(SCHEDULE_PART + TIME_PART + SAMPLING_PART + GUIDANCE_PART + RANGE_SOLUTION + BLOCK_PART + UNET_INIT + UNET_FORWARD + LOSS_SOLUTION))

DIGITS2_LESSON = prose(
    """
    Training is diffusion2's loop on digits. Pick a step for each image and
    noise the image to that step in one jump with `noise_to`. Then score the
    model on how well it predicts the noise. The schedule is the DDPM
    paper's, 1,000 steps with betas from 0.0001 to 0.02.

    One time in ten the label is swapped for the null label, as diffusion4's
    `drop_labels` does. That trains one model to work both with a digit and
    with none, which guidance needs in digits3.

    The random draws come from a generator on the CPU. So the noise and the
    steps are made there, then moved to the device, as cifar2 moved its
    augmented batches.

    Once the checks above the last one pass, train from the project folder
    with `.venv/bin/python run.py train digits2`. It runs 4,000 steps of 128
    digits. The run behind this project took about 21 minutes on MPS while
    other training shared the machine. One batch's loss jumps around, from
    0.029 at step 500 to between 0.018 and 0.025 after that.

    The last check measures the trained model's loss on 1,000 test digits it
    never saw, at random steps. It asks for less than 0.05, and the run
    behind this project got 0.017.
    """,
)

DIGITS2_SOURCE = source_file(
    '''
    # digits2.py
    #
    # Train once the checks above the last one pass, from the project folder.
    #
    #     .venv/bin/python run.py train digits2

    import math
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_mnist
    from tensorlings.training import best_device, save_checkpoint

    NULL = 10
    steps = torch.randint(0, 1000, (4,), generator=torch.Generator().manual_seed(0))    # one step for each of 4 images


    # The schedule and noise_to from diffusion1, and drop_labels from diffusion4.
''' + indented(SCHEDULE_PART) + '''


''' + indented(GUIDANCE_PART) + '''


    # The pieces of digits1.
''' + indented(TIME_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(RANGE_SOLUTION) + '''


''' + indented(UNET_INIT) + '''

''' + indented(UNET_FORWARD) + '''


    # TODO: Return the noise prediction loss for a batch of x0, digits in the
    #   model's range, with their labels. First drop labels to NULL with
    #   drop_labels at 0.1, then draw a step for each image with torch.randint,
    #   then the noise with torch.randn of x0's shape, all with generator. Noise
    #   x0 to its step with noise_to, move what the model reads and the noise
    #   to device, and take the mean squared error.
    def digit_loss(
        model: nn.Module,
        x0: Tensor,
        labels: Tensor,
        alpha_bars: Tensor,
        generator: torch.Generator,
        device: torch.device,
    ) -> Tensor:
        ...

''' + DIGITS2_MAIN,
)

DIGITS2_SOLUTION = source_file(
    '''
    # digits2.py, a solution for comparison

    import math
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_mnist
    from tensorlings.training import best_device, save_checkpoint

    NULL = 10
    steps = torch.randint(0, 1000, (4,), generator=torch.Generator().manual_seed(0))


''' + indented(SCHEDULE_PART) + '''


''' + indented(GUIDANCE_PART) + '''


''' + indented(TIME_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(RANGE_SOLUTION) + '''


''' + indented(UNET_INIT) + '''

''' + indented(UNET_FORWARD) + '''


''' + indented(LOSS_SOLUTION) + '''

''' + DIGITS2_MAIN,
)


def schedule() -> Tensor:
    return DIGITS_REFERENCE["alpha_bars_of"](DIGITS_REFERENCE["linear_betas"](STEPS, 1e-4, 0.02))


def check_digit_loss(found: object) -> None:
    model = seeded_unet(DIGITS_REFERENCE["DigitUNet"])
    x0 = DIGITS_REFERENCE["to_model_range"](digit_images(714, 6))
    labels = torch.tensor([0, 3, 5, 7, 9, 2])
    expected = DIGITS_REFERENCE["digit_loss"](model, x0, labels, schedule(), torch.Generator().manual_seed(715), torch.device("cpu"))
    actual = found(model, x0, labels, schedule(), torch.Generator().manual_seed(715), torch.device("cpu"))
    expect_close(
        call="digit_loss on 6 digits",
        actual=actual.item() if isinstance(actual, Tensor) else actual,
        expected=expected.item(),
        tolerance=1e-5,
    )


def trained_unet(found: type) -> nn.Module:
    torch.manual_seed(0)
    model = found()

    try:
        load_checkpoint(model, "digits2")
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None
    except RuntimeError:
        raise Mismatch(
            "The saved model doesn't fit DigitUNet as it is now. Train it again from the project folder.\n\n"
            "    .venv/bin/python run.py train digits2",
        ) from None

    return model


@functools.cache
def test_digits() -> tuple[Tensor, Tensor]:
    from ..data import load_mnist

    images, labels = load_mnist("test")

    return DIGITS_REFERENCE["to_model_range"](images[:1000]), labels[:1000]


def test_loss(model: nn.Module) -> float:
    x0, labels = test_digits()
    generator = torch.Generator().manual_seed(716)
    losses = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(x0), 250):
            t = torch.randint(0, STEPS, (250,), generator=generator)
            noise = torch.randn(250, 1, 32, 32, generator=generator)
            noisy = DIGITS_REFERENCE["noise_to"](x0[start : start + 250], t, schedule(), noise)
            losses.append(F.mse_loss(model(noisy, t, labels[start : start + 250]), noise).item())

    return sum(losses) / len(losses)


def check_trained_unet(found: type) -> None:
    loss = test_loss(trained_unet(found))

    if loss >= 0.05:
        raise Mismatch(f"The trained model's loss on 1,000 test digits is {loss:.4f}, but it should be below 0.05.")


def check_digits2(checker: Checker) -> None:
    checker.test(
        function="digit_loss",
        description="drops labels, noises each digit to its step, scores the noise",
        run=check_digit_loss,
    )
    checker.test_method(
        class_name="DigitUNet",
        method="forward",
        description="trained, predicts the noise in test digits",
        run=check_trained_unet,
    )


DIGITS2 = Exercise(
    name="digits2",
    chapter=DIGIT_DIFFUSION,
    title="Project: training the denoiser",
    lesson=DIGITS2_LESSON,
    source=DIGITS2_SOURCE,
    solution=DIGITS2_SOLUTION,
    hints={
        "digit_loss": "labels = drop_labels(labels, 0.1, NULL, generator), then t and noise in that order. "
        "noise_to(x0, t, alpha_bars, noise) gives the noisy digits. Move them, t, the labels and the noise "
        "to device before the model and F.mse_loss.",
    },
    introduces=(),
    check=check_digits2,
    time_limit=180,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# digits3
# ---------------------------------------------------------------------------

SAMPLE_SOLUTION = '''
def clipped_ddim(
    predict_noise: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, ...],
    alpha_bars: Tensor,
    steps: int,
    generator: torch.Generator,
) -> Tensor:
    x = torch.randn(shape, generator=generator)
    times = list(range(0, len(alpha_bars), len(alpha_bars) // steps))[::-1]

    with torch.no_grad():
        for position, t in enumerate(times):
            eps = predict_noise(x, torch.full((shape[0],), t))
            x0 = ((x - (1 - alpha_bars[t]).sqrt() * eps) / alpha_bars[t].sqrt()).clamp(-1, 1)
            eps = (x - alpha_bars[t].sqrt() * x0) / (1 - alpha_bars[t]).sqrt()
            before = alpha_bars[times[position + 1]] if position + 1 < len(times) else torch.tensor(1.0)
            x = before.sqrt() * x0 + (1 - before).sqrt() * eps

    return x


def guided_digits(
    model: nn.Module,
    labels: Tensor,
    w: float,
    steps: int,
    generator: torch.Generator,
) -> Tensor:
    alpha_bars = alpha_bars_of(linear_betas(1000, 1e-4, 0.02))
    predict = lambda x, t: guided_noise(model, x, t, labels, NULL, w)

    return clipped_ddim(predict, (len(labels), 1, 32, 32), alpha_bars, steps, generator)


def share_right(
    classifier: nn.Module,
    pictures: Tensor,
    labels: Tensor,
) -> float:
    with torch.no_grad():
        return (classifier(pictures).argmax(dim=1) == labels).float().mean().item()
'''

DIGITS_REFERENCE.update(namespace_of(SCHEDULE_PART + TIME_PART + SAMPLING_PART + GUIDANCE_PART + RANGE_SOLUTION + BLOCK_PART + UNET_INIT + UNET_FORWARD + SAMPLE_SOLUTION))

DIGITS3_LESSON = prose(
    """
    digits2's model predicts noise. Sampling runs it backwards from pure
    noise, with DDIM's 50 steps in place of DDPM's 1,000, as diffusion3 did.
    Guidance pushes each step toward the digit asked for, with diffusion4's
    `guided_noise`.

    Images add one rule the 2D points never needed. Each step makes a guess
    at the finished digit, and a pixel guess outside -1 to 1 is impossible.
    So `clipped_ddim` clamps the guess, then works out the noise again from
    the clamped guess, so the two still agree. Without that second line,
    strong guidance pulls the steps apart.

    To score the samples, a LeNet from lenet1, trained on real digits, reads
    each one. The score is the share it reads as the digit that was asked
    for. The LeNet gets 95.7% of real test digits right.

    Here is the trained model from digits2, 20 samples of each digit, 50
    steps each.

    | Sampler | w = 0 | w = 1 | w = 2 | w = 4 |
    |---|---|---|---|---|
    | `clipped_ddim` | W0 | W1 | W2 | W4 |
    | Clamped, noise not worked out again | C0 | C1 | C2 | C4 |
    | diffusion3's `ddim_sample`, no clamp | D0 | D1 | D2 | D4 |

    GUIDANCE_SENTENCE Ho and Salimans found the same trade, guidance buys
    samples that look more like their class at the cost of variety.
    """,
)

DIGITS3_SOURCE = source_file(
    '''
    # digits3.py

    import math
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    NULL = 10
    clamped = torch.tensor([-1.7, 0.2, 1.3]).clamp(-1, 1)    # tensor([-1.0000,  0.2000,  1.0000])


    # The schedule, ddim_sample and guided_noise of 69_diffusion.
''' + indented(SCHEDULE_PART) + '''


''' + indented(SAMPLING_PART) + '''


''' + indented(GUIDANCE_PART) + '''


    # The pieces of digits1.
''' + indented(TIME_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(RANGE_SOLUTION) + '''


''' + indented(UNET_INIT) + '''

''' + indented(UNET_FORWARD) + '''


    # TODO: Finish the loop, diffusion3's ddim_sample with one change. The guess
    #   at the clean image, x0, is clamped to -1 to 1, and then eps is worked
    #   out again from x and the clamped x0, before the step to the next time.
    def clipped_ddim(
        predict_noise: Callable[[Tensor, Tensor], Tensor],
        shape: tuple[int, ...],
        alpha_bars: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> Tensor:
        x = torch.randn(shape, generator=generator)
        times = list(range(0, len(alpha_bars), len(alpha_bars) // steps))[::-1]

        with torch.no_grad():
            for position, t in enumerate(times):
                ...

        return x


    # TODO: Return images of the digits in labels, sampled from model with
    #   clipped_ddim for steps steps, guided with guided_noise at strength w
    #   and NULL as the null label. The schedule is linear_betas(1000, 1e-4,
    #   0.02) through alpha_bars_of, and the shape is (len(labels), 1, 32, 32).
    def guided_digits(
        model: nn.Module,
        labels: Tensor,
        w: float,
        steps: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the share of pictures that classifier reads as their label,
    #   as a plain number, without recording for autograd.
    def share_right(
        classifier: nn.Module,
        pictures: Tensor,
        labels: Tensor,
    ) -> float:
        ...
    ''',
)

DIGITS3_SOLUTION = source_file(
    '''
    # digits3.py, a solution for comparison

    import math
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    NULL = 10
    clamped = torch.tensor([-1.7, 0.2, 1.3]).clamp(-1, 1)


''' + indented(SCHEDULE_PART) + '''


''' + indented(SAMPLING_PART) + '''


''' + indented(GUIDANCE_PART) + '''


''' + indented(TIME_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(RANGE_SOLUTION) + '''


''' + indented(UNET_INIT) + '''

''' + indented(UNET_FORWARD) + '''


''' + indented(SAMPLE_SOLUTION) + '''
    ''',
)


def recorded_noise(recording: list[bool]) -> Callable[[Tensor, Tensor], Tensor]:
    """A stand in for a model's noise, a fixed function of x and t, that records whether autograd was on."""

    def predict(
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        recording.append(torch.is_grad_enabled())

        return 0.8 * x + 0.001 * t.float().reshape(-1, *[1] * (x.dim() - 1))

    return predict


def check_clipped_ddim(found: object) -> None:
    recording: list[bool] = []
    actual = found(recorded_noise(recording), (3, 1, 4, 4), schedule(), 10, torch.Generator().manual_seed(717))
    expected = DIGITS_REFERENCE["clipped_ddim"](recorded_noise([]), (3, 1, 4, 4), schedule(), 10, torch.Generator().manual_seed(717))

    if any(recording):
        raise Mismatch("clipped_ddim ran the model while recording for autograd. Keep the loop inside torch.no_grad().")

    expect_tensor(
        call="clipped_ddim of a stand in model for 10 steps on 3 images of 4 by 4",
        actual=actual,
        expected=expected,
    )


def check_guided_digits(found: object) -> None:
    model = seeded_unet(DIGITS_REFERENCE["DigitUNet"])
    labels = torch.tensor([1, 4, 9])
    expected = DIGITS_REFERENCE["guided_digits"](model, labels, 2.0, 4, torch.Generator().manual_seed(718))
    expect_tensor(
        call="guided_digits of an untrained DigitUNet(1000, 8) for digits 1, 4 and 9, w = 2, 4 steps",
        actual=found(model, labels, 2.0, 4, torch.Generator().manual_seed(718)),
        expected=expected,
    )


@functools.cache
def judge() -> nn.Module:
    """lenet1's LeNet, trained on real digits, to read the samples."""
    images = mnist_images()
    torch.manual_seed(0)
    model = ReferenceLeNet()
    reference_train_lenet(model, images["train_x"], images["train_y"], 800, torch.Generator().manual_seed(1))
    model.eval()

    return model


def check_share_right(found: object) -> None:
    torch.manual_seed(719)
    classifier = nn.Sequential(nn.Flatten(), nn.Linear(16, 3))
    pictures = torch.randn(30, 1, 4, 4, generator=torch.Generator().manual_seed(720))

    with torch.no_grad():
        read = classifier(pictures).argmax(dim=1)

    labels = torch.where(torch.arange(30) < 18, read, (read + 1) % 3)
    recording: list[bool] = []
    hook = classifier.register_forward_pre_hook(lambda module, inputs: recording.append(torch.is_grad_enabled()))

    try:
        actual = found(classifier, pictures, labels)
    finally:
        hook.remove()

    if any(recording):
        raise Mismatch("share_right recorded for autograd. Run the classifier inside torch.no_grad().")

    expect_close(
        call="share_right on 30 pictures, 18 of them read as their label",
        actual=actual,
        expected=0.6,
    )


def sample_share(
    model: nn.Module,
    w: float,
    sampler: str = "clipped",
) -> float:
    """The share of 20 samples of each digit that the judge reads as asked, for the lesson's table."""
    labels = torch.arange(10).repeat(20)
    predict = lambda x, t: DIGITS_REFERENCE["guided_noise"](model, x, t, labels, NULL, w)
    shape = (len(labels), 1, 32, 32)
    generator = torch.Generator().manual_seed(721)

    if sampler == "plain":
        samples = DIGITS_REFERENCE["ddim_sample"](predict, shape, schedule(), 50, generator)
    else:
        samples = DIGITS_REFERENCE["clipped_ddim"](predict, shape, schedule(), 50, generator)

    return DIGITS_REFERENCE["share_right"](judge(), DIGITS_REFERENCE["to_picture"](samples), labels)


def check_trained_samples(found: type) -> None:
    model = trained_unet(DIGITS_REFERENCE["DigitUNet"])
    model.eval()
    labels = torch.arange(10).repeat(5)
    samples = DIGITS_REFERENCE["guided_digits"](model, labels, 2.0, 50, torch.Generator().manual_seed(722))
    share = DIGITS_REFERENCE["share_right"](judge(), DIGITS_REFERENCE["to_picture"](samples), labels)

    if share < 0.85:
        raise Mismatch(f"The LeNet reads {share:.0%} of 50 guided samples as the digit asked for, but it should pass 85%.")


def check_digits3(checker: Checker) -> None:
    checker.test(
        function="clipped_ddim",
        description="clamps the guess at the clean image and works the noise out again",
        run=check_clipped_ddim,
    )
    checker.test(
        function="guided_digits",
        description="samples each digit with guidance",
        run=check_guided_digits,
    )
    checker.test(
        function="share_right",
        description="gives the share read as the digit asked for",
        run=check_share_right,
    )
    checker.test(
        function="guided_digits",
        description="from the trained model, samples a LeNet reads right",
        run=check_trained_samples,
    )


DIGITS3 = Exercise(
    name="digits3",
    chapter=DIGIT_DIFFUSION,
    title="Project: guided digits",
    lesson=DIGITS3_LESSON,
    source=DIGITS3_SOURCE,
    solution=DIGITS3_SOLUTION,
    hints={
        "clipped_ddim": "Take eps from predict_noise and x0 from it as ddim_sample does, with .clamp(-1, 1). Then eps "
        "becomes (x - alpha_bars[t].sqrt() * x0) / (1 - alpha_bars[t]).sqrt(). The step to before follows ddim_sample.",
        "guided_digits": "predict = lambda x, t: guided_noise(model, x, t, labels, NULL, w). Pass it to clipped_ddim "
        "with the shape, the schedule, steps and generator.",
        "share_right": "Inside torch.no_grad(), compare classifier(pictures).argmax(dim=1) with labels and average.",
    },
    introduces=(),
    check=check_digits3,
    time_limit=300,
)
