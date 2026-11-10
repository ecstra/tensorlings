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
    DeviceRecorder,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_on_device,
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

        This is Section 9's project. You build a small U-Net that predicts the
        noise in an MNIST digit. It also takes the digit the image should
        show as an input, called a condition. You train it for a few minutes
        outside the app, then sample new digits with classifier free
        guidance. A LeNet from lenet1 then checks whether each sample is the
        digit that was asked for.

        ## Further reading

        - [Ho, Jain and Abbeel, denoising diffusion probabilistic models](https://arxiv.org/abs/2006.11239)
        - [Song, Meng and Ermon, denoising diffusion implicit models](https://arxiv.org/abs/2010.02502)
        - [Ho and Salimans, classifier free diffusion guidance](https://arxiv.org/abs/2207.12598)
        """,
    ),
)

STEPS = 1000
NULL = 10

# digits2 needs only diffusion4's drop_labels, and digits3 only its guided_noise.
DROP_LABELS_PART = GUIDANCE_PART[: GUIDANCE_PART.index("def guided_noise")]
GUIDED_NOISE_PART = GUIDANCE_PART[GUIDANCE_PART.index("def guided_noise") : GUIDANCE_PART.index("def guided_sample")]

RANGE_SOLUTION = '''
def to_model_range(images: Tensor) -> Tensor:
    x = images.float().unsqueeze(1) / 127.5 - 1

    return F.pad(x, (2, 2, 2, 2), value=-1.0)


def to_picture(x: Tensor) -> Tensor:
    return ((x[:, :, 2:30, 2:30] + 1) / 2).clamp(0, 1)
'''

BLOCK_PART = '''
# Two convolutions with GroupNorm and SiLU. The condition is added to every channel between them.
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
        # One row per digit, and one more for the null label, which means no digit.
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
    the noise in 2D points. Here a U-Net like the one unet1 built predicts
    the noise in MNIST digits. Like diffusion4's model for points, it is
    conditioned on a label, here the digit the image should show.

    Diffusion works on numbers centered around 0, so pixel values from 0 to
    255 are rescaled to the range -1 to 1. Each 28 by 28 digit also gets a
    black border 2 pixels wide, with value -1, which makes it 32 by 32. That
    is the size of DDPM's CIFAR-10 images. 32 can be halved again and again
    down to 1, so a deeper U-Net would fit it too. This U-Net halves the size
    only twice, which 28 by 28 would also allow.

    The condition is one vector per image, the sum of two parts. The first is
    the step's time features passed through a small MLP. The second is the
    digit's row of an `nn.Embedding`. Row 10 is the null label, which
    guidance uses for its prediction without a label, as in diffusion4. Every
    block adds the condition to its channels after its first convolution,
    one number of the condition per channel.

    ```python
    condition = torch.ones(2, 4)
    condition[:, :, None, None].shape    # torch.Size([2, 4, 1, 1]), one number per channel
    ```

    On the way down, the U-Net keeps the output of each level. On the way
    back up, it concatenates each kept output with the upsampled tensor of
    the same size, along the channels. These are unet1's skip connections.

    The blocks use `nn.GroupNorm(8, channels)` instead of BatchNorm.
    GroupNorm splits the channels into 8 groups. It normalizes each group of
    each image separately, with that group's own mean and standard deviation.
    LayerNorm is the special case with a single group. Since nothing is
    shared across the batch, one sample's output does not depend on the other
    samples in its batch.
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


    # TODO: images holds uint8 digits of shape (n, 28, 28), with pixel values
    #   from 0 to 255. Return them as floats of shape (n, 1, 32, 32). Map 0 to
    #   -1 and 255 to 1, and add a border of 2 pixels with value -1 on every
    #   side.
    def to_model_range(images: Tensor) -> Tensor:
        ...


    # TODO: Undo to_model_range for x, of shape (n, 1, 32, 32). Cut off the
    #   2 pixel border, map -1 to 0 and 1 to 1, and clamp every pixel to the
    #   range 0 to 1. Return pictures of shape (n, 1, 28, 28).
    def to_picture(x: Tensor) -> Tensor:
        ...


    # TODO: Finish forward. The condition is self.time applied to the time
    #   features of t, plus self.labels applied to labels. Every block takes its
    #   input and the condition. down1 reads x. down2 reads down1's output,
    #   halved in size with F.avg_pool2d. middle reads down2's output, halved
    #   again. On the way up, F.interpolate doubles the size, in its default
    #   nearest neighbor mode. Concatenate each doubled result with the output
    #   of the same size from the way down, the doubled one first, and feed
    #   that to up2 and then to up1. self.out of up1's output is the predicted
    #   noise.
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
        description="conditions every block and concatenates the skip connections",
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
        "to_model_range": "Convert the images to floats and add a channel dimension of size 1. Divide by 127.5 and "
        "subtract 1, so 0 becomes -1 and 255 becomes 1. Then pad 2 pixels on every side with F.pad, filling with -1.0.",
        "to_picture": "Slice [:, :, 2:30, 2:30], add 1, divide by 2, and call .clamp(0, 1).",
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
    Training uses diffusion2's loop on digits. Pick a random step for each
    image and noise the image to that step in one jump with `noise_to`. The
    loss is the mean squared error between the model's predicted noise and
    the true noise. The schedule is the one from the DDPM paper, 1,000 steps
    with betas from 0.0001 to 0.02.

    One time in ten, the label is replaced by the null label, with
    diffusion4's `drop_labels`. That trains one model to predict the noise
    both with a digit label and with no label. Guidance in digits3 needs both
    predictions.

    The random numbers come from a generator on the CPU. So the noise and the
    steps are created on the CPU and then moved to the device, as dna2 moved
    its batches.

    Once every check except the last one passes, train from the project
    folder with `.venv/bin/python run.py train digits2`. It runs 4,000 steps
    with batches of 128 digits. The run behind this project took about 21
    minutes on MPS while other training jobs shared the machine. The printed
    loss is for a single batch, so it jumps around. It was 0.029 at step 500
    and between 0.018 and 0.025 after that.

    The last check measures the trained model's loss on 1,000 test digits it
    never saw, at random steps. It requires a loss below 0.05, and the run
    behind this project got 0.017.
    """,
)

DIGITS2_SOURCE = source_file(
    '''
    # digits2.py
    #
    # Once every check except the last one passes, train from the project folder.
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


''' + indented(DROP_LABELS_PART) + '''


    # The pieces of digits1.
''' + indented(TIME_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(RANGE_SOLUTION) + '''


''' + indented(UNET_INIT) + '''

''' + indented(UNET_FORWARD) + '''


    # TODO: Return the noise prediction loss for a batch x0 of digits, already
    #   in the model's range, with their labels. Draw everything with generator,
    #   in this order. First replace labels with NULL using drop_labels with
    #   p = 0.1. Then draw a step for each image with torch.randint. Then draw
    #   noise of x0's shape with torch.randn. Noise x0 to its steps with
    #   noise_to. Move the model's three inputs and the noise to device, and
    #   return the mean squared error between the model's output and the noise.
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


''' + indented(DROP_LABELS_PART) + '''


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

    # Both seeds drop exactly one label at 0.1, and neither would at 0.05.
    for seed in (724, 743):
        expected = DIGITS_REFERENCE["digit_loss"](model, x0, labels, schedule(), torch.Generator().manual_seed(seed), torch.device("cpu"))
        actual = found(model, x0, labels, schedule(), torch.Generator().manual_seed(seed), torch.device("cpu"))
        call = f"digit_loss on 6 digits with a generator seeded with {seed}, which drops one label"

        if not isinstance(actual, Tensor):
            raise Mismatch(f"{call} gave {actual!r}, but it should give the loss as a tensor. main calls loss.backward() on it.")

        if not actual.requires_grad:
            raise Mismatch(
                f"{call} gave a loss that doesn't reach the model's weights, so main's loss.backward() couldn't train "
                "them. Return F.mse_loss as it comes, with no .item(), .detach() or torch.no_grad().",
            )

        if actual.shape != ():
            raise Mismatch(f"{call} gave a loss of shape {tuple(actual.shape)}, but it should be the mean, a single number.")

        expect_close(
            call=call,
            actual=actual.item(),
            expected=expected.item(),
            tolerance=1e-5,
        )

    recorder = DeviceRecorder(
        names=("the noisy digits", "t", "the labels"),
        output=lambda x, t, labels: x,
    )
    expect_on_device(
        function="digit_loss",
        run=lambda: found(recorder, x0, labels, schedule(), torch.Generator().manual_seed(724), torch.device("meta")),
        recorders=[recorder],
        advice="Draw and noise on the CPU, then move what the model reads and the noise to device.",
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
        description="drops labels, noises each digit to its step, and compares the predicted noise with the true noise",
        run=check_digit_loss,
    )
    checker.test_method(
        class_name="DigitUNet",
        method="forward",
        description="after training, predicts the noise in test digits",
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
        "digit_loss": "Pass the labels through drop_labels first, with 0.1 and NULL. Then draw t, then the noise, all "
        "from generator. noise_to gives the noisy digits. Move them, t, the labels and the noise to device before "
        "calling the model and F.mse_loss.",
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
    digits2's model predicts noise. Sampling starts from pure noise and
    removes it step by step, with 50 DDIM steps instead of DDPM's 1,000, as
    in diffusion3. Guidance with diffusion4's `guided_noise` pushes each step
    toward the digit that was asked for.

    Images need one rule that the 2D points did not. Each DDIM step computes
    an estimate of the finished digit, `x0`. A pixel value outside -1 to 1 is
    impossible in a real image. So `clipped_ddim` clamps `x0` to the range -1
    to 1. Then it recomputes the noise `eps` from `x` and the clamped `x0`,
    so that `eps` and `x0` agree again. Without that recomputation, strong
    guidance ruins the samples. DDPM's own code clamps `x0` when it samples.
    OpenAI's guided diffusion code also recomputes the noise from it for
    DDIM.

    To score the samples, a LeNet from lenet1, trained on real digits,
    classifies each one. The score is the share of samples it classifies as
    the digit that was asked for. After 3,000 training steps, the LeNet
    classifies 97.8% of real test digits correctly.

    Here are the scores for the trained model from digits2, with 20 samples
    of each digit and 50 steps per sample.

    | Sampler | w = 0 | w = 1 | w = 2 | w = 4 |
    |---|---|---|---|---|
    | `clipped_ddim` | 71.0% | 92.5% | 95.5% | 97.5% |
    | Clamped, noise not recomputed | 76.5% | 33.0% | 7.0% | 1.0% |
    | diffusion3's `ddim_sample`, no clamp | 44.0% | 59.5% | 67.5% | 76.5% |

    Guidance raises the score of `clipped_ddim` from 71% to 97.5%. At w = 4
    the LeNet classifies the samples about as well as real test digits.
    Clamping without recomputing the noise gets worse as w grows, down to 1%.
    With no clamp at all, guidance helps, but the score stays well below
    `clipped_ddim`.

    Ho and Salimans found the same tradeoff. Guidance gives samples that look
    more like their class, but with less variety.
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


    # The schedule, ddim_sample and guided_noise from 69_diffusion.
''' + indented(SCHEDULE_PART) + '''


''' + indented(SAMPLING_PART) + '''


''' + indented(GUIDED_NOISE_PART) + '''


    # The pieces of digits1.
''' + indented(TIME_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(RANGE_SOLUTION) + '''


''' + indented(UNET_INIT) + '''

''' + indented(UNET_FORWARD) + '''


    # TODO: Finish the loop. It is diffusion3's ddim_sample with one change.
    #   After computing the estimate of the clean image, x0, clamp it to the
    #   range -1 to 1. Then recompute eps from x and the clamped x0, by solving
    #   the jump formula for the noise. Then take the step to the next time as
    #   ddim_sample does.
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


    # TODO: Return one sampled image for each digit in labels, as a tensor of
    #   shape (len(labels), 1, 32, 32). Sample from model with clipped_ddim for
    #   steps steps. Its noise prediction is guided_noise at strength w, with
    #   NULL as the null label. The alpha bars come from alpha_bars_of applied
    #   to linear_betas(1000, 1e-4, 0.02).
    def guided_digits(
        model: nn.Module,
        labels: Tensor,
        w: float,
        steps: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the fraction of pictures whose highest scoring class from
    #   classifier equals their label. Return it as a Python float, and run the
    #   classifier with autograd recording turned off.
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


''' + indented(GUIDED_NOISE_PART) + '''


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
    reference_train_lenet(model, images["train_x"], images["train_y"], 3000, torch.Generator().manual_seed(1))
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
        raise Mismatch(f"The LeNet classifies {share:.0%} of 50 guided samples as the digit asked for, but it should be at least 85%.")


DIGITS3_TIME_LIMIT = 300


def check_digits3(checker: Checker) -> None:
    checker.test(
        function="clipped_ddim",
        description="clamps the clean image estimate and recomputes the noise",
        run=check_clipped_ddim,
    )
    checker.test(
        function="guided_digits",
        description="samples each digit with guidance",
        run=check_guided_digits,
    )
    checker.test(
        function="share_right",
        description="returns the fraction classified as the digit asked for",
        run=check_share_right,
    )
    # This test runs only the course's own sampler on the saved model, so running out of time means a busy machine.
    checker.test(
        function="guided_digits",
        description="draws samples from the trained model that a LeNet classifies correctly",
        run=check_trained_samples,
        when_late=f"The course's own sampler was still drawing digits from your trained model after "
        f"{DIGITS3_TIME_LIMIT} seconds. That means the machine is too busy, not that your file has a problem. Save "
        "the file again once other work on the machine has finished.",
    )


DIGITS3 = Exercise(
    name="digits3",
    chapter=DIGIT_DIFFUSION,
    title="Project: guided digits",
    lesson=DIGITS3_LESSON,
    source=DIGITS3_SOURCE,
    solution=DIGITS3_SOLUTION,
    hints={
        "clipped_ddim": "Get eps from predict_noise, and x0 from eps, as ddim_sample does. Then clamp x0 to -1 to 1. "
        "Recompute eps by solving the jump formula for the noise, using x and the clamped x0. The step to the "
        "earlier time is the same as in ddim_sample.",
        "guided_digits": "Make a lambda of x and t that calls guided_noise with model, labels, NULL and w. Pass it to "
        "clipped_ddim with the shape, the alpha bars, steps and generator.",
        "share_right": "Inside torch.no_grad(), compare classifier(pictures).argmax(dim=1) with labels, turn the "
        "matches into floats, and take their mean.",
    },
    introduces=(),
    check=check_digits3,
    time_limit=DIGITS3_TIME_LIMIT,
    imports_first=OPTIMIZER_IMPORTS,
    # The judge trains for 3,000 steps once per run of the app, which on a busy machine took over 200 seconds.
    prepare=(judge,),
)


# ---------------------------------------------------------------------------
# The runs behind digits3's lesson
# ---------------------------------------------------------------------------


def clamped_only(
    predict_noise: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, ...],
    alpha_bars: Tensor,
    steps: int,
    generator: torch.Generator,
) -> Tensor:
    """clipped_ddim without its second line: the guess of x_0 is clamped, but the noise is never worked out again."""
    x = torch.randn(shape, generator=generator)
    times = list(range(0, len(alpha_bars), len(alpha_bars) // steps))[::-1]

    with torch.no_grad():
        for position, t in enumerate(times):
            eps = predict_noise(x, torch.full((shape[0],), t))
            x0 = ((x - (1 - alpha_bars[t]).sqrt() * eps) / alpha_bars[t].sqrt()).clamp(-1, 1)
            before = alpha_bars[times[position + 1]] if position + 1 < len(times) else torch.tensor(1.0)
            x = before.sqrt() * x0 + (1 - before).sqrt() * eps

    return x


def measure_guidance() -> dict[str, object]:
    """The judge's share of real test digits, and digits3's table: 20 samples of each digit, 50 steps, for each
    sampler and each w, from the trained model in checkpoints/digits2.pt."""
    from ..data import load_mnist

    before = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        reader = judge()
        images, labels = load_mnist("test")

        with torch.no_grad():
            real = (reader(images.float().unsqueeze(1) / 255).argmax(dim=1) == labels).float().mean().item()

        model = trained_unet(DIGITS_REFERENCE["DigitUNet"])
        model.eval()
        wanted = torch.arange(10).repeat(20)
        shape = (len(wanted), 1, 32, 32)
        samplers = {
            "clipped": DIGITS_REFERENCE["clipped_ddim"],
            "clamped": clamped_only,
            "plain": DIGITS_REFERENCE["ddim_sample"],
        }
        table: dict[tuple[str, float], float] = {}

        for w in (0.0, 1.0, 2.0, 4.0):
            for name, sampler in samplers.items():
                predict = lambda x, t: DIGITS_REFERENCE["guided_noise"](model, x, t, wanted, NULL, w)
                samples = sampler(predict, shape, schedule(), 50, torch.Generator().manual_seed(721))
                table[(name, w)] = DIGITS_REFERENCE["share_right"](reader, DIGITS_REFERENCE["to_picture"](samples), wanted)
    finally:
        torch.set_num_threads(before)

    return {"real test digits": real, "table": table}
