"""69_diffusion, noise added step by step and in one jump, a model that predicts the noise, DDPM and DDIM sampling, and classifier free guidance, on points in 2D."""

from __future__ import annotations

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
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

DIFFUSION = Chapter(
    folder="69_diffusion",
    title="Diffusion",
    readme=source_file(
        """
        # Diffusion

        A diffusion model learns to undo noise. Training adds Gaussian noise
        to the data in many small steps, and the model learns to guess the
        noise that was added. Sampling starts from pure noise and takes the
        noise away a step at a time. This chapter builds DDPM on points in 2D,
        then DDIM's faster sampling and classifier free guidance, the pieces
        most image generators are made of.

        ## Further reading

        - [Ho, Jain and Abbeel, denoising diffusion probabilistic models](https://arxiv.org/abs/2006.11239)
        - [Song, Meng and Ermon, denoising diffusion implicit models](https://arxiv.org/abs/2010.02502)
        - [Ho and Salimans, classifier free diffusion guidance](https://arxiv.org/abs/2207.12598)
        """,
    ),
)

STEPS = 1000

# The pieces each exercise hands on to the next, written once so the files never drift apart.
SCHEDULE_PART = '''
def linear_betas(
    steps: int,
    start: float,
    end: float,
) -> Tensor:
    return torch.linspace(start, end, steps)


def alpha_bars_of(betas: Tensor) -> Tensor:
    return torch.cumprod(1 - betas, dim=0)


def noise_step(
    x: Tensor,
    beta: float,
    noise: Tensor,
) -> Tensor:
    return (1 - beta) ** 0.5 * x + beta**0.5 * noise


def noise_to(
    x0: Tensor,
    t: Tensor,
    alpha_bars: Tensor,
    noise: Tensor,
) -> Tensor:
    kept = alpha_bars[t].reshape(len(t), *[1] * (x0.dim() - 1))

    return kept.sqrt() * x0 + (1 - kept).sqrt() * noise
'''

TIME_PART = '''
# The sine and cosine features of position2, here for the step t out of steps.
def time_features(
    t: Tensor,
    steps: int,
    size: int = 16,
) -> Tensor:
    frequencies = torch.arange(1, size // 2 + 1).float()
    angles = (t.float() / steps).unsqueeze(1) * frequencies * math.pi

    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
'''

DENOISER_INIT = '''
class Denoiser(nn.Module):
    def __init__(
        self,
        steps: int,
        features: int = 2,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(
            nn.Linear(features + 16, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, features),
        )
'''

DENOISER_FORWARD = '''
    def forward(
        self,
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        return self.net(torch.cat([x, time_features(t, self.steps)], dim=1))
'''

LOSS_PART = '''
def diffusion_loss(
    model: nn.Module,
    x0: Tensor,
    alpha_bars: Tensor,
    generator: torch.Generator,
) -> Tensor:
    t = torch.randint(0, len(alpha_bars), (len(x0),), generator=generator)
    noise = torch.randn(x0.shape, generator=generator)

    return F.mse_loss(model(noise_to(x0, t, alpha_bars, noise), t), noise)
'''

SAMPLING_PART = '''
def ddpm_step(
    eps: Tensor,
    x: Tensor,
    t: int,
    betas: Tensor,
    alpha_bars: Tensor,
    noise: Tensor,
) -> Tensor:
    mean = (x - betas[t] / (1 - alpha_bars[t]).sqrt() * eps) / (1 - betas[t]).sqrt()

    return mean + betas[t].sqrt() * noise


def ddpm_sample(
    predict_noise: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, ...],
    betas: Tensor,
    alpha_bars: Tensor,
    generator: torch.Generator,
) -> Tensor:
    x = torch.randn(shape, generator=generator)

    with torch.no_grad():
        for t in range(len(betas) - 1, -1, -1):
            eps = predict_noise(x, torch.full((shape[0],), t))
            noise = torch.randn(shape, generator=generator) if t > 0 else torch.zeros(shape)
            x = ddpm_step(eps, x, t, betas, alpha_bars, noise)

    return x


def ddim_sample(
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
            x0 = (x - (1 - alpha_bars[t]).sqrt() * eps) / alpha_bars[t].sqrt()
            before = alpha_bars[times[position + 1]] if position + 1 < len(times) else torch.tensor(1.0)
            x = before.sqrt() * x0 + (1 - before).sqrt() * eps

    return x
'''

CONDITIONAL_INIT = '''
class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        steps: int,
        classes: int,
        features: int = 2,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.steps = steps
        # One row more than the classes, for the null label that means no label at all.
        self.labels = nn.Embedding(classes + 1, 16)
        self.net = nn.Sequential(
            nn.Linear(features + 32, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, features),
        )
'''

CONDITIONAL_FORWARD = '''
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        labels: Tensor,
    ) -> Tensor:
        return self.net(torch.cat([x, time_features(t, self.steps), self.labels(labels)], dim=1))
'''

GUIDANCE_PART = '''
def drop_labels(
    labels: Tensor,
    p: float,
    null: int,
    generator: torch.Generator,
) -> Tensor:
    return torch.where(torch.rand(len(labels), generator=generator) < p, null, labels)


def guided_noise(
    model: nn.Module,
    x: Tensor,
    t: Tensor,
    labels: Tensor,
    null: int,
    w: float,
) -> Tensor:
    unconditional = torch.full_like(labels, null)

    return (1 + w) * model(x, t, labels) - w * model(x, t, unconditional)


def guided_sample(
    model: nn.Module,
    labels: Tensor,
    null: int,
    alpha_bars: Tensor,
    steps: int,
    w: float,
    generator: torch.Generator,
) -> Tensor:
    return ddim_sample(lambda x, t: guided_noise(model, x, t, labels, null, w), (len(labels), 2), alpha_bars, steps, generator)
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def reference_namespace() -> dict[str, object]:
    """Every solution piece, run once, so the checks compare against exactly the code the files hold."""
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "Callable": Callable,
    }
    pieces = SCHEDULE_PART + TIME_PART + DENOISER_INIT + DENOISER_FORWARD + LOSS_PART + SAMPLING_PART
    exec(pieces + CONDITIONAL_INIT + CONDITIONAL_FORWARD + GUIDANCE_PART, namespace)

    return namespace


REFERENCE = reference_namespace()


def ring(
    count: int,
    generator: torch.Generator,
) -> Tensor:
    """Points on a circle of radius 1, with a little noise in the radius."""
    angle = torch.rand(count, generator=generator) * 2 * math.pi
    radius = 1 + 0.05 * torch.randn(count, generator=generator)

    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)


def ring_distance(points: Tensor) -> float:
    """How far points sit from the circle on average, about 0.04 for the data itself."""
    return (points.norm(dim=1) - 1).abs().mean().item()


def arcs(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """The ring cut into its four quarters, each quarter one class."""
    labels = torch.randint(0, 4, (count,), generator=generator)
    angle = (labels.float() + torch.rand(count, generator=generator)) * (math.pi / 2)
    radius = 1 + 0.05 * torch.randn(count, generator=generator)

    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1), labels


def quarter_of(points: Tensor) -> Tensor:
    angle = torch.atan2(points[:, 1], points[:, 0]) % (2 * math.pi)

    return (angle // (math.pi / 2)).long().clamp(max=3)


def schedule() -> tuple[Tensor, Tensor]:
    betas = REFERENCE["linear_betas"](STEPS, 1e-4, 0.02)

    return betas, REFERENCE["alpha_bars_of"](betas)


def train_ring_denoiser(
    model: nn.Module,
    steps: int,
    predict: str = "noise",
    seed: int = 1,
) -> list[float]:
    """The run behind the lessons' tables, Adam at 0.001 on batches of 256 ring points. With predict "x0"
    the model is trained to guess the data instead of the noise, for diffusion2's comparison."""
    _, alpha_bars = schedule()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    losses = []

    for _ in range(steps):
        x0 = ring(256, generator)
        t = torch.randint(0, STEPS, (256,), generator=generator)
        noise = torch.randn(x0.shape, generator=generator)
        target = noise if predict == "noise" else x0
        loss = F.mse_loss(model(REFERENCE["noise_to"](x0, t, alpha_bars, noise), t), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def noise_guess(
    model: nn.Module,
    predict: str,
) -> Callable[[Tensor, Tensor], Tensor]:
    """The model as predict_noise, turning a guess of the data into the noise it implies."""
    _, alpha_bars = schedule()

    def predict_noise(
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        guess = model(x, t)

        if predict == "noise":
            return guess

        kept = alpha_bars[t].unsqueeze(1)

        return (x - kept.sqrt() * guess) / (1 - kept).sqrt()

    return predict_noise


def train_arcs_denoiser(
    model: nn.Module,
    steps: int,
    p: float = 0.1,
    seed: int = 1,
) -> nn.Module:
    _, alpha_bars = schedule()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        x0, labels = arcs(256, generator)
        labels = REFERENCE["drop_labels"](labels, p, 4, generator)
        t = torch.randint(0, STEPS, (256,), generator=generator)
        noise = torch.randn(x0.shape, generator=generator)
        loss = F.mse_loss(model(REFERENCE["noise_to"](x0, t, alpha_bars, noise), t, labels), noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def seeded_model(
    make: type,
    *arguments: int,
) -> nn.Module:
    torch.manual_seed(0)

    return make(*arguments)


def run_watching_autograd(
    run: Callable[[Callable[[Tensor, Tensor], Tensor]], Tensor],
) -> tuple[Tensor, list[bool]]:
    """Runs a sampler with a made up noise guess that records whether autograd was on at every call."""
    recorded: list[bool] = []

    def predict_noise(
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        recorded.append(torch.is_grad_enabled())

        return 0.3 * x + 0.001 * t.float().unsqueeze(1)

    return run(predict_noise), recorded


# ---------------------------------------------------------------------------
# diffusion1
# ---------------------------------------------------------------------------

DIFFUSION1_LESSON = prose(
    """
    A diffusion model learns to undo noise. First comes the forward process,
    which adds a little Gaussian noise at each of many steps until nothing of
    the data is left. DDPM takes 1,000 steps, with an amount `beta` that
    rises in a line from 0.0001 to 0.02.

    ```text
    x_t = sqrt(1 - beta_t) * x_(t-1) + sqrt(beta_t) * noise
    ```

    The `sqrt(1 - beta_t)` shrinks the point a little as the noise grows.
    Together they keep the spread at 1, so after enough steps every point is
    plain Gaussian noise of spread 1.

    Running 500 steps one at a time would make training slow. The steps
    combine into one jump. With `alpha_t = 1 - beta_t` and `alpha_bar_t` the
    product of every `alpha` up to step t, any step comes straight from the
    data.

    ```text
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    ```

    `alpha_bar_t` says how much of the data is left. Here it is on DDPM's
    schedule.

    | Step | 1 | 100 | 250 | 500 | 1,000 |
    |---|---|---|---|---|---|
    | `alpha_bar` | 0.9999 | 0.897 | 0.524 | 0.0786 | 0.00004 |

    To check the jump, 100,000 copies of the number 3 went through 500 steps
    one at a time. They came out with a mean of 0.838 and a spread of 0.959.
    The jump gives 0.841 and 0.960.

    The code counts steps from 0, so `alpha_bars[t]` is the paper's
    `alpha_bar` at step t + 1.
    """,
)

DIFFUSION1_SOURCE = source_file(
    '''
    # diffusion1.py

    import math

    import torch
    from torch import Tensor

    products = torch.cumprod(torch.tensor([2.0, 3.0, 4.0]), dim=0)    # tensor([ 2.,  6., 24.])
    column = torch.tensor([0.5, 0.9]).reshape(2, 1, 1)                # shape (2, 1, 1), one number per row


    # TODO: Return steps amounts of noise, rising in a line from start to end.
    def linear_betas(
        steps: int,
        start: float,
        end: float,
    ) -> Tensor:
        ...


    # TODO: Return alpha_bar for every step, the product of 1 - beta over every
    #   step up to and including it.
    def alpha_bars_of(betas: Tensor) -> Tensor:
        ...


    # TODO: Return x after one step of noise, beta of it, using noise.
    def noise_step(
        x: Tensor,
        beta: float,
        noise: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return x0 after the jump to step t, in one go. t holds one step per
    #   row of x0, of shape (batch,), and x0 may have any shape after the
    #   batch, like (batch, 2) or (batch, 1, 28, 28).
    def noise_to(
        x0: Tensor,
        t: Tensor,
        alpha_bars: Tensor,
        noise: Tensor,
    ) -> Tensor:
        ...
    ''',
)

DIFFUSION1_SOLUTION = source_file(
    '''
    # diffusion1.py, a solution for comparison

    import math

    import torch
    from torch import Tensor

    products = torch.cumprod(torch.tensor([2.0, 3.0, 4.0]), dim=0)
    column = torch.tensor([0.5, 0.9]).reshape(2, 1, 1)


''' + indented(SCHEDULE_PART) + '''
    ''',
)


def check_diffusion1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="linear_betas",
        description="rises in a line from start to end",
        inputs=(
            (5, 0.1, 0.5),
            (1000, 1e-4, 0.02),
        ),
        reference=REFERENCE["linear_betas"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="alpha_bars_of",
        description="multiplies 1 - beta up to each step",
        inputs=(
            (torch.tensor([0.1, 0.2, 0.5, 0.05]),),
            (REFERENCE["linear_betas"](50, 1e-3, 0.2),),
        ),
        reference=REFERENCE["alpha_bars_of"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="noise_step",
        description="shrinks x and adds beta of the noise",
        inputs=(
            (small_tensor(690, 3, 2), 0.36, small_tensor(691, 3, 2)),
            (small_tensor(692, 4), 0.02, small_tensor(693, 4)),
        ),
        reference=REFERENCE["noise_step"],
        expect=expect_tensor,
    )
    alpha_bars = REFERENCE["alpha_bars_of"](REFERENCE["linear_betas"](20, 0.01, 0.3))
    check_on_inputs(
        checker=checker,
        function="noise_to",
        description="jumps each row to its own step",
        inputs=(
            (small_tensor(694, 3, 2), torch.tensor([0, 7, 19]), Described(alpha_bars, "alpha_bars of 20 steps"), small_tensor(695, 3, 2)),
            (small_tensor(696, 2, 1, 2, 3), torch.tensor([4, 12]), Described(alpha_bars, "alpha_bars of 20 steps"), small_tensor(697, 2, 1, 2, 3)),
        ),
        reference=REFERENCE["noise_to"],
        expect=expect_tensor,
    )


DIFFUSION1 = Exercise(
    name="diffusion1",
    chapter=DIFFUSION,
    title="Noise, step by step and in one jump",
    lesson=DIFFUSION1_LESSON,
    source=DIFFUSION1_SOURCE,
    solution=DIFFUSION1_SOLUTION,
    hints={
        "linear_betas": "torch.linspace gives evenly spaced numbers from start to end, steps of them.",
        "alpha_bars_of": "torch.cumprod of 1 - betas along dim=0.",
        "noise_step": "Multiply x by the square root of 1 - beta, and add the square root of beta times noise.",
        "noise_to": "alpha_bars[t] picks one number per row. Reshape it to (len(t), 1, 1, ...) with as many 1s as x0 "
        "has dimensions after the batch, then use the jump's formula.",
    },
    introduces=(),
    check=check_diffusion1,
)


# ---------------------------------------------------------------------------
# diffusion2
# ---------------------------------------------------------------------------

DIFFUSION2_LESSON = prose(
    """
    The model learns the way back. It sees a noisy point `x_t` and its step
    `t`, and guesses the noise that was mixed in. Knowing the noise gives
    the data back, since the jump's formula turns around.

    ```text
    x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
    ```

    Training repeats one move. Draw a random step for each point of a batch,
    and fresh noise. Jump the batch to those steps, and score the model's
    guess of the noise by the mean squared error. That is DDPM's whole loss.

    The step goes in as sine and cosine features, as position2 did for
    positions, so the model can tell step 10 from step 900.

    Why guess the noise, and not the data? The noise has a spread of 1 at
    every step, so one loss treats every step alike. DDPM's authors found
    that guessing `x_0` gave worse samples on images. Guessing the noise
    reached a score of 3.17 on CIFAR-10, the best at the time.

    That edge is about images. This chapter trains on points of a 2D ring.
    There, after 12,000 steps of training, guessing `x_0` came closer to the
    ring than guessing the noise, 0.035 against 0.065. Which target trains
    best depends on the data.
    """,
)

DIFFUSION2_SOURCE = source_file(
    '''
    # diffusion2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    joined = torch.cat([torch.zeros(4, 2), torch.ones(4, 16)], dim=1)    # shape (4, 18)


    # The schedule and the jump from diffusion1.
''' + indented(SCHEDULE_PART) + '''


''' + indented(TIME_PART) + '''


    # TODO: Finish forward. Join x, of shape (batch, features), with the time
    #   features of t, one step per row, and run net.
''' + indented(DENOISER_INIT) + '''

        def forward(
            self,
            x: Tensor,
            t: Tensor,
        ) -> Tensor:
            ...


    # TODO: Return DDPM's loss on x0. Draw one step per row with
    #   torch.randint(0, len(alpha_bars), (len(x0),), generator=generator),
    #   then noise of x0's shape with torch.randn and generator, in that order.
    #   Jump x0 to those steps, and return the mean squared error between
    #   model's guess and the noise.
    def diffusion_loss(
        model: nn.Module,
        x0: Tensor,
        alpha_bars: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        ...
    ''',
)

DIFFUSION2_SOLUTION = source_file(
    '''
    # diffusion2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    joined = torch.cat([torch.zeros(4, 2), torch.ones(4, 16)], dim=1)


''' + indented(SCHEDULE_PART) + '''


''' + indented(TIME_PART) + '''


''' + indented(DENOISER_INIT) + '''

''' + indented(DENOISER_FORWARD) + '''


''' + indented(LOSS_PART) + '''
    ''',
)


def check_denoiser_forward(found: type) -> None:
    model = expect_module("Denoiser(1000)", seeded_model(found, STEPS))
    x = small_tensor(698, 5, 2) / 3
    t = torch.tensor([0, 10, 400, 999, 37])

    with torch.no_grad():
        expected = seeded_model(REFERENCE["Denoiser"], STEPS)(x, t)

    expect_tensor(
        call="Denoiser(1000) on 5 points at steps 0, 10, 400, 999 and 37",
        actual=model(x, t),
        expected=expected,
    )


def check_diffusion_loss(found: object) -> None:
    _, alpha_bars = schedule()
    model = seeded_model(REFERENCE["Denoiser"], STEPS)
    x0 = ring(64, torch.Generator().manual_seed(699))

    with torch.no_grad():
        expected = REFERENCE["diffusion_loss"](model, x0, alpha_bars, torch.Generator().manual_seed(700)).item()
        actual = found(model, x0, alpha_bars, torch.Generator().manual_seed(700))

    expect_close(
        call="diffusion_loss on 64 ring points, a generator seeded with 700",
        actual=actual.item() if isinstance(actual, Tensor) else actual,
        expected=expected,
        tolerance=1e-6,
    )


def check_denoiser_learns(found: type) -> None:
    """Trains the file's Denoiser with the file's diffusion_loss, which the checks above held to the reference."""
    loss_of = found.forward.__globals__.get("diffusion_loss")

    if not callable(loss_of):
        raise Mismatch("diffusion2.py has no diffusion_loss to train with.")

    _, alpha_bars = schedule()
    model = seeded_model(found, STEPS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(1)
    losses = []

    for _ in range(600):
        loss = loss_of(model, ring(256, generator), alpha_bars, generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    last = sum(losses[-100:]) / 100

    if last > 0.4:
        raise Mismatch(f"After 600 steps the loss averages {last:.3f}, but it should fall below 0.4. A guess of 0 scores 1.")


def check_diffusion2(checker: Checker) -> None:
    checker.test_method(
        class_name="Denoiser",
        method="forward",
        description="joins the point with its step's features",
        run=check_denoiser_forward,
    )
    checker.test(
        function="diffusion_loss",
        description="scores the guess of the noise at a random step",
        run=check_diffusion_loss,
    )
    checker.test_method(
        class_name="Denoiser",
        method="forward",
        description="learns to guess the noise on the ring",
        run=check_denoiser_learns,
    )


DIFFUSION2 = Exercise(
    name="diffusion2",
    chapter=DIFFUSION,
    title="Predicting the noise",
    lesson=DIFFUSION2_LESSON,
    source=DIFFUSION2_SOURCE,
    solution=DIFFUSION2_SOLUTION,
    hints={
        "Denoiser.forward": "time_features(t, self.steps) gives 16 numbers per row. torch.cat them after x on dim=1, "
        "then run self.net.",
        "diffusion_loss": "Draw t, then the noise. noise_to(x0, t, alpha_bars, noise) is the noisy batch, and "
        "F.mse_loss compares model of it and t with the noise.",
    },
    introduces=(),
    check=check_diffusion2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# diffusion3
# ---------------------------------------------------------------------------

DIFFUSION3_LESSON = prose(
    """
    Sampling runs the forward process backwards. Start from pure noise at
    step 1,000, and at every step take away the model's guess of the noise.
    DDPM's step back is this, with a little fresh noise added at every step
    but the last.

    ```text
    x_(t-1) = (x_t - beta_t / sqrt(1 - alpha_bar_t) * guess) / sqrt(1 - beta_t) + sqrt(beta_t) * noise
    ```

    That is 1,000 runs of the model for one batch of samples. DDIM, from
    Song and colleagues, takes a shortcut. From the guess it works out where
    `x_0` would be. Then it jumps straight to an earlier step with that same
    guess of the noise, and adds no fresh noise. So it can skip steps,
    visiting only every 20th or every 50th.

    ```text
    x_0 guess = (x_t - sqrt(1 - alpha_bar_t) * guess) / sqrt(alpha_bar_t)
    x_before  = sqrt(alpha_bar_before) * x_0 guess + sqrt(1 - alpha_bar_before) * guess
    ```

    Here is diffusion2's model after 12,000 steps of training, drawing 2,000
    points. The score is how far they sit from the ring on average. The data
    itself scores 0.039 and pure noise 0.541.

    | Sampler | Steps | Distance from the ring |
    |---|---|---|
    | DDPM | 1,000 | 0.065 |
    | DDIM | 100 | 0.067 |
    | DDIM | 50 | 0.065 |
    | DDIM | 20 | 0.058 |
    | DDIM | 10 | 0.092 |

    DDIM with 20 steps matched DDPM's 1,000 here, and ran about 45 times
    faster in two runs. The DDIM paper reports 10 to 50 times faster
    sampling on images.

    Both samplers take the model as `predict_noise(x, t)`, a function. That
    lets the same samplers serve any model, the conditional one of diffusion4
    included.
    """,
)

DIFFUSION3_SOURCE = source_file(
    '''
    # diffusion3.py

    import math
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    times = list(range(0, 1000, 1000 // 50))[::-1]    # [980, 960, ..., 20, 0], every 20th step
    batch_steps = torch.full((3,), 980)               # tensor([980, 980, 980]), one step per row


    # The schedule from diffusion1.
''' + indented(SCHEDULE_PART) + '''


    # TODO: Return x one step back, DDPM's step from step t, given eps, the
    #   model's guess of the noise, and fresh noise to add.
    def ddpm_step(
        eps: Tensor,
        x: Tensor,
        t: int,
        betas: Tensor,
        alpha_bars: Tensor,
        noise: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return a batch of samples of shape, by DDPM. Start x from
    #   torch.randn(shape, generator=generator). For every step t from the
    #   last down to 0, ask predict_noise(x, a tensor of shape[0] copies of t),
    #   then draw noise of shape with generator, or use zeros at step 0, and
    #   take ddpm_step. Record nothing for autograd.
    def ddpm_sample(
        predict_noise: Callable[[Tensor, Tensor], Tensor],
        shape: tuple[int, ...],
        betas: Tensor,
        alpha_bars: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return a batch of samples of shape, by DDIM with steps steps. Start x
    #   from torch.randn(shape, generator=generator). Visit the steps
    #   range(0, len(alpha_bars), len(alpha_bars) // steps), last first. At each,
    #   ask predict_noise as ddpm_sample does, work out the x_0 guess, and jump
    #   to the next step visited, or to alpha_bar 1 after step 0. Record
    #   nothing for autograd.
    def ddim_sample(
        predict_noise: Callable[[Tensor, Tensor], Tensor],
        shape: tuple[int, ...],
        alpha_bars: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...
    ''',
)

DIFFUSION3_SOLUTION = source_file(
    '''
    # diffusion3.py, a solution for comparison

    import math
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    times = list(range(0, 1000, 1000 // 50))[::-1]
    batch_steps = torch.full((3,), 980)


''' + indented(SCHEDULE_PART) + '''


''' + indented(SAMPLING_PART) + '''
    ''',
)

SMALL_BETAS = REFERENCE["linear_betas"](20, 0.01, 0.3)
SMALL_ALPHA_BARS = REFERENCE["alpha_bars_of"](SMALL_BETAS)


def check_ddpm_sample(found: object) -> None:
    expected, _ = run_watching_autograd(
        lambda predict: REFERENCE["ddpm_sample"](predict, (6, 2), SMALL_BETAS, SMALL_ALPHA_BARS, torch.Generator().manual_seed(701)),
    )
    actual, recorded = run_watching_autograd(
        lambda predict: found(predict, (6, 2), SMALL_BETAS, SMALL_ALPHA_BARS, torch.Generator().manual_seed(701)),
    )

    if len(recorded) != 20:
        raise Mismatch(f"ddpm_sample over 20 steps asked predict_noise {len(recorded)} times, but it should ask once per step.")

    if any(recorded):
        raise Mismatch("ddpm_sample recorded for autograd. Run the loop inside torch.no_grad().")

    expect_tensor(
        call="ddpm_sample of 6 points over 20 steps, with a made up noise guess",
        actual=actual,
        expected=expected,
    )


def check_ddim_sample(found: object) -> None:
    for steps, wanted_calls in ((5, 5), (20, 20)):
        expected, _ = run_watching_autograd(
            lambda predict: REFERENCE["ddim_sample"](predict, (6, 2), SMALL_ALPHA_BARS, steps, torch.Generator().manual_seed(702)),
        )
        actual, recorded = run_watching_autograd(
            lambda predict: found(predict, (6, 2), SMALL_ALPHA_BARS, steps, torch.Generator().manual_seed(702)),
        )

        if len(recorded) != wanted_calls:
            raise Mismatch(f"ddim_sample with {steps} steps asked predict_noise {len(recorded)} times, but it should ask {wanted_calls}.")

        if any(recorded):
            raise Mismatch("ddim_sample recorded for autograd. Run the loop inside torch.no_grad().")

        expect_tensor(
            call=f"ddim_sample of 6 points, {steps} of 20 steps, with a made up noise guess",
            actual=actual,
            expected=expected,
        )


def check_diffusion3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="ddpm_step",
        description="takes away the guess and adds a little noise",
        inputs=(
            (small_tensor(703, 4, 2) / 3, small_tensor(704, 4, 2) / 3, 12, Described(SMALL_BETAS, "betas of 20 steps"), Described(SMALL_ALPHA_BARS, "their alpha_bars"), small_tensor(705, 4, 2) / 3),
            (small_tensor(706, 3, 2) / 3, small_tensor(707, 3, 2) / 3, 0, Described(SMALL_BETAS, "betas of 20 steps"), Described(SMALL_ALPHA_BARS, "their alpha_bars"), torch.zeros(3, 2)),
        ),
        reference=REFERENCE["ddpm_step"],
        expect=expect_tensor,
    )
    checker.test(
        function="ddpm_sample",
        description="walks every step back from noise",
        run=check_ddpm_sample,
    )
    checker.test(
        function="ddim_sample",
        description="jumps back through every few steps",
        run=check_ddim_sample,
    )


DIFFUSION3 = Exercise(
    name="diffusion3",
    chapter=DIFFUSION,
    title="Sampling with DDPM and DDIM",
    lesson=DIFFUSION3_LESSON,
    source=DIFFUSION3_SOURCE,
    solution=DIFFUSION3_SOLUTION,
    hints={
        "ddpm_step": "Take betas[t] / (1 - alpha_bars[t]).sqrt() * eps from x, divide by (1 - betas[t]).sqrt(), and add "
        "betas[t].sqrt() * noise.",
        "ddpm_sample": "Loop t over range(len(betas) - 1, -1, -1) inside torch.no_grad(). torch.full((shape[0],), t) "
        "is the step for every row.",
        "ddim_sample": "Keep the visited steps in a list, last first. The step after position i is times[i + 1], and "
        "after the last one use torch.tensor(1.0) for alpha_bar.",
    },
    introduces=(),
    check=check_diffusion3,
)


# ---------------------------------------------------------------------------
# diffusion4
# ---------------------------------------------------------------------------

DIFFUSION4_LESSON = prose(
    """
    A model that should draw a chosen class takes the label as one more
    input, as an embedding. Classifier free guidance, from Ho and Salimans,
    then makes it follow the label harder, with no separate classifier.

    Training drops the label now and then, 10% of the time here. A dropped
    label becomes a null id, one more row of the embedding. So one model
    learns both, a guess of the noise given the class and one given nothing.

    Sampling mixes the two guesses with a strength `w`.

    ```text
    guess = (1 + w) * guess(x, t, label) - w * guess(x, t, null)
    ```

    The difference between the two guesses points toward what makes the
    class that class. A larger `w` pushes further along it.

    Here is a model trained on the ring cut into four quarters, one class
    each, drawing 2,000 points with DDIM in 50 steps. Spread is how widely
    the points fan out along their quarter. Points spread evenly would score
    0.453.

    | `w` | In the asked quarter | Spread | Distance from the ring |
    |---|---|---|---|
    | 0 | 99.4% | 0.417 | 0.043 |
    | 1 | 100% | 0.267 | 0.034 |
    | 3 | 100% | 0.175 | 0.047 |
    | 7 | 100% | 0.135 | 0.059 |

    Guidance trades variety for certainty. The points crowd into the middle
    of their quarter, and at large `w` they drift off the ring. The paper
    found the same on images, less variety and more fidelity as `w` grows.
    """,
)

DIFFUSION4_SOURCE = source_file(
    '''
    # diffusion4.py

    import math
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    nulls = torch.full_like(torch.tensor([2, 0, 1]), 4)    # tensor([4, 4, 4])


    # The schedule from diffusion1, the time features from diffusion2 and DDIM from diffusion3.
''' + indented(SCHEDULE_PART) + '''


''' + indented(TIME_PART) + '''


''' + indented(SAMPLING_PART) + '''


    # TODO: Finish forward. Join x, the time features of t and the label
    #   embedding of labels, in that order, and run net.
''' + indented(CONDITIONAL_INIT) + '''

        def forward(
            self,
            x: Tensor,
            t: Tensor,
            labels: Tensor,
        ) -> Tensor:
            ...


    # TODO: Return labels with each one replaced by null where its draw from
    #   torch.rand(len(labels), generator=generator) is below p.
    def drop_labels(
        labels: Tensor,
        p: float,
        null: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the guided guess of the noise, model's guess given labels
    #   and its guess given null labels, mixed with strength w.
    def guided_noise(
        model: nn.Module,
        x: Tensor,
        t: Tensor,
        labels: Tensor,
        null: int,
        w: float,
    ) -> Tensor:
        ...


    # TODO: Return one 2D sample for each of labels, by ddim_sample with steps
    #   steps and guided_noise as the guess of the noise.
    def guided_sample(
        model: nn.Module,
        labels: Tensor,
        null: int,
        alpha_bars: Tensor,
        steps: int,
        w: float,
        generator: torch.Generator,
    ) -> Tensor:
        ...
    ''',
)

DIFFUSION4_SOLUTION = source_file(
    '''
    # diffusion4.py, a solution for comparison

    import math
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    nulls = torch.full_like(torch.tensor([2, 0, 1]), 4)


''' + indented(SCHEDULE_PART) + '''


''' + indented(TIME_PART) + '''


''' + indented(SAMPLING_PART) + '''


''' + indented(CONDITIONAL_INIT) + '''

''' + indented(CONDITIONAL_FORWARD) + '''


''' + indented(GUIDANCE_PART) + '''
    ''',
)


def check_conditional_forward(found: type) -> None:
    model = expect_module("ConditionalDenoiser(1000, 4)", seeded_model(found, STEPS, 4))
    x = small_tensor(708, 5, 2) / 3
    t = torch.tensor([0, 10, 400, 999, 37])
    labels = torch.tensor([0, 4, 2, 3, 1])

    with torch.no_grad():
        expected = seeded_model(REFERENCE["ConditionalDenoiser"], STEPS, 4)(x, t, labels)

    expect_tensor(
        call="ConditionalDenoiser(1000, 4) on 5 points, the second with the null label",
        actual=model(x, t, labels),
        expected=expected,
    )


def check_guided_noise(found: object) -> None:
    model = seeded_model(REFERENCE["ConditionalDenoiser"], STEPS, 4)
    x = small_tensor(709, 4, 2) / 3
    t = torch.tensor([3, 500, 999, 120])
    labels = torch.tensor([0, 1, 2, 3])

    for w in (0.0, 2.5):
        with torch.no_grad():
            expected = REFERENCE["guided_noise"](model, x, t, labels, 4, w)
            actual = found(model, x, t, labels, 4, w)

        expect_tensor(
            call=f"guided_noise with w = {w}",
            actual=actual,
            expected=expected,
        )


def check_guided_sample(found: object) -> None:
    """The file's ddim_sample is swapped for one that records what it was given, so guided_sample is held to
    passing the right shape and steps, and a guess of the noise that is the guided one."""
    model = seeded_model(REFERENCE["ConditionalDenoiser"], STEPS, 4)
    labels = torch.tensor([3, 0, 2])
    _, alpha_bars = schedule()
    calls: list[tuple[Callable[[Tensor, Tensor], Tensor], tuple[int, ...], int]] = []
    marker = torch.full((3, 2), 7.0)

    def scripted_ddim_sample(
        predict_noise: Callable[[Tensor, Tensor], Tensor],
        shape: tuple[int, ...],
        alpha_bars: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> Tensor:
        calls.append((predict_noise, tuple(shape), steps))

        return marker

    namespace = found.__globals__
    original = namespace.get("ddim_sample")
    namespace["ddim_sample"] = scripted_ddim_sample

    try:
        actual = found(model, labels, 4, alpha_bars, 25, 3.0, torch.Generator().manual_seed(710))
    finally:
        namespace["ddim_sample"] = original

    if not calls:
        raise Mismatch("guided_sample never called ddim_sample.")

    predict_noise, shape, steps = calls[0]

    if shape != (3, 2) or steps != 25:
        raise Mismatch(f"guided_sample passed ddim_sample the shape {shape} and {steps} steps, but it should pass (3, 2) and 25.")

    x = small_tensor(711, 3, 2) / 3
    t = torch.full((3,), 600)

    with torch.no_grad():
        expected = REFERENCE["guided_noise"](model, x, t, labels, 4, 3.0)
        guess = predict_noise(x, t)

    expect_tensor(
        call="the guess of the noise guided_sample hands ddim_sample, at w = 3",
        actual=guess,
        expected=expected,
    )

    if actual is not marker:
        raise Mismatch("guided_sample should return what ddim_sample gives back.")


def check_diffusion4(checker: Checker) -> None:
    checker.test_method(
        class_name="ConditionalDenoiser",
        method="forward",
        description="joins the point, its step and its label",
        run=check_conditional_forward,
    )
    check_on_inputs(
        checker=checker,
        function="drop_labels",
        description="swaps about p of the labels for the null one",
        inputs=(
            (torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1]), 0.3, 4, seeded(712)),
            (torch.tensor([5, 5, 2, 7, 1, 0]), 0.5, 9, seeded(713)),
        ),
        reference=REFERENCE["drop_labels"],
        expect=expect_tensor,
    )
    checker.test(
        function="guided_noise",
        description="mixes the guesses given the label and given nothing",
        run=check_guided_noise,
    )
    checker.test(
        function="guided_sample",
        description="samples with DDIM and the guided guess",
        run=check_guided_sample,
    )


DIFFUSION4 = Exercise(
    name="diffusion4",
    chapter=DIFFUSION,
    title="Classifier free guidance",
    lesson=DIFFUSION4_LESSON,
    source=DIFFUSION4_SOURCE,
    solution=DIFFUSION4_SOLUTION,
    hints={
        "ConditionalDenoiser.forward": "torch.cat x, time_features(t, self.steps) and self.labels(labels) on dim=1, "
        "then run self.net.",
        "drop_labels": "torch.where picks null where the draws are below p, and the label everywhere else.",
        "guided_noise": "torch.full_like(labels, null) gives null labels. Then the formula in the lesson.",
        "guided_sample": "Hand ddim_sample a lambda of x and t that calls guided_noise, the shape (len(labels), 2), "
        "steps and generator.",
    },
    introduces=(),
    check=check_diffusion4,
)
