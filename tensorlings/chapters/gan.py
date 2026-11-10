"""67_gan, a generator and a discriminator trained against each other, mode collapse on a ring of points, and when a GAN is worth its unstable training."""

from __future__ import annotations

import math
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
    expect_value,
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from .classic_nets import pinned_threads

GAN = Chapter(
    folder="67_gan",
    title="GANs",
    readme=source_file(
        """
        # GANs

        A generative adversarial network, or GAN, trains two models against
        each other. A generator turns random noise into samples. A
        discriminator learns to tell those samples apart from real data. This
        chapter builds both losses from the original paper and trains the two
        models on a ring of points in 2D. It measures mode collapse, and it
        weighs the GAN's sharp samples against its unstable training.

        ## Further reading

        - [Goodfellow and colleagues, generative adversarial networks](https://arxiv.org/abs/1406.2661)
        - [Metz and colleagues, unrolled generative adversarial networks](https://arxiv.org/abs/1611.02163)
        - [Salimans and colleagues, improved techniques for training GANs](https://arxiv.org/abs/1606.03498)
        """,
    ),
)

# The pieces later exercises are handed, written once so the files never drift apart.
LOSS_PART = '''
def discriminator_loss(
    real_scores: Tensor,
    fake_scores: Tensor,
) -> Tensor:
    real = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores))
    fake = F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))

    return real + fake


def generator_loss(fake_scores: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))
'''

MINIMAX_PART = '''
def minimax_generator_loss(fake_scores: Tensor) -> Tensor:
    return F.logsigmoid(-fake_scores).mean()
'''

RING_PART = '''
def mode_centers(
    modes: int,
    radius: float = 2.0,
) -> Tensor:
    angles = 2 * math.pi * torch.arange(modes) / modes

    return radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


def mlp(
    inputs: int,
    outputs: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, outputs),
    )
'''

RING_POINTS_PART = '''
def ring_points(
    count: int,
    modes: int,
    generator: torch.Generator,
) -> Tensor:
    which = torch.randint(0, modes, (count,), generator=generator)
    noise = 0.05 * torch.randn(count, 2, generator=generator)

    return mode_centers(modes)[which] + noise
'''

TRAIN_STEP_PART = '''
def train_step(
    generator: nn.Module,
    discriminator: nn.Module,
    g_opt: torch.optim.Optimizer,
    d_opt: torch.optim.Optimizer,
    real: Tensor,
    noise: Tensor,
) -> tuple[float, float]:
    fake = generator(noise)
    d_loss = discriminator_loss(discriminator(real), discriminator(fake.detach()))
    d_opt.zero_grad()
    d_loss.backward()
    d_opt.step()

    g_loss = generator_loss(discriminator(fake))
    g_opt.zero_grad()
    g_loss.backward()
    g_opt.step()

    return d_loss.item(), g_loss.item()
'''

COVERAGE_PART = '''
def coverage(
    samples: Tensor,
    centers: Tensor,
    near: float,
) -> tuple[int, float]:
    nearest = torch.cdist(samples, centers).min(dim=1)
    good = nearest.values < near
    counts = torch.bincount(nearest.indices[good], minlength=len(centers))

    return int((counts >= 0.02 * len(samples)).sum()), good.float().mean().item()


def sample(
    generator: nn.Module,
    count: int,
    rng: torch.Generator,
) -> Tensor:
    with torch.no_grad():
        return generator(torch.randn(count, 2, generator=rng))
'''

BUGGY_STEP_PART = '''
def train_step(
    generator: nn.Module,
    discriminator: nn.Module,
    g_opt: torch.optim.Optimizer,
    d_opt: torch.optim.Optimizer,
    real: Tensor,
    noise: Tensor,
) -> tuple[float, float]:
    g_opt.zero_grad()
    d_opt.zero_grad()

    d_loss = discriminator_loss(discriminator(real), discriminator(generator(noise)))
    d_loss.backward()
    d_opt.step()

    g_loss = generator_loss(discriminator(generator(noise)))
    g_loss.backward()
    g_opt.step()

    return d_loss.item(), g_loss.item()
'''

FIXED_STEP_PART = BUGGY_STEP_PART.replace("discriminator(generator(noise)))\n    d_loss", "discriminator(generator(noise).detach()))\n    d_loss")

NEAREST_PART = '''
def nearest_distance(
    samples: Tensor,
    centers: Tensor,
) -> float:
    return torch.cdist(samples, centers).min(dim=1).values.mean().item()
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def reference_namespace(*parts: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec("\n\n".join(parts), namespace)

    return namespace


REFERENCE = reference_namespace(LOSS_PART, MINIMAX_PART, RING_PART, RING_POINTS_PART, TRAIN_STEP_PART, COVERAGE_PART, NEAREST_PART)
BUGGY = reference_namespace(LOSS_PART, RING_PART, BUGGY_STEP_PART)


def seeded_pair(seed: int) -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(seed)
    generator = REFERENCE["mlp"](2, 2)
    discriminator = REFERENCE["mlp"](2, 1)

    return generator, discriminator


def train_gan(
    step: object,
    seed: int,
    steps: int,
    lr: float,
) -> nn.Module:
    """The run every number in this chapter comes from, Adam with betas (0.5, 0.999) on batches of 256."""
    generator, discriminator = seeded_pair(seed)
    g_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    rng = torch.Generator().manual_seed(seed + 100)

    for _ in range(steps):
        real = REFERENCE["ring_points"](256, 8, rng)
        noise = torch.randn(256, 2, generator=rng)
        step(generator, discriminator, g_opt, d_opt, real, noise)

    return generator


def ring_scores(generator: nn.Module) -> tuple[int, float]:
    """How many of the 8 modes a generator covers, and the share of its samples near one, on 2,000 samples."""
    samples = REFERENCE["sample"](generator, 2000, torch.Generator().manual_seed(7))

    return REFERENCE["coverage"](samples, REFERENCE["mode_centers"](8), 0.2)


def train_ring_vae(
    seed: int,
    steps: int,
    sigma: float,
) -> nn.Module:
    """gan3's VAE on the ring, with the GAN's layers and batches. Its reconstruction is the squared error over
    2 * sigma ** 2, the negative log density of a normal of spread sigma, so halving sigma weighs it 4 times higher."""
    torch.manual_seed(seed)
    encoder = REFERENCE["mlp"](2, 4)
    decoder = REFERENCE["mlp"](2, 2)
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=1e-3)
    rng = torch.Generator().manual_seed(seed + 100)

    for _ in range(steps):
        real = REFERENCE["ring_points"](256, 8, rng)
        mean, log_variance = encoder(real).chunk(2, dim=1)
        code = mean + torch.exp(0.5 * log_variance) * torch.randn(256, 2, generator=rng)

        reconstruction = ((real - decoder(code)) ** 2).sum(dim=1) / (2 * sigma**2)
        kl = 0.5 * (mean**2 + log_variance.exp() - 1 - log_variance).sum(dim=1)
        loss = (reconstruction + kl).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return decoder


def measure_ring_models() -> dict[str, list[tuple[float, int, float]]]:
    """Every number of gan3's first table and of its stronger reconstruction, as the mean distance to the nearest
    mode, the modes covered and the share near a mode for each seed. Both models train 8,000 steps, since at 2,000 the
    VAE is still far from where it settles. A few minutes on a laptop CPU. Not run by any check."""
    centers = REFERENCE["mode_centers"](8)

    def scores(samples: Tensor) -> tuple[float, int, float]:
        covered, near = REFERENCE["coverage"](samples, centers, 0.2)

        return REFERENCE["nearest_distance"](samples, centers), covered, near

    def drawn(model: nn.Module) -> Tensor:
        return REFERENCE["sample"](model, 2000, torch.Generator().manual_seed(7))

    seeds = range(3)

    with pinned_threads():
        return {
            "data": [scores(REFERENCE["ring_points"](2000, 8, torch.Generator().manual_seed(7)))],
            "gan": [scores(drawn(train_gan(REFERENCE["train_step"], seed, 8000, 1e-3))) for seed in seeds],
            "vae, sigma 0.1": [scores(drawn(train_ring_vae(seed, 8000, 0.1))) for seed in seeds],
            "vae, sigma 0.05": [scores(drawn(train_ring_vae(seed, 8000, 0.05))) for seed in seeds],
        }


def measure_learning_rates() -> dict[float, list[tuple[int, float]]]:
    """gan2's table, the modes covered and the share near a mode after 4,000 steps, on seeds 0 to 4 at each rate. A
    few minutes on a laptop CPU. Not run by any check."""
    with pinned_threads():
        return {
            lr: [ring_scores(train_gan(REFERENCE["train_step"], seed, 4000, lr)) for seed in range(5)]
            for lr in (1e-4, 1e-3, 5e-3)
        }


def measure_modes_over_time(lr: float = 5e-3) -> list[list[int]]:
    """gan2's runs at 5e-3 carried on to 12,000 steps, with the modes covered every 1,000 steps, on seeds 0 to 4.
    The same run as train_gan, counted along the way. A few minutes on a laptop CPU. Not run by any check."""
    tracks: list[list[int]] = []

    with pinned_threads():
        for seed in range(5):
            generator, discriminator = seeded_pair(seed)
            g_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
            d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
            rng = torch.Generator().manual_seed(seed + 100)
            covered: list[int] = []

            for step in range(1, 12_001):
                real = REFERENCE["ring_points"](256, 8, rng)
                noise = torch.randn(256, 2, generator=rng)
                REFERENCE["train_step"](generator, discriminator, g_opt, d_opt, real, noise)

                if step % 1000 == 0:
                    covered.append(ring_scores(generator)[0])

            tracks.append(covered)

    return tracks


# ---------------------------------------------------------------------------
# gan1
# ---------------------------------------------------------------------------

GAN1_LESSON = prose(
    """
    Goodfellow and colleagues introduced the GAN, which trains two networks
    against each other. The generator turns random noise into a sample. The
    discriminator gives each sample a score. The score should be high for
    real data and low for the generator's samples, which are called fakes.

    The discriminator is trained as an ordinary binary classifier of real
    against fake, with binary cross entropy. Its scores are logits.
    `F.binary_cross_entropy_with_logits` applies the sigmoid to them itself,
    as in activations2 and conv4. Real samples get the label 1 and fakes get
    the label 0.

    The paper first writes the generator's goal as minimizing
    `log(1 - D(G(z)))`. Here z is the noise and G(z) is a fake. D is the
    probability the discriminator gives that a sample is real, the sigmoid of
    its score. Minimizing this loss pushes D(G(z)) up, so the generator
    learns to make the discriminator wrong. Then the paper points out a
    problem. Early in training the fakes are poor, so the discriminator
    rejects them confidently. There this loss saturates, which means it is
    nearly flat and its gradient is close to 0.

    Here is the derivative of each loss with respect to a fake's score.

    | Score | D | `log(1 - D)` | `-log D` |
    |---|---|---|---|
    | -5 | 0.007 | -0.0067 | -0.9933 |
    | -2 | 0.119 | -0.1192 | -0.8808 |
    | 0 | 0.5 | -0.5 | -0.5 |
    | 2 | 0.881 | -0.8808 | -0.1192 |

    At a score of -5, the derivative of the first loss is about 150 times
    smaller than the derivative of `-log D`. So the paper trains the
    generator to maximize `log D(G(z))` instead. That is the same as
    minimizing the binary cross entropy of the fake scores against the label
    1, which means real. Its gradient is largest exactly where the fakes are
    worst. Most GANs use this loss.

    Because `1 - sigmoid(s)` equals `sigmoid(-s)`, `log(1 - D)` equals
    `F.logsigmoid(-s)`. `F.logsigmoid` stays accurate even when the sigmoid
    would round to 1 and `log(1 - D)` would become `log(0)`.
    """,
)

GAN1_SOURCE = source_file(
    '''
    # gan1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    ones = torch.ones_like(torch.tensor([2.0, -1.0]))                          # tensor([1., 1.])
    loss = F.binary_cross_entropy_with_logits(torch.tensor([2.0]), torch.tensor([0.0]))    # tensor(2.1269)
    logs = F.logsigmoid(torch.tensor([0.0, 3.0]))                              # tensor([-0.6931, -0.0486])


    # TODO: Return the discriminator's loss. It is the binary cross entropy of
    #   real_scores against the label 1, plus the binary cross entropy of
    #   fake_scores against the label 0. Both inputs are 1D tensors of logits,
    #   one per sample.
    #   Example: discriminator_loss(torch.tensor([2.0]), torch.tensor([2.0]))
    #   returns about tensor(2.2539).
    def discriminator_loss(
        real_scores: Tensor,
        fake_scores: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the generator loss that most GANs use, the binary cross
    #   entropy of fake_scores against the label 1.
    #   Example: generator_loss(torch.tensor([0.0])) returns about
    #   tensor(0.6931).
    def generator_loss(fake_scores: Tensor) -> Tensor:
        ...


    # TODO: Return the paper's first generator loss, the mean of
    #   log(1 - sigmoid(s)) over the fake scores s. Compute it in a way that
    #   stays accurate for any score, as the end of the lesson shows.
    #   Example: minimax_generator_loss(torch.tensor([0.0, 40.0])) returns
    #   about tensor(-20.3466).
    def minimax_generator_loss(fake_scores: Tensor) -> Tensor:
        ...
    ''',
)

GAN1_SOLUTION = source_file(
    '''
    # gan1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    ones = torch.ones_like(torch.tensor([2.0, -1.0]))
    loss = F.binary_cross_entropy_with_logits(torch.tensor([2.0]), torch.tensor([0.0]))
    logs = F.logsigmoid(torch.tensor([0.0, 3.0]))


''' + indented(LOSS_PART) + '''


''' + indented(MINIMAX_PART) + '''
    ''',
)


def scores_for_check(
    seed: int,
    count: int,
) -> Tensor:
    return small_tensor(seed, count) * 2


def expect_scalar_tensor(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    if isinstance(actual, Tensor) and actual.dim() != 0:
        raise Mismatch(f"{call} has shape {tuple(actual.shape)}, but a loss is one number. Take the mean over the samples.")

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def check_gan1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="discriminator_loss",
        description="labels the real scores 1 and the fake ones 0",
        inputs=(
            (scores_for_check(670, 5), scores_for_check(671, 5)),
            (scores_for_check(672, 3), scores_for_check(673, 4)),
        ),
        reference=REFERENCE["discriminator_loss"],
        expect=expect_scalar_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="generator_loss",
        description="labels the fake scores 1",
        inputs=(
            (scores_for_check(674, 5),),
            (torch.tensor([-12.0, 0.5, 30.0]),),
        ),
        reference=REFERENCE["generator_loss"],
        expect=expect_scalar_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="minimax_generator_loss",
        description="returns the mean of log(1 - D), accurate for any score",
        inputs=(
            (scores_for_check(675, 5),),
            (torch.tensor([-3.0, 0.0, 40.0]),),
        ),
        reference=REFERENCE["minimax_generator_loss"],
        expect=expect_scalar_tensor,
    )


GAN1 = Exercise(
    name="gan1",
    chapter=GAN,
    title="Two networks, two losses",
    lesson=GAN1_LESSON,
    source=GAN1_SOURCE,
    solution=GAN1_SOLUTION,
    hints={
        "discriminator_loss": "Call F.binary_cross_entropy_with_logits twice, once on the real scores with "
        "torch.ones_like of them, once on the fake scores with torch.zeros_like, and add the two.",
        "generator_loss": "Here the fakes get the label 1, meaning real, so the targets are "
        "torch.ones_like(fake_scores).",
        "minimax_generator_loss": "log(1 - sigmoid(s)) is log(sigmoid(-s)), which F.logsigmoid of the negated scores "
        "gives. Take its mean.",
    },
    introduces=(),
    check=check_gan1,
)


# ---------------------------------------------------------------------------
# gan2
# ---------------------------------------------------------------------------

GAN2_LESSON = prose(
    """
    GAN training alternates between the two networks. Each step first trains
    the discriminator on a batch of real points and a batch of fakes. Then it
    trains the generator to fool the discriminator, as the discriminator is
    after that update.

    The fakes go into the discriminator's loss through `.detach()`. The
    discriminator's update should change only the discriminator. `.detach()`
    cuts the fakes off from the generator's computation graph, so no
    gradient from that loss flows back into the generator. gan3 shows what
    goes wrong without it.

    The data is a ring of 8 clusters of points around 8 evenly spaced
    centers on a circle of radius 2. Each cluster is called a mode. The
    generator turns 2 random numbers into a point, and it should learn to put
    its points on all 8 modes.

    ```python
    centers = mode_centers(8)           # shape (8, 2), on a circle of radius 2
    centers[torch.tensor([0, 0, 3])]    # rows 0, 0 and 3, repeats allowed
    ```

    Two numbers measure how well a generator does, computed on 2,000 of its
    samples. The first is the number of modes covered. A mode counts as
    covered when at least 2% of the samples are within 0.2 of its center. The
    second is the share of all samples that are within 0.2 of any mode's
    center.

    Here are training runs of 4,000 steps, with 5 seeds at each learning
    rate.

    | Learning rate | Modes covered, 5 seeds | Samples near a mode |
    |---|---|---|
    | 1e-4 | 4, 7, 1, 1, 1 | 4% to 32% |
    | 1e-3 | 8, 8, 8, 8, 8 | 89% to 93% |
    | 5e-3 | 8, 5, 7, 8, 7 | 92% to 96% |

    At 1e-4 the generator has barely started learning. At 1e-3 every run
    finds all 8 modes. At 5e-3 the samples sit closest to the modes, but 3 of
    the 5 runs are missing whole modes at 4,000 steps.

    Missing modes like this is called **mode collapse**. The generator only
    has to fool the discriminator, so it can put all its samples on the modes
    the discriminator accepts as real. Nothing in its loss asks it to cover
    every mode. Metz and colleagues saw a GAN on a ring like this one jump
    from mode to mode, covering one at a time. When the runs at 5e-3 here
    continue to 12,000 steps, they also lose modes and regain them. 4 of the
    5 end up missing one mode.

    `torch.cdist(a, b)` returns the distance from every row of `a` to every
    row of `b`, as a matrix with one row for each row of `a`. `.min(dim=1)`
    then gives, for each row of `a`, the smallest distance and the index of
    the nearest row of `b`.
    """,
)

GAN2_SOURCE = source_file(
    '''
    # gan2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    nearest = torch.cdist(torch.tensor([[0.0, 0.1]]), torch.tensor([[0.0, 0.0], [2.0, 0.0]])).min(dim=1)
    # values tensor([0.1000]), indices tensor([0]), so the first center is closest


    # discriminator_loss and generator_loss from gan1, then the ring and the networks.
''' + indented(LOSS_PART) + '''


''' + indented(RING_PART) + '''


    # TODO: Return count points from a ring of modes clusters, as a tensor of
    #   shape (count, 2). First draw the mode of each point with
    #   torch.randint(0, modes, (count,), generator=generator). Then draw the
    #   noise with 0.05 * torch.randn(count, 2, generator=generator). Keep that
    #   order. Each point is the center of its mode plus its noise.
    def ring_points(
        count: int,
        modes: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Do one training step. Make the fakes by passing noise through
    #   generator. First update the discriminator with d_opt, using its loss on
    #   real and on the detached fakes. Then update the generator with g_opt,
    #   using its loss on the discriminator's scores of the same fakes. Zero
    #   each optimizer's gradients before the matching backward. Return both
    #   losses as Python floats, the discriminator's loss first.
    def train_step(
        generator: nn.Module,
        discriminator: nn.Module,
        g_opt: torch.optim.Optimizer,
        d_opt: torch.optim.Optimizer,
        real: Tensor,
        noise: Tensor,
    ) -> tuple[float, float]:
        ...


    # TODO: Return a tuple of two numbers, how many modes the samples cover and
    #   the share of samples near a mode. A sample is near a mode when the
    #   distance to its nearest center is less than near. A mode is covered
    #   when at least 2% of all samples are near it.
    #   Example: coverage(torch.tensor([[2.0, 0.0], [2.0, 0.1], [0.0, 0.0]]),
    #   mode_centers(8), 0.2) returns (1, 0.6666666865348816).
    def coverage(
        samples: Tensor,
        centers: Tensor,
        near: float,
    ) -> tuple[int, float]:
        ...


    # TODO: Return count points from generator. Pass it the noise
    #   torch.randn(count, 2, generator=rng), with autograd recording turned
    #   off.
    def sample(
        generator: nn.Module,
        count: int,
        rng: torch.Generator,
    ) -> Tensor:
        ...
    ''',
)

GAN2_SOLUTION = source_file(
    '''
    # gan2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    nearest = torch.cdist(torch.tensor([[0.0, 0.1]]), torch.tensor([[0.0, 0.0], [2.0, 0.0]])).min(dim=1)


''' + indented(LOSS_PART) + '''


''' + indented(RING_PART) + '''


''' + indented(RING_POINTS_PART) + '''


''' + indented(TRAIN_STEP_PART) + '''


''' + indented(COVERAGE_PART) + '''
    ''',
)


def expect_coverage(
    call: str,
    actual: object,
    expected: tuple[int, float],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should give the modes covered and the share near a mode, as a tuple of two.")

    expect_value(
        call=f"the modes covered of {call}",
        actual=actual[0],
        expected=expected[0],
    )
    expect_close(
        call=f"the share near a mode of {call}",
        actual=actual[1],
        expected=expected[1],
    )


def uneven_samples() -> Tensor:
    """200 points at mode 0, 5 at mode 1, and 100 far out but nearest mode 2. Only modes 0 and 1 get near points.
    Mode 1's 5 are under 2% of all the samples but over 2% of the near ones, so only mode 0 counts as covered."""
    rng = torch.Generator().manual_seed(688)
    centers = REFERENCE["mode_centers"](8)

    return torch.cat([
        centers[0] + 0.05 * torch.randn(200, 2, generator=rng),
        centers[1] + 0.05 * torch.randn(5, 2, generator=rng),
        0.5 * centers[2] + 0.05 * torch.randn(100, 2, generator=rng),
    ])


def naming_a_missing_detach(found: object) -> object:
    """found, with PyTorch's error about a freed graph turned into the change a learner should make."""

    def step(*arguments: object) -> object:
        try:
            return found(*arguments)
        except RuntimeError as error:
            if "backward through the graph a second time" not in str(error):
                raise

            raise Mismatch(
                "train_step's second backward needed the part of the graph that its first backward had already freed. "
                "The fakes went into the discriminator's loss still connected to the generator. Detach them there, "
                "with fake.detach().",
            ) from error

    return step


def keeps_old_gradients(found: object) -> bool:
    """Whether found steps with gradients left over from before. Every gradient starts filled, which a step that
    zeroes each optimizer's gradients before its backward clears."""
    rng = torch.Generator().manual_seed(676)
    real = REFERENCE["ring_points"](32, 8, rng)
    noise = torch.randn(32, 2, generator=rng)
    weights = []

    for step in (found, REFERENCE["train_step"]):
        generator, discriminator = seeded_pair(677)

        for parameter in [*generator.parameters(), *discriminator.parameters()]:
            parameter.grad = torch.full_like(parameter, 0.5)

        g_opt = torch.optim.SGD(generator.parameters(), lr=0.1)
        d_opt = torch.optim.SGD(discriminator.parameters(), lr=0.1)
        step(generator, discriminator, g_opt, d_opt, real, noise)
        weights.append(torch.cat([generator[0].weight.detach().flatten(), discriminator[0].weight.detach().flatten()]))

    return not torch.allclose(weights[0], weights[1], atol=1e-6)


def check_train_step(found: object) -> None:
    found = naming_a_missing_detach(found)
    rng = torch.Generator().manual_seed(676)
    real = REFERENCE["ring_points"](32, 8, rng)
    noise = torch.randn(32, 2, generator=rng)
    results = []

    for step in (found, REFERENCE["train_step"]):
        generator, discriminator = seeded_pair(677)
        g_opt = torch.optim.SGD(generator.parameters(), lr=0.1)
        d_opt = torch.optim.SGD(discriminator.parameters(), lr=0.1)
        losses = step(generator, discriminator, g_opt, d_opt, real, noise)
        results.append((losses, generator[0].weight.detach().clone(), discriminator[0].weight.detach().clone()))

    (losses, generator_weight, discriminator_weight), (expected_losses, expected_generator, expected_discriminator) = results

    if not isinstance(losses, tuple) or len(losses) != 2:
        raise Mismatch("train_step should return both losses as a tuple, the discriminator's first.")

    for name, actual, expected in zip(("discriminator", "generator"), losses, expected_losses):
        expect_close(
            call=f"the {name}'s loss from train_step",
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        )

    expect_tensor(
        call="the discriminator's first weights after one train_step with SGD",
        actual=discriminator_weight,
        expected=expected_discriminator,
    )
    expect_tensor(
        call="the generator's first weights after one train_step with SGD",
        actual=generator_weight,
        expected=expected_generator,
    )

    if keeps_old_gradients(found):
        raise Mismatch(
            "train_step stepped with gradients left over from before it ran. Zero each optimizer's gradients before "
            "its backward.",
        )


def check_sample(found: object) -> None:
    generator, _ = seeded_pair(678)
    actual = found(generator, 5, torch.Generator().manual_seed(679))
    expected = REFERENCE["sample"](generator, 5, torch.Generator().manual_seed(679))
    expect_tensor(
        call="sample(a generator, 5, a generator seeded with 679)",
        actual=actual,
        expected=expected,
    )

    if actual.requires_grad:
        raise Mismatch("sample's points are still recorded for autograd. Make them inside torch.no_grad().")


def check_learns_ring(found: object) -> None:
    covered, share = ring_scores(train_gan(naming_a_missing_detach(found), 0, 1000, 1e-3))

    if covered < 6 or share < 0.4:
        raise Mismatch(
            f"After 1,000 steps of train_step, the generator covers {covered} of 8 modes with {share:.1%} of its "
            "samples near one, but it should cover at least 6 with at least 40% near.",
        )


def check_gan2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="ring_points",
        description="draws points around the ring's centers",
        inputs=(
            (6, 8, seeded(680)),
            (4, 3, seeded(681)),
        ),
        reference=REFERENCE["ring_points"],
        expect=expect_tensor,
    )
    checker.test(
        function="train_step",
        description="trains the discriminator on detached fakes, then the generator",
        run=check_train_step,
    )
    check_on_inputs(
        checker=checker,
        function="coverage",
        description="counts the modes covered and the share near a mode",
        inputs=(
            (Described(REFERENCE["ring_points"](300, 8, torch.Generator().manual_seed(682)), "300 ring points"), REFERENCE["mode_centers"](8), 0.2),
            (Described(REFERENCE["ring_points"](300, 4, torch.Generator().manual_seed(683)) * 1.1, "300 points of a ring of 4, stretched"), REFERENCE["mode_centers"](8), 0.15),
            (Described(uneven_samples(), "200 points at one mode, 5 at the next, and 100 far from any"), REFERENCE["mode_centers"](8), 0.2),
        ),
        reference=REFERENCE["coverage"],
        expect=expect_coverage,
    )
    checker.test(
        function="sample",
        description="draws points without recording for autograd",
        run=check_sample,
    )
    checker.test(
        function="train_step",
        description="trains a generator that covers the ring",
        run=check_learns_ring,
    )


GAN2 = Exercise(
    name="gan2",
    chapter=GAN,
    title="Training on a ring, and mode collapse",
    lesson=GAN2_LESSON,
    source=GAN2_SOURCE,
    solution=GAN2_SOLUTION,
    hints={
        "ring_points": "Index mode_centers(modes) with the drawn modes to get each point's center. Then add the noise.",
        "train_step": "Make the fakes once, by calling the generator on the noise. The discriminator's loss takes "
        "discriminator(fake.detach()). After the discriminator's step, the generator's loss takes "
        "discriminator(fake). Return d_loss.item() and g_loss.item().",
        "coverage": "torch.cdist(samples, centers).min(dim=1) gives each sample's distance to its nearest center and "
        "that center's index. torch.bincount with minlength counts the near samples at each center.",
        "sample": "Inside torch.no_grad(), run generator on the noise.",
    },
    introduces=(),
    check=check_gan2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# gan3
# ---------------------------------------------------------------------------

GAN3_LESSON = prose(
    """
    GANs became popular because of the quality of their samples. The table
    shows how far samples are from the nearest mode of the ring. A GAN and a
    VAE like the one in 66_vae were both trained for 8,000 steps with the
    same layers.

    | Model, 3 seeds | Mean distance to the nearest mode | Samples near a mode |
    |---|---|---|
    | The data itself | 0.063 | 99.9% |
    | GAN | 0.064 to 0.072 | 95% to 96% |
    | VAE | 0.20 to 0.23 | 64% to 67% |

    The GAN's samples are almost as close to the modes as the real data is.
    The VAE reaches every mode, but it spreads its samples along the ring
    between the modes. The VAE's loss rewards a decoder whose outputs are
    near the data on average, and a point between two modes is fairly near
    both. The discriminator does not average like this. It gives a low score
    to any point that does not look real. Weighting the VAE's reconstruction
    loss 4 times higher still reached every mode, but it put fewer samples
    near a mode, 53% to 57%.

    The cost of a GAN is harder training. In gan2, a learning rate of 1e-4
    had barely started after 4,000 steps, and 5e-3 lost modes. A VAE, a flow
    or a diffusion model minimizes a single loss, and a falling loss tells
    you that training is working. A GAN has two losses that work against
    each other, and neither one tells you whether the samples are good.

    The `train_step` in this exercise raises no error but still trains
    incorrectly. Its fakes go into the discriminator's loss without
    `.detach()`. So the discriminator's backward also fills the generator's
    gradients, pushing the generator toward the discriminator's goal. The
    gradients are zeroed only at the start of the step. So the generator's
    own backward adds its gradients on top of those wrong ones.

    | Step, 2,000 steps at 1e-3 | Modes covered, 5 seeds | Samples near a mode |
    |---|---|---|
    | Fixed | 8, 8, 8, 8, 7 | 73% to 87% |
    | Missing detach | 4, 5, 5, 4, 5 | 86% to 89% |

    The broken step looks healthy, with sharp samples and no error, but it
    drops modes on every seed. The two gradients added together push each
    fake's score toward 0 instead of toward real. A score of 0 means a
    probability of 0.5, so the discriminator cannot tell whether that fake
    is real. Adding `.detach()` to the fakes in the discriminator's loss
    fixes the bug.

    So use a GAN when samples must be sharp and there is time to tune its
    training. Diffusion models, which 69_diffusion builds, have taken over
    image generation. Neural audio codecs like SoundStream and EnCodec, the
    kind sound2 quantizes, still train against a discriminator.
    """,
)

GAN3_SOURCE = source_file(
    '''
    # gan3.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    distances = torch.cdist(torch.tensor([[0.0, 0.3]]), torch.tensor([[0.0, 0.0], [2.0, 0.0]]))
    closest = distances.min(dim=1).values    # tensor([0.3000])


    # discriminator_loss and generator_loss from gan1, then the ring and the networks.
''' + indented(LOSS_PART) + '''


''' + indented(RING_PART) + '''


    # TODO: This step raises no error but trains incorrectly, as the lesson
    #   explains. Fix the one line that lets the discriminator's backward send
    #   gradients into the generator.
''' + indented(BUGGY_STEP_PART) + '''


    # TODO: Find the distance from each sample to its nearest center, and
    #   return the mean of those distances as a Python float.
    #   Example: nearest_distance(torch.tensor([[0.0, 0.3]]),
    #   torch.tensor([[0.0, 0.0], [2.0, 0.0]])) returns about 0.3.
    def nearest_distance(
        samples: Tensor,
        centers: Tensor,
    ) -> float:
        ...
    ''',
)

GAN3_SOLUTION = source_file(
    '''
    # gan3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    distances = torch.cdist(torch.tensor([[0.0, 0.3]]), torch.tensor([[0.0, 0.0], [2.0, 0.0]]))
    closest = distances.min(dim=1).values


''' + indented(LOSS_PART) + '''


''' + indented(RING_PART) + '''


''' + indented(FIXED_STEP_PART) + '''


''' + indented(NEAREST_PART) + '''
    ''',
)

FIXED = reference_namespace(LOSS_PART, RING_PART, FIXED_STEP_PART)


def measure_detach() -> dict[str, list[tuple[int, float]]]:
    """gan3's second table, the modes covered and the share near a mode after 2,000 steps at 1e-3 with the fixed step
    and with the step missing its detach, on seeds 0 to 4. A minute or two on a laptop CPU. Not run by any check."""
    with pinned_threads():
        return {
            name: [ring_scores(train_gan(namespace["train_step"], seed, 2000, 1e-3)) for seed in range(5)]
            for name, namespace in (("fixed", FIXED), ("missing detach", BUGGY))
        }


def check_fixed_step(found: object) -> None:
    rng = torch.Generator().manual_seed(684)
    real = REFERENCE["ring_points"](32, 8, rng)
    noise = torch.randn(32, 2, generator=rng)
    weights = []

    for step in (found, FIXED["train_step"]):
        generator, discriminator = seeded_pair(685)
        g_opt = torch.optim.SGD(generator.parameters(), lr=0.1)
        d_opt = torch.optim.SGD(discriminator.parameters(), lr=0.1)
        step(generator, discriminator, g_opt, d_opt, real, noise)
        weights.append((generator[0].weight.detach().clone(), discriminator[0].weight.detach().clone()))

    (generator_weight, discriminator_weight), (expected_generator, expected_discriminator) = weights
    expect_tensor(
        call="the discriminator's first weights after one train_step with SGD",
        actual=discriminator_weight,
        expected=expected_discriminator,
    )
    expect_tensor(
        call="the generator's first weights after one train_step with SGD",
        actual=generator_weight,
        expected=expected_generator,
    )


def check_gan3(checker: Checker) -> None:
    checker.test(
        function="train_step",
        description="updates the generator with only its own gradients",
        run=check_fixed_step,
    )
    check_on_inputs(
        checker=checker,
        function="nearest_distance",
        description="gives the mean distance to the nearest center",
        inputs=(
            (Described(REFERENCE["ring_points"](50, 8, torch.Generator().manual_seed(686)), "50 ring points"), REFERENCE["mode_centers"](8)),
            (small_tensor(687, 6, 2) / 2, REFERENCE["mode_centers"](4)),
        ),
        reference=REFERENCE["nearest_distance"],
        expect=expect_close,
    )


GAN3 = Exercise(
    name="gan3",
    chapter=GAN,
    title="When a GAN, and a silent bug",
    lesson=GAN3_LESSON,
    source=GAN3_SOURCE,
    solution=GAN3_SOLUTION,
    hints={
        "train_step": "The discriminator's loss should get fakes that are cut off from the generator. Add .detach() "
        "where that loss takes the generator's output.",
        "nearest_distance": "Take torch.cdist of the samples and the centers, then the smallest value in each row with "
        ".min(dim=1). Return the mean of those as a Python float.",
    },
    introduces=(),
    check=check_gan3,
    imports_first=OPTIMIZER_IMPORTS,
)
