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

GAN = Chapter(
    folder="67_gan",
    title="GANs",
    readme=source_file(
        """
        # GANs

        A generative adversarial network trains two models against each
        other. A generator turns noise into samples, and a discriminator
        learns to tell them from real data. This chapter builds both losses
        from the paper, trains them on a ring of points in 2D, measures mode
        collapse, and weighs sharp samples against unstable training.

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


# ---------------------------------------------------------------------------
# gan1
# ---------------------------------------------------------------------------

GAN1_LESSON = prose(
    """
    Goodfellow and colleagues set two networks against each other. The
    generator turns random noise into a sample. The discriminator scores a
    sample, high for real data and low for the generator's fakes.

    The discriminator trains as a plain classifier, real against fake, with
    binary cross entropy. Its scores are logits, and
    `F.binary_cross_entropy_with_logits` takes the sigmoid itself, as in
    activations2 and conv4. Real samples get the label 1 and fakes 0.

    The paper first writes the generator's goal as minimizing
    `log(1 - D(G(z)))`, with D the discriminator's chance that a sample is
    real. That is the generator making the discriminator wrong. Then it points
    out a problem with it. Early on, when the fakes are poor, the
    discriminator rejects them with confidence, and this loss saturates.

    Here is the derivative of each loss with respect to a fake's score.

    | Score | D's chance of real | `log(1 - D)` | `-log D` |
    |---|---|---|---|
    | -5 | 0.007 | -0.0067 | -0.9933 |
    | -2 | 0.119 | -0.1192 | -0.8808 |
    | 0 | 0.5 | -0.5 | -0.5 |
    | 2 | 0.881 | -0.8808 | -0.1192 |

    At a score of -5 the paper's first loss pushes 150 times weaker. So the
    paper trains the generator to maximize `log D(G(z))` instead, which is
    binary cross entropy of the fake scores against the label real. It pulls
    hardest exactly where the fakes are worst. Most GANs use this one.

    `1 - sigmoid(s)` equals `sigmoid(-s)`, so `log(1 - D)` is
    `F.logsigmoid(-s)`. That stays exact even when the sigmoid rounds to 1.
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


    # TODO: Return the discriminator's loss, the binary cross entropy of
    #   real_scores against the label 1 plus that of fake_scores against the
    #   label 0. Both are logits, one per sample.
    def discriminator_loss(
        real_scores: Tensor,
        fake_scores: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the generator's loss that most GANs use, the binary cross
    #   entropy of fake_scores against the label 1.
    def generator_loss(fake_scores: Tensor) -> Tensor:
        ...


    # TODO: Return the paper's first generator loss, the mean of
    #   log(1 - sigmoid(s)) over the fake scores s, written so it stays exact
    #   for any score.
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
        description="gives the mean of log(1 - D), exact for any score",
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
        "generator_loss": "The fakes get the label real here, so the targets are torch.ones_like(fake_scores).",
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
    Training alternates. Each step first trains the discriminator on a batch
    of real points and a batch of fakes. Then it trains the generator to fool
    the discriminator as it now stands.

    The fakes reach the discriminator's loss through `.detach()`. That step
    should change the discriminator alone, and detach cuts the fakes off from
    the generator's graph, so no gradient flows back into the generator. gan3
    shows what goes wrong without it.

    The data is a ring of 8 clusters, points around 8 places on a circle of
    radius 2. Each cluster is one mode. A generator that turns 2 random numbers
    into a point should learn to put its points on all 8.

    ```python
    centers = mode_centers(8)           # shape (8, 2), on a circle of radius 2
    centers[torch.tensor([0, 0, 3])]    # rows 0, 0 and 3, repeats allowed
    ```

    Two numbers say how a generator does, on 2,000 of its samples. One is how
    many modes get at least 2% of the samples within 0.2 of their center. The
    other is the share of all samples that land that close to any mode.

    Here are runs of 4,000 steps, 5 seeds at each learning rate.

    | Learning rate | Modes covered, 5 seeds | Samples near a mode |
    |---|---|---|
    | 1e-4 | 4, 7, 1, 1, 1 | 4% to 32% |
    | 1e-3 | 8, 8, 8, 8, 8 | 89% to 93% |
    | 5e-3 | 8, 5, 7, 8, 7 | 92% to 96% |

    At 1e-4 the generator has barely started. At 1e-3 every run finds all 8
    modes. At 5e-3 the samples are the sharpest of all, yet 3 runs of 5 lose
    whole modes for good.

    At 5e-3 this is **mode collapse**. The generator only has to fool the
    discriminator, so it can pile its samples onto the modes the discriminator
    accepts. Nothing in its loss asks for every mode. Metz and colleagues
    saw a GAN on this same ring hop from mode to mode, one at a time.

    `torch.cdist(a, b)` gives the distance from every row of `a` to every row
    of `b`, and `.min(dim=1)` the nearest one with its index.
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
    # values tensor([0.1000]), indices tensor([0]), the first center is closest


    # discriminator_loss and generator_loss from gan1, then the ring and the networks.
''' + indented(LOSS_PART) + '''


''' + indented(RING_PART) + '''


    # TODO: Return count points of a ring of modes clusters. Draw which mode each
    #   point belongs to with torch.randint(0, modes, (count,), generator=generator),
    #   then its noise with 0.05 * torch.randn(count, 2, generator=generator), in
    #   that order. Each point is its mode's center plus its noise.
    def ring_points(
        count: int,
        modes: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Take one step of training. First the discriminator, on real and on
    #   the generator's fakes made from noise and detached, with d_opt. Then the
    #   generator, on the discriminator's scores of the same fakes, with g_opt.
    #   Zero each optimizer's gradients before its backward. Return both losses
    #   as plain numbers, the discriminator's first.
    def train_step(
        generator: nn.Module,
        discriminator: nn.Module,
        g_opt: torch.optim.Optimizer,
        d_opt: torch.optim.Optimizer,
        real: Tensor,
        noise: Tensor,
    ) -> tuple[float, float]:
        ...


    # TODO: Return how many modes samples cover and the share of samples near a
    #   mode. A sample is near when its nearest center is less than near away. A
    #   mode is covered when at least 2% of all samples are near it.
    def coverage(
        samples: Tensor,
        centers: Tensor,
        near: float,
    ) -> tuple[int, float]:
        ...


    # TODO: Return count points from generator, made from
    #   torch.randn(count, 2, generator=rng), without recording for autograd.
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
    """200 points at mode 0, 3 at mode 1, under 2% of all, and 100 far out but nearest mode 2. Only modes 0 and 1
    get near points, and only mode 0 gets enough to count as covered."""
    rng = torch.Generator().manual_seed(688)
    centers = REFERENCE["mode_centers"](8)

    return torch.cat([
        centers[0] + 0.05 * torch.randn(200, 2, generator=rng),
        centers[1] + 0.05 * torch.randn(3, 2, generator=rng),
        0.5 * centers[2] + 0.05 * torch.randn(100, 2, generator=rng),
    ])


def check_train_step(found: object) -> None:
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
    covered, share = ring_scores(train_gan(found, 0, 1000, 1e-3))

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
            (Described(uneven_samples(), "200 points at one mode, 3 at the next, and 100 far from any"), REFERENCE["mode_centers"](8), 0.2),
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
        "ring_points": "mode_centers(modes) indexed with the drawn modes gives each point's center. Add the noise.",
        "train_step": "fake = generator(noise). The discriminator's loss takes discriminator(fake.detach()). After its "
        "step, the generator's loss takes discriminator(fake). Return d_loss.item() and g_loss.item().",
        "coverage": "torch.cdist(samples, centers).min(dim=1) gives each sample's nearest distance and center. "
        "torch.bincount with minlength counts the near samples per center.",
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
    GANs earned their place with their samples. Here is how far samples land
    from the nearest mode of the ring. The GAN and a VAE of 66_vae's kind both
    trained for 2,000 steps with the same layers.

    | Model, 3 seeds | Mean distance to the nearest mode | Samples near a mode |
    |---|---|---|
    | The data itself | 0.063 | 99.9% |
    | GAN | 0.13 to 0.18 | 73% to 84% |
    | VAE | 0.35 to 0.38 | 39% to 44% |

    The VAE reaches every mode but smears its samples along the ring between
    them. Its loss rewards a decoder that lands near the data on average, and
    a point between two modes is near both. The discriminator has no such
    average, since any point that doesn't look real loses. Weighting the VAE's
    reconstruction 4 times higher did worse here in 2,000 steps, with 1 to 3
    modes.

    The price is training. In gan2, a rate of 1e-4 learned almost nothing in
    4,000 steps, and 5e-3 lost modes. A VAE, a flow or a diffusion model
    lowers one loss, and a falling loss says training works. A GAN's two
    losses fight each other, and neither says the samples are good.

    The `train_step` below raises no error and still trains wrong. Its fakes
    reach the discriminator's loss without `.detach()`, so that backward also
    fills the generator's gradients, with the discriminator's goal. They are
    zeroed only at the top, so the generator's own backward adds to them.

    | Step, 2,000 steps at 1e-3 | Modes covered, 5 seeds | Samples near a mode |
    |---|---|---|
    | Fixed | 8, 8, 8, 8, 7 | 73% to 87% |
    | Missing detach | 4, 5, 5, 4, 5 | 86% to 89% |

    The broken step looks healthy, with sharp samples and no error, yet it
    drops modes on every seed. Added together, the two gradients pull each
    fake's score toward 0, where the discriminator can't call it either way,
    instead of toward real. Adding `.detach()` to the fakes in the
    discriminator's loss fixes it.

    So reach for a GAN when samples must be sharp and there is time to tune
    it, as StyleGAN did for faces. Many recent image generators, like DALL-E 2
    and Stable Diffusion, use diffusion instead, which 69_diffusion builds.
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


    # TODO: This step raises no error and trains wrong, as the lesson says. Fix
    #   the one line that lets the discriminator's backward reach the generator.
''' + indented(BUGGY_STEP_PART) + '''


    # TODO: Return the mean distance from each sample to its nearest center, as a
    #   plain number.
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
        description="updates the generator with its own gradients alone",
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
        "train_step": "The discriminator's loss should see the fakes cut off from the generator. Add .detach() where "
        "its loss takes the generator's output.",
        "nearest_distance": "torch.cdist of the samples and centers, the smallest of each row with .min(dim=1), then "
        "the mean as a plain number.",
    },
    introduces=(),
    check=check_gan3,
    imports_first=OPTIMIZER_IMPORTS,
)
