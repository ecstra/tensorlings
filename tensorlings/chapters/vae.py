"""66_vae, an encoder that outputs a mean and a log variance, the ELBO of a small VAE on MNIST, and what a VAE's code space gives that an autoencoder's doesn't."""

from __future__ import annotations

import functools
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
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from .classic_nets import mnist_images
from .linear_layers import expect_module

VAE = Chapter(
    folder="66_vae",
    title="Variational autoencoders",
    readme=source_file(
        """
        # Variational autoencoders

        A variational autoencoder, or VAE, is an autoencoder whose code is
        random. During training, an extra loss term pulls the distribution of
        every code toward a standard normal distribution. To make new data, you
        draw a code from the standard normal and decode it. This chapter builds
        the encoder and its KL term, trains a small VAE on MNIST with its ELBO
        loss, and measures what a VAE's codes can do that a plain
        autoencoder's codes cannot.

        ## Further reading

        - [Kingma and Welling, auto-encoding variational Bayes](https://arxiv.org/abs/1312.6114)
        - [Doersch, tutorial on variational autoencoders](https://arxiv.org/abs/1606.05908)
        - [Kingma and Welling, an introduction to variational autoencoders](https://arxiv.org/abs/1906.02691)
        - [Ghosh and colleagues, from variational to deterministic autoencoders](https://arxiv.org/abs/1903.12436)
        """,
    ),
)

ENCODER_INIT = '''
class Encoder(nn.Module):
    def __init__(
        self,
        code: int,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(784, hidden)
        self.mean = nn.Linear(hidden, code)
        self.log_variance = nn.Linear(hidden, code)
'''

ENCODER_FORWARD = '''
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = torch.relu(self.hidden(x))

        return self.mean(h), self.log_variance(h)
'''

KL_PARTS = '''
def kl_to_standard(
    mean: Tensor,
    log_variance: Tensor,
) -> Tensor:
    return 0.5 * (mean**2 + log_variance.exp() - log_variance - 1).sum(dim=-1)


def log_normal(
    z: Tensor,
    mean: Tensor,
    log_variance: Tensor,
) -> Tensor:
    return -0.5 * (math.log(2 * math.pi) + log_variance + (z - mean) ** 2 / log_variance.exp())


def kl_by_sampling(
    mean: Tensor,
    log_variance: Tensor,
    count: int,
    generator: torch.Generator,
) -> Tensor:
    noise = torch.randn(count, *mean.shape, generator=generator)
    z = mean + torch.exp(0.5 * log_variance) * noise
    gaps = log_normal(z, mean, log_variance) - log_normal(z, torch.zeros_like(z), torch.zeros_like(z))

    return gaps.sum(dim=-1).mean(dim=0)
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


KL_REFERENCE = reference_namespace(ENCODER_INIT + ENCODER_FORWARD, KL_PARTS)


# ---------------------------------------------------------------------------
# vae1
# ---------------------------------------------------------------------------

VAE1_LESSON = prose(
    """
    An autoencoder compresses an input into a small code and then rebuilds
    the input from that code, as in autoencoder1. A variational autoencoder,
    or VAE, from Kingma and Welling, makes the code random. For each number in
    the code, the encoder outputs a mean and a log variance. The code is then
    drawn from a normal distribution with that mean and variance, using the
    reparameterization trick from sampling3.

    The encoder is one hidden layer followed by two heads. A head is a final
    Linear layer that reads the hidden layer. One head outputs the means and
    the other outputs the log variances. sampling3 explained why a model
    outputs the log variance instead of the standard deviation.

    Training pulls the normal distribution of every code toward the standard
    normal, which has mean 0 and standard deviation 1. The distance between
    two distributions is measured by the **KL divergence**. It is 0 when the
    two distributions are the same, and it grows as they become more
    different. For a normal distribution against the standard normal, the KL
    divergence has a closed form, which means an exact formula with no
    sampling in it. The KL of a whole code is that formula summed over the
    numbers in the code.

    ```text
    KL = 0.5 * sum(mean ** 2 + variance - log_variance - 1)
    ```

    Each part of the formula pulls toward a different target. `mean ** 2` is
    smallest when the mean is 0, so it pulls the mean to 0. The part
    `variance - log_variance - 1` is 0 when the variance is 1 and grows when
    the variance is above or below 1. So it pulls the variance to 1.

    You can also get the KL by sampling. Call the code's own normal
    distribution q and the standard normal p. Draw many codes z from q,
    compute `log q(z) - log p(z)` for each, and average. The closed form is
    exactly the value this average approaches. The log density of a normal
    distribution at a point z is given by this formula.

    ```text
    log q(z) = -0.5 * (log(2 * pi) + log_variance + ((z - mean) ** 2) / variance)
    ```

    The sampled average gets close to the closed form only slowly as the
    number of draws grows. Here is a code of 2 numbers, with means 0.5 and -1
    and variances 1 and 0.25. Each cell gives the range of results over 3
    random seeds.

    | Draws | 100 | 10,000 | Closed form |
    |---|---|---|---|
    | KL | 0.89 to 1.02 | 0.933 to 0.951 | 0.943 |

    The closed form is one line of code and has no random noise, which is why
    VAEs use it.
    """,
)

VAE1_SOURCE = source_file(
    '''
    # vae1.py

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    log_variance = torch.tensor([0.0, math.log(0.25)])
    variance = log_variance.exp()                       # tensor([1.0000, 0.2500])


    # TODO: Finish forward. Pass x, of shape (batch, 784), through self.hidden
    #   and then a ReLU. Feed the result to both heads and return two tensors,
    #   the output of self.mean and the output of self.log_variance, each of
    #   shape (batch, code).
''' + indented(ENCODER_INIT) + '''

        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
            ...


    # TODO: Return the KL divergence between each code's normal distribution
    #   and the standard normal, using the closed form from the lesson. Sum over
    #   the last dimension. mean and log_variance have shape (batch, code), and
    #   the result has shape (batch,).
    #   Example: kl_to_standard(torch.tensor([[0.5, -1.0]]),
    #   log_variance.unsqueeze(0)) returns about tensor([0.9431]).
    def kl_to_standard(
        mean: Tensor,
        log_variance: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the log density of a normal distribution with the given mean
    #   and log_variance, evaluated at each entry of z. Use the lesson's
    #   formula entry by entry, so the result has the same shape as z.
    #   Example: log_normal(torch.tensor(0.0), torch.tensor(0.0),
    #   torch.tensor(0.0)) returns about tensor(-0.9189).
    def log_normal(
        z: Tensor,
        mean: Tensor,
        log_variance: Tensor,
    ) -> Tensor:
        ...


    # TODO: Estimate the KL divergence of each code by sampling, and return it
    #   with shape (batch,). Draw noise of shape (count, *mean.shape) with
    #   torch.randn and generator. Turn the noise into codes z with the
    #   reparameterization trick. For each draw, compute log q(z) - log p(z)
    #   with log_normal, where q is the code's normal and p is the standard
    #   normal. Sum that over the code dimension, then average over the count
    #   draws.
    def kl_by_sampling(
        mean: Tensor,
        log_variance: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...
    ''',
)

VAE1_SOLUTION = source_file(
    '''
    # vae1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    log_variance = torch.tensor([0.0, math.log(0.25)])
    variance = log_variance.exp()


''' + indented(ENCODER_INIT) + '''

''' + indented(ENCODER_FORWARD) + '''


''' + indented(KL_PARTS) + '''
    ''',
)


def seeded_encoder(make: type) -> nn.Module:
    torch.manual_seed(660)

    return make(3, 8)


def check_encoder(found: type) -> None:
    model = expect_module("Encoder(3, 8)", seeded_encoder(found))
    reference = seeded_encoder(KL_REFERENCE["Encoder"])
    x = torch.rand(4, 784, generator=torch.Generator().manual_seed(661))
    result = model(x)

    if not isinstance(result, tuple) or len(result) != 2:
        raise Mismatch("Encoder(3, 8) should give the mean and the log variance as a tuple of two tensors.")

    with torch.no_grad():
        expected_mean, expected_log_variance = reference(x)

    expect_tensor(
        call="the mean from Encoder(3, 8) on 4 images",
        actual=result[0],
        expected=expected_mean,
    )
    expect_tensor(
        call="the log variance from Encoder(3, 8) on 4 images",
        actual=result[1],
        expected=expected_log_variance,
    )


EXAMPLE_MEAN = torch.tensor([[0.5, -1.0]])
EXAMPLE_LOG_VARIANCE = torch.tensor([[0.0, math.log(0.25)]])


def measure_kl_by_sampling(seed: int = 0) -> dict[str, float]:
    """vae1's table for one seed, the sampled KL of the lesson's code with a fresh generator for each count, and the
    closed form. The lesson gives seeds 0 to 2. Not run by any check."""
    table = {
        f"{count} draws": KL_REFERENCE["kl_by_sampling"](
            EXAMPLE_MEAN,
            EXAMPLE_LOG_VARIANCE,
            count,
            torch.Generator().manual_seed(seed),
        ).item()
        for count in (100, 10_000)
    }
    table["closed form"] = KL_REFERENCE["kl_to_standard"](EXAMPLE_MEAN, EXAMPLE_LOG_VARIANCE).item()

    return table


def check_vae1(checker: Checker) -> None:
    checker.test_method(
        class_name="Encoder",
        method="forward",
        description="returns a mean and a log variance for each number in the code",
        run=check_encoder,
    )
    check_on_inputs(
        checker=checker,
        function="kl_to_standard",
        description="gives each code's KL by the closed form",
        inputs=(
            (EXAMPLE_MEAN, EXAMPLE_LOG_VARIANCE),
            (small_tensor(662, 3, 4) / 2, small_tensor(663, 3, 4) / 3),
        ),
        reference=KL_REFERENCE["kl_to_standard"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="log_normal",
        description="gives the log density entry by entry",
        inputs=(
            (torch.tensor([0.0, 1.0, -2.0]), torch.tensor([0.0, 0.5, -1.0]), torch.tensor([0.0, -1.0, 1.0])),
            (small_tensor(664, 2, 3), small_tensor(665, 2, 3) / 2, small_tensor(666, 2, 3) / 3),
        ),
        reference=KL_REFERENCE["log_normal"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="kl_by_sampling",
        description="averages log q minus log p over the draws",
        inputs=(
            (EXAMPLE_MEAN, EXAMPLE_LOG_VARIANCE, 50, seeded(667)),
            (small_tensor(668, 3, 4) / 2, small_tensor(669, 3, 4) / 3, 20, seeded(670)),
        ),
        reference=KL_REFERENCE["kl_by_sampling"],
        expect=expect_tensor,
    )


VAE1 = Exercise(
    name="vae1",
    chapter=VAE,
    title="A random code and its KL",
    lesson=VAE1_LESSON,
    source=VAE1_SOURCE,
    solution=VAE1_SOLUTION,
    hints={
        "Encoder.forward": "Pass x through self.hidden and torch.relu. Feed that to self.mean and self.log_variance "
        "and return both.",
        "kl_to_standard": "log_variance.exp() is the variance. Combine the four terms of the formula, sum over "
        "dim=-1, and multiply by 0.5.",
        "log_normal": "Write the lesson's formula with math.log(2 * math.pi) and log_variance.exp() as the variance.",
        "kl_by_sampling": "The standard deviation is torch.exp of half the log variance. A sample is the mean plus "
        "the standard deviation times the noise. For log p, call log_normal with zeros for the mean and the log "
        "variance. Sum log q minus log p over dim=-1, then average over dim=0.",
    },
    introduces=(),
    check=check_vae1,
)


# ---------------------------------------------------------------------------
# vae2
# ---------------------------------------------------------------------------

VAE_INIT = '''
class VAE(nn.Module):
    def __init__(
        self,
        code: int,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(code, hidden)
        self.decoder = nn.Sequential(
            nn.Linear(code, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 784),
        )
'''

VAE_FORWARD = '''
    def forward(
        self,
        x: Tensor,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_variance = self.encoder(x)
        noise = torch.randn(mean.shape, generator=generator)
        code = mean + torch.exp(0.5 * log_variance) * noise

        return self.decoder(code), mean, log_variance
'''

ELBO_PARTS = '''
def negative_elbo(
    logits: Tensor,
    x: Tensor,
    mean: Tensor,
    log_variance: Tensor,
) -> Tensor:
    reconstruction = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
    kl = 0.5 * (mean**2 + log_variance.exp() - log_variance - 1).sum()

    return (reconstruction + kl) / len(x)


def sample_digits(
    model: nn.Module,
    count: int,
    generator: torch.Generator,
) -> Tensor:
    model.eval()

    with torch.no_grad():
        codes = torch.randn(count, model.decoder[0].in_features, generator=generator)
        digits = torch.sigmoid(model.decoder(codes))

    model.train()

    return digits
'''

VAE_REFERENCE = reference_namespace(ENCODER_INIT + ENCODER_FORWARD, VAE_INIT + VAE_FORWARD, ELBO_PARTS)


def train_vae(
    model: nn.Module,
    x: Tensor,
    steps: int,
    seed: int,
) -> nn.Module:
    """The run behind vae2's table, Adam at 0.001 on batches of 128 images."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        batch = torch.randint(0, len(x), (128,), generator=generator)
        logits, mean, log_variance = model(x[batch], generator)
        loss = VAE_REFERENCE["negative_elbo"](logits, x[batch], mean, log_variance)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


@functools.cache
def flat_mnist() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    images = mnist_images()

    return (
        images["train_x"].reshape(-1, 784),
        images["train_y"],
        images["validation_x"].reshape(-1, 784),
        images["validation_y"],
    )


def elbo_parts(
    model: nn.Module,
    x: Tensor,
) -> tuple[float, float]:
    """The average reconstruction and KL per image, the two halves of vae2's table."""
    with torch.no_grad():
        logits, mean, log_variance = model(x, torch.Generator().manual_seed(5))
        reconstruction = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
        kl = KL_REFERENCE["kl_to_standard"](mean, log_variance).mean()

    return reconstruction.item(), kl.item()


VAE2_LESSON = prose(
    """
    A VAE is trained to minimize the negative ELBO. ELBO stands for evidence
    lower bound, a term from Kingma and Welling. The ELBO is a lower bound on
    the log likelihood of each image. So raising the ELBO trains the VAE by
    likelihood, as Section 2 did for other models. The negative ELBO of an
    image is the sum of two parts.

    - The reconstruction loss measures how badly the decoder rebuilds the image from a sampled code, as a binary cross entropy summed over 784 pixels.
    - The KL term is vae1's KL divergence between the code's normal distribution and the standard normal.

    Each pixel value is between 0 and 1, and it is treated as the probability
    that the pixel has ink. The decoder's 784 outputs are logits for those
    probabilities. `reduction="sum"` adds up the loss over every pixel and
    every image instead of averaging it. Dividing that sum by the batch size
    gives the loss per image.

    ```python
    F.binary_cross_entropy_with_logits(torch.zeros(2, 3), torch.ones(2, 3), reduction="sum")    # tensor(4.1589)
    ```

    The two parts work against each other. The reconstruction loss is lowest
    when each code says exactly which image it came from. The KL term is
    lowest when every code is a draw from the same standard normal, which
    says nothing about the image. Training ends up at a balance between the
    two.

    Here is the training run behind this exercise, on MNIST with a code of 64
    numbers and a hidden layer of 256 units. Each loss is the average per
    image on 2,000 images the model never trained on.

    | Steps | 0 | 400 | 2,000 |
    |---|---|---|---|
    | Reconstruction | 551.4 | 129.3 | 90.0 |
    | KL | 0.35 | 19.1 | 24.4 |

    The KL starts near 0, because a freshly initialized encoder outputs means
    near 0 and variances near 1. It rises as the codes start to carry
    information about each image.

    Making new digits needs only the decoder. Draw codes from the standard
    normal and decode them. Training pulled every code toward that normal, so
    these random codes look like the codes the decoder saw in training. After
    4,000 training steps, a digit classifier gives one digit a probability
    above 0.9 for 44.3% to 46.1% of 1,000 such samples, over 3 seeds. vae3
    shows what the same random codes give for a plain autoencoder.
    """,
)

VAE2_SOURCE = source_file(
    '''
    # vae2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    pixels = F.binary_cross_entropy_with_logits(torch.zeros(2, 3), torch.ones(2, 3), reduction="sum")    # tensor(4.1589)


    # Encoder from vae1.
''' + indented(ENCODER_INIT) + '''

''' + indented(ENCODER_FORWARD) + '''


    # TODO: Finish forward. Encode x to get the mean and the log variance. Draw
    #   noise with the shape of the mean, using torch.randn and generator, and
    #   turn it into a code with the reparameterization trick. Decode the code
    #   and return three tensors, the decoder's logits, the mean and the log
    #   variance.
''' + indented(VAE_INIT) + '''

        def forward(
            self,
            x: Tensor,
            generator: torch.Generator,
        ) -> tuple[Tensor, Tensor, Tensor]:
            ...


    # TODO: Return the negative ELBO per image, as a tensor holding one number.
    #   Compute the binary cross entropy of logits against x, summed over every
    #   pixel and every image. Add vae1's KL, summed over every code number and
    #   every image. Then divide the total by the number of images.
    #   Example: negative_elbo(torch.zeros(1, 784), torch.ones(1, 784),
    #   torch.zeros(1, 4), torch.zeros(1, 4)) returns about tensor(543.4275).
    def negative_elbo(
        logits: Tensor,
        x: Tensor,
        mean: Tensor,
        log_variance: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return count new digits as a tensor of shape (count, 784), with
    #   pixel values from 0 to 1. Draw count codes from the standard normal with
    #   torch.randn and generator. Each code has as many numbers as
    #   model.decoder[0] takes as input. Decode the codes and apply a sigmoid.
    #   Do this in eval mode with autograd recording turned off, then put the
    #   model back in training mode.
    def sample_digits(
        model: nn.Module,
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...
    ''',
)

VAE2_SOLUTION = source_file(
    '''
    # vae2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    pixels = F.binary_cross_entropy_with_logits(torch.zeros(2, 3), torch.ones(2, 3), reduction="sum")


''' + indented(ENCODER_INIT) + '''

''' + indented(ENCODER_FORWARD) + '''


''' + indented(VAE_INIT) + '''

''' + indented(VAE_FORWARD) + '''


''' + indented(ELBO_PARTS) + '''
    ''',
)


def seeded_vae(
    make: type,
    code: int = 4,
    hidden: int = 16,
) -> nn.Module:
    torch.manual_seed(671)

    return make(code, hidden)


def check_vae_forward(found: type) -> None:
    model = expect_module("VAE(4, 16)", seeded_vae(found))
    reference = seeded_vae(VAE_REFERENCE["VAE"])
    x = torch.rand(3, 784, generator=torch.Generator().manual_seed(672))
    result = model(x, torch.Generator().manual_seed(673))

    if not isinstance(result, tuple) or len(result) != 3:
        raise Mismatch("VAE(4, 16) should give the logits, the mean and the log variance as a tuple of three tensors.")

    with torch.no_grad():
        expected = reference(x, torch.Generator().manual_seed(673))

    for name, part, wanted in zip(("logits", "mean", "log variance"), result, expected):
        expect_tensor(
            call=f"the {name} from VAE(4, 16) on 3 images",
            actual=part,
            expected=wanted,
        )


def check_vae_learns(found: type) -> None:
    train_x, _, validation_x, _ = flat_mnist()
    model = train_vae(seeded_vae(found, 64, 256), train_x[:10000], 400, 674)
    model.eval()

    with torch.no_grad():
        logits, mean, log_variance = model(validation_x[:1000], torch.Generator().manual_seed(675))
        loss = VAE_REFERENCE["negative_elbo"](logits, validation_x[:1000], mean, log_variance).item()

    if loss > 170:
        raise Mismatch(f"After 400 steps the VAE's negative ELBO on new images is {loss:.1f}, but it should be below 170.")


def check_sample_digits(found: object) -> None:
    model = seeded_vae(VAE_REFERENCE["VAE"])
    expected = VAE_REFERENCE["sample_digits"](model, 5, torch.Generator().manual_seed(676))
    modes: list[tuple[bool, bool]] = []
    hook = model.decoder.register_forward_pre_hook(lambda module, inputs: modes.append((module.training, torch.is_grad_enabled())))

    try:
        actual = found(model, 5, torch.Generator().manual_seed(676))
    finally:
        hook.remove()

    if any(training for training, _ in modes):
        raise Mismatch("sample_digits ran the decoder in training mode. Call model.eval() first.")

    if any(recording for _, recording in modes):
        raise Mismatch("sample_digits recorded for autograd. Decode inside torch.no_grad().")

    if not model.training:
        raise Mismatch("sample_digits left the model in eval mode. Call model.train() before returning.")

    expect_tensor(
        call="sample_digits(VAE(4, 16), 5, a generator seeded with 676)",
        actual=actual,
        expected=expected,
    )


def check_vae2(checker: Checker) -> None:
    checker.test_method(
        class_name="VAE",
        method="forward",
        description="encodes, draws the code with the reparameterization trick, and decodes",
        run=check_vae_forward,
    )
    check_on_inputs(
        checker=checker,
        function="negative_elbo",
        description="adds the reconstruction loss and the KL, per image",
        inputs=(
            (
                small_tensor(677, 3, 784) / 2,
                Described(torch.rand(3, 784, generator=torch.Generator().manual_seed(678)), "3 images of 784 pixels"),
                small_tensor(679, 3, 4) / 2,
                small_tensor(680, 3, 4) / 3,
            ),
        ),
        reference=VAE_REFERENCE["negative_elbo"],
        expect=expect_tensor,
    )
    checker.test(
        function="sample_digits",
        description="decodes standard normal codes in eval mode",
        run=check_sample_digits,
    )
    checker.test_method(
        class_name="VAE",
        method="forward",
        description="learns MNIST, reaching a negative ELBO below 170 after 400 steps",
        run=check_vae_learns,
    )


VAE2 = Exercise(
    name="vae2",
    chapter=VAE,
    title="The ELBO on MNIST",
    lesson=VAE2_LESSON,
    source=VAE2_SOURCE,
    solution=VAE2_SOLUTION,
    hints={
        "VAE.forward": "self.encoder returns the mean and the log variance of x. The noise is torch.randn(mean.shape, "
        "generator=generator). Decode the mean plus the standard deviation times the noise. Return the logits, the "
        "mean and the log variance.",
        "negative_elbo": "F.binary_cross_entropy_with_logits with reduction=\"sum\" gives the reconstruction loss. Add "
        "vae1's KL formula summed over everything, then divide by len(x).",
        "sample_digits": "model.decoder[0].in_features is the code size. Inside torch.no_grad(), decode "
        "torch.randn(count, size, generator=generator) and take torch.sigmoid.",
    },
    introduces=(),
    check=check_vae2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# vae3
# ---------------------------------------------------------------------------

SHARE_PARTS = '''
def confident_share(
    classifier: nn.Module,
    images: Tensor,
    threshold: float,
) -> float:
    classifier.eval()

    with torch.no_grad():
        sure = torch.softmax(classifier(images), dim=1).max(dim=1).values > threshold

    classifier.train()

    return sure.float().mean().item()


def matched_codes(
    codes: Tensor,
    count: int,
    generator: torch.Generator,
) -> Tensor:
    noise = torch.randn(count, codes.shape[1], generator=generator)

    return codes.mean(dim=0) + codes.std(dim=0) * noise


def midpoint_codes(
    codes: Tensor,
    count: int,
    generator: torch.Generator,
) -> Tensor:
    first = torch.randint(0, len(codes), (count,), generator=generator)
    second = torch.randint(0, len(codes), (count,), generator=generator)

    return (codes[first] + codes[second]) / 2


def decoded_share(
    decoder: nn.Module,
    classifier: nn.Module,
    codes: Tensor,
    threshold: float,
) -> float:
    with torch.no_grad():
        images = torch.sigmoid(decoder(codes))

    return confident_share(classifier, images, threshold)
'''

SHARE_REFERENCE = reference_namespace(SHARE_PARTS)


class ReferenceAutoencoder(nn.Module):
    """A VAE's layers with the noise and the KL taken away, the plain autoencoder of vae3's table."""

    def __init__(
        self,
        code: int,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = VAE_REFERENCE["Encoder"](code, hidden)
        self.decoder = nn.Sequential(
            nn.Linear(code, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 784),
        )

    def forward(
        self,
        x: Tensor,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_variance = self.encoder(x)

        return self.decoder(mean), mean, log_variance


def train_autoencoder(
    model: nn.Module,
    x: Tensor,
    steps: int,
    seed: int,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        batch = torch.randint(0, len(x), (128,), generator=generator)
        logits, _, _ = model(x[batch], generator)
        loss = F.binary_cross_entropy_with_logits(logits, x[batch], reduction="sum") / len(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_classifier(
    x: Tensor,
    y: Tensor,
) -> nn.Module:
    """The digit classifier that judges the samples, an MLP that gets about 97% of validation images right."""
    torch.manual_seed(3)
    classifier = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(4)

    for _ in range(3000):
        batch = torch.randint(0, len(x), (128,), generator=generator)
        loss = F.cross_entropy(classifier(x[batch]), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return classifier


@functools.cache
def judge() -> nn.Module:
    train_x, train_y, _, _ = flat_mnist()

    return train_classifier(train_x, train_y)


def measure_code_spaces(
    seed: int = 0,
    code: int = 64,
) -> dict[str, object]:
    """Every number of vae2's and vae3's tables for one seed, with 4 threads, a few minutes on a laptop CPU. The
    lessons give seeds 0 to 2. Not run by any check."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        return code_space_numbers(seed, code)
    finally:
        torch.set_num_threads(threads)


def code_space_numbers(
    seed: int,
    code: int,
) -> dict[str, object]:
    train_x, _, validation_x, validation_y = flat_mnist()
    classifier = judge()
    share = SHARE_REFERENCE["confident_share"]
    decoded = SHARE_REFERENCE["decoded_share"]

    with torch.no_grad():
        accuracy = (classifier(validation_x).argmax(dim=1) == validation_y).float().mean().item()

    results: dict[str, object] = {
        "classifier accuracy": accuracy,
        "real images": share(classifier, validation_x[:1000], 0.9),
    }

    torch.manual_seed(671 + seed)
    vae = VAE_REFERENCE["VAE"](code, 256)
    results["vae reconstruction, KL at 0"] = elbo_parts(vae, validation_x[:2000])
    train_vae(vae, train_x, 400, 1 + 10 * seed)
    results["vae reconstruction, KL at 400"] = elbo_parts(vae, validation_x[:2000])
    train_vae(vae, train_x, 1600, 2 + 10 * seed)
    results["vae reconstruction, KL at 2000"] = elbo_parts(vae, validation_x[:2000])
    train_vae(vae, train_x, 2000, 3 + 10 * seed)
    # The same draw as the standard normal row of vae3's table, so both lessons quote one number.
    results["vae samples"] = share(classifier, VAE_REFERENCE["sample_digits"](vae, 1000, torch.Generator().manual_seed(8)), 0.9)

    torch.manual_seed(671 + seed)
    autoencoder = ReferenceAutoencoder(code, 256)
    train_autoencoder(autoencoder, train_x, 4000, 1 + 10 * seed)

    for name, model in (("autoencoder", autoencoder), ("vae", vae)):
        with torch.no_grad():
            codes, _ = model.encoder(train_x[:20000])

        results[f"{name} code spread"] = codes.std(dim=0).mean().item()
        results[f"{name} own codes"] = decoded(model.decoder, classifier, codes[:1000], 0.9)
        standard = torch.randn(1000, code, generator=torch.Generator().manual_seed(8))
        results[f"{name} standard normal"] = decoded(model.decoder, classifier, standard, 0.9)
        matched = SHARE_REFERENCE["matched_codes"](codes, 1000, torch.Generator().manual_seed(9))
        results[f"{name} matched"] = decoded(model.decoder, classifier, matched, 0.9)
        midpoints = SHARE_REFERENCE["midpoint_codes"](codes, 1000, torch.Generator().manual_seed(10))
        results[f"{name} midpoints"] = decoded(model.decoder, classifier, midpoints, 0.9)

    with torch.no_grad():
        means, log_variances = vae.encoder(train_x[:20000])

    own_spreads = torch.exp(0.5 * log_variances)
    drawn = means + own_spreads * torch.randn(means.shape, generator=torch.Generator().manual_seed(0))
    kl_per_number = KL_REFERENCE["kl_to_standard"](means.unsqueeze(-1), log_variances.unsqueeze(-1)).mean(dim=0)
    results["vae own spread"] = own_spreads.mean().item()
    results["vae drawn code spread"] = drawn.std(dim=0).mean().item()
    results["vae numbers in use"] = int((kl_per_number > 0.1).sum())

    return results


VAE3_LESSON = prose(
    """
    The plain autoencoder from autoencoder1 also compresses images into codes
    and decodes them back. So why add the noise and the KL term? The answer
    is in where the codes end up.

    An autoencoder's codes end up wherever training happened to push them.
    Here an autoencoder and a VAE, each with a code of 64 numbers, were
    trained for 4,000 steps on MNIST, on 3 seeds. For the autoencoder, the
    standard deviation of each code number across images is about 2.2 on
    average.

    The VAE stores information about each image in only 22 or 23 of its 64
    numbers. On the other numbers the KL term wins. Their mean stays at 0 and
    their variance at 1, so they carry nothing about the image. Take the codes
    of all images, drawn with their noise. Each code number then has a
    standard deviation of about 1.01 across images, close to the standard
    normal's 1.

    To make a new digit, you draw a code and decode it. A digit classifier
    judges each result. It is a small MLP that classifies 97.4% of unseen
    images correctly. The table gives the share of 1,000 decoded samples for
    which the classifier gives one digit a probability above 0.9. On real
    images that share is 93.9%.

    | Codes drawn from | Autoencoder | VAE |
    |---|---|---|
    | its own images' codes | 94.2% to 95.1% | 92.2% to 93.8% |
    | the standard normal | 17.0% to 22.3% | 44.3% to 46.1% |
    | a normal matched to its codes | 24.6% to 27.0% | 46.2% to 48.1% |
    | halfway between two images' codes | 44.4% to 46.7% | 51.3% to 52.8% |

    Codes drawn from the standard normal mostly fall away from the
    autoencoder's real codes, so its decoder has never seen codes like them.
    Even a normal matched to the mean and standard deviation of each code
    number gives a confident digit only about a quarter of the time. The
    autoencoder's real codes fill only part of that normal, and the rest of
    it is empty holes. The VAE's KL term packed its codes into the standard
    normal, so drawing from the standard normal works nearly twice as well.

    The last row tests for holes directly. It decodes the point halfway
    between the codes of two real images. At those points the VAE decodes a
    confident digit more often than the autoencoder, on every seed.

    Ghosh and colleagues measured the same gap on MNIST with FID. FID scores
    how far a set of generated images is from a set of real ones, and lower
    is better. A VAE sampled from the standard normal scored 19.21. An
    autoencoder sampled from a normal fitted to its codes scored 58.73.
    Fitting a mixture of 10 normals to the autoencoder's codes after training
    brought it to 10.66, better than the VAE. So the advantage of a VAE is
    that sampling from a plain standard normal works, with no second fitting
    step after training.

    gan3 compares a VAE with a GAN and measures how close each one's samples
    are to real data.
    """,
)

VAE3_SOURCE = source_file(
    '''
    # vae3.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    codes = torch.tensor([[1.0, 4.0], [3.0, 8.0]])
    centre, spread = codes.mean(dim=0), codes.std(dim=0)    # tensor([2., 6.]), tensor([1.4142, 2.8284])
    halfway = (codes[0] + codes[1]) / 2                      # tensor([2., 6.])


    # TODO: Return the fraction of images that classifier is confident about,
    #   as a Python float. An image counts as confident when its largest
    #   softmax probability is above threshold. Run the classifier in eval mode
    #   with autograd recording turned off, then put it back in training mode.
    def confident_share(
        classifier: nn.Module,
        images: Tensor,
        threshold: float,
    ) -> float:
        ...


    # TODO: Return count new codes, of shape (count, size), drawn from a normal
    #   distribution matched to codes, which has shape (rows, size). Each code
    #   number gets the mean and the standard deviation of its column over the
    #   rows of codes. Draw the noise with torch.randn and generator.
    def matched_codes(
        codes: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return count codes, each halfway between two randomly chosen rows
    #   of codes, as a tensor of shape (count, size). First draw the indices of
    #   the first rows, then the indices of the second rows, each with
    #   torch.randint(0, len(codes), (count,), generator=generator).
    def midpoint_codes(
        codes: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Turn codes into images by passing them through decoder and applying
    #   a sigmoid, with autograd recording turned off. Return confident_share of
    #   classifier on those images at threshold.
    def decoded_share(
        decoder: nn.Module,
        classifier: nn.Module,
        codes: Tensor,
        threshold: float,
    ) -> float:
        ...
    ''',
)

VAE3_SOLUTION = source_file(
    '''
    # vae3.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    codes = torch.tensor([[1.0, 4.0], [3.0, 8.0]])
    centre, spread = codes.mean(dim=0), codes.std(dim=0)
    halfway = (codes[0] + codes[1]) / 2


''' + indented(SHARE_PARTS) + '''
    ''',
)


def check_classifier(shift: float = 0.0) -> nn.Module:
    """A classifier whose confidence varies between images, so a share lands strictly between 0 and 1.
    A shift moves every score alike, which changes no softmax chance."""
    torch.manual_seed(681)
    classifier = nn.Linear(784, 10)

    with torch.no_grad():
        classifier.weight *= 20
        classifier.bias += shift

    return classifier


def check_confident_share(found: object) -> None:
    images = torch.rand(40, 784, generator=torch.Generator().manual_seed(682))

    for shift in (0.0, -10.0):
        classifier = check_classifier(shift)
        call = f"confident_share of 40 images at 0.6, every score shifted by {shift:g}"
        expected = SHARE_REFERENCE["confident_share"](classifier, images, 0.6)
        modes: list[tuple[bool, bool]] = []
        hook = classifier.register_forward_pre_hook(lambda module, inputs: modes.append((module.training, torch.is_grad_enabled())))

        try:
            actual = found(classifier, images, 0.6)
        finally:
            hook.remove()

        if any(training for training, _ in modes):
            raise Mismatch("confident_share ran the classifier in training mode. Call classifier.eval() first.")

        if any(recording for _, recording in modes):
            raise Mismatch("confident_share recorded for autograd. Run the classifier inside torch.no_grad().")

        if not classifier.training:
            raise Mismatch("confident_share left the classifier in eval mode. Call classifier.train() before returning.")

        expect_close(
            call=call,
            actual=actual,
            expected=expected,
        )


def check_decoded_share(found: object) -> None:
    """The file's confident_share is swapped for one that records the images it gets, so decoded_share is held
    to decoding with a sigmoid and no autograd, and to handing the result on."""
    torch.manual_seed(683)
    decoder = nn.Linear(4, 784)
    classifier = check_classifier()
    codes = torch.randn(6, 4, generator=torch.Generator().manual_seed(684))
    seen: list[tuple[Tensor, bool]] = []

    def scripted_share(
        classifier: nn.Module,
        images: Tensor,
        threshold: float,
    ) -> float:
        seen.append((images, images.requires_grad))

        return 0.25

    namespace = found.__globals__
    original = namespace.get("confident_share")
    namespace["confident_share"] = scripted_share

    try:
        actual = found(decoder, classifier, codes, 0.6)
    finally:
        namespace["confident_share"] = original

    if not seen:
        raise Mismatch("decoded_share never called confident_share on the decoded images.")

    images, recorded = seen[0]

    if recorded:
        raise Mismatch("decoded_share recorded for autograd while decoding. Decode inside torch.no_grad().")

    with torch.no_grad():
        expected = torch.sigmoid(decoder(codes))

    expect_tensor(
        call="the images decoded_share hands to confident_share",
        actual=images,
        expected=expected,
    )
    expect_close(
        call="decoded_share's result",
        actual=actual,
        expected=0.25,
    )


def check_vae3(checker: Checker) -> None:
    checker.test(
        function="confident_share",
        description="returns the fraction of images the classifier is confident about, in eval mode",
        run=check_confident_share,
    )
    check_on_inputs(
        checker=checker,
        function="matched_codes",
        description="draws from a normal matched to each code number",
        inputs=(
            (small_tensor(685, 6, 3), 4, seeded(686)),
            (small_tensor(687, 10, 2) * 3, 5, seeded(688)),
        ),
        reference=SHARE_REFERENCE["matched_codes"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="midpoint_codes",
        description="gives codes halfway between two rows",
        inputs=(
            (small_tensor(689, 6, 3), 4, seeded(690)),
            (small_tensor(691, 9, 2) * 3, 7, seeded(692)),
        ),
        reference=SHARE_REFERENCE["midpoint_codes"],
        expect=expect_tensor,
    )
    checker.test(
        function="decoded_share",
        description="decodes the codes and scores the images",
        run=check_decoded_share,
    )


VAE3 = Exercise(
    name="vae3",
    chapter=VAE,
    title="Why a VAE",
    lesson=VAE3_LESSON,
    source=VAE3_SOURCE,
    solution=VAE3_SOLUTION,
    hints={
        "confident_share": "Inside torch.no_grad(), torch.softmax(classifier(images), dim=1).max(dim=1).values is each "
        "image's largest probability. Compare it with threshold, turn the booleans into floats, and take their mean.",
        "matched_codes": "codes.mean(dim=0) and codes.std(dim=0) give one mean and one standard deviation per code "
        "number. Add the mean to the standard deviation times torch.randn(count, codes.shape[1], "
        "generator=generator).",
        "midpoint_codes": "Draw two tensors of indices, one after the other, with torch.randint and the generator. "
        "Index codes with each, add the two sets of rows, and divide by 2.",
        "decoded_share": "Decode inside torch.no_grad(), take torch.sigmoid, and pass the images to confident_share.",
    },
    introduces=(),
    check=check_vae3,
)
