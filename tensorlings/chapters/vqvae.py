"""70_vqvae, vectors swapped for their nearest codebook entry, the straight through gradient, the codebook and commitment losses, and images as grids of tokens."""

from __future__ import annotations

import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, QUIZZES, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file

VQVAE = Chapter(
    folder="70_vqvae",
    title="Vector quantization",
    readme=source_file(
        """
        # Vector quantization

        A vector quantizer swaps each vector for the nearest of a fixed set of
        vectors, a codebook, so a model's output becomes a list of code ids.
        This chapter builds the swap, the straight through trick that lets a
        gradient pass it, and the two losses of VQ-VAE that keep the codebook
        and the encoder together. An image comes out as a small grid of
        tokens, which a model like your GPT could learn to write.

        ## Further reading

        - [van den Oord, Vinyals and Kavukcuoglu, neural discrete representation learning](https://arxiv.org/abs/1711.00937)
        - [Bengio, Leonard and Courville, the straight through estimator](https://arxiv.org/abs/1308.3432)
        - [Esser, Rombach and Ommer, taming transformers for high resolution image synthesis](https://arxiv.org/abs/2012.09841)
        """,
    ),
)

QUANTIZE_PART = '''
def squared_distances(
    z: Tensor,
    codebook: Tensor,
) -> Tensor:
    return ((z.unsqueeze(1) - codebook.unsqueeze(0)) ** 2).sum(dim=2)


def nearest_codes(
    z: Tensor,
    codebook: Tensor,
) -> Tensor:
    return squared_distances(z, codebook).argmin(dim=1)


def quantize(
    z: Tensor,
    codebook: Tensor,
) -> Tensor:
    return codebook[nearest_codes(z, codebook)]


def straight_through(
    z: Tensor,
    quantized: Tensor,
) -> Tensor:
    return z + (quantized - z).detach()
'''

LOSSES_PART = '''
def vq_losses(
    z: Tensor,
    quantized: Tensor,
    beta: float,
) -> tuple[Tensor, Tensor]:
    codebook_loss = F.mse_loss(quantized, z.detach())
    commitment_loss = beta * F.mse_loss(z, quantized.detach())

    return codebook_loss, commitment_loss


def image_tokens(
    z: Tensor,
    codebook: Tensor,
) -> Tensor:
    batch, size, height, width = z.shape
    vectors = z.permute(0, 2, 3, 1).reshape(-1, size)

    return nearest_codes(vectors, codebook).reshape(batch, height, width)


def codes_used(
    tokens: Tensor,
    codes: int,
) -> int:
    return int((torch.bincount(tokens.flatten(), minlength=codes) > 0).sum())
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def reference_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(QUANTIZE_PART + LOSSES_PART, namespace)

    return namespace


REFERENCE = reference_namespace()

SIZE = 8
CODES = 32


class TokenAutoencoder(nn.Module):
    """vq2's model, an MNIST digit to a 7 by 7 grid of codes and back, or with no quantizer at all."""

    def __init__(
        self,
        quantized: bool = True,
    ) -> None:
        super().__init__()
        self.quantized = quantized
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, SIZE, 1),
        )
        self.codebook = nn.Embedding(CODES, SIZE)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(SIZE, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        z = self.encoder(images)

        if not self.quantized:
            return self.decoder(z), z, z, None

        tokens = REFERENCE["image_tokens"](z, self.codebook.weight)
        quantized = self.codebook(tokens).permute(0, 3, 1, 2)

        return self.decoder(REFERENCE["straight_through"](z, quantized)), z, quantized, tokens


def train_token_autoencoder(
    losses: str,
    beta: float,
    steps: int,
    device: torch.device,
) -> tuple[float, int | None, float]:
    """The runs behind vq2's table, Adam at 0.002 on batches of 128 MNIST digits. losses is "both",
    "no codebook" or "none" for the model with no quantizer. Gives the reconstruction error, the codes
    used and the average size of the encoder's vectors, on 2,000 validation digits."""
    from .classic_nets import mnist_images

    torch.manual_seed(0)
    model = TokenAutoencoder(quantized=losses != "none").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    images = mnist_images()
    train, validation = images["train_x"].to(device), images["validation_x"][:2000].to(device)
    generator = torch.Generator().manual_seed(1)

    for _ in range(steps):
        batch = train[torch.randint(0, len(train), (128,), generator=generator).to(device)]
        rebuilt, z, quantized, _ = model(batch)
        loss = F.mse_loss(rebuilt, batch)

        if losses != "none":
            codebook_loss, commitment_loss = REFERENCE["vq_losses"](z, quantized, beta)
            loss = loss + commitment_loss + (codebook_loss if losses == "both" else 0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        rebuilt, z, _, tokens = model(validation)
        used = None if tokens is None else REFERENCE["codes_used"](tokens.cpu(), CODES)

        return F.mse_loss(rebuilt, validation).item(), used, z.norm(dim=1).mean().item()


# ---------------------------------------------------------------------------
# vq1
# ---------------------------------------------------------------------------

VQ1_LESSON = prose(
    """
    A vector quantizer keeps a codebook, K vectors the model learns. It
    swaps each vector it is given for the nearest codebook entry, by squared
    distance. After the swap a vector is an id from 0 to K - 1, a token.
    VQ-VAE used a codebook of 512.

    ```python
    codebook = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    z = torch.tensor([[0.9, 0.1], [-0.8, 0.3]])
    # nearest entries: 0 and 2, so z becomes [[1, 0], [-1, 0]]
    ```

    Broadcasting gives every distance at once. `z.unsqueeze(1)` has shape
    `(n, 1, d)` and `codebook.unsqueeze(0)` has shape `(1, K, d)`, so their
    difference is `(n, K, d)`, every vector against every entry.

    The swap is a hard step, an `argmin`, and soft1 showed what that means.
    No gradient passes it, so the encoder in front of it could never learn.

    VQ-VAE's answer is the straight through trick. Going forwards, use the
    codebook entry. Going backwards, hand the gradient to the encoder's
    vector as if the swap weren't there.

    ```python
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(5.0)
    out = a + (b - a).detach()    # tensor(5.), the value of b
    ```

    The two `a`s cancel, so `out` has the value of `b`. But `.detach()`
    hides the bracket from autograd, so the gradient of `out` with respect to
    `a` is 1, as if `out` were `a` itself. With the encoder's vector for `a`
    and its codebook entry for `b`, that is the trick. The codebook gets no
    gradient this way, and vq2 gives it a loss of its own.
    """,
)

VQ1_SOURCE = source_file(
    '''
    # vq1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    pairs = torch.zeros(4, 1, 2) - torch.zeros(1, 3, 2)    # shape (4, 3, 2), every row against every row
    smallest = torch.tensor([[3.0, 1.0, 2.0]]).argmin(dim=1)    # tensor([1])


    # TODO: Return the squared distance from every row of z, of shape (n, d), to
    #   every row of codebook, of shape (K, d), as a tensor of shape (n, K).
    def squared_distances(
        z: Tensor,
        codebook: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the id of the nearest codebook row for every row of z, of
    #   shape (n,).
    def nearest_codes(
        z: Tensor,
        codebook: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return every row of z swapped for its nearest codebook row.
    def quantize(
        z: Tensor,
        codebook: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return a tensor whose value is quantized, but whose gradient goes
    #   to z unchanged, as if the swap weren't there.
    def straight_through(
        z: Tensor,
        quantized: Tensor,
    ) -> Tensor:
        ...
    ''',
)

VQ1_SOLUTION = source_file(
    '''
    # vq1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    pairs = torch.zeros(4, 1, 2) - torch.zeros(1, 3, 2)
    smallest = torch.tensor([[3.0, 1.0, 2.0]]).argmin(dim=1)


''' + indented(QUANTIZE_PART) + '''
    ''',
)

CODEBOOK = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.5, -2.0]])


def check_straight_through(found: object) -> None:
    z = small_tensor(720, 5, 2).div(3).requires_grad_()
    codebook = CODEBOOK.clone().requires_grad_()
    quantized = REFERENCE["quantize"](z, codebook)
    out = found(z, quantized)
    expect_tensor(
        call="straight_through(z, quantize(z, codebook))",
        actual=out.detach() if isinstance(out, Tensor) else out,
        expected=quantized.detach(),
    )
    upstream = small_tensor(721, 5, 2)
    z_gradient, codebook_gradient = torch.autograd.grad((out * upstream).sum(), [z, codebook], allow_unused=True)

    if z_gradient is None or not torch.allclose(z_gradient, upstream):
        raise Mismatch("The gradient that reaches z should be the gradient handed to the output, unchanged.")

    if codebook_gradient is not None and codebook_gradient.abs().sum() > 0:
        raise Mismatch("The codebook got a gradient through straight_through. Detach the bracket, so only z gets it.")


def check_vq1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="squared_distances",
        description="gives every row's squared distance to every entry",
        inputs=(
            (small_tensor(722, 3, 2), CODEBOOK),
            (small_tensor(723, 4, 3), small_tensor(724, 5, 3)),
        ),
        reference=REFERENCE["squared_distances"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="nearest_codes",
        description="picks the nearest entry for every row",
        inputs=(
            (torch.tensor([[0.9, 0.1], [-0.8, 0.3], [0.6, -1.5], [0.1, 0.8]]), CODEBOOK),
            (small_tensor(725, 6, 3), small_tensor(726, 4, 3)),
        ),
        reference=REFERENCE["nearest_codes"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="quantize",
        description="swaps every row for its nearest entry",
        inputs=(
            (small_tensor(727, 5, 2) / 3, CODEBOOK),
        ),
        reference=REFERENCE["quantize"],
        expect=expect_tensor,
    )
    checker.test(
        function="straight_through",
        description="gives the entry forwards and the gradient to z backwards",
        run=check_straight_through,
    )


VQ1 = Exercise(
    name="vq1",
    chapter=VQVAE,
    title="The nearest code, and a gradient past it",
    lesson=VQ1_LESSON,
    source=VQ1_SOURCE,
    solution=VQ1_SOLUTION,
    hints={
        "squared_distances": "z.unsqueeze(1) minus codebook.unsqueeze(0) has shape (n, K, d). Square it and sum over "
        "dim=2.",
        "nearest_codes": "The argmin of squared_distances over dim=1.",
        "quantize": "Index codebook with nearest_codes.",
        "straight_through": "z plus the detached difference between quantized and z.",
    },
    introduces=(),
    check=check_vq1,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# vq2
# ---------------------------------------------------------------------------

VQ2_LESSON = prose(
    """
    The straight through trick trains the encoder, but nothing moves the
    codebook. VQ-VAE adds two losses, each with one side held still by
    `.detach()`.

    - The codebook loss pulls each chosen entry toward the encoder's vector, `mse(quantized, z.detach())`.
    - The commitment loss pulls the encoder's vector toward its entry, `beta * mse(z, quantized.detach())`.

    The paper used `beta` of 0.25 throughout, and found results changed
    little from 0.1 to 2.0. Without the commitment loss nothing stops the
    encoder's outputs from wandering away from every entry.

    An image becomes tokens. Two stride 2 convolutions turn an MNIST digit
    into a grid of 7 by 7 vectors, and each becomes its nearest entry's id.
    So a digit is 49 tokens from a codebook of 32 here, and the decoder draws
    it back from them.

    ```python
    z = torch.zeros(2, 8, 7, 7)                  # (batch, vector size, height, width)
    z.permute(0, 2, 3, 1).reshape(-1, 8).shape    # torch.Size([98, 8]), one row per grid cell
    ```

    Here is that model trained 1,500 steps on MNIST on MPS, next to the same
    network with no quantizer, on 2,000 validation digits.

    | Model | Reconstruction error | Codes used, of 32 |
    |---|---|---|
    | No quantizer | 0.0009 | none |
    | VQ-VAE, both losses | 0.0139 | 6 |
    | No codebook loss | 0.0262 | 4 |
    | No commitment loss | 0.0809 | 1 |

    Without the commitment loss, the encoder's vectors grew to an average
    size of 1.7 million, far from every entry, and every cell took one code.
    Without the codebook loss the entries never moved, and the error nearly
    doubled. The quantizer costs detail, 15 times the error of no quantizer,
    for 49 small ids per digit. Only 6 of the 32 codes got used, and later
    VQ models add tricks to bring unused entries back.

    Tokens are what other models can learn to write. VQ-VAE trained a second
    model, a PixelCNN, on its grids to draw new images, and VQGAN later used
    a transformer for that.
    """,
)

VQ2_SOURCE = source_file(
    '''
    # vq2.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    z = torch.zeros(2, 8, 7, 7)
    rows = z.permute(0, 2, 3, 1).reshape(-1, 8)       # shape (98, 8), one row per grid cell
    counts = torch.bincount(torch.tensor([0, 2, 2]), minlength=4)    # tensor([1, 0, 2, 0])


    # The nearest code from vq1.
''' + indented(QUANTIZE_PART) + '''


    # TODO: Return the codebook loss and the commitment loss, in that order. The
    #   codebook loss is the mean squared error of quantized against z held
    #   still. The commitment loss is beta times the mean squared error of z
    #   against quantized held still.
    def vq_losses(
        z: Tensor,
        quantized: Tensor,
        beta: float,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return the token id of every cell of z, of shape (batch, size,
    #   height, width), as a tensor of shape (batch, height, width).
    def image_tokens(
        z: Tensor,
        codebook: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return how many of the codes, from 0 to codes - 1, appear at least
    #   once in tokens, as a plain int.
    def codes_used(
        tokens: Tensor,
        codes: int,
    ) -> int:
        ...
    ''',
)

VQ2_SOLUTION = source_file(
    '''
    # vq2.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    z = torch.zeros(2, 8, 7, 7)
    rows = z.permute(0, 2, 3, 1).reshape(-1, 8)
    counts = torch.bincount(torch.tensor([0, 2, 2]), minlength=4)


''' + indented(QUANTIZE_PART) + '''


''' + indented(LOSSES_PART) + '''
    ''',
)


def check_vq_losses(found: object) -> None:
    z = small_tensor(730, 6, 3).div(3).requires_grad_()
    quantized = small_tensor(731, 6, 3).div(3).requires_grad_()
    actual = found(z, quantized, 0.25)

    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch("vq_losses should give the codebook loss and the commitment loss as two tensors in a tuple.")

    codebook_loss, commitment_loss = actual
    expected_codebook, expected_commitment = REFERENCE["vq_losses"](z, quantized, 0.25)

    for name, mine, wanted in (("codebook", codebook_loss, expected_codebook), ("commitment", commitment_loss, expected_commitment)):
        expect_close(
            call=f"the {name} loss of vq_losses at beta 0.25",
            actual=mine.item() if isinstance(mine, Tensor) else mine,
            expected=wanted.item(),
            tolerance=1e-6,
        )

    codebook_to_z, codebook_to_q = torch.autograd.grad(codebook_loss, [z, quantized], allow_unused=True)
    commitment_to_z, commitment_to_q = torch.autograd.grad(commitment_loss, [z, quantized], allow_unused=True)

    if codebook_to_z is not None and codebook_to_z.abs().sum() > 0:
        raise Mismatch("The codebook loss sent a gradient to z. Hold z still with .detach() in it.")

    if commitment_to_q is not None and commitment_to_q.abs().sum() > 0:
        raise Mismatch("The commitment loss sent a gradient to quantized. Hold quantized still with .detach() in it.")

    if codebook_to_q is None or commitment_to_z is None:
        raise Mismatch("Each loss should send a gradient to its own side, the codebook loss to quantized and the commitment loss to z.")


def check_vq2(checker: Checker) -> None:
    checker.test(
        function="vq_losses",
        description="pulls each side toward the other, one at a time",
        run=check_vq_losses,
    )
    check_on_inputs(
        checker=checker,
        function="image_tokens",
        description="gives every grid cell its token",
        inputs=(
            (small_tensor(732, 2, 3, 2, 4), small_tensor(733, 5, 3)),
            (small_tensor(734, 1, 2, 3, 3), CODEBOOK),
        ),
        reference=REFERENCE["image_tokens"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="codes_used",
        description="counts the codes that appear",
        inputs=(
            (torch.tensor([[0, 3, 3], [3, 0, 5]]), 8),
            (torch.tensor([[[1, 1], [1, 1]]]), 4),
            (torch.tensor([6, 6, 0]), 7),
        ),
        reference=REFERENCE["codes_used"],
        expect=expect_value,
    )


VQ2 = Exercise(
    name="vq2",
    chapter=VQVAE,
    title="The codebook and commitment losses",
    lesson=VQ2_LESSON,
    source=VQ2_SOURCE,
    solution=VQ2_SOLUTION,
    hints={
        "vq_losses": "F.mse_loss(quantized, z.detach()) is the codebook loss. The commitment loss holds quantized "
        "still instead, and is scaled by beta.",
        "image_tokens": "permute z to (batch, height, width, size), reshape to rows of size, take nearest_codes, and "
        "reshape the ids to (batch, height, width).",
        "codes_used": "torch.bincount of the flattened tokens with minlength=codes counts each code. Count the counts "
        "above 0, and turn the result into an int.",
    },
    introduces=(),
    check=check_vq2,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# quiz12
# ---------------------------------------------------------------------------

QUIZ12_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines moves
    from Section 9. If a step is unclear, the exercise in the table shows it.

    | Function | Moves | Shown in |
    |---|---|---|
    | `reparameterize` | a Gaussian sample as the mean plus the spread times noise | sampling3 |
    | `kl_to_standard` | the KL divergence from a diagonal Gaussian to the standard one, in closed form | vae1 |
    | `discriminator_loss` | binary cross entropy that calls real samples 1 and generated ones 0 | gan1 |
    | `noise_to` | a clean sample noised to its step in one jump | diffusion1 |
    | `nearest_codes` | each vector's closest codebook row by squared distance | vq1 |
    """,
)

QUIZ12_SOURCE = source_file(
    '''
    # quiz12.py
    #
    # No examples and no new tools. Each function combines moves from Section 9.

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Return a sample of a Gaussian with mean and log_variance, given its
    #   standard normal noise of the same shape.
    def reparameterize(
        mean: Tensor,
        log_variance: Tensor,
        noise: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the KL divergence of each row's diagonal Gaussian, with mean
    #   and log_variance, from the standard normal, one number per row.
    def kl_to_standard(
        mean: Tensor,
        log_variance: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the discriminator's loss for its scores, logits, on real
    #   samples and on generated ones, summed over the two kinds.
    def discriminator_loss(
        real_scores: Tensor,
        fake_scores: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return x0 noised to step t in one jump, for x0 of any shape whose
    #   first dimension is the batch. alpha_bars holds every step's alpha bar,
    #   t one step per sample, and noise is the shape of x0.
    def noise_to(
        x0: Tensor,
        t: Tensor,
        alpha_bars: Tensor,
        noise: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the index of each row of z's closest codebook row by squared
    #   distance.
    def nearest_codes(
        z: Tensor,
        codebook: Tensor,
    ) -> Tensor:
        ...
    ''',
)

QUIZ12_SOLUTION = source_file(
    '''
    # quiz12.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    def reparameterize(
        mean: Tensor,
        log_variance: Tensor,
        noise: Tensor,
    ) -> Tensor:
        return mean + torch.exp(0.5 * log_variance) * noise


    def kl_to_standard(
        mean: Tensor,
        log_variance: Tensor,
    ) -> Tensor:
        return 0.5 * (mean**2 + log_variance.exp() - log_variance - 1).sum(dim=-1)


    def discriminator_loss(
        real_scores: Tensor,
        fake_scores: Tensor,
    ) -> Tensor:
        real = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores))
        fake = F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))

        return real + fake


    def noise_to(
        x0: Tensor,
        t: Tensor,
        alpha_bars: Tensor,
        noise: Tensor,
    ) -> Tensor:
        kept = alpha_bars[t].reshape(len(t), *[1] * (x0.dim() - 1))

        return kept.sqrt() * x0 + (1 - kept).sqrt() * noise


    def nearest_codes(
        z: Tensor,
        codebook: Tensor,
    ) -> Tensor:
        return ((z.unsqueeze(1) - codebook.unsqueeze(0)) ** 2).sum(dim=-1).argmin(dim=1)
    ''',
)


def quiz12_reference() -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(QUIZ12_SOLUTION, namespace)

    return namespace


QUIZ12_REFERENCE = quiz12_reference()


def quiz12_tensor(
    seed: int,
    *shape: int,
) -> Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def check_quiz12(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="reparameterize",
        description="adds the spread times the noise to the mean",
        inputs=(
            (quiz12_tensor(1210, 3, 2), quiz12_tensor(1211, 3, 2), quiz12_tensor(1212, 3, 2)),
        ),
        reference=QUIZ12_REFERENCE["reparameterize"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="kl_to_standard",
        description="gives each row's KL in closed form",
        inputs=(
            (quiz12_tensor(1213, 4, 3), quiz12_tensor(1214, 4, 3)),
        ),
        reference=QUIZ12_REFERENCE["kl_to_standard"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="discriminator_loss",
        description="calls the real samples 1 and the generated ones 0",
        inputs=(
            (quiz12_tensor(1215, 5), quiz12_tensor(1216, 5) - 1),
        ),
        reference=QUIZ12_REFERENCE["discriminator_loss"],
        expect=expect_tensor,
    )
    alpha_bars = torch.cumprod(1 - torch.linspace(1e-4, 0.02, 1000), dim=0)
    check_on_inputs(
        checker=checker,
        function="noise_to",
        description="noises each sample to its own step",
        inputs=(
            (quiz12_tensor(1217, 3, 2), torch.tensor([0, 400, 999]), Described(alpha_bars, "DDPM's 1,000 alpha bars"), quiz12_tensor(1218, 3, 2)),
            (quiz12_tensor(1219, 2, 1, 3, 3), torch.tensor([100, 700]), Described(alpha_bars, "DDPM's 1,000 alpha bars"), quiz12_tensor(1220, 2, 1, 3, 3)),
        ),
        reference=QUIZ12_REFERENCE["noise_to"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="nearest_codes",
        description="picks each vector's closest code",
        inputs=(
            (quiz12_tensor(1221, 6, 3), quiz12_tensor(1222, 4, 3)),
        ),
        reference=QUIZ12_REFERENCE["nearest_codes"],
        expect=expect_tensor,
    )


QUIZ12 = Exercise(
    name="quiz12",
    chapter=QUIZZES,
    title="Generative models, start to finish",
    lesson=QUIZ12_LESSON,
    source=QUIZ12_SOURCE,
    solution=QUIZ12_SOLUTION,
    hints={
        "reparameterize": "The spread is torch.exp(0.5 * log_variance). Add the spread times the noise to the mean.",
        "kl_to_standard": "Half of mean squared plus the variance minus the log variance minus 1, summed over the last "
        "dimension.",
        "discriminator_loss": "F.binary_cross_entropy_with_logits of the real scores against ones, plus that of the "
        "fake scores against zeros.",
        "noise_to": "alpha_bars[t] reshaped to (len(t), 1, ...) lines each sample up with its step. Then the square "
        "root of it times x0 plus the square root of 1 minus it times the noise.",
        "nearest_codes": "z.unsqueeze(1) - codebook.unsqueeze(0) gives every difference. Square, sum over the last "
        "dimension, and argmin over the codes.",
    },
    introduces=(),
    check=check_quiz12,
)
