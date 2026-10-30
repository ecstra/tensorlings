"""63_autoencoders, a small code that rebuilds its input, why the code must be small or the input noisy, and why a linear one finds what PCA finds."""

from __future__ import annotations

import math
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
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module
from .mnist import mnist_sets
from .word2vec import expect_pairs

AUTOENCODERS = Chapter(
    folder="63_autoencoders",
    title="Autoencoders",
    readme=source_file(
        """
        # Autoencoders

        An autoencoder squeezes its input into a small code and rebuilds the
        input from it, with no labels. This chapter builds one for MNIST
        digits, measures why the code has to be small or the input noisy for
        it to learn anything, and shows that a linear one finds the same
        directions as PCA.

        ## Further reading

        - [Hinton and Salakhutdinov, reducing the dimensionality of data with neural networks](https://www.cs.toronto.edu/~hinton/absps/science.pdf)
        - [Vincent and colleagues, extracting and composing robust features with denoising autoencoders](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
        - [PyTorch, torch.pca_lowrank](https://docs.pytorch.org/docs/2.14/generated/torch.pca_lowrank.html)
        """,
    ),
)


def digit_rows(
    name: str,
    count: int,
) -> Tensor:
    """MNIST digits as rows of 784 pixels from 0 to 1, from the shared splits."""
    return mnist_sets()[name][:count].clone()


# The model autoencoder1 builds and the later exercises are handed.
AUTOENCODER_INIT = '''
class Autoencoder(nn.Module):
    def __init__(
        self,
        code: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, code),
        )
        self.decoder = nn.Sequential(
            nn.Linear(code, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),
        )
'''

AUTOENCODER_FORWARD = '''
    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))
'''

TRAIN_PART = '''
def reconstruction_loss(
    model: nn.Module,
    x: Tensor,
) -> Tensor:
    return F.mse_loss(model(x), x)


def train_autoencoder(
    model: nn.Module,
    x: Tensor,
    steps: int,
    generator: torch.Generator,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(steps):
        batch = x[torch.randint(0, len(x), (128,), generator=generator)]
        loss = reconstruction_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
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
    exec(AUTOENCODER_INIT + AUTOENCODER_FORWARD + TRAIN_PART, namespace)

    return namespace


REFERENCE = reference_namespace()


def seeded_autoencoder(
    make: type,
    code: int,
) -> nn.Module:
    torch.manual_seed(0)

    return make(code)


def validation_error(
    model: nn.Module,
    count: int = 2000,
) -> float:
    x = digit_rows("validation_x", count)

    with torch.no_grad():
        return F.mse_loss(model(x), x).item()


def code_size_table() -> dict[int, float]:
    """The run behind autoencoder1's table, 1,500 steps on 20,000 digits for each size of code."""
    x = digit_rows("train_x", 20000)
    errors = {}

    for code in (2, 8, 32, 128):
        model = seeded_autoencoder(REFERENCE["Autoencoder"], code)
        REFERENCE["train_autoencoder"](model, x, 1500, torch.Generator().manual_seed(0))
        errors[code] = validation_error(model)

    return errors


# ---------------------------------------------------------------------------
# autoencoder1
# ---------------------------------------------------------------------------

AUTOENCODER1_LESSON = prose(
    """
    Section 8 learns from data with no labels. An **autoencoder** is the
    plainest way. An encoder squeezes the input into a small code, and a
    decoder rebuilds the input from that code. The loss is how far the
    rebuild lands from the input.

    For an MNIST digit, the encoder maps 784 pixels down to, say, 32 numbers,
    and the decoder maps them back to 784. If 32 numbers rebuild the digit
    well, they hold most of what makes it that digit. Nothing tells the model
    which digit it is.

    The decoder ends in a sigmoid, since pixels run from 0 to 1, the output
    rule of activations2. The loss is the mean squared error between the
    rebuild and the input, so the input is its own target.

    ```python
    rebuilt = torch.tensor([0.9, 0.1])
    F.mse_loss(rebuilt, torch.tensor([1.0, 0.0]))    # tensor(0.0100)
    ```

    Here are codes of four sizes, each trained for 1,500 steps on 20,000
    digits, measured on 2,000 digits it never saw.

    | Code size | 2 | 8 | 32 | 128 |
    |---|---|---|---|---|
    | Error on new digits | 0.049 | 0.026 | 0.014 | 0.011 |

    Two numbers already rebuild a blurry digit. More numbers rebuild finer
    strokes, and the error keeps falling as the code grows. autoencoder2
    asks what happens when it grows past the input.
    """,
)

AUTOENCODER1_SOURCE = source_file(
    '''
    # autoencoder1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    rebuilt = torch.tensor([0.9, 0.1])
    error = F.mse_loss(rebuilt, torch.tensor([1.0, 0.0]))    # tensor(0.0100)


    # TODO: Finish forward. It returns the decoder's rebuild of the encoder's
    #   code for x, of the same shape as x.
''' + indented(AUTOENCODER_INIT) + '''

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return the mean squared error between model's rebuild of x and x.
    def reconstruction_loss(
        model: nn.Module,
        x: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish the loop. Each step draws a batch of 128 rows of x with
    #   torch.randint(0, len(x), (128,), generator=generator), takes
    #   reconstruction_loss on it, and steps Adam. Return the last loss as a
    #   plain number.
    def train_autoencoder(
        model: nn.Module,
        x: Tensor,
        steps: int,
        generator: torch.Generator,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(steps):
            ...

        return loss.item()
    ''',
)

AUTOENCODER1_SOLUTION = source_file(
    '''
    # autoencoder1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    rebuilt = torch.tensor([0.9, 0.1])
    error = F.mse_loss(rebuilt, torch.tensor([1.0, 0.0]))


''' + indented(AUTOENCODER_INIT) + '''

''' + indented(AUTOENCODER_FORWARD) + '''


''' + indented(TRAIN_PART) + '''
    ''',
)


def check_autoencoder_forward(found: type) -> None:
    model = expect_module("Autoencoder(16)", seeded_autoencoder(found, 16))
    x = digit_rows("validation_x", 5)

    with torch.no_grad():
        expected = seeded_autoencoder(REFERENCE["Autoencoder"], 16)(x)

    expect_tensor(
        call="Autoencoder(16) on 5 digits",
        actual=model(x),
        expected=expected,
    )


def check_train_autoencoder(found: object) -> None:
    x = digit_rows("train_x", 600)
    model = seeded_autoencoder(REFERENCE["Autoencoder"], 8)
    expected_model = seeded_autoencoder(REFERENCE["Autoencoder"], 8)
    expected = REFERENCE["train_autoencoder"](expected_model, x, 15, torch.Generator().manual_seed(470))
    actual = found(model, x, 15, torch.Generator().manual_seed(470))
    expect_close(
        call="train_autoencoder for 15 steps on 600 digits",
        actual=actual,
        expected=expected,
        tolerance=1e-5,
    )
    expect_tensor(
        call="the decoder's last weights after those steps",
        actual=model.decoder[2].weight.detach(),
        expected=expected_model.decoder[2].weight.detach(),
    )


def check_autoencoder_learns(found: type) -> None:
    model = seeded_autoencoder(found, 32)
    REFERENCE["train_autoencoder"](model, digit_rows("train_x", 5000), 300, torch.Generator().manual_seed(0))
    error = validation_error(model, 1000)

    if error > LEARNS_BELOW:
        raise Mismatch(f"After 300 steps, Autoencoder(32) rebuilds new digits with an error of {error:.4f}, but it should be below {LEARNS_BELOW}.")


# The solution reaches about 0.033 in these 300 steps, and an untrained model sits far above.
LEARNS_BELOW = 0.05


def check_autoencoder1(checker: Checker) -> None:
    checker.test_method(
        class_name="Autoencoder",
        method="forward",
        description="rebuilds x from its code",
        run=check_autoencoder_forward,
    )
    check_on_inputs(
        checker=checker,
        function="reconstruction_loss",
        description="gives the mean squared error of the rebuild",
        inputs=(
            (Described(seeded_autoencoder(REFERENCE["Autoencoder"], 4), "Autoencoder(4)"), Described(digit_rows("validation_x", 3), "3 digits")),
            (Described(seeded_autoencoder(REFERENCE["Autoencoder"], 8), "Autoencoder(8)"), Described(digit_rows("validation_x", 5), "5 digits")),
        ),
        reference=lambda model, x: REFERENCE["reconstruction_loss"](model, x),
        expect=expect_tensor,
    )
    checker.test(
        function="train_autoencoder",
        description="draws batches of 128 and steps Adam on the rebuild",
        run=check_train_autoencoder,
    )
    checker.test_method(
        class_name="Autoencoder",
        method="forward",
        description="learns to rebuild digits it never saw",
        run=check_autoencoder_learns,
    )


AUTOENCODER1 = Exercise(
    name="autoencoder1",
    chapter=AUTOENCODERS,
    title="A code that rebuilds its input",
    lesson=AUTOENCODER1_LESSON,
    source=AUTOENCODER1_SOURCE,
    solution=AUTOENCODER1_SOLUTION,
    hints={
        "Autoencoder.forward": "Run self.encoder on x, then self.decoder on the code it gives.",
        "reconstruction_loss": "F.mse_loss of model(x) against x itself.",
        "train_autoencoder": "batch = x[torch.randint(...)] with the generator. Take the loss on the batch, then "
        "zero_grad, backward and step.",
    },
    introduces=("autoencoder",),
    check=check_autoencoder1,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# autoencoder2
# ---------------------------------------------------------------------------

NOISE_PARTS = '''
def add_noise(
    x: Tensor,
    std: float,
    generator: torch.Generator,
) -> Tensor:
    return x + std * torch.randn(x.shape, generator=generator)


def denoising_loss(
    model: nn.Module,
    x: Tensor,
    std: float,
    generator: torch.Generator,
) -> Tensor:
    return F.mse_loss(model(add_noise(x, std, generator)), x)


def noisy_error(
    model: nn.Module,
    x: Tensor,
    std: float,
    generator: torch.Generator,
) -> float:
    model.eval()

    with torch.no_grad():
        error = F.mse_loss(model(add_noise(x, std, generator)), x).item()

    model.train()

    return error
'''

NOISE_REFERENCE = dict(REFERENCE)
exec(NOISE_PARTS, NOISE_REFERENCE)


class Shallow(nn.Module):
    """One layer each way, so a code at least as wide as the input can copy it. autoencoder2's table compares these."""

    def __init__(
        self,
        code: int,
        linear: bool,
    ) -> None:
        super().__init__()

        if linear:
            self.encoder = nn.Linear(784, code)
            self.decoder = nn.Linear(code, 784)
            return

        self.encoder = nn.Sequential(nn.Linear(784, code), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(code, 784), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))


def probe_accuracy(model: nn.Module | None) -> float:
    """A Linear head trained on the codes of 1,000 labelled digits, scored on 2,000 others. None probes the raw pixels."""
    sets = mnist_sets()
    train_x, train_y = sets["train_x"][20000:21000], sets["train_y"][20000:21000]
    test_x, test_y = sets["validation_x"][:2000], sets["validation_y"][:2000]

    with torch.no_grad():
        train_codes = train_x if model is None else model.encoder(train_x)
        test_codes = test_x if model is None else model.encoder(test_x)

    mean, spread = train_codes.mean(dim=0), train_codes.std(dim=0) + 1e-5
    train_codes, test_codes = (train_codes - mean) / spread, (test_codes - mean) / spread
    torch.manual_seed(0)
    head = nn.Linear(train_codes.shape[1], 10)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)

    for _ in range(300):
        loss = F.cross_entropy(head(train_codes), train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return (head(test_codes).argmax(dim=1) == test_y).float().mean().item()


def bottleneck_table() -> dict[str, tuple[float, float, float]]:
    """The run behind autoencoder2's table: 1,500 steps on 20,000 digits, then the error on clean digits,
    the error on digits with noise of spread 0.5 measured against the clean ones, and a probe on the code."""
    x = digit_rows("train_x", 20000)
    clean = digit_rows("validation_x", 2000)
    rows = {}

    for name, code, linear, noise in (
        ("linear, code 1,024", 1024, True, 0.0),
        ("linear, code 32", 32, True, 0.0),
        ("ReLU, code 1,024", 1024, False, 0.0),
        ("ReLU, code 1,024, denoising", 1024, False, 0.5),
    ):
        torch.manual_seed(0)
        model = Shallow(code, linear)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        generator = torch.Generator().manual_seed(0)

        for _ in range(1500):
            batch = x[torch.randint(0, len(x), (128,), generator=generator)]
            loss = NOISE_REFERENCE["denoising_loss"](model, batch, noise, generator) if noise else F.mse_loss(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        noisy = NOISE_REFERENCE["noisy_error"](model, clean, 0.5, torch.Generator().manual_seed(5))
        rows[name] = (validation_error(model), noisy, probe_accuracy(model))

    return rows


AUTOENCODER2_LESSON = prose(
    """
    Rebuilding the input is an easy job for a model that can copy it. Give an
    autoencoder a code as wide as its input and nothing forces it to learn
    what digits look like.

    The models here have one layer each way, so a code of 1,024 numbers is
    wider than the 784 pixels. Each trained for 1,500 steps. The noisy error
    feeds in digits with noise of spread 0.5 added and compares the rebuild
    with the clean digit. The probe trains a Linear head on the codes of
    1,000 labelled digits, as a test of how useful the code is.

    | Autoencoder | Clean error | Noisy error | Probe |
    |---|---|---|---|
    | linear, code 1,024 | 0.0004 | 0.183 | 83.0% |
    | linear, code 32 | 0.018 | 0.032 | 84.0% |
    | ReLU, code 1,024 | 0.0030 | 0.022 | 82.6% |
    | ReLU, code 1,024, denoising | 0.0074 | 0.013 | 87.6% |

    The wide linear one rebuilds clean digits almost perfectly and passes
    most of the noise through. It learned to copy. With a code of 32 it can't
    copy, so it keeps what digits share and drops most of the noise. The wide
    ReLU one copied less in its 1,500 steps, but its code probes no better
    than the narrow one's. The pixels alone probe at 81.5%.

    A **denoising autoencoder** removes the shortcut another way. It reads a
    noisy digit and is scored against the clean one, so copying its input
    can never be right. Vincent and colleagues proposed it in 2008. Here it
    gives the best code of the four, even at 1,024 wide. These probes come
    from one run each, so read differences of a point or two as noise.

    The same idea runs through Section 8. Masking in 62_masked_modeling is a
    kind of noise too, with the hidden part as the answer.
    """,
)

AUTOENCODER2_SOURCE = source_file(
    '''
    # autoencoder2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    noise = 0.5 * torch.randn(3, generator=torch.Generator().manual_seed(0))    # tensor([ 0.7705, -0.1467, -1.0894])


    # TODO: Return x with noise added, drawn with torch.randn(x.shape,
    #   generator=generator) and scaled by std.
    def add_noise(
        x: Tensor,
        std: float,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the mean squared error between model's rebuild of a noisy
    #   copy of x and the clean x.
    def denoising_loss(
        model: nn.Module,
        x: Tensor,
        std: float,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the same error as a plain number, measured in eval mode and
    #   without recording for autograd. Then put model back in training mode.
    def noisy_error(
        model: nn.Module,
        x: Tensor,
        std: float,
        generator: torch.Generator,
    ) -> float:
        ...
    ''',
)

AUTOENCODER2_SOLUTION = source_file(
    '''
    # autoencoder2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    noise = 0.5 * torch.randn(3, generator=torch.Generator().manual_seed(0))


''' + indented(NOISE_PARTS) + '''
    ''',
)


def check_noisy_error(found: object) -> None:
    for code, count, std, seed in ((8, 6, 0.3, 472), (4, 9, 0.7, 489)):
        model = seeded_autoencoder(REFERENCE["Autoencoder"], code)
        x = digit_rows("validation_x", count)
        expected = NOISE_REFERENCE["noisy_error"](model, x, std, torch.Generator().manual_seed(seed))
        modes: list[tuple[bool, bool]] = []
        hook = model.register_forward_pre_hook(lambda module, inputs: modes.append((module.training, torch.is_grad_enabled())))

        try:
            actual = found(model, x, std, torch.Generator().manual_seed(seed))
        finally:
            hook.remove()

        if any(training for training, _ in modes):
            raise Mismatch("noisy_error ran the model in training mode. Call model.eval() first.")

        if any(recording for _, recording in modes):
            raise Mismatch("noisy_error recorded for autograd. Run the model inside torch.no_grad().")

        if not model.training:
            raise Mismatch("noisy_error left the model in eval mode. Call model.train() before returning.")

        expect_close(
            call=f"noisy_error of Autoencoder({code}) on {count} digits, noise of spread {std}",
            actual=actual,
            expected=expected,
            tolerance=1e-6,
        )


def check_autoencoder2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="add_noise",
        description="adds noise of the given spread",
        inputs=(
            (small_tensor(473, 2, 3), 0.5, seeded(474)),
            (small_tensor(475, 4), 2.0, seeded(476)),
        ),
        reference=NOISE_REFERENCE["add_noise"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="denoising_loss",
        description="scores the rebuild of a noisy copy against the clean one",
        inputs=(
            (Described(seeded_autoencoder(REFERENCE["Autoencoder"], 4), "Autoencoder(4)"), Described(digit_rows("validation_x", 3), "3 digits"), 0.4, seeded(477)),
            (Described(seeded_autoencoder(REFERENCE["Autoencoder"], 8), "Autoencoder(8)"), Described(digit_rows("validation_x", 5), "5 digits"), 0.8, seeded(488)),
        ),
        reference=NOISE_REFERENCE["denoising_loss"],
        expect=expect_tensor,
    )
    checker.test(
        function="noisy_error",
        description="measures the noisy error in eval mode, without autograd",
        run=check_noisy_error,
    )


AUTOENCODER2 = Exercise(
    name="autoencoder2",
    chapter=AUTOENCODERS,
    title="Why the code must be small, or the input noisy",
    lesson=AUTOENCODER2_LESSON,
    source=AUTOENCODER2_SOURCE,
    solution=AUTOENCODER2_SOLUTION,
    hints={
        "add_noise": "x plus std times torch.randn of x's shape, drawn with the generator.",
        "denoising_loss": "Run model on add_noise(x, std, generator), and take F.mse_loss of that against x, not "
        "against the noisy copy.",
        "noisy_error": "model.eval(), then inside torch.no_grad() take the same error as denoising_loss and .item() it. "
        "Call model.train() before returning.",
    },
    introduces=("denoising autoencoder",),
    check=check_autoencoder2,
    time_limit=60,
)


# ---------------------------------------------------------------------------
# autoencoder3
# ---------------------------------------------------------------------------

SUBSPACE_PARTS = '''
def projection(basis: Tensor) -> Tensor:
    q, _ = torch.linalg.qr(basis)

    return q @ q.T


def subspace_gap(
    a: Tensor,
    b: Tensor,
) -> float:
    k = a.shape[1]

    return ((projection(a) - projection(b)).norm() / math.sqrt(2 * k)).item()


def pca_basis(
    x: Tensor,
    k: int,
) -> Tensor:
    asked = min(2 * k + 10, x.shape[0], x.shape[1])
    _, _, v = torch.pca_lowrank(x, q=asked, center=True, niter=4)

    return v[:, :k]
'''

LINEAR_AUTOENCODER = '''
class LinearAutoencoder(nn.Module):
    def __init__(
        self,
        size: int,
        code: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(size, code, bias=False)
        self.decoder = nn.Linear(code, size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))


def train_linear(
    model: nn.Module,
    x: Tensor,
    steps: int,
    generator: torch.Generator,
) -> nn.Module:
    centered = x - x.mean(dim=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _ in range(steps):
        batch = centered[torch.randint(0, len(x), (256,), generator=generator)]
        loss = F.mse_loss(model(batch), batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
'''

SUBSPACE_REFERENCE: dict[str, object] = {
    "math": math,
    "torch": torch,
    "nn": nn,
    "F": F,
    "Tensor": Tensor,
}
exec(SUBSPACE_PARTS + LINEAR_AUTOENCODER, SUBSPACE_REFERENCE)


def stretched_points(
    seed: int,
    count: int,
) -> Tensor:
    """Points in 8 dimensions with spreads from 4 down to 0.25 along random directions, and a shift."""
    generator = torch.Generator().manual_seed(seed)
    spreads = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.25])
    turn, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))

    return (torch.randn(count, 8, generator=generator) * spreads) @ turn.T + 3.0


def mnist_pca_run() -> tuple[float, float, float, float]:
    """The run behind autoencoder3's numbers: a linear autoencoder with a code of 16 on 20,000 digits for
    1,500 steps, its gap to PCA's 16 directions, a random basis's gap, the gap of pca_lowrank asked for
    exactly 16, and how close its first column is to PCA's first direction."""
    x = digit_rows("train_x", 20000)
    torch.manual_seed(0)
    model = SUBSPACE_REFERENCE["LinearAutoencoder"](784, 16)
    SUBSPACE_REFERENCE["train_linear"](model, x, 1500, torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    principal = SUBSPACE_REFERENCE["pca_basis"](x, 16)
    _, _, exactly_sixteen = torch.pca_lowrank(x, q=16, center=True)
    learned = model.decoder.weight.detach()
    random_basis = torch.randn(784, 16, generator=torch.Generator().manual_seed(1))
    first = F.cosine_similarity(learned[:, 0], principal[:, 0], dim=0).abs().item()

    return (
        SUBSPACE_REFERENCE["subspace_gap"](learned, principal),
        SUBSPACE_REFERENCE["subspace_gap"](random_basis, principal),
        SUBSPACE_REFERENCE["subspace_gap"](exactly_sixteen, principal),
        first,
    )


AUTOENCODER3_LESSON = prose(
    """
    Take the activations out, and an autoencoder is two matrices. The
    encoder maps a centered input down to a code of k numbers and the decoder
    maps it back. Baldi and Hornik showed in 1989 that the best such pair
    keeps the same k directions as principal component analysis, PCA. Those
    are the directions along which the data spreads the most.

    A set of k directions **spans** a **space**, every point you can reach by
    scaling the directions and adding them up. The directions are a
    **basis** of that space, and many different bases span the same one.

    The best pair keeps PCA's space but picks its own axes. Any basis of that
    space rebuilds the input equally well, so the decoder's columns come out
    as a mix of PCA's directions. Comparing the two means comparing the
    spaces they span.

    A **projection** onto a space maps every point to its nearest point in
    that space. Two bases span the same space exactly when they give the same
    projection. `torch.linalg.qr` turns a basis of columns into one of
    perpendicular unit columns spanning the same space, and with that basis
    `q @ q.T` is the projection.

    ```python
    q, _ = torch.linalg.qr(torch.tensor([[2.0], [0.0]]))
    q @ q.T    # tensor([[1., 0.], [0., 0.]]), up to the sign of q
    ```

    The gap between two spaces of k directions is the size of the difference
    of their projections, divided by the square root of 2k. It is 0 for the
    same space and 1 for spaces at right angles.

    `torch.pca_lowrank(x, q=q, center=True)` gives PCA's first q directions
    as the columns of its third result. It is a fast random method, and on
    MNIST, asked for exactly 16 directions, it lands 0.30 from the true
    space. Asked for 2k + 10 with 4 rounds of refining, `niter=4`, and cut
    back to k, it lands on it.

    Here is a linear autoencoder with a code of 16, trained for 1,500 steps on
    20,000 digits, against PCA's first 16 directions.

    - Its gap to PCA's space: 0.05.
    - The gap of 16 random directions to it: 0.99.
    - The cosine between its first column and PCA's first direction: 0.52.

    So it found PCA's space, in a basis of its own. With activations, an
    autoencoder can find curved shapes PCA can't, which is what made the deep
    ones of Hinton and Salakhutdinov worth training.
    """,
)

AUTOENCODER3_SOURCE = source_file(
    '''
    # autoencoder3.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    q, _ = torch.linalg.qr(torch.tensor([[2.0], [0.0]]))
    onto_first = q @ q.T          # tensor([[1., 0.], [0., 0.]]), up to the sign of q


    # TODO: Return the projection onto the space the columns of basis span, of
    #   shape (rows, rows). Make the columns perpendicular with torch.linalg.qr
    #   first.
    def projection(basis: Tensor) -> Tensor:
        ...


    # TODO: Return the gap between the spaces a and b span, each a basis of k
    #   columns, as a plain number: the norm of the difference of their
    #   projections, divided by the square root of 2 * k.
    def subspace_gap(
        a: Tensor,
        b: Tensor,
    ) -> float:
        ...


    # TODO: Return PCA's first k directions of x, as k columns. Ask
    #   torch.pca_lowrank for q = 2 * k + 10 directions, or fewer when x has
    #   fewer rows or columns, with center=True and niter=4. Keep the first k
    #   columns of its third result.
    def pca_basis(
        x: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # Two matrices and no activations, trained on centered data.
''' + indented(LINEAR_AUTOENCODER) + '''
    ''',
)

AUTOENCODER3_SOLUTION = source_file(
    '''
    # autoencoder3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    q, _ = torch.linalg.qr(torch.tensor([[2.0], [0.0]]))
    onto_first = q @ q.T


''' + indented(SUBSPACE_PARTS) + '''


''' + indented(LINEAR_AUTOENCODER) + '''
    ''',
)


def check_pca_basis(found: object) -> None:
    for seed, k in ((480, 2), (481, 3)):
        x = stretched_points(seed, 400)
        actual = found(x, k)

        if not isinstance(actual, Tensor) or tuple(actual.shape) != (8, k):
            raise Mismatch(f"pca_basis of 400 points in 8 dimensions, k={k}, should give 8 rows of {k} columns.")

        _, _, exact = torch.linalg.svd(x - x.mean(dim=0), full_matrices=False)
        gap = SUBSPACE_REFERENCE["subspace_gap"](actual, exact[:k].T)

        if gap > 0.01:
            raise Mismatch(f"pca_basis(x, {k}) spans a space {gap:.3f} away from PCA's first {k} directions.")


def check_matches_pca(found: object) -> None:
    """The learner's gap between a trained linear autoencoder and their own PCA should be near 0, and the
    gap to a random basis should not."""
    x = stretched_points(482, 2000)
    torch.manual_seed(0)
    model = SUBSPACE_REFERENCE["train_linear"](SUBSPACE_REFERENCE["LinearAutoencoder"](8, 3), x, 600, torch.Generator().manual_seed(0))
    learned = model.decoder.weight.detach()
    principal = SUBSPACE_REFERENCE["pca_basis"](x, 3)
    random_basis = torch.randn(8, 3, generator=torch.Generator().manual_seed(483))
    near = found(learned, principal)
    far = found(random_basis, principal)
    expect_close(
        call="subspace_gap of a trained linear autoencoder and PCA",
        actual=near,
        expected=SUBSPACE_REFERENCE["subspace_gap"](learned, principal),
        tolerance=1e-4,
    )
    expect_close(
        call="subspace_gap of 3 random directions and PCA",
        actual=far,
        expected=SUBSPACE_REFERENCE["subspace_gap"](random_basis, principal),
        tolerance=1e-4,
    )

    if near > 0.05:
        raise Mismatch(f"The trained linear autoencoder's space is {near:.3f} from PCA's, but it should be near 0.")


def check_autoencoder3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="projection",
        description="projects onto the space the columns span",
        inputs=(
            (small_tensor(484, 4, 2),),
            (Described(torch.tensor([[2.0, 1.0], [0.0, 1.0], [0.0, 0.0]]), "two columns in 3 dimensions"),),
        ),
        reference=SUBSPACE_REFERENCE["projection"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="subspace_gap",
        description="is 0 for the same space and grows as they part",
        inputs=(
            (small_tensor(485, 5, 2), small_tensor(485, 5, 2) @ torch.tensor([[2.0, 1.0], [0.0, 3.0]])),
            (small_tensor(486, 5, 2), small_tensor(487, 5, 2)),
            (torch.eye(4)[:, :2], torch.eye(4)[:, 2:]),
        ),
        reference=SUBSPACE_REFERENCE["subspace_gap"],
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )
    checker.test(
        function="pca_basis",
        description="spans PCA's first k directions",
        run=check_pca_basis,
    )
    checker.test(
        function="subspace_gap",
        description="finds a trained linear autoencoder in PCA's space",
        run=check_matches_pca,
    )


AUTOENCODER3 = Exercise(
    name="autoencoder3",
    chapter=AUTOENCODERS,
    title="A linear autoencoder finds PCA",
    lesson=AUTOENCODER3_LESSON,
    source=AUTOENCODER3_SOURCE,
    solution=AUTOENCODER3_SOLUTION,
    hints={
        "projection": "q, _ = torch.linalg.qr(basis), then q times its own transpose.",
        "subspace_gap": "Take projection of a minus projection of b, its .norm(), and divide by math.sqrt(2 * k) with k "
        "the number of columns. Return it with .item().",
        "pca_basis": "min(2 * k + 10, x.shape[0], x.shape[1]) is how many to ask for. torch.pca_lowrank gives three "
        "results, the directions are the last, and [:, :k] keeps the first k columns.",
    },
    introduces=("span", "basis", "projection"),
    check=check_autoencoder3,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# quiz11
# ---------------------------------------------------------------------------

QUIZ11_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines moves
    from Section 8. If a step is unclear, the exercise in the table shows it.

    | Function | Moves | Shown in |
    |---|---|---|
    | `skipgram_pairs` | every word paired with each neighbour inside the window | word2vec1 |
    | `sgns_loss` | the log sigmoid of the real pair's score, plus that of each negative's score turned negative | word2vec2 |
    | `nearest` | the k rows closest to one row by cosine similarity, leaving the row itself out | word2vec3 |
    | `info_nce` | cosine scores over a temperature, each row's partner as the right class | contrastive1 |
    | `masked_loss` | cross entropy at the chosen positions alone | masked1 |
    | `subspace_gap` | the norm of the difference of two projections, each from a QR basis, over the square root of 2k | autoencoder3 |
    """,
)

QUIZ11_SOURCE = source_file(
    '''
    # quiz11.py
    #
    # No examples and no new tools. Each function combines moves from Section 8.

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Return every (center, context) pair of ids, a context being any
    #   other position at most window away, centers in order and contexts in
    #   order within each, as a long tensor of shape (pairs, 2).
    def skipgram_pairs(
        ids: list[int],
        window: int,
    ) -> Tensor:
        ...


    # TODO: Return the mean over the batch of minus the log sigmoid of each
    #   center's dot product with its context, minus the log sigmoids of minus
    #   its dot products with each of its negatives. centers and contexts are
    #   (batch, size), and negatives (batch, k, size).
    def sgns_loss(
        centers: Tensor,
        contexts: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the indices of the k rows of vectors with the highest cosine
    #   similarity to row index, highest first, leaving row index out.
    def nearest(
        vectors: Tensor,
        index: int,
        k: int,
    ) -> list[int]:
        ...


    # TODO: Return the InfoNCE loss of z1 against z2, each (batch, size): the
    #   cosine similarity of every row of z1 with every row of z2, over
    #   temperature, with row i of z2 as the right class for row i of z1.
    def info_nce(
        z1: Tensor,
        z2: Tensor,
        temperature: float,
    ) -> Tensor:
        ...


    # TODO: Return the cross entropy of scores, (positions, vocabulary), against
    #   ids, at the positions where chosen is True alone.
    def masked_loss(
        scores: Tensor,
        ids: Tensor,
        chosen: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the gap between the spaces the columns of a and b span,
    #   each a basis of k columns, as a plain number: the norm of the
    #   difference of their projections, divided by the square root of 2 * k.
    def subspace_gap(
        a: Tensor,
        b: Tensor,
    ) -> float:
        ...
    ''',
)

QUIZ11_SOLUTION = source_file(
    '''
    # quiz11.py, a solution for comparison

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    def skipgram_pairs(
        ids: list[int],
        window: int,
    ) -> Tensor:
        pairs = []

        for center in range(len(ids)):
            for context in range(max(0, center - window), min(len(ids), center + window + 1)):
                if context != center:
                    pairs.append((ids[center], ids[context]))

        return torch.tensor(pairs, dtype=torch.long).view(-1, 2)


    def sgns_loss(
        centers: Tensor,
        contexts: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        real = F.logsigmoid((centers * contexts).sum(dim=-1))
        made_up = F.logsigmoid(-(negatives @ centers.unsqueeze(2)).squeeze(2)).sum(dim=-1)

        return -(real + made_up).mean()


    def nearest(
        vectors: Tensor,
        index: int,
        k: int,
    ) -> list[int]:
        similarity = F.normalize(vectors, dim=1) @ F.normalize(vectors[index], dim=0)
        similarity[index] = float("-inf")

        return similarity.topk(k).indices.tolist()


    def info_nce(
        z1: Tensor,
        z2: Tensor,
        temperature: float,
    ) -> Tensor:
        scores = F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T / temperature

        return F.cross_entropy(scores, torch.arange(len(z1)))


    def masked_loss(
        scores: Tensor,
        ids: Tensor,
        chosen: Tensor,
    ) -> Tensor:
        return F.cross_entropy(scores[chosen], ids[chosen])


    def subspace_gap(
        a: Tensor,
        b: Tensor,
    ) -> float:
        q_a, _ = torch.linalg.qr(a)
        q_b, _ = torch.linalg.qr(b)
        difference = q_a @ q_a.T - q_b @ q_b.T

        return (difference.norm() / math.sqrt(2 * a.shape[1])).item()
    ''',
)


def quiz11_reference() -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(QUIZ11_SOLUTION, namespace)

    return namespace


QUIZ11_REFERENCE = quiz11_reference()


def quiz11_tensor(
    seed: int,
    *shape: int,
) -> Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def check_quiz11(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="skipgram_pairs",
        description="pairs every word with its neighbours in the window",
        inputs=(
            ([4, 7, 1, 9, 2], 2),
            ([3, 3, 8], 1),
        ),
        reference=QUIZ11_REFERENCE["skipgram_pairs"],
        expect=expect_pairs,
    )
    check_on_inputs(
        checker=checker,
        function="sgns_loss",
        description="scores the real pair up and the negatives down",
        inputs=(
            (quiz11_tensor(1110, 4, 3), quiz11_tensor(1111, 4, 3), quiz11_tensor(1112, 4, 5, 3)),
            (quiz11_tensor(1120, 6, 2), quiz11_tensor(1121, 6, 2), quiz11_tensor(1122, 6, 3, 2)),
        ),
        reference=QUIZ11_REFERENCE["sgns_loss"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="nearest",
        description="gives the closest rows by cosine, leaving the row out",
        inputs=(
            (quiz11_tensor(1113, 8, 4), 3, 3),
            (quiz11_tensor(1114, 6, 5), 0, 2),
        ),
        reference=QUIZ11_REFERENCE["nearest"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="info_nce",
        description="takes each row's partner as the right class",
        inputs=(
            (quiz11_tensor(1115, 5, 4), quiz11_tensor(1116, 5, 4), 0.5),
            (quiz11_tensor(1117, 3, 6) * 4, quiz11_tensor(1118, 3, 6), 0.1),
        ),
        reference=QUIZ11_REFERENCE["info_nce"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="masked_loss",
        description="scores the chosen positions alone",
        inputs=(
            (quiz11_tensor(1119, 6, 5), torch.tensor([1, 4, 0, 2, 2, 3]), torch.tensor([True, False, False, True, True, False])),
            (quiz11_tensor(1123, 5, 7), torch.tensor([6, 0, 3, 3, 5]), torch.tensor([False, True, True, False, True])),
        ),
        reference=QUIZ11_REFERENCE["masked_loss"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="subspace_gap",
        description="is 0 for the same space and grows as they part",
        inputs=(
            (quiz11_tensor(1124, 5, 2), quiz11_tensor(1124, 5, 2) @ torch.tensor([[1.0, 2.0], [-1.0, 1.0]])),
            (quiz11_tensor(1125, 6, 3), quiz11_tensor(1126, 6, 3)),
            (torch.eye(5)[:, :2], torch.eye(5)[:, 3:]),
        ),
        reference=QUIZ11_REFERENCE["subspace_gap"],
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )


QUIZ11 = Exercise(
    name="quiz11",
    chapter=QUIZZES,
    title="Representation learning, start to finish",
    lesson=QUIZ11_LESSON,
    source=QUIZ11_SOURCE,
    solution=QUIZ11_SOLUTION,
    hints={
        "skipgram_pairs": "Loop over each center, then over each context from max(0, center - window) up to and "
        "including center + window, stopping at the last id. Skip the center itself.",
        "sgns_loss": "F.logsigmoid of the row by row dot product, plus the sum over k of F.logsigmoid of minus "
        "negatives @ the center. Take minus the mean.",
        "nearest": "Normalize the rows and the chosen row, multiply, set the chosen row's score to minus "
        "infinity, then .topk(k).indices.tolist().",
        "info_nce": "F.normalize both on dim=1, multiply z1 by z2's transpose, divide by temperature, and "
        "F.cross_entropy against torch.arange(len(z1)).",
        "masked_loss": "Index scores and ids with chosen, then F.cross_entropy.",
        "subspace_gap": "Each projection is torch.linalg.qr's first result times its own transpose. The gap is the "
        "norm of their difference over the square root of 2k, as a plain number.",
    },
    introduces=(),
    check=check_quiz11,
)
