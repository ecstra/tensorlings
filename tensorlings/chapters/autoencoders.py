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

        An autoencoder compresses its input into a small code, a short vector
        of numbers, and then reconstructs the input from that code. It needs
        no labels. This chapter builds one for MNIST digits. It measures why
        the code has to be small, or the input noisy, for the autoencoder to
        learn anything useful. Then it shows that a linear autoencoder finds
        the same directions as PCA.

        ## Further reading

        - [Hinton and Salakhutdinov, reducing the dimensionality of data with neural networks](https://www.cs.toronto.edu/~hinton/absps/science.pdf)
        - [Vincent and colleagues, extracting and composing robust features with denoising autoencoders](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
        - [Vincent and colleagues, stacked denoising autoencoders](https://jmlr.org/papers/v11/vincent10a.html)
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
    simplest way to do that. It has two parts. An encoder compresses the
    input into a small code, a short vector of numbers. A decoder then
    reconstructs the input from that code. The loss measures how far the
    reconstruction is from the original input.

    For an MNIST digit, the encoder might map the 784 pixels down to 32
    numbers. The decoder then maps those 32 numbers back to 784 pixels. If 32
    numbers are enough to reconstruct the digit well, they must hold most of
    what makes it that digit. The model is never told which digit it is.

    The decoder ends in a sigmoid, because pixels run from 0 to 1. This
    follows the rule for output activations from activations2. The loss is
    the mean squared error between the reconstruction and the input, so the
    input is its own target.

    ```python
    rebuilt = torch.tensor([0.9, 0.1])
    F.mse_loss(rebuilt, torch.tensor([1.0, 0.0]))    # tensor(0.0100)
    ```

    The table compares codes of four sizes. Each autoencoder trained for
    1,500 steps on 20,000 digits, and was then measured on 2,000 digits it
    never saw.

    | Code size | 2 | 8 | 32 | 128 |
    |---|---|---|---|---|
    | Error on new digits | 0.049 | 0.026 | 0.014 | 0.011 |

    A code of only 2 numbers already reconstructs a blurry digit. Larger
    codes reconstruct finer strokes, and the error keeps falling as the code
    grows. autoencoder2 asks what happens when the code grows larger than the
    input.
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


    # TODO: Finish forward. Pass x, of shape (batch, 784), through self.encoder
    #   to get the code. Then pass the code through self.decoder and return the
    #   reconstruction, which has the same shape as x.
''' + indented(AUTOENCODER_INIT) + '''

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return the mean squared error between model(x), the reconstruction
    #   of x, and x itself.
    def reconstruction_loss(
        model: nn.Module,
        x: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish the loop. In each step, pick a batch of 128 rows of x at the
    #   random indices torch.randint(0, len(x), (128,), generator=generator).
    #   Compute reconstruction_loss of model on the batch, and take one step of
    #   the Adam optimizer. Store the loss in a variable named loss, because the
    #   last line returns loss.item().
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
        call="the weights of the decoder's last Linear layer after those steps",
        actual=model.decoder[2].weight.detach(),
        expected=expected_model.decoder[2].weight.detach(),
    )


def check_autoencoder_learns(found: type) -> None:
    model = seeded_autoencoder(found, 32)
    REFERENCE["train_autoencoder"](model, digit_rows("train_x", 5000), 300, torch.Generator().manual_seed(0))
    error = validation_error(model, 1000)

    if error > LEARNS_BELOW:
        raise Mismatch(f"After 300 steps, Autoencoder(32) reconstructs new digits with an error of {error:.4f}. It should be below {LEARNS_BELOW}.")


# The solution reaches about 0.033 in these 300 steps, and an untrained model sits far above.
LEARNS_BELOW = 0.05


def check_autoencoder1(checker: Checker) -> None:
    checker.test_method(
        class_name="Autoencoder",
        method="forward",
        description="reconstructs x from its code",
        run=check_autoencoder_forward,
    )
    check_on_inputs(
        checker=checker,
        function="reconstruction_loss",
        description="returns the mean squared error of the reconstruction",
        inputs=(
            (Described(seeded_autoencoder(REFERENCE["Autoencoder"], 4), "Autoencoder(4)"), Described(digit_rows("validation_x", 3), "3 digits")),
            (Described(seeded_autoencoder(REFERENCE["Autoencoder"], 8), "Autoencoder(8)"), Described(digit_rows("validation_x", 5), "5 digits")),
        ),
        reference=lambda model, x: REFERENCE["reconstruction_loss"](model, x),
        expect=expect_tensor,
    )
    checker.test(
        function="train_autoencoder",
        description="draws batches of 128 and takes Adam steps on the reconstruction loss",
        run=check_train_autoencoder,
    )
    checker.test_method(
        class_name="Autoencoder",
        method="forward",
        description="learns to reconstruct digits it never saw",
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
        "Autoencoder.forward": "Run self.encoder on x, then run self.decoder on the code it returns.",
        "reconstruction_loss": "Call F.mse_loss on model(x) and x itself.",
        "train_autoencoder": "Index x with torch.randint(...), passing the generator, to get the batch. Compute the "
        "loss on the batch, then call zero_grad, backward and step.",
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


def measure_bottlenecks(seeds: range = range(3)) -> dict[int, dict[str, tuple[float, float, float]]]:
    """autoencoder2's table for each seed, with 4 threads. Not run by any check."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        return {seed: bottleneck_table(seed) for seed in seeds}
    finally:
        torch.set_num_threads(threads)


def bottleneck_table(seed: int = 0) -> dict[str, tuple[float, float, float]]:
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
        torch.manual_seed(seed)
        model = Shallow(code, linear)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        generator = torch.Generator().manual_seed(seed)

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
    Reconstructing the input is easy for a model that can copy it. If an
    autoencoder's code is as wide as its input, nothing forces it to learn
    what digits look like.

    The models here have one layer in the encoder and one in the decoder. A
    code of 1,024 numbers is wider than the 784 pixels. Each model trained
    for 1,500 steps, on 3 seeds. The clean error is the usual reconstruction
    error. For the noisy error, each digit gets noise with a standard
    deviation of 0.5 added, and the reconstruction is compared with the clean
    digit.

    | Autoencoder | Clean error | Noisy error |
    |---|---|---|
    | linear, code 1,024 | 0.0001 to 0.0004 | 0.182 |
    | linear, code 32 | 0.018 | 0.032 |
    | ReLU, code 1,024 | 0.0030 to 0.0032 | 0.022 to 0.023 |
    | ReLU, code 1,024, denoising | 0.0074 | 0.013 |

    The wide linear autoencoder reconstructs clean digits almost perfectly,
    but it also passes most of the noise through to its output. It learned to
    copy its input. With a code of 32 it cannot copy. So it keeps what digits
    have in common and drops most of the noise.

    A **denoising autoencoder** stops the copying without shrinking the code.
    Its input is a noisy digit, and its loss compares the output with the
    clean digit. So copying the input can never give the right answer.
    Vincent and colleagues proposed it in 2008. In 2010 they showed that it
    learns stroke detectors from digits, and an ordinary autoencoder does
    not.

    A linear probe, from masked2, measures how useful a code is. Here it
    trains a Linear layer to classify the codes of 1,000 labelled digits, then
    tests it on 2,000 other digits. On the raw pixels it gets 81.5% right. On
    the two wide ReLU codes, over 3 seeds, the plain one gets 82.4% to 82.7%.
    The denoising one gets 87.4% to 87.9%. Training with noise made a better
    code on every seed.

    The same idea appears across Section 8. The masking in
    62_masked_modeling is also a kind of noise, with the hidden part as the
    answer.
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


    # TODO: Return x with random noise added. Draw the noise with
    #   torch.randn(x.shape, generator=generator) and multiply it by std, so
    #   std is the standard deviation of the noise.
    def add_noise(
        x: Tensor,
        std: float,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the denoising loss. Make a noisy copy of x with add_noise,
    #   pass it through model, and return the mean squared error between that
    #   output and the clean x.
    def denoising_loss(
        model: nn.Module,
        x: Tensor,
        std: float,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the same error as denoising_loss, as a Python float. Put
    #   model in eval mode first, and compute the error inside torch.no_grad()
    #   so autograd records nothing. Put model back in training mode before
    #   returning.
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
            raise Mismatch("noisy_error let autograd record the computation. Run the model inside torch.no_grad().")

        if not model.training:
            raise Mismatch("noisy_error left the model in eval mode. Call model.train() before returning.")

        expect_close(
            call=f"noisy_error of Autoencoder({code}) on {count} digits, noise with standard deviation {std}",
            actual=actual,
            expected=expected,
            tolerance=1e-6,
        )


def check_autoencoder2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="add_noise",
        description="adds noise with the given standard deviation",
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
        description="compares the reconstruction of a noisy copy with the clean input",
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
        "add_noise": "Add std times torch.randn of x's shape, drawn with the generator, to x.",
        "denoising_loss": "Run model on add_noise(x, std, generator). Then take F.mse_loss of the output against the "
        "clean x, not the noisy copy.",
        "noisy_error": "Call model.eval(). Then inside torch.no_grad(), compute the same error as denoising_loss and "
        "convert it with .item(). Call model.train() before returning.",
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


def slowly_spread_points(
    seed: int,
    count: int,
) -> Tensor:
    """Points in 40 dimensions whose spreads fall slowly, from 4 down to 0.63, along random directions, and a shift.
    In 8 dimensions pca_lowrank asked for 2k + 10 reads every column and is exact however it is run. Here one that
    skips its refining lands 0.2 to 0.3 away, and one asked for exactly k lands 0.4 to 0.7 away."""
    generator = torch.Generator().manual_seed(seed)
    spreads = 4 / torch.arange(1, 41).float().sqrt()
    turn, _ = torch.linalg.qr(torch.randn(40, 40, generator=generator))

    return (torch.randn(count, 40, generator=generator) * spreads) @ turn.T + 3.0


def exact_pca(
    x: Tensor,
    k: int,
) -> Tensor:
    _, _, rows = torch.linalg.svd(x - x.mean(dim=0), full_matrices=False)

    return rows[:k].T


def mnist_pca_runs(seeds: range = range(3)) -> dict[int, dict[str, float]]:
    """The runs behind autoencoder3's numbers, for each seed, against PCA's exact 16 directions of 20,000 digits: a
    linear autoencoder with a code of 16 trained 1,500 steps, 16 random directions, pca_lowrank asked for exactly 16,
    pca_basis, and the cosine between the autoencoder's first column and PCA's first direction."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        x = digit_rows("train_x", 20000)
        exact = exact_pca(x, 16)
        gap = SUBSPACE_REFERENCE["subspace_gap"]
        results = {}

        for seed in seeds:
            torch.manual_seed(seed)
            model = SUBSPACE_REFERENCE["LinearAutoencoder"](784, 16)
            SUBSPACE_REFERENCE["train_linear"](model, x, 1500, torch.Generator().manual_seed(seed))
            learned = model.decoder.weight.detach()
            torch.manual_seed(seed)
            _, _, exactly_sixteen = torch.pca_lowrank(x, q=16, center=True)
            torch.manual_seed(seed)
            principal = SUBSPACE_REFERENCE["pca_basis"](x, 16)
            random_basis = torch.randn(784, 16, generator=torch.Generator().manual_seed(seed + 1))
            results[seed] = {
                "autoencoder": gap(learned, exact),
                "random": gap(random_basis, exact),
                "exactly 16": gap(exactly_sixteen, exact),
                "pca_basis": gap(principal, exact),
                "first cosine": F.cosine_similarity(learned[:, 0], exact[:, 0], dim=0).abs().item(),
            }

        return results
    finally:
        torch.set_num_threads(threads)


# The code sizes where autoencoder3 sets autoencoder1's model against PCA.
SMALL_CODES = (2, 8, 32)


def against_pca(seeds: range = range(3)) -> dict[int, tuple[list[float], float]]:
    """autoencoder3's table: for each small code, autoencoder1's model trained as its table was, 1,500 steps on 20,000
    digits, on each seed, and PCA with the same number of directions, rebuilding each digit as its projection onto
    them plus the mean. Both are scored on the same 2,000 new digits."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        x = digit_rows("train_x", 20000)
        new = digit_rows("validation_x", 2000)
        mean = x.mean(dim=0)
        results = {}

        for code in SMALL_CODES:
            torch.manual_seed(0)
            basis = SUBSPACE_REFERENCE["pca_basis"](x, code)
            rebuilt = (new - mean) @ basis @ basis.T + mean
            errors = []

            for seed in seeds:
                torch.manual_seed(seed)
                model = REFERENCE["Autoencoder"](code)
                REFERENCE["train_autoencoder"](model, x, 1500, torch.Generator().manual_seed(seed))
                errors.append(validation_error(model))

            results[code] = (errors, F.mse_loss(rebuilt, new).item())

        return results
    finally:
        torch.set_num_threads(threads)


AUTOENCODER3_LESSON = prose(
    """
    Without activations, an autoencoder is two matrices. The encoder maps a
    centered input, one with the average input subtracted, down to a code of
    k numbers. The decoder maps the code back to the input's size. Baldi and
    Hornik showed in 1989 that the best such pair of matrices keeps the same
    k directions as principal component analysis, PCA. PCA's directions are
    the ones along which the data varies the most.

    A set of k directions **spans** a **space**. The space is every point you
    can reach by scaling the directions and adding them up. The directions
    are called a **basis** of that space. Many different bases span the same
    space.

    The best pair of matrices keeps PCA's space but picks its own basis for
    it. Any basis of that space reconstructs the input equally well. So the
    decoder's columns come out as a mix of PCA's directions, not PCA's
    directions themselves. To compare the two, compare the spaces they span.

    A **projection** onto a space maps every point to its nearest point in
    that space. Two bases span the same space exactly when they give the same
    projection. `torch.linalg.qr` turns a basis, stored as columns, into a
    new basis `q` for the same space. The columns of `q` have length 1 and
    are perpendicular to each other. With that basis, the projection matrix
    is `q @ q.T`.

    ```python
    q, _ = torch.linalg.qr(torch.tensor([[2.0], [0.0]]))
    q @ q.T    # tensor([[1., 0.], [0., 0.]]), whatever the sign of q
    ```

    To measure the gap between two spaces of k directions, subtract their
    projection matrices and take the norm of the difference. The norm of a
    matrix here is the square root of the sum of its squared entries, which
    `.norm()` computes. Then divide by the square root of `2 * k`. The gap is
    0 for the same space, and 1 for spaces at right angles to each other.

    `torch.pca_lowrank(x, q=q, center=True)` returns three results, and the
    columns of the third are PCA's first q directions. It is a fast method
    that uses random numbers, so it is only approximate. On MNIST, asked for
    exactly 16 directions, its space has a gap of 0.30 to 0.38 from the true
    space, depending on the random draw. Ask it for `2 * k + 10` directions
    with 4 refining rounds, `niter=4`, then keep the first k. The gap is then
    within 0.001.

    Here is a linear autoencoder with a code of 16, trained for 1,500 steps
    on 20,000 digits with 3 seeds. It is compared with PCA's first 16
    directions.

    - The gap from the decoder's columns to PCA's space: 0.05 to 0.08.
    - The gap from 16 random directions to PCA's space: 0.99.
    - The cosine similarity of the decoder's first column and PCA's first direction: 0.01 to 0.52.

    So the autoencoder found PCA's space, but with a basis of its own. With
    activations, an autoencoder can fit curved shapes in the data that PCA
    cannot. PCA reconstructs a digit in three steps. It subtracts the average
    digit, projects the result onto its space, and adds the average back. The
    table compares autoencoder1's model and PCA on the same 2,000 new digits.

    | Code size | 2 | 8 | 32 |
    |---|---|---|---|
    | Error of autoencoder1's model, 3 seeds | 0.048 to 0.049 | 0.026 | 0.014 |
    | Error of PCA | 0.056 | 0.038 | 0.018 |

    For these small codes, the autoencoder with activations has an error 13%
    to 33% lower than PCA's. That advantage is what made the deep
    autoencoders of Hinton and Salakhutdinov worth training.
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
    onto_first = q @ q.T          # tensor([[1., 0.], [0., 0.]]), whatever the sign of q


    # TODO: Return the projection matrix onto the space that the columns of
    #   basis span. basis has shape (rows, k), and the result has shape
    #   (rows, rows). First make the columns perpendicular, with length 1,
    #   using torch.linalg.qr as the lesson shows.
    #   Example: projection(torch.tensor([[1.0], [1.0]])) returns
    #   tensor([[0.5000, 0.5000], [0.5000, 0.5000]]).
    def projection(basis: Tensor) -> Tensor:
        ...


    # TODO: Return the gap between the spaces spanned by the columns of a and
    #   of b, as a Python float. a and b each have shape (rows, k). Subtract
    #   their projection matrices, take the norm of the difference, and divide
    #   it by the square root of 2 * k.
    #   Example: subspace_gap(torch.eye(4)[:, :2], torch.eye(4)[:, 2:]) returns
    #   1.0, because those two spaces are at right angles.
    def subspace_gap(
        a: Tensor,
        b: Tensor,
    ) -> float:
        ...


    # TODO: Return PCA's first k directions of x, as a tensor of k columns. x
    #   has shape (points, size), so the result has shape (size, k). Call
    #   torch.pca_lowrank with center=True and niter=4, and ask for
    #   q = 2 * k + 10 directions, or fewer if x has fewer rows or columns than
    #   that. Keep the first k columns of its third result.
    def pca_basis(
        x: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # A linear autoencoder, made of two matrices with no activations. It trains
    #   on centered data.
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
        x = slowly_spread_points(seed, 400)
        actual = found(x, k)

        if not isinstance(actual, Tensor) or tuple(actual.shape) != (40, k):
            raise Mismatch(f"pca_basis of 400 points in 40 dimensions, k={k}, should return a tensor of shape (40, {k}).")

        gap = SUBSPACE_REFERENCE["subspace_gap"](actual, exact_pca(x, k))

        if gap > 0.01:
            raise Mismatch(
                f"pca_basis of 400 points in 40 dimensions, k={k}, spans a space with a gap of {gap:.3f} from PCA's first "
                f"{k} directions. torch.pca_lowrank uses random numbers, so it needs the extra directions and the 4 "
                "refining rounds that the TODO asks for.",
            )


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
        raise Mismatch(f"The gap between the trained linear autoencoder's space and PCA's space is {near:.3f}, but it should be near 0.")


def check_autoencoder3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="projection",
        description="returns the projection onto the space the columns span",
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
        description="is 0 for the same space and grows as the spaces differ more",
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
        description="shows that a trained linear autoencoder lies in PCA's space",
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
        "projection": "torch.linalg.qr of the basis returns q, whose columns are perpendicular with length 1, and a "
        "second factor you can ignore. Return q matrix multiplied by its own transpose.",
        "subspace_gap": "Subtract projection(b) from projection(a) and take the .norm() of the result. Divide it by "
        "math.sqrt(2 * k), where k is the number of columns, and convert it with .item().",
        "pca_basis": "Ask for min(2 * k + 10, x.shape[0], x.shape[1]) directions. torch.pca_lowrank returns three "
        "results, and the directions are the last one. Then [:, :k] keeps the first k columns.",
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
    This quiz has no examples and no new tools. Each function combines steps
    from Section 8. If a step is unclear, the exercise listed in the table
    shows how to do it.

    | Function | Steps | Shown in |
    |---|---|---|
    | `vocabulary_pairs` | keep the common words as a vocabulary, replace each kept word by its index in it, then pair every id with its neighbours | word2vec1 |
    | `skipgram_loss` | look up rows in the two tables by id, then compute the negative sampling loss of the real and made up pairs | embedding1, word2vec2 |
    | `nearest` | scale every row and the chosen row to length 1, take their cosines, and keep the k highest except the chosen row | word2vec3 |
    | `info_nce` | divide the cosine of every pair of rows by a temperature, then take the cross entropy with each row's partner as the target | contrastive1 |
    | `masked_loss` | choose the positions where a random draw is below the rate, then take the cross entropy at those positions only | masked1 |
    | `subspace_gap` | a QR basis and its projection matrix for each space, then the norm of their difference divided by the square root of 2k | autoencoder3 |
    """,
)

QUIZ11_SOURCE = source_file(
    '''
    # quiz11.py
    #
    # No examples and no new tools. Each function combines steps from Section 8.

    import math
    from collections import Counter

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Return the skip gram pairs of words as a long tensor of shape
    #   (pairs, 2). The vocabulary is the words seen at least min_count times,
    #   with the most common first and ties in alphabetical order. Replace each
    #   kept word by its index in the vocabulary, and drop the other words. Then
    #   pair each id with every other id at most window places away. List the
    #   pairs center by center, and for each center from its leftmost neighbour
    #   to its rightmost. Put the center id first in each pair.
    def vocabulary_pairs(
        words: list[str],
        min_count: int,
        window: int,
    ) -> Tensor:
        ...


    # TODO: Return the negative sampling loss, averaged over the batch. centers
    #   and contexts are the two embedding tables, each of shape (vocabulary,
    #   size). Each row of pairs, of shape (batch, 2), holds a center id and
    #   then a context id. negatives has shape (batch, k) and holds the made up
    #   ids for each pair. Look up the center ids in centers, and look up the
    #   context ids and the made up ids in contexts.
    def skipgram_loss(
        centers: Tensor,
        contexts: Tensor,
        pairs: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the indices of the k rows of vectors with the highest cosine
    #   similarity to vectors[index], as a list with the highest first. Leave
    #   out the row at index itself.
    def nearest(
        vectors: Tensor,
        index: int,
        k: int,
    ) -> list[int]:
        ...


    # TODO: Return the InfoNCE loss of z1 against z2, which each have shape
    #   (batch, size). The scores are the cosine similarity of every row of z1
    #   with every row of z2, divided by temperature. Take the cross entropy of
    #   the scores, with column i as the target of row i.
    def info_nce(
        z1: Tensor,
        z2: Tensor,
        temperature: float,
    ) -> Tensor:
        ...


    # TODO: Choose the positions where torch.rand(ids.shape, generator=generator)
    #   is below rate. scores has shape (positions, vocabulary), and ids has
    #   shape (positions,). Return the cross entropy of scores against ids at
    #   the chosen positions only.
    def masked_loss(
        scores: Tensor,
        ids: Tensor,
        rate: float,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the gap between the spaces spanned by the columns of a and
    #   of b, as a Python float. a and b each have k columns. Subtract their
    #   projection matrices, take the norm of the difference, and divide it by
    #   the square root of 2 * k.
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
    from collections import Counter

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    def vocabulary_pairs(
        words: list[str],
        min_count: int,
        window: int,
    ) -> Tensor:
        counted = Counter(words)
        kept = sorted([word for word in counted if counted[word] >= min_count], key=lambda word: (-counted[word], word))
        place = {word: position for position, word in enumerate(kept)}
        ids = [place[word] for word in words if word in place]
        pairs = []

        for center in range(len(ids)):
            for context in range(max(0, center - window), min(len(ids), center + window + 1)):
                if context != center:
                    pairs.append((ids[center], ids[context]))

        return torch.tensor(pairs, dtype=torch.long).view(-1, 2)


    def skipgram_loss(
        centers: Tensor,
        contexts: Tensor,
        pairs: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        center = centers[pairs[:, 0]]
        real = F.logsigmoid((center * contexts[pairs[:, 1]]).sum(dim=-1))
        made_up = F.logsigmoid(-(contexts[negatives] @ center.unsqueeze(2)).squeeze(2)).sum(dim=-1)

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
        rate: float,
        generator: torch.Generator,
    ) -> Tensor:
        chosen = torch.rand(ids.shape, generator=generator) < rate

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


def quiz11_ids(
    seed: int,
    vocabulary: int,
    *shape: int,
) -> Tensor:
    return torch.randint(0, vocabulary, shape, generator=torch.Generator().manual_seed(seed))


def check_quiz11(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="vocabulary_pairs",
        description="keeps the common words as ids and pairs each with its neighbours",
        inputs=(
            ("to be or not to be that is the question to be".split(), 2, 1),
            ("b a c b a c d c a".split(), 2, 2),
            ("x y x z y x".split(), 1, 3),
        ),
        reference=QUIZ11_REFERENCE["vocabulary_pairs"],
        expect=expect_pairs,
    )
    check_on_inputs(
        checker=checker,
        function="skipgram_loss",
        description="looks up the rows and computes the negative sampling loss",
        inputs=(
            (quiz11_tensor(1110, 6, 3), quiz11_tensor(1111, 6, 3), quiz11_ids(1112, 6, 4, 2), quiz11_ids(1113, 6, 4, 5)),
            (quiz11_tensor(1120, 5, 2), quiz11_tensor(1121, 5, 2), quiz11_ids(1122, 5, 6, 2), quiz11_ids(1123, 5, 6, 3)),
        ),
        reference=QUIZ11_REFERENCE["skipgram_loss"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="nearest",
        description="returns the most similar rows by cosine, leaving out the row itself",
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
        description="uses column i as the target of row i",
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
        description="computes the cross entropy at the chosen positions only",
        inputs=(
            (quiz11_tensor(1119, 6, 5), torch.tensor([1, 4, 0, 2, 2, 3]), 0.5, seeded(1130)),
            (quiz11_tensor(1127, 5, 7), torch.tensor([6, 0, 3, 3, 5]), 0.4, seeded(1131)),
        ),
        reference=QUIZ11_REFERENCE["masked_loss"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="subspace_gap",
        description="is 0 for the same space and grows as the spaces differ more",
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
        "vocabulary_pairs": "Count with Counter, then keep and sort the words as word2vec1's vocabulary did. Make a dict "
        "from each kept word to its index. The ids are the indices of the kept words, in text order. Then loop over "
        "each center and each context up to window away, skipping the center itself.",
        "skipgram_loss": "Index centers with pairs[:, 0] and contexts with pairs[:, 1]. Index contexts with negatives "
        "to get rows of shape (batch, k, size). Add F.logsigmoid of the real dot products to the sum over k of "
        "F.logsigmoid of minus the made up dot products. Return minus the mean.",
        "nearest": "Normalize the rows and the chosen row, and multiply them to get one cosine per row. Set the chosen "
        "row's score to minus infinity, then use .topk(k).indices.tolist().",
        "info_nce": "Normalize both with F.normalize on dim=1, multiply z1 by the transpose of z2, and divide by "
        "temperature. Then call F.cross_entropy against torch.arange(len(z1)).",
        "masked_loss": "chosen is the draw compared with rate using <. Index scores and ids with chosen, then call "
        "F.cross_entropy.",
        "subspace_gap": "Each projection matrix is the first result of torch.linalg.qr times its own transpose. The gap "
        "is the norm of their difference divided by the square root of 2k, as a Python float.",
    },
    introduces=(),
    check=check_quiz11,
)
