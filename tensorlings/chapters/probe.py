"""64_probe, Section 8's project, SimCLR on CIFAR-10 with no labels, then a linear probe on its frozen features with a few labels."""

from __future__ import annotations

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
)
from ..text import prose, source_file
from ..training import MissingCheckpoint, load_checkpoint
from .cifar import BLOCK_PART, PREPARE_PART, SMALL_RESNET_SOLUTION, cifar_sets

PROBE = Chapter(
    folder="64_probe",
    title="Project: features without labels",
    readme=source_file(
        """
        # Project: features without labels

        Section 8's project. You train cifar2's small ResNet on CIFAR-10 with
        SimCLR, which never sees a label, then freeze it and fit a linear
        probe on its features with a few labels. Then you measure what the
        unlabelled training bought, against random features and against a
        network trained on the same few labels. The training runs for minutes
        outside the app.

        ## Further reading

        - [Chen and colleagues, SimCLR](https://arxiv.org/abs/2002.05709)
        - [Alain and Bengio, linear classifier probes](https://arxiv.org/abs/1610.01644)
        - [Wang and Isola, alignment and uniformity on the hypersphere](https://arxiv.org/abs/2005.10242)
        """,
    ),
)

# contrastive1's loss, with its partners made on the scores' device so it also runs on MPS and CUDA.
LOSS_PART = '''
def nt_xent(
    z1: Tensor,
    z2: Tensor,
    temperature: float,
) -> Tensor:
    both = F.normalize(torch.cat([z1, z2]), dim=1)
    scores = both @ both.T / temperature
    scores.fill_diagonal_(float("-inf"))
    count = len(z1)
    partners = torch.cat([torch.arange(count, 2 * count), torch.arange(count)]).to(scores.device)

    return F.cross_entropy(scores, partners)
'''

FEATURES_PART = '''
# What the ResNet's blocks see, averaged over the image, before any head.
def features(
    encoder: nn.Module,
    x: Tensor,
) -> Tensor:
    return encoder.blocks(encoder.stem(x)).mean(dim=(2, 3))


# augment2's shifts and flips, with the edges reflected instead of padded with zeros.
def shift_and_flip(
    images: Tensor,
    generator: torch.Generator,
) -> Tensor:
    pad = 4
    h, w = images.shape[2], images.shape[3]
    tops = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator).tolist()
    lefts = torch.randint(0, 2 * pad + 1, (len(images),), generator=generator).tolist()
    padded = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    shifted = torch.stack([padded[i, :, top : top + h, left : left + w] for i, (top, left) in enumerate(zip(tops, lefts))])
    flips = torch.rand(len(images), 1, 1, 1, generator=generator) < 0.5

    return torch.where(flips, shifted.flip(-1), shifted)
'''

VIEWS_SOLUTION = '''
def jitter(
    images: Tensor,
    generator: torch.Generator,
    strength: float,
) -> Tensor:
    brightness = 1 + strength * (torch.rand(len(images), 1, 1, 1, generator=generator) - 0.5)
    contrast = 1 + strength * (torch.rand(len(images), 1, 1, 1, generator=generator) - 0.5)
    mean = images.mean(dim=(1, 2, 3), keepdim=True)

    return (images - mean) * contrast + mean * brightness


def random_gray(
    images: Tensor,
    generator: torch.Generator,
    chance: float,
) -> Tensor:
    gray = images.mean(dim=1, keepdim=True).expand_as(images)
    picked = torch.rand(len(images), 1, 1, 1, generator=generator) < chance

    return torch.where(picked, gray, images)


def view(
    images: Tensor,
    generator: torch.Generator,
) -> Tensor:
    return random_gray(jitter(shift_and_flip(images, generator), generator, 0.8), generator, 0.2)
'''

STEP_SOLUTION = '''
def simclr_step(
    encoder: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: Tensor,
    generator: torch.Generator,
    device: torch.device,
    temperature: float,
) -> float:
    first = view(images, generator).to(device)
    second = view(images, generator).to(device)
    loss = nt_xent(head(features(encoder, first)), head(features(encoder, second)), temperature)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
'''

PROBE1_MAIN = '''
    # Run by .venv/bin/python run.py train probe1, never by the app.
    def main() -> None:
        images, labels = load_cifar10("train")
        train_x, _, _, _ = prepare(images, labels, 5000, torch.Generator().manual_seed(0))
        device = best_device()
        torch.manual_seed(0)
        encoder = SmallResNet().to(device)
        head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64)).to(device)
        steps = 2000
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        generator = torch.Generator().manual_seed(0)
        print(f"Training on {device} for {steps} steps, with no labels.")

        for step in range(steps):
            batch = torch.randint(0, len(train_x), (256,), generator=generator)
            loss = simclr_step(encoder, head, optimizer, train_x[batch], generator, device, 0.5)
            scheduler.step()

            if (step + 1) % 250 == 0:
                print(f"Step {step + 1}, loss {loss:.3f}")

        save_checkpoint(encoder, "probe1")
        print("Saved the encoder in checkpoints/probe1.pt. probe2 reads it back.")
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(code, namespace)

    return namespace


PROBE_REFERENCE = namespace_of(BLOCK_PART + SMALL_RESNET_SOLUTION + LOSS_PART + FEATURES_PART + VIEWS_SOLUTION + STEP_SOLUTION)


# ---------------------------------------------------------------------------
# probe1
# ---------------------------------------------------------------------------

PROBE1_LESSON = prose(
    """
    This is Section 8's project. cifar2's small ResNet learned from 45,000
    labelled photos. Here it learns from the same photos with every label
    thrown away, as SimCLR does, and probe2 measures what it learned.

    contrastive2 made two views of each digit. Here a view of a photo is a
    shift and a flip, then a change of brightness and contrast. One time in
    five it is also turned grey. SimCLR's paper found one pair of changes
    stands out, crops together with color changes. Without the color changes
    two crops of one photo can be matched by their colors alone, and the
    network learns little about shapes.

    - Brightness scales the image's mean, and contrast scales how far each pixel sits from that mean, each by a random factor around 1.
    - Grey is the mean over the three channels, copied back into all three.

    A pair of views of one photo should score high against each other and low
    against every other view in the batch. That is contrastive1's `nt_xent`,
    with a temperature of 0.5. A small head maps the features to 64 numbers
    for the loss, and probe2 throws the head away and keeps the features.

    contrastive2 liked 0.1, which SimCLR found best on ImageNet with batches
    of 4,096. On CIFAR-10 its paper tried 0.1, 0.5 and 1, and 0.5 did best at
    every batch size once trained long. These batches are 256.

    ```python
    images = torch.ones(2, 3, 1, 1)
    images.mean(dim=(1, 2, 3), keepdim=True).shape    # torch.Size([2, 1, 1, 1]), one mean per image
    ```

    Once every check passes, train from the project folder with
    `.venv/bin/python run.py train probe1`. It runs 2,000 steps of 256
    photos and saves the encoder. The run behind this project took about 21
    minutes on MPS while other training shared the machine. Its loss fell
    from 4.56 at step 250 to 4.38. A network that matched nothing would sit
    near the log of the 511 other views, 6.24.
    """,
)

PROBE1_SOURCE = source_file(
    '''
    # probe1.py
    #
    # Train once every check passes, from the project folder.
    #
    #     .venv/bin/python run.py train probe1

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_cifar10
    from tensorlings.training import best_device, save_checkpoint

    images = torch.ones(2, 3, 1, 1)
    means = images.mean(dim=(1, 2, 3), keepdim=True)    # shape (2, 1, 1, 1), one mean per image


''' + indented(PREPARE_PART) + '''


''' + indented(BLOCK_PART) + '''


    # SmallResNet from cifar2.
''' + indented(SMALL_RESNET_SOLUTION) + '''


''' + indented(LOSS_PART) + '''


''' + indented(FEATURES_PART) + '''


    # TODO: Return images with a random brightness and contrast for each image.
    #   Draw the brightness factors, then the contrast factors, each as
    #   1 + strength * (torch.rand(len(images), 1, 1, 1, generator=generator)
    #   - 0.5). Scale each pixel's distance from its image's mean by the
    #   contrast, and the mean itself by the brightness.
    def jitter(
        images: Tensor,
        generator: torch.Generator,
        strength: float,
    ) -> Tensor:
        ...


    # TODO: Return images with each one, with probability chance, replaced by its
    #   grey copy, the mean over the channels in all three. Draw one number per
    #   image with torch.rand and generator, and pick grey where it is below
    #   chance.
    def random_gray(
        images: Tensor,
        generator: torch.Generator,
        chance: float,
    ) -> Tensor:
        ...


    # TODO: Return one view of images: shift_and_flip, then jitter with a
    #   strength of 0.8, then random_gray with a chance of 0.2, all with
    #   generator.
    def view(
        images: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Take one SimCLR step and return the loss as a plain number. Make two
    #   views of images, first then second, move each to device, and take
    #   nt_xent of head on each one's features at temperature. Run each view
    #   through the encoder on its own. One joined batch would work too, but
    #   BatchNorm's batch statistics would give other numbers than the check's.
    def simclr_step(
        encoder: nn.Module,
        head: nn.Module,
        optimizer: torch.optim.Optimizer,
        images: Tensor,
        generator: torch.Generator,
        device: torch.device,
        temperature: float,
    ) -> float:
        ...

''' + PROBE1_MAIN,
)

PROBE1_SOLUTION = source_file(
    '''
    # probe1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_cifar10
    from tensorlings.training import best_device, save_checkpoint

    images = torch.ones(2, 3, 1, 1)
    means = images.mean(dim=(1, 2, 3), keepdim=True)


''' + indented(PREPARE_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(SMALL_RESNET_SOLUTION) + '''


''' + indented(LOSS_PART) + '''


''' + indented(FEATURES_PART) + '''


''' + indented(VIEWS_SOLUTION) + '''


''' + indented(STEP_SOLUTION) + '''

''' + PROBE1_MAIN,
)


def small_images(
    seed: int,
    count: int,
    size: int = 6,
) -> Tensor:
    return torch.randn(count, 3, size, size, generator=torch.Generator().manual_seed(seed))


def tiny_encoder() -> nn.Module:
    torch.manual_seed(640)

    return PROBE_REFERENCE["SmallResNet"]()


def check_simclr_step(found: object) -> None:
    images = small_images(641, 6, 12)
    results = []

    for step in (found, PROBE_REFERENCE["simclr_step"]):
        encoder = tiny_encoder()
        torch.manual_seed(642)
        head = nn.Sequential(nn.Linear(128, 16), nn.ReLU(), nn.Linear(16, 8))
        optimizer = torch.optim.SGD(list(encoder.parameters()) + list(head.parameters()), lr=0.1)
        generator = torch.Generator().manual_seed(643)
        # Two steps, since gradients left from a first step only show in a second.
        # The second runs at another temperature, so one typed in as 0.5 shows.
        step(encoder, head, optimizer, images, generator, torch.device("cpu"), 0.5)
        loss = step(encoder, head, optimizer, images, generator, torch.device("cpu"), 0.2)
        results.append((loss, encoder.stem[0].weight.detach().clone()))

    (loss, weight), (expected_loss, expected_weight) = results
    expect_close(
        call="simclr_step's loss on 6 images, at the second step, at a temperature of 0.2",
        actual=loss,
        expected=expected_loss,
        tolerance=1e-5,
    )
    expect_tensor(
        call="the stem's weights after two simclr_steps",
        actual=weight,
        expected=expected_weight,
    )


def check_probe1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="jitter",
        description="scales each image's brightness and contrast",
        inputs=(
            (Described(small_images(644, 4), "4 images of 6 by 6"), seeded(645), 0.8),
            (Described(small_images(646, 2), "2 images of 6 by 6"), seeded(647), 0.3),
        ),
        reference=PROBE_REFERENCE["jitter"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="random_gray",
        description="turns about chance of the images grey",
        inputs=(
            (Described(small_images(648, 8), "8 images of 6 by 6"), seeded(649), 0.5),
            (Described(small_images(666, 8), "8 images of 6 by 6"), seeded(667), 0.2),
        ),
        reference=PROBE_REFERENCE["random_gray"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="view",
        description="shifts and flips, jitters, then greys some",
        inputs=(
            (Described(small_images(650, 5, 10), "5 images of 10 by 10"), seeded(651)),
        ),
        reference=PROBE_REFERENCE["view"],
        expect=expect_tensor,
    )
    checker.test(
        function="simclr_step",
        description="trains on two views of each image",
        run=check_simclr_step,
    )


PROBE1 = Exercise(
    name="probe1",
    chapter=PROBE,
    title="Project: SimCLR with no labels",
    lesson=PROBE1_LESSON,
    source=PROBE1_SOURCE,
    solution=PROBE1_SOLUTION,
    hints={
        "jitter": "Draw brightness, then contrast, as the TODO writes them. The mean over dim=(1, 2, 3) with "
        "keepdim=True gives each image's mean. Then (images - mean) * contrast + mean * brightness.",
        "random_gray": "images.mean(dim=1, keepdim=True).expand_as(images) is the grey copy. torch.where picks it where "
        "the draw is below chance.",
        "view": "random_gray of jitter of shift_and_flip, with 0.8 and 0.2.",
        "simclr_step": "first = view(images, generator).to(device), then second the same way. The loss is nt_xent of "
        "head(features(encoder, first)) and the same for second. Then zero_grad, backward and step.",
    },
    introduces=(),
    check=check_probe1,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# probe2
# ---------------------------------------------------------------------------

PROBE_SOLUTION = '''
def frozen_features(
    encoder: nn.Module,
    x: Tensor,
    batch_size: int = 500,
) -> Tensor:
    encoder.eval()

    with torch.no_grad():
        return torch.cat([features(encoder, x[start : start + batch_size]) for start in range(0, len(x), batch_size)])


def standardize(
    train_features: Tensor,
    other_features: Tensor,
) -> tuple[Tensor, Tensor]:
    mean = train_features.mean(dim=0)
    spread = train_features.std(dim=0) + 1e-6

    return (train_features - mean) / spread, (other_features - mean) / spread


def train_probe(
    probe: nn.Linear,
    train_features: Tensor,
    labels: Tensor,
    steps: int,
) -> nn.Linear:
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)

    for _ in range(steps):
        loss = F.cross_entropy(probe(train_features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return probe


def probe_accuracy(
    probe: nn.Linear,
    features: Tensor,
    labels: Tensor,
) -> float:
    with torch.no_grad():
        return (probe(features).argmax(dim=1) == labels).float().mean().item()
'''

# The few labelled photos every probe gets, the same ones each run.
LABELLED = 2500

PROBE2_GIVEN = '''
# The first count photos of the training part in a fixed shuffle, the few that keep their labels.
def labelled_photos(
    train_x: Tensor,
    train_y: Tensor,
    count: int,
) -> tuple[Tensor, Tensor]:
    picked = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(5))[:count]

    return train_x[picked], train_y[picked]
'''

PROBE_REFERENCE.update(namespace_of(BLOCK_PART + SMALL_RESNET_SOLUTION + FEATURES_PART + PROBE_SOLUTION + PROBE2_GIVEN))

PROBE2_LESSON = prose(
    """
    probe1's encoder never saw a label. A linear probe measures what it
    learned anyway. Freeze the encoder and run every photo through it once.
    Then fit a single Linear layer from its 128 features to the 10 classes,
    with a few labelled photos.

    The probe is linear on purpose. A deeper head could learn the classes by
    itself, and then the score would say little about the features. A linear
    one can only use what the features already lay out in straight lines,
    which is what Alain and Bengio's probes measure.

    The features get standardized first, each one shifted to a mean of 0 and
    scaled to a spread of 1, as cifar1 did for pixels. The mean and spread
    come from the labelled photos alone.

    ```python
    rows = torch.tensor([[1.0, 10.0], [3.0, 30.0]])
    rows.mean(dim=0), rows.std(dim=0)    # (tensor([ 2., 20.]), tensor([ 1.4142, 14.1421]))
    ```

    `labelled_photos` picks 2,500 labelled photos, 1 in 18 of the training
    part. With them, the probe on the trained encoder labels 53.7% of the
    validation set right. The check asks for more than 45%. probe3 says
    what that number is worth.
    """,
)

PROBE2_SOURCE = source_file(
    '''
    # probe2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_cifar10
    from tensorlings.training import load_checkpoint

    rows = torch.tensor([[1.0, 10.0], [3.0, 30.0]])
    centred = rows - rows.mean(dim=0)    # tensor([[-1., -10.], [1., 10.]])


''' + indented(PREPARE_PART) + '''


''' + indented(BLOCK_PART) + '''


    # SmallResNet from cifar2.
''' + indented(SMALL_RESNET_SOLUTION) + '''


''' + indented(FEATURES_PART) + '''


''' + indented(PROBE2_GIVEN) + '''


    # TODO: Return encoder's features of every image of x, going through x in
    #   batches of batch_size, in eval mode and without recording for autograd.
    def frozen_features(
        encoder: nn.Module,
        x: Tensor,
        batch_size: int = 500,
    ) -> Tensor:
        ...


    # TODO: Return both sets of features standardized with train_features' own
    #   mean and spread for each feature, the spread from .std plus 1e-6.
    def standardize(
        train_features: Tensor,
        other_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Train probe with Adam at a rate of 0.01 and weight decay of 0.0001,
    #   for steps steps of cross entropy on all of train_features at once.
    #   Return the trained probe.
    def train_probe(
        probe: nn.Linear,
        train_features: Tensor,
        labels: Tensor,
        steps: int,
    ) -> nn.Linear:
        ...


    # TODO: Return the share of features probe labels right, as a plain number,
    #   without recording for autograd.
    def probe_accuracy(
        probe: nn.Linear,
        features: Tensor,
        labels: Tensor,
    ) -> float:
        ...
    ''',
)

PROBE2_SOLUTION = source_file(
    '''
    # probe2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_cifar10
    from tensorlings.training import load_checkpoint

    rows = torch.tensor([[1.0, 10.0], [3.0, 30.0]])
    centred = rows - rows.mean(dim=0)


''' + indented(PREPARE_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(SMALL_RESNET_SOLUTION) + '''


''' + indented(FEATURES_PART) + '''


''' + indented(PROBE2_GIVEN) + '''


''' + indented(PROBE_SOLUTION) + '''
    ''',
)


def check_frozen_features(found: object) -> None:
    encoder = tiny_encoder()
    x = small_images(652, 7, 8)
    # features runs the stem and the blocks, never the encoder's own forward, so the hook sits on the stem.
    stem_calls: list[tuple[bool, bool, int]] = []
    hook = encoder.stem.register_forward_pre_hook(lambda module, inputs: stem_calls.append((module.training, torch.is_grad_enabled(), len(inputs[0]))))

    try:
        actual = found(encoder, x, 3)
    finally:
        hook.remove()

    if any(training for training, _, _ in stem_calls):
        raise Mismatch("frozen_features ran the encoder in training mode. Call encoder.eval() first.")

    if any(recording for _, recording, _ in stem_calls):
        raise Mismatch("frozen_features recorded for autograd. Run the batches inside torch.no_grad().")

    batch_sizes = [size for _, _, size in stem_calls]

    if batch_sizes != [3, 3, 1]:
        raise Mismatch(
            f"frozen_features of 7 images in batches of 3 should run the encoder on 3, 3, then 1 images, but ran it on {batch_sizes}."
        )

    with torch.no_grad():
        encoder.eval()
        expected = PROBE_REFERENCE["features"](encoder, x)

    expect_tensor(
        call="frozen_features of 7 images in batches of 3",
        actual=actual,
        expected=expected,
    )


def check_train_probe(found: object) -> None:
    generator = torch.Generator().manual_seed(653)
    train_features = torch.randn(40, 6, generator=generator)
    labels = torch.randint(0, 3, (40,), generator=generator)
    results = []

    for train in (found, PROBE_REFERENCE["train_probe"]):
        torch.manual_seed(654)
        probe = nn.Linear(6, 3)
        trained = train(probe, train_features, labels, 25)
        results.append(trained.weight.detach().clone() if isinstance(trained, nn.Linear) else None)

    if results[0] is None:
        raise Mismatch("train_probe should return the trained nn.Linear.")

    expect_tensor(
        call="the probe's weights after train_probe for 25 steps",
        actual=results[0],
        expected=results[1],
    )


def check_accuracy_without_autograd(found: object) -> None:
    torch.manual_seed(657)
    probe = nn.Linear(4, 3)
    recording: list[bool] = []
    probe.register_forward_pre_hook(lambda module, inputs: recording.append(torch.is_grad_enabled()))
    found(probe, small_tensor_rows(658, 20, 4), torch.randint(0, 3, (20,), generator=torch.Generator().manual_seed(659)))

    if any(recording):
        raise Mismatch("probe_accuracy recorded for autograd. Score inside torch.no_grad().")


def trained_encoder() -> nn.Module:
    torch.manual_seed(0)
    encoder = PROBE_REFERENCE["SmallResNet"]()

    try:
        load_checkpoint(encoder, "probe1")
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None

    return encoder


def reference_probe_score(
    encoder: nn.Module,
    labelled: int,
) -> float:
    """The probe of probe2 on encoder, with the labelled photos of the training part, scored on validation."""
    train_x, train_y, validation_x, validation_y = cifar_sets()
    few_x, few_y = PROBE_REFERENCE["labelled_photos"](train_x, train_y, labelled)
    train_features, validation_features = PROBE_REFERENCE["standardize"](
        PROBE_REFERENCE["frozen_features"](encoder, few_x),
        PROBE_REFERENCE["frozen_features"](encoder, validation_x),
    )
    torch.manual_seed(0)
    probe = PROBE_REFERENCE["train_probe"](nn.Linear(128, 10), train_features, few_y, 1000)

    return PROBE_REFERENCE["probe_accuracy"](probe, validation_features, validation_y)


def check_trained_probe(found: object) -> None:
    score = reference_probe_score(trained_encoder(), LABELLED)

    if score <= 0.45:
        raise Mismatch(f"A probe on the trained encoder labels {score:.1%} of the validation set right, but it should pass 45%.")


def check_probe2(checker: Checker) -> None:
    checker.test(
        function="frozen_features",
        description="gives every image's features in eval mode, without autograd",
        run=check_frozen_features,
    )
    train_features = small_tensor_rows(655, 6, 4)
    check_on_inputs(
        checker=checker,
        function="standardize",
        description="scales both sets by the training features' mean and spread",
        inputs=(
            (train_features, small_tensor_rows(656, 3, 4)),
        ),
        reference=PROBE_REFERENCE["standardize"],
        expect=expect_pair,
    )
    checker.test(
        function="train_probe",
        description="trains the Linear with Adam on the cross entropy",
        run=check_train_probe,
    )
    torch.manual_seed(657)
    probe = nn.Linear(4, 3)
    torch.manual_seed(668)
    wider_probe = nn.Linear(5, 4)
    check_on_inputs(
        checker=checker,
        function="probe_accuracy",
        description="gives the share labelled right",
        inputs=(
            (Described(probe, "a Linear from 4 to 3"), small_tensor_rows(658, 20, 4), torch.randint(0, 3, (20,), generator=torch.Generator().manual_seed(659))),
            (Described(wider_probe, "a Linear from 5 to 4"), small_tensor_rows(669, 30, 5), torch.randint(0, 4, (30,), generator=torch.Generator().manual_seed(670))),
        ),
        reference=PROBE_REFERENCE["probe_accuracy"],
        expect=expect_close,
    )
    checker.test(
        function="probe_accuracy",
        description="scores without recording for autograd",
        run=check_accuracy_without_autograd,
    )
    checker.test(
        function="probe_accuracy",
        description="on the trained encoder, passes 45 percent with 2,500 labels",
        run=check_trained_probe,
    )


def small_tensor_rows(
    seed: int,
    rows: int,
    columns: int,
) -> Tensor:
    return torch.randn(rows, columns, generator=torch.Generator().manual_seed(seed)) * 3 + 1


def expect_pair(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should give the two standardized sets in a tuple, the training ones first.")

    for name, part, wanted in zip(("first", "second"), actual, expected):
        expect_tensor(
            call=f"the {name} set of {call}",
            actual=part,
            expected=wanted,
        )


PROBE2 = Exercise(
    name="probe2",
    chapter=PROBE,
    title="Project: a linear probe",
    lesson=PROBE2_LESSON,
    source=PROBE2_SOURCE,
    solution=PROBE2_SOLUTION,
    hints={
        "frozen_features": "encoder.eval(), then inside torch.no_grad() torch.cat the features of each slice "
        "x[start : start + batch_size].",
        "standardize": "Take the mean and std over dim=0 of train_features, add 1e-6 to the std, and apply both to "
        "each set.",
        "train_probe": "torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4). Each step, F.cross_entropy "
        "of probe(train_features) and labels, then zero_grad, backward, step.",
        "probe_accuracy": "Inside torch.no_grad(), compare probe(features).argmax(dim=1) with labels and average as "
        "floats.",
    },
    introduces=(),
    check=check_probe2,
    time_limit=240,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# probe3
# ---------------------------------------------------------------------------

SCORES_SOLUTION = '''
def probe_score(
    encoder: nn.Module,
    few_x: Tensor,
    few_y: Tensor,
    validation_x: Tensor,
    validation_y: Tensor,
    steps: int,
) -> float:
    train_features, validation_features = standardize(frozen_features(encoder, few_x), frozen_features(encoder, validation_x))
    torch.manual_seed(0)
    probe = train_probe(nn.Linear(train_features.shape[1], 10), train_features, few_y, steps)

    return probe_accuracy(probe, validation_features, validation_y)


def scores_by_labels(
    encoder: nn.Module,
    train_x: Tensor,
    train_y: Tensor,
    validation_x: Tensor,
    validation_y: Tensor,
    counts: list[int],
) -> list[float]:
    scores = []

    for count in counts:
        few_x, few_y = labelled_photos(train_x, train_y, count)
        scores.append(probe_score(encoder, few_x, few_y, validation_x, validation_y, 1000))

    return scores
'''

PROBE_REFERENCE.update(namespace_of(BLOCK_PART + SMALL_RESNET_SOLUTION + FEATURES_PART + PROBE_SOLUTION + PROBE2_GIVEN + SCORES_SOLUTION))

PROBE3_LESSON = prose(
    """
    A probe score means little alone. Two comparisons give it weight.

    - Random features: the same ResNet, untrained, frozen and probed the same way. Whatever beats it is what SimCLR's training added.
    - A supervised model: the same ResNet trained on the labelled photos alone, with labels, as cifar2 trained on all of them.

    Here is the run behind this project, on 5,000 validation photos, for
    several numbers of labelled photos. The supervised models trained for
    2,000 steps with cifar2's shifts and flips.

    | Labelled photos | SimCLR probe | Random probe | Supervised |
    |---|---|---|---|
    | 100 | 37.2% | 21.6% | 28.7% |
    | 500 | 44.6% | 27.4% | 45.5% |
    | 2,500 | 53.7% | 38.7% | 65.8% |
    | 10,000 | 57.9% | 42.7% | 75.2% |

    SimCLR's features beat the random ones by 15 to 17 points at every
    count, so its training taught the encoder something real. Labels still
    win once there are enough of them. SimCLR leads only at 100 labelled
    photos, and the two are about even at 500. From 2,500 on, the supervised
    model leads by 12 to 17 points.

    This is the honest picture at this size. SimCLR's own paper reached 76.5%
    on ImageNet with a probe. That took a ResNet-50 four times as wide,
    batches of 4,096 and 1,000 epochs. probe1 ran 2,000 steps of 256, about
    11 epochs of the training part. The method pays off with scale, and this
    is a small taste of it.
    """,
)

PROBE3_SOURCE = source_file(
    '''
    # probe3.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_cifar10
    from tensorlings.training import load_checkpoint

    counts = [100, 500]
    pairs = [(count, count // 10) for count in counts]    # [(100, 10), (500, 50)], per class on average


''' + indented(PREPARE_PART) + '''


''' + indented(BLOCK_PART) + '''


    # SmallResNet from cifar2.
''' + indented(SMALL_RESNET_SOLUTION) + '''


''' + indented(FEATURES_PART) + '''


    # labelled_photos and the probe of probe2.
''' + indented(PROBE2_GIVEN) + '''


''' + indented(PROBE_SOLUTION) + '''


    # TODO: Return the validation share right for a probe on encoder's frozen
    #   features of few_x, standardized with few_x's statistics. Make the probe
    #   as nn.Linear from the number of features to 10, right after
    #   torch.manual_seed(0), and train it for steps steps.
    def probe_score(
        encoder: nn.Module,
        few_x: Tensor,
        few_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        steps: int,
    ) -> float:
        ...


    # TODO: Return probe_score for each count of labelled photos in counts, in
    #   order, each picked with labelled_photos and trained for 1,000 steps.
    def scores_by_labels(
        encoder: nn.Module,
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        counts: list[int],
    ) -> list[float]:
        ...
    ''',
)

PROBE3_SOLUTION = source_file(
    '''
    # probe3.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_cifar10
    from tensorlings.training import load_checkpoint

    counts = [100, 500]
    pairs = [(count, count // 10) for count in counts]


''' + indented(PREPARE_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(SMALL_RESNET_SOLUTION) + '''


''' + indented(FEATURES_PART) + '''


''' + indented(PROBE2_GIVEN) + '''


''' + indented(PROBE_SOLUTION) + '''


''' + indented(SCORES_SOLUTION) + '''
    ''',
)


def same_tensor(
    actual: object,
    expected: Tensor,
) -> bool:
    return isinstance(actual, Tensor) and actual.shape == expected.shape and torch.allclose(actual, expected, atol=1e-6)


# What the scripted probe_accuracy answers, a share no real probe on 20 images could give.
SCRIPTED_SHARE = 0.4321


def check_probe_score(found: object) -> None:
    """The file's train_probe and probe_accuracy are swapped for ones that record what they get."""
    encoder = tiny_encoder()
    generator = torch.Generator().manual_seed(660)
    few_x, validation_x = small_images(661, 30, 8), small_images(662, 20, 8)
    few_y, validation_y = torch.randint(0, 10, (30,), generator=generator), torch.randint(0, 10, (20,), generator=generator)
    trained: list[tuple[nn.Module, Tensor, Tensor, Tensor, int]] = []
    scored: list[tuple[nn.Module, Tensor, Tensor]] = []

    def scripted_train_probe(
        probe: nn.Linear,
        train_features: Tensor,
        labels: Tensor,
        steps: int,
    ) -> nn.Linear:
        trained.append((probe, probe.weight.detach().clone(), train_features.detach().clone(), labels, steps))

        return probe

    def scripted_probe_accuracy(
        probe: nn.Linear,
        features: Tensor,
        labels: Tensor,
    ) -> float:
        scored.append((probe, features.detach().clone(), labels))

        return SCRIPTED_SHARE

    namespace = found.__globals__
    originals = {name: namespace.get(name) for name in ("train_probe", "probe_accuracy")}
    namespace["train_probe"] = scripted_train_probe
    namespace["probe_accuracy"] = scripted_probe_accuracy

    try:
        actual = found(encoder, few_x, few_y, validation_x, validation_y, 50)
    finally:
        namespace.update(originals)

    train_features, validation_features = PROBE_REFERENCE["standardize"](
        PROBE_REFERENCE["frozen_features"](encoder, few_x),
        PROBE_REFERENCE["frozen_features"](encoder, validation_x),
    )
    torch.manual_seed(0)
    fresh_probe = nn.Linear(train_features.shape[1], 10)

    if len(trained) != 1:
        raise Mismatch(f"probe_score should train its probe with train_probe once, but called it {len(trained)} times.")

    probe, starting_weight, seen_train_features, seen_labels, steps = trained[0]

    if not same_tensor(starting_weight, fresh_probe.weight.detach()):
        raise Mismatch("The probe's starting weights differ from an nn.Linear to 10 made right after torch.manual_seed(0).")

    if not same_tensor(seen_train_features, train_features) or seen_labels is not few_y:
        raise Mismatch("train_probe should get few_x's frozen features, standardized with their own mean and spread, and few_y.")

    if steps != 50:
        raise Mismatch(f"probe_score for 50 steps handed train_probe {steps} steps. Pass steps through.")

    if len(scored) != 1:
        raise Mismatch(f"probe_score should score with probe_accuracy once, but called it {len(scored)} times.")

    scored_probe, seen_validation_features, seen_validation_y = scored[0]

    if scored_probe is not probe:
        raise Mismatch("probe_accuracy should score the probe that train_probe gave back.")

    if not same_tensor(seen_validation_features, validation_features) or seen_validation_y is not validation_y:
        raise Mismatch(
            "probe_accuracy should get validation_x's frozen features, standardized with few_x's mean and spread, and validation_y."
        )

    expect_close(
        call=f"probe_score, when probe_accuracy gives {SCRIPTED_SHARE}",
        actual=actual,
        expected=SCRIPTED_SHARE,
    )


def check_scores_by_labels(found: object) -> None:
    """The file's probe_score is swapped for one that records what it gets and answers by its count."""
    seen: list[tuple[nn.Module, Tensor, Tensor, Tensor, Tensor, int]] = []

    def scripted_probe_score(
        encoder: nn.Module,
        few_x: Tensor,
        few_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        steps: int,
    ) -> float:
        seen.append((encoder, few_x, few_y, validation_x, validation_y, steps))

        return len(few_x) / 1000

    encoder = nn.Identity()
    train_x, train_y = small_images(663, 60, 4), torch.arange(60) % 10
    validation_x, validation_y = small_images(664, 7, 4), torch.arange(7)
    counts = [20, 5, 40]
    namespace = found.__globals__
    original = namespace.get("probe_score")
    namespace["probe_score"] = scripted_probe_score

    try:
        actual = found(encoder, train_x, train_y, validation_x, validation_y, counts)
    finally:
        namespace["probe_score"] = original

    expect_value(
        call="scores_by_labels for counts [20, 5, 40]",
        actual=actual,
        expected=[count / 1000 for count in counts],
    )

    if len(seen) != len(counts):
        raise Mismatch(f"scores_by_labels for 3 counts should call probe_score 3 times, but called it {len(seen)} times.")

    for count, (seen_encoder, few_x, few_y, seen_validation_x, seen_validation_y, steps) in zip(counts, seen):
        picked_x, picked_y = PROBE_REFERENCE["labelled_photos"](train_x, train_y, count)

        if seen_encoder is not encoder:
            raise Mismatch("scores_by_labels should hand probe_score the encoder it was given.")

        if not same_tensor(few_x, picked_x) or not same_tensor(few_y, picked_y):
            raise Mismatch(f"scores_by_labels should pick the {count} labelled photos with labelled_photos(train_x, train_y, {count}).")

        if seen_validation_x is not validation_x or seen_validation_y is not validation_y:
            raise Mismatch("scores_by_labels should score every count on validation_x and validation_y.")

        if steps != 1000:
            raise Mismatch(f"scores_by_labels should train each probe for 1,000 steps, but asked for {steps}.")


def check_probe3(checker: Checker) -> None:
    checker.test(
        function="probe_score",
        description="probes the frozen features, standardized, from seed 0",
        run=check_probe_score,
    )
    checker.test(
        function="scores_by_labels",
        description="scores each count of labelled photos in turn",
        run=check_scores_by_labels,
    )


PROBE3 = Exercise(
    name="probe3",
    chapter=PROBE,
    title="Project: what the features are worth",
    lesson=PROBE3_LESSON,
    source=PROBE3_SOURCE,
    solution=PROBE3_SOLUTION,
    hints={
        "probe_score": "standardize the frozen_features of few_x and of validation_x. Then torch.manual_seed(0), "
        "nn.Linear(train_features.shape[1], 10), train_probe, and probe_accuracy on the validation features.",
        "scores_by_labels": "For each count, labelled_photos(train_x, train_y, count), then probe_score with 1000 steps.",
    },
    introduces=(),
    check=check_probe3,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)
