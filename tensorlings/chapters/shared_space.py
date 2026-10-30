"""99_shared_space, two kinds of input put in one space as CLIP does: a signal encoder and a text encoder trained with
a symmetric contrastive loss and a learned temperature, then a new signal named by its nearest description with no
training for that name, and what the batch size and the temperature do to the space."""

from __future__ import annotations

import contextlib
import itertools
import math
import textwrap
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Mismatch,
    expect_close,
    expect_tensor,
    reject_missing_return,
    small_tensor,
    value_text,
)
from ..text import prose, source_file
from .linear_layers import expect_module

SHARED_SPACE = Chapter(
    folder="99_shared_space",
    title="Two kinds, one space",
    readme=source_file(
        """
        # Two kinds, one space

        CLIP trained an image encoder and a text encoder together until each
        image sat next to its caption. Then a new picture could be named by
        the nearest caption, even for names it never trained on. This
        chapter does the same with signals and short descriptions of them,
        then measures what the batch size and the temperature do.

        ## Further reading

        - [Radford and colleagues, CLIP, learning transferable visual models from natural language supervision](https://arxiv.org/abs/2103.00020)
        - [van den Oord, Li and Vinyals, contrastive predictive coding](https://arxiv.org/abs/1807.03748)
        - [Wang and Isola, alignment and uniformity on the hypersphere](https://arxiv.org/abs/2005.10242)
        """,
    ),
)

MEASURE_THREADS = 4

DATA_PART = '''
LENGTH = 32
SHAPES = {
    "rises": ("rises", ".", "."),
    "falls": ("falls", ".", "."),
    "flat": ("stays", "flat", "."),
    "peak": ("rises", "then", "falls"),
    "dip": ("falls", "then", "rises"),
    "jump": ("jumps", "up", "."),
}
PERIODS = {
    0: ("never", "repeats", "."),
    4: ("repeats", "every", "4"),
    8: ("repeats", "every", "8"),
    16: ("repeats", "every", "16"),
}
LEVELS = {-1.0: ("low",), 1.0: ("high",)}
WORDS = sorted({word for words in [*SHAPES.values(), *PERIODS.values(), *LEVELS.values()] for word in words})
COMBOS = list(itertools.product(SHAPES, PERIODS, LEVELS))

# Six pairs of a shape and a period that training never shows together.
HELD_OUT = {("peak", 8), ("dip", 4), ("jump", 16), ("flat", 4), ("rises", 16), ("falls", 8)}
SEEN = [combo for combo in COMBOS if combo[:2] not in HELD_OUT]
UNSEEN = [combo for combo in COMBOS if combo[:2] in HELD_OUT]


def description(combo: tuple[str, int, float]) -> list[str]:
    shape, period, level = combo

    return [*SHAPES[shape], *PERIODS[period], *LEVELS[level]]


def description_ids(combos: list[tuple[str, int, float]]) -> Tensor:
    return torch.tensor([[WORDS.index(word) for word in description(combo)] for combo in combos])


def trend(
    shape: str,
    t: Tensor,
) -> Tensor:
    if shape == "rises":
        return 2 * t - 1

    if shape == "falls":
        return 1 - 2 * t

    if shape == "flat":
        return torch.zeros_like(t)

    if shape == "peak":
        return 1 - 4 * (t - 0.5).abs()

    if shape == "dip":
        return 4 * (t - 0.5).abs() - 1

    return torch.where(t < 0.5, -1.0, 1.0)


def make_pairs(
    combos: list[tuple[str, int, float]],
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """count signals drawn from combos, their descriptions' word ids, and each one's place in COMBOS."""
    picks = torch.randint(0, len(combos), (count,), generator=generator)
    chosen = [combos[pick] for pick in picks.tolist()]
    t = torch.linspace(0, 1, LENGTH)
    steps = torch.arange(LENGTH).float()
    trends = torch.stack([trend(shape, t) for shape, _, _ in chosen])
    periods = torch.tensor([[float(period)] for _, period, _ in chosen])
    levels = torch.tensor([[level] for _, _, level in chosen])
    phases = torch.rand(count, 1, generator=generator) * 2 * math.pi
    waves = torch.where(periods > 0, 0.6 * torch.sin(2 * math.pi * steps / periods.clamp(min=1) + phases), 0.0)
    noise = 0.1 * torch.randn(count, LENGTH, generator=generator)
    places = torch.tensor([COMBOS.index(combo) for combo in chosen])

    return trends + waves + levels + noise, description_ids(chosen), places
'''

SIGNAL_ENCODER_PART = '''
class SignalEncoder(nn.Module):
    def __init__(self, size: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LENGTH, 64), nn.ReLU(), nn.Linear(64, size))

    def forward(self, signals: Tensor) -> Tensor:
        return self.net(signals)
'''

TEXT_INIT = '''
class TextEncoder(nn.Module):
    def __init__(
        self,
        size: int = 32,
        slots: int = 7,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(len(WORDS), 16)
        self.proj = nn.Linear(16 * slots, size)
'''

TEXT_FORWARD = '''
    def forward(self, ids: Tensor) -> Tensor:
        return self.proj(self.embed(ids).flatten(1))
'''

LOSS_SOLUTION = '''
def clip_loss(
    signal_features: Tensor,
    text_features: Tensor,
    scale: Tensor,
) -> Tensor:
    scores = scale * F.normalize(signal_features, dim=1) @ F.normalize(text_features, dim=1).T
    targets = torch.arange(len(scores))

    return (F.cross_entropy(scores, targets) + F.cross_entropy(scores.T, targets)) / 2


def clip_step(
    signal_encoder: nn.Module,
    text_encoder: nn.Module,
    log_scale: Tensor,
    optimizer: torch.optim.Optimizer,
    signals: Tensor,
    ids: Tensor,
) -> float:
    scale = log_scale.exp().clamp(max=100.0)
    loss = clip_loss(signal_encoder(signals), text_encoder(ids), scale)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
'''

TRAIN_PART = '''
def train_clip(
    steps: int,
    batch: int,
    temperature: float | None,
    seed: int,
) -> tuple[nn.Module, nn.Module, float]:
    """Both encoders trained on SEEN. temperature None learns it, starting from 0.07 as CLIP did."""
    torch.manual_seed(seed)
    signal_encoder = SignalEncoder()
    text_encoder = TextEncoder()
    parameters = [*signal_encoder.parameters(), *text_encoder.parameters()]

    if temperature is None:
        log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        parameters.append(log_scale)
    else:
        log_scale = torch.tensor(math.log(1 / temperature))

    optimizer = torch.optim.Adam(parameters, lr=3e-3)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        signals, ids, _ = make_pairs(SEEN, batch, generator)
        clip_step(signal_encoder, text_encoder, log_scale, optimizer, signals, ids)

    return signal_encoder, text_encoder, log_scale.exp().clamp(max=100.0).item()
'''

NAMING_SOLUTION = '''
def describe(
    text_encoder: nn.Module,
    combos: list[tuple[str, int, float]],
) -> Tensor:
    text_encoder.eval()

    with torch.no_grad():
        return F.normalize(text_encoder(description_ids(combos)), dim=1)


def name_signals(
    signal_encoder: nn.Module,
    descriptions: Tensor,
    signals: Tensor,
) -> Tensor:
    signal_encoder.eval()

    with torch.no_grad():
        return (F.normalize(signal_encoder(signals), dim=1) @ descriptions.T).argmax(dim=1)


def naming_accuracy(
    signal_encoder: nn.Module,
    text_encoder: nn.Module,
    combos: list[tuple[str, int, float]],
    generator: torch.Generator,
) -> float:
    signals, _, places = make_pairs(combos, 1000, generator)
    names = name_signals(signal_encoder, describe(text_encoder, COMBOS), signals)

    return (names == places).float().mean().item()
'''

SPACE_SOLUTION = '''
def top_share(
    similarities: Tensor,
    scale: float,
) -> float:
    return torch.softmax(scale * similarities, dim=1).amax(dim=1).mean().item()


def similarity_by_difference(text_encoder: nn.Module) -> list[float]:
    features = describe(text_encoder, COMBOS)
    cosines = features @ features.T
    parts = torch.tensor([[SHAPE_NAMES.index(shape), period, level] for shape, period, level in COMBOS])
    differences = (parts.unsqueeze(0) != parts.unsqueeze(1)).sum(dim=2)

    return [cosines[differences == count].mean().item() for count in (1, 2, 3)]
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "itertools": itertools,
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(code, namespace)

    return namespace


SHAPE_NAMES_PART = '''
SHAPE_NAMES = list(SHAPES)
'''

REFERENCE = namespace_of(
    DATA_PART + SHAPE_NAMES_PART + SIGNAL_ENCODER_PART + TEXT_INIT + TEXT_FORWARD + LOSS_SOLUTION + TRAIN_PART + NAMING_SOLUTION + SPACE_SOLUTION,
)


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(MEASURE_THREADS)

    try:
        yield
    finally:
        torch.set_num_threads(threads)


FILE_IMPORTS = '''
    import itertools
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
'''


# ---------------------------------------------------------------------------
# clip1
# ---------------------------------------------------------------------------

CLIP1_LESSON = prose(
    """
    CLIP, from Radford and colleagues, trained on 400 million pictures with
    their captions from the internet. Its task was to predict which caption
    goes with which picture. Here the pictures become signals of 32 steps,
    and the captions short descriptions made from three parts.

    ```text
    rises then falls | repeats every 8 | high
    ```

    A signal encoder and a text encoder each map their input to 32 numbers.
    For a batch of N pairs, contrastive1's similarity matrix compares every
    signal with every description, and the right pairs sit on the diagonal.

    CLIP's loss is **symmetric**. Each row is a choice among N descriptions,
    as contrastive1's InfoNCE was, and each column a choice among N signals.
    The loss is the average of the two cross entropies.

    ```python
    scores = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
    scores.T    # tensor([[2., 1.], [0., 3.]]), the columns as rows
    ```

    CLIP learns its temperature too. It keeps `log_scale`, a single number
    that training moves, and multiplies the cosines by its exp. It starts at
    the scale of a temperature of 0.07. The paper clips it so the logits are
    never scaled by more than 100, which they found kept training stable.

    The text encoder reads a description's 7 word slots. It embeds each word
    and flattens the 7 embeddings into one row before a Linear, so the order
    of the words counts. "rises then falls" and "falls then rises" share
    every word, so an encoder that only averaged them could not tell them
    apart.

    CLIP's batches held 32,768 pairs. clip3 measures what the batch size does
    here.
    """,
)


def clip1_source(solved: bool) -> str:
    header = "# clip1.py, a solution for comparison" if solved else "# clip1.py"
    forward = indented(TEXT_FORWARD) if solved else '''
        def forward(self, ids: Tensor) -> Tensor:
            ...'''
    forward_todo = "" if solved else '''
    # TODO: Finish forward. ids has shape (batch, 7). Embed every word with
    #   self.embed, flatten each row's 7 embeddings into one row of 112
    #   numbers, and return self.proj of it.'''
    loss = indented(LOSS_SOLUTION) if solved else '''
    # TODO: Return CLIP's symmetric loss. Scale both features to length 1,
    #   multiply their similarity matrix by scale, and average the cross
    #   entropy of the rows with that of the columns. Pair i is the right
    #   answer for row i and for column i.
    def clip_loss(
        signal_features: Tensor,
        text_features: Tensor,
        scale: Tensor,
    ) -> Tensor:
        ...


    # TODO: Take one training step and return the loss as a plain number.
    #   The scale is log_scale's exp, clamped to at most 100. Score both
    #   encoders with clip_loss, zero the old gradients, backpropagate, and
    #   step the optimizer.
    def clip_step(
        signal_encoder: nn.Module,
        text_encoder: nn.Module,
        log_scale: Tensor,
        optimizer: torch.optim.Optimizer,
        signals: Tensor,
        ids: Tensor,
    ) -> float:
        ...'''

    return source_file(
        f'''
    {header}
''' + FILE_IMPORTS + '''
    flipped = torch.tensor([[2.0, 0.0], [1.0, 3.0]]).T                    # tensor([[2., 1.], [0., 3.]])
    rows = nn.Embedding(5, 4)(torch.zeros(3, 7, dtype=torch.long)).flatten(1).shape    # torch.Size([3, 28])
    capped = torch.tensor(150.0).clamp(max=100.0)                          # tensor(100.)
    start = math.exp(math.log(1 / 0.07))                                   # 14.285..., a temperature of 0.07


    # Signals of 32 steps and their descriptions, in 48 combinations.
''' + indented(DATA_PART) + '''


    # The signal encoder, an MLP.
''' + indented(SIGNAL_ENCODER_PART) + '''

''' + forward_todo + '''
''' + indented(TEXT_INIT) + '''
''' + forward + '''

''' + loss + '''


    # 300 steps of 64 pairs with a learned temperature. The loss falls from 6.28 to 1.00.
    torch.manual_seed(0)
    signal_encoder = SignalEncoder()
    text_encoder = TextEncoder()
    log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    optimizer = torch.optim.Adam([*signal_encoder.parameters(), *text_encoder.parameters(), log_scale], lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    losses = [clip_step(signal_encoder, text_encoder, log_scale, optimizer, *make_pairs(SEEN, 64, generator)[:2]) for _ in range(300)]
    first_loss = losses[0]
    last_loss = losses[-1]
    ''',
    )


def reference_run() -> tuple[float, float]:
    torch.manual_seed(0)
    signal_encoder = REFERENCE["SignalEncoder"]()
    text_encoder = REFERENCE["TextEncoder"]()
    log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    optimizer = torch.optim.Adam([*signal_encoder.parameters(), *text_encoder.parameters(), log_scale], lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    losses = [
        REFERENCE["clip_step"](signal_encoder, text_encoder, log_scale, optimizer, *REFERENCE["make_pairs"](REFERENCE["SEEN"], 64, generator)[:2])
        for _ in range(300)
    ]

    return losses[0], losses[-1]


def seeded_text_encoder(
    make: type,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)

    return make()


def check_text_forward(found: type) -> None:
    for seed, count in ((1200, 3), (1201, 5)):
        ids = REFERENCE["make_pairs"](REFERENCE["COMBOS"], count, torch.Generator().manual_seed(seed))[1]
        model = seeded_text_encoder(found, seed)
        expect_module(
            call=f"TextEncoder() seeded with {seed}",
            made=model,
        )
        features = model(ids)
        expect_tensor(
            call=f"TextEncoder() seeded with {seed}, on {count} descriptions",
            actual=features.detach() if isinstance(features, Tensor) else features,
            expected=seeded_text_encoder(REFERENCE["TextEncoder"], seed)(ids).detach(),
        )


def check_clip_loss(found: object) -> None:
    for signal_features, text_features, scale in (
        (small_tensor(1210, 3, 4).float(), small_tensor(1211, 3, 4).float(), torch.tensor(2.0)),
        (small_tensor(1212, 5, 3).float(), small_tensor(1213, 5, 3).float(), torch.tensor(14.0)),
        (small_tensor(1214, 4, 6).float(), small_tensor(1215, 4, 6).float(), torch.tensor(0.5)),
    ):
        loss = found(signal_features, text_features, scale)
        expect_close(
            call=f"clip_loss({value_text(signal_features)}, {value_text(text_features)}, {value_text(scale)})",
            actual=loss,
            expected=REFERENCE["clip_loss"](signal_features, text_features, scale),
            tolerance=1e-5,
        )


def check_clip_step(found: object) -> None:
    for seed in (1220, 1221):
        made: list[tuple[nn.Module, nn.Module, Tensor, torch.optim.Optimizer]] = []

        for _ in range(2):
            torch.manual_seed(seed)
            signal_encoder = REFERENCE["SignalEncoder"]()
            text_encoder = REFERENCE["TextEncoder"]()
            log_scale = nn.Parameter(torch.tensor(math.log(150.0)))
            optimizer = torch.optim.SGD([*signal_encoder.parameters(), *text_encoder.parameters(), log_scale], lr=0.1)
            made.append((signal_encoder, text_encoder, log_scale, optimizer))

        generator = torch.Generator().manual_seed(seed)
        batches = [REFERENCE["make_pairs"](REFERENCE["SEEN"], 8, generator)[:2] for _ in range(2)]
        call = f"two clip_steps with log_scale at log(150), SGD at 0.1, seeded with {seed}"

        for signals, ids in batches:
            loss = found(*made[0], signals, ids)
            expected = REFERENCE["clip_step"](*made[1], signals, ids)
            expect_close(
                call=f"the loss of {call}",
                actual=loss,
                expected=expected,
                tolerance=1e-5,
            )

        for (name, parameter), wanted in zip(made[0][0].named_parameters(), made[1][0].parameters()):
            if not torch.allclose(parameter, wanted, atol=1e-5):
                raise Mismatch(
                    f"After {call}, the signal encoder's {name} differs from the reference's. Zero the gradients "
                    "each step, and clamp the scale at 100 before the loss.",
                )

        if not torch.allclose(made[0][2], made[1][2], atol=1e-5):
            raise Mismatch(f"After {call}, log_scale is {made[0][2].item():.5g} where it should be {made[1][2].item():.5g}.")


def check_losses(name: str) -> Callable[[object], None]:
    def check(value: object) -> None:
        expect_close(
            call=name,
            actual=value,
            expected=dict(zip(("first_loss", "last_loss"), reference_run()))[name],
            tolerance=1e-3,
        )

    return check


def check_clip1(checker: Checker) -> None:
    checker.test_method(
        class_name="TextEncoder",
        method="forward",
        description="flattens the 7 word embeddings before the Linear",
        run=check_text_forward,
    )
    checker.test(
        function="clip_loss",
        description="averages the cross entropy of the rows and of the columns",
        run=check_clip_loss,
    )
    checker.test(
        function="clip_step",
        description="takes one step with the scale clamped at 100",
        run=check_clip_step,
    )

    for name in ("first_loss", "last_loss"):
        checker.check_name(
            name=name,
            description="comes from 300 steps of training",
            run=check_losses(name),
        )


CLIP1 = Exercise(
    name="clip1",
    chapter=SHARED_SPACE,
    title="Signals and their descriptions",
    lesson=CLIP1_LESSON,
    source=clip1_source(solved=False),
    solution=clip1_source(solved=True),
    hints={
        "TextEncoder.forward": "self.embed(ids) has shape (batch, 7, 16). flatten(1) makes it (batch, 112) for "
        "self.proj.",
        "clip_loss": "F.normalize both with dim=1, take the first times the second's transpose, and multiply by scale. "
        "The targets are torch.arange of the batch size. Take the cross entropy of the scores and of their "
        "transpose, and average the two.",
        "clip_step": "The scale is log_scale.exp() clamped with max=100.0. Then the loss, zero_grad, backward and "
        "step, and the loss as a plain number.",
    },
    introduces=(),
    check=check_clip1,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# clip2
# ---------------------------------------------------------------------------

CLIP2_LESSON = prose(
    """
    A classifier trained on the 36 seen combinations has one output per
    combination. It can never answer with a name it has no output for.

    CLIP names an input another way. Encode every candidate description,
    encode the signal, and pick the description with the largest cosine. The
    candidates are only text, so any new description can join them. CLIP
    called this zero shot transfer, and wrote its candidates as prompts like
    "A photo of a {label}."

    Training here held out six pairs of a shape and a period, such as "rises
    then falls" with "repeats every 8". Each word was seen, but never in
    those pairs. The 12 unseen combinations are `UNSEEN`.

    ```python
    torch.tensor([[0.1, 0.9, 0.3], [0.8, 0.2, 0.0]]).argmax(dim=1)    # tensor([1, 0])
    ```

    `train_clip`, given in the file, runs clip1's step. Here are both
    encoders after 1,500 steps of 64 pairs, on 3 seeds. Each score names
    1,000 new signals among all 48 descriptions, where a guess is right
    about 2% of the time.

    | Signals from | Named right |
    |---|---|
    | the 36 seen combinations | 100% on every seed |
    | the 12 unseen combinations | 91.8% to 92.4% |

    The text encoder learned each word's place in the space, so a new
    combination of seen words lands near the signals that match it. That is
    why CLIP's text side mattered. It turned a fixed list of classes into
    any list you can write.
    """,
)


def clip2_source(solved: bool) -> str:
    header = "# clip2.py, a solution for comparison" if solved else "# clip2.py"
    naming = indented(NAMING_SOLUTION) if solved else '''
    # TODO: Return text_encoder's features for each of combos' descriptions,
    #   scaled to length 1. Put text_encoder in eval mode and record nothing
    #   for autograd.
    def describe(
        text_encoder: nn.Module,
        combos: list[tuple[str, int, float]],
    ) -> Tensor:
        ...


    # TODO: Return, for each signal, the index of the description whose
    #   cosine with the signal's features is largest. descriptions are
    #   describe's rows. Put signal_encoder in eval mode and record nothing
    #   for autograd.
    def name_signals(
        signal_encoder: nn.Module,
        descriptions: Tensor,
        signals: Tensor,
    ) -> Tensor:
        ...


    # TODO: Draw 1,000 signals from combos with make_pairs and generator, name
    #   them among the descriptions of every combination in COMBOS, and return
    #   the share named right as a plain number. make_pairs' third result is
    #   each signal's place in COMBOS.
    def naming_accuracy(
        signal_encoder: nn.Module,
        text_encoder: nn.Module,
        combos: list[tuple[str, int, float]],
        generator: torch.Generator,
    ) -> float:
        ...'''

    return source_file(
        f'''
    {header}
''' + FILE_IMPORTS + '''
    nearest = torch.tensor([[0.1, 0.9, 0.3], [0.8, 0.2, 0.0]]).argmax(dim=1)    # tensor([1, 0])
    unit = F.normalize(torch.tensor([[3.0, 4.0]]), dim=1)                       # tensor([[0.6000, 0.8000]])


''' + indented(DATA_PART) + '''


''' + indented(SIGNAL_ENCODER_PART) + '''


''' + indented(TEXT_INIT) + '''
''' + indented(TEXT_FORWARD) + '''


    # clip1's loss and step, and a loop around them.
''' + indented(LOSS_SOLUTION) + '''


''' + indented(TRAIN_PART) + '''


''' + naming + '''


    # Trained on SEEN, then asked to name signals from both lists.
    signal_encoder, text_encoder, scale = train_clip(1500, 64, None, 0)
    seen_score = naming_accuracy(signal_encoder, text_encoder, SEEN, torch.Generator().manual_seed(500))
    unseen_score = naming_accuracy(signal_encoder, text_encoder, UNSEEN, torch.Generator().manual_seed(501))
    ''',
    )


def trained_pair(seed: int) -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(seed)
    signal_encoder = REFERENCE["SignalEncoder"]()
    text_encoder = REFERENCE["TextEncoder"]()

    return signal_encoder, text_encoder


def recorded_modes(model: nn.Module) -> list[tuple[bool, bool]]:
    """Every forward pass's training flag and whether autograd was recording, through a forward pre hook."""
    modes: list[tuple[bool, bool]] = []
    model.register_forward_pre_hook(lambda module, inputs: modes.append((module.training, torch.is_grad_enabled())))

    return modes


def expect_quiet(
    name: str,
    modes: list[tuple[bool, bool]],
) -> None:
    if any(training for training, _ in modes):
        raise Mismatch(f"{name} ran the encoder in training mode. Call eval() on it first.")

    if any(recording for _, recording in modes):
        raise Mismatch(f"{name} recorded for autograd. Run the encoder inside torch.no_grad().")


def check_describe(found: object) -> None:
    combos = REFERENCE["COMBOS"]

    for seed, picked in ((1230, combos[:5]), (1231, combos[10:13])):
        _, text_encoder = trained_pair(seed)
        features = found(text_encoder, picked)
        expect_tensor(
            call=f"describe(TextEncoder() seeded with {seed}, {len(picked)} combinations)",
            actual=features.detach() if isinstance(features, Tensor) else features,
            expected=REFERENCE["describe"](trained_pair(seed)[1], picked),
        )

    _, text_encoder = trained_pair(1232)
    text_encoder.train()
    modes = recorded_modes(text_encoder)
    found(text_encoder, combos[:2])
    expect_quiet(
        name="describe",
        modes=modes,
    )


def check_name_signals(found: object) -> None:
    for seed, count in ((1240, 6), (1241, 9)):
        signal_encoder, text_encoder = trained_pair(seed)
        descriptions = REFERENCE["describe"](text_encoder, REFERENCE["COMBOS"])
        signals = REFERENCE["make_pairs"](REFERENCE["COMBOS"], count, torch.Generator().manual_seed(seed))[0]
        names = found(signal_encoder, descriptions, signals)
        expect_tensor(
            call=f"name_signals(SignalEncoder() seeded with {seed}, the 48 descriptions, {count} signals)",
            actual=names,
            expected=REFERENCE["name_signals"](trained_pair(seed)[0], descriptions, signals),
        )

    signal_encoder, text_encoder = trained_pair(1242)
    signal_encoder.train()
    modes = recorded_modes(signal_encoder)
    found(signal_encoder, REFERENCE["describe"](text_encoder, REFERENCE["COMBOS"]), torch.zeros(2, 32))
    expect_quiet(
        name="name_signals",
        modes=modes,
    )


def check_naming_accuracy(found: object) -> None:
    for seed, combos, text in ((1250, "SEEN", "SEEN"), (1251, "UNSEEN", "UNSEEN")):
        signal_encoder, text_encoder = trained_pair(seed)
        expect_close(
            call=f"naming_accuracy(encoders seeded with {seed}, {text}, a generator seeded with {seed})",
            actual=found(signal_encoder, text_encoder, REFERENCE[combos], torch.Generator().manual_seed(seed)),
            expected=REFERENCE["naming_accuracy"](*trained_pair(seed), REFERENCE[combos], torch.Generator().manual_seed(seed)),
        )


def check_score(
    name: str,
    low: float,
    advice: str,
) -> Callable[[object], None]:
    def check(value: object) -> None:
        reject_missing_return(
            call=name,
            actual=value,
        )

        if isinstance(value, Tensor) or not isinstance(value, (int, float)):
            raise Mismatch(f"{name} is {value_text(value)}, but it should be a plain number.")

        if not low <= value <= 1.0:
            raise Mismatch(f"{name} is {value:.4g}, but it should be at least {low:g}. {advice}")

    return check


def check_clip2(checker: Checker) -> None:
    checker.test(
        function="describe",
        description="gives each description's features, scaled to length 1",
        run=check_describe,
    )
    checker.test(
        function="name_signals",
        description="picks the description with the largest cosine",
        run=check_name_signals,
    )
    checker.test(
        function="naming_accuracy",
        description="names 1,000 signals among all 48 descriptions",
        run=check_naming_accuracy,
    )
    checker.check_name(
        name="seen_score",
        description="names signals from the combinations training saw",
        run=check_score(
            name="seen_score",
            low=0.95,
            advice="Check describe, name_signals and naming_accuracy against their TODOs.",
        ),
    )
    checker.check_name(
        name="unseen_score",
        description="names signals from combinations training never saw",
        run=check_score(
            name="unseen_score",
            low=0.8,
            advice="Check that naming_accuracy names among every description in COMBOS.",
        ),
    )


CLIP2 = Exercise(
    name="clip2",
    chapter=SHARED_SPACE,
    title="Naming what training never named",
    lesson=CLIP2_LESSON,
    source=clip2_source(solved=False),
    solution=clip2_source(solved=True),
    hints={
        "describe": "description_ids(combos) gives the word ids. Run text_encoder on them inside torch.no_grad() and "
        "F.normalize the result along dim 1.",
        "name_signals": "Scale signal_encoder(signals) to length 1, multiply by descriptions' transpose, and take the "
        "argmax over dim 1.",
        "naming_accuracy": "make_pairs(combos, 1000, generator) gives the signals and their places. Name them among "
        "describe(text_encoder, COMBOS), then the share of names equal to the places.",
    },
    introduces=(),
    check=check_clip2,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# clip3
# ---------------------------------------------------------------------------

CLIP3_LESSON = prose(
    """
    What did the space learn? Take the cosine of every two descriptions, and
    group the pairs by how many of their three parts differ. Here are the
    averages after clip2's training, on 3 seeds.

    | Parts that differ | Average cosine |
    |---|---|
    | 1 | 0.51 |
    | 2 | 0.05 |
    | 3 | -0.41 to -0.42 |

    The space is laid out by the parts. Descriptions that share more of
    them sit closer, and a signal lands near the parts it shows.

    The temperature decides how sharp each row's softmax can get. A cosine
    is at most 1. So at a temperature of 1, the right description out of 64
    gets at most `e**2 / (e**2 + 63)`, about 0.105. That holds even when
    every rival scores -1. `top_share` measures the share it really gets.

    ```python
    torch.softmax(torch.tensor([[1.0, 0.0, 0.0]]) * 10, dim=1).amax(dim=1)    # tensor([0.9999])
    ```

    Here is clip2's training at fixed temperatures and with a learned one,
    naming the 12 unseen combinations, on 3 seeds.

    | Temperature | Unseen named right |
    |---|---|
    | 1 | 32.2%, 57.4% and 73.4% |
    | 0.3 | 91.2% to 94.6% |
    | 0.1 | 93.4% to 95.1% |
    | 0.03 | 92.0% to 95.4% |
    | learned, from 0.07 | 91.8% to 92.4% |

    A temperature of 1 caps how sure the model can be, so the loss keeps
    pulling at pairs it already has right. On a batch of 64, the trained
    model gives the right description 0.04 of the softmax at a temperature
    of 1. At its learned one, which settled at 0.046 to 0.049, it gives 0.47.

    The batch size decides how many rivals each pair meets per step. Here is
    the same training stopped after 300 steps.

    | Pairs per batch | Unseen named right after 300 steps |
    |---|---|
    | 4 | 0.7% to 16.3% |
    | 16 | 86.4% to 87.5% |
    | 64 | 88.8% to 90.4% |
    | 256 | 87.9% to 90.0% |

    With 4 pairs a signal meets only 3 rivals a step, and each step sees
    only 4 pairs. A batch of 256 holds each of the 36 seen descriptions about
    7 times, and the loss counts those repeats as rivals too. CLIP's 400
    million captions rarely repeat, and its batches held 32,768 pairs.
    """,
)


def clip3_source(solved: bool) -> str:
    header = "# clip3.py, a solution for comparison" if solved else "# clip3.py"
    space = indented(SPACE_SOLUTION) if solved else '''
    # TODO: Return the share softmax gives each row's largest entry, averaged
    #   over the rows, as a plain number. Multiply similarities by scale first.
    def top_share(
        similarities: Tensor,
        scale: float,
    ) -> float:
        ...


    # TODO: Return the average cosine between the descriptions of every two
    #   combinations in COMBOS that differ in exactly 1, 2 and 3 of their
    #   parts, as a list of three plain numbers. Take the features with
    #   describe. Two combinations differ in a part when their shapes, periods
    #   or levels are not equal.
    def similarity_by_difference(text_encoder: nn.Module) -> list[float]:
        ...'''

    return source_file(
        f'''
    {header}
''' + FILE_IMPORTS + '''
    sure = torch.softmax(torch.tensor([[1.0, 0.0, 0.0]]) * 10, dim=1).amax(dim=1)    # tensor([0.9999])
    differ = (torch.tensor([[1, 2, 3]]) != torch.tensor([[1, 5, 4]])).sum(dim=1)        # tensor([2])
    picked = torch.tensor([0.5, 0.9, 0.1])[torch.tensor([False, True, True])]           # tensor([0.9000, 0.1000])


''' + indented(DATA_PART) + '''


    SHAPE_NAMES = list(SHAPES)


''' + indented(SIGNAL_ENCODER_PART) + '''


''' + indented(TEXT_INIT) + '''
''' + indented(TEXT_FORWARD) + '''


    # clip1's loss and step, clip2's training loop and describe.
''' + indented(LOSS_SOLUTION) + '''


''' + indented(TRAIN_PART) + '''


''' + indented(NAMING_SOLUTION.split("\n\n\ndef name_signals(")[0]) + '''


''' + space + '''


    # clip2's training, then the space's layout and how sure a batch of 64 lets the model be.
    signal_encoder, text_encoder, scale = train_clip(1500, 64, None, 0)
    by_difference = similarity_by_difference(text_encoder)
    signals, ids, _ = make_pairs(SEEN, 64, torch.Generator().manual_seed(502))

    with torch.no_grad():
        batch_cosines = F.normalize(signal_encoder(signals), dim=1) @ F.normalize(text_encoder(ids), dim=1).T

    share_at_one = top_share(batch_cosines, 1.0)
    share_learned = top_share(batch_cosines, scale)
    ''',
    )


def reference_space(seed: int) -> tuple[nn.Module, nn.Module, float]:
    return REFERENCE["train_clip"](1500, 64, None, seed)


def check_top_share(found: object) -> None:
    for similarities, scale in (
        (small_tensor(1260, 4, 4).float() / 3, 1.0),
        (small_tensor(1261, 5, 5).float() / 3, 20.0),
        (small_tensor(1262, 3, 6).float() / 3, 4.0),
    ):
        expect_close(
            call=f"top_share({value_text(similarities)}, {scale})",
            actual=found(similarities, scale),
            expected=REFERENCE["top_share"](similarities, scale),
            tolerance=1e-5,
        )


def check_similarity_by_difference(found: object) -> None:
    for seed in (1270, 1271):
        _, text_encoder = trained_pair(seed)
        value = found(text_encoder)
        expected = REFERENCE["similarity_by_difference"](trained_pair(seed)[1])

        if not isinstance(value, list) or len(value) != 3:
            raise Mismatch(
                f"similarity_by_difference(TextEncoder() seeded with {seed}) is {value_text(value)}, but it should be "
                "a list of three plain numbers, for 1, 2 and 3 parts that differ.",
            )

        for count, actual, wanted in zip((1, 2, 3), value, expected):
            expect_close(
                call=f"similarity_by_difference(TextEncoder() seeded with {seed}), its average for {count} parts",
                actual=actual,
                expected=wanted,
                tolerance=1e-5,
            )


def check_by_difference(value: object) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise Mismatch(f"by_difference is {value_text(value)}, but it should be a list of three plain numbers.")

    if not value[0] > value[1] > value[2]:
        raise Mismatch(
            f"by_difference is {value_text(value)}, but descriptions that differ in fewer parts should sit closer. "
            "Check similarity_by_difference.",
        )


def check_clip3(checker: Checker) -> None:
    checker.test(
        function="top_share",
        description="gives the softmax's largest share, averaged over the rows",
        run=check_top_share,
    )
    checker.test(
        function="similarity_by_difference",
        description="averages the cosines by how many parts differ",
        run=check_similarity_by_difference,
    )
    checker.check_name(
        name="by_difference",
        description="is the trained space's cosines by parts that differ",
        run=check_by_difference,
    )
    checker.check_name(
        name="share_at_one",
        description="is how sure a temperature of 1 lets the model be",
        run=check_score_range(
            name="share_at_one",
            low=0.0,
            high=0.11,
            advice="At a temperature of 1 the right description out of 64 can get at most about 0.105.",
        ),
    )
    checker.check_name(
        name="share_learned",
        description="is how sure the learned temperature lets it be",
        run=check_score_range(
            name="share_learned",
            low=0.3,
            high=1.0,
            advice="Pass the learned scale to top_share.",
        ),
    )


def check_score_range(
    name: str,
    low: float,
    high: float,
    advice: str,
) -> Callable[[object], None]:
    def check(value: object) -> None:
        reject_missing_return(
            call=name,
            actual=value,
        )

        if isinstance(value, Tensor) or not isinstance(value, (int, float)):
            raise Mismatch(f"{name} is {value_text(value)}, but it should be a plain number.")

        if not low <= value <= high:
            raise Mismatch(f"{name} is {value:.4g}, but it should be between {low:g} and {high:g}. {advice}")

    return check


CLIP3 = Exercise(
    name="clip3",
    chapter=SHARED_SPACE,
    title="What the space learns",
    lesson=CLIP3_LESSON,
    source=clip3_source(solved=False),
    solution=clip3_source(solved=True),
    hints={
        "top_share": "torch.softmax of scale times similarities along dim 1, the largest entry of each row with "
        "amax, then the mean as a plain number.",
        "similarity_by_difference": "describe(text_encoder, COMBOS) times its transpose gives every cosine. Make a "
        "tensor of each combination's shape index, period and level. Compare every row with every row and sum the "
        "parts that differ. Then average the cosines where that count is 1, 2 or 3.",
    },
    introduces=(),
    check=check_clip3,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_clip1() -> tuple[float, float]:
    """clip1's file: the first and last loss of 300 steps of 64 pairs."""
    with pinned_threads():
        return reference_run()


def measure_naming(seeds: range = range(3)) -> dict[str, list[float]]:
    """clip2's table: both encoders trained 1,500 steps of 64 pairs on SEEN, with a learned temperature, each naming
    1,000 new signals from SEEN and from UNSEEN among all 48 descriptions, on 4 threads."""
    results: dict[str, list[float]] = {}

    with pinned_threads():
        for seed in seeds:
            signal_encoder, text_encoder, scale = reference_space(seed)

            for name in ("SEEN", "UNSEEN"):
                generator = torch.Generator().manual_seed(500 + seed)
                results.setdefault(name, []).append(REFERENCE["naming_accuracy"](signal_encoder, text_encoder, REFERENCE[name], generator))

            results.setdefault("scale", []).append(scale)
            results.setdefault("by difference", []).append(REFERENCE["similarity_by_difference"](text_encoder))

    return results


def measure_temperatures(seeds: range = range(3)) -> dict[str, list[float]]:
    """clip3's temperature table: clip2's training at fixed temperatures and a learned one, naming UNSEEN."""
    results: dict[str, list[float]] = {}

    with pinned_threads():
        for temperature in (1.0, 0.3, 0.1, 0.03, None):
            for seed in seeds:
                signal_encoder, text_encoder, scale = REFERENCE["train_clip"](1500, 64, temperature, seed)
                generator = torch.Generator().manual_seed(500 + seed)
                score = REFERENCE["naming_accuracy"](signal_encoder, text_encoder, REFERENCE["UNSEEN"], generator)
                results.setdefault(str(temperature), []).append(score)

                if temperature is None:
                    results.setdefault("learned temperature", []).append(1 / scale)

    return results


def measure_batches(seeds: range = range(3)) -> dict[int, list[float]]:
    """clip3's batch table: clip2's training stopped after 300 steps, at four batch sizes, naming UNSEEN."""
    results: dict[int, list[float]] = {}

    with pinned_threads():
        for batch in (4, 16, 64, 256):
            for seed in seeds:
                signal_encoder, text_encoder, _ = REFERENCE["train_clip"](300, batch, None, seed)
                generator = torch.Generator().manual_seed(500 + seed)
                results.setdefault(batch, []).append(REFERENCE["naming_accuracy"](signal_encoder, text_encoder, REFERENCE["UNSEEN"], generator))

    return results


def measure_shares() -> dict[str, float]:
    """clip3's file: how sure a batch of 64 lets the trained model be, at a temperature of 1 and at the learned one."""
    with pinned_threads():
        signal_encoder, text_encoder, scale = reference_space(0)
        signals, ids, _ = REFERENCE["make_pairs"](REFERENCE["SEEN"], 64, torch.Generator().manual_seed(502))

        with torch.no_grad():
            cosines = F.normalize(signal_encoder(signals), dim=1) @ F.normalize(text_encoder(ids), dim=1).T

    return {
        "at one": REFERENCE["top_share"](cosines, 1.0),
        "learned": REFERENCE["top_share"](cosines, scale),
        "bound": math.exp(2) / (math.exp(2) + 63),
    }
