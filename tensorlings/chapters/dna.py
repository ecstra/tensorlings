"""37_dna, Section 5's project, a 1D conv net that finds where genes are spliced in primate DNA, trained on the best device and read back as motifs."""

from __future__ import annotations

import functools
import textwrap
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import load_splice
from ..engine import (
    Checker,
    Described,
    DeviceRecorder,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_on_device,
    expect_tensor,
    expect_value,
    fresh_copy,
    seeded,
)
from ..text import prose, source_file
from ..training import MissingCheckpoint, load_checkpoint
from .linear_layers import expect_module

DNA = Chapter(
    folder="37_dna",
    title="Project: DNA motifs",
    readme=source_file(
        """
        # Project: DNA motifs

        Section 5's project. A gene's DNA holds stretches the cell keeps,
        exons, and stretches it cuts out, introns. You train a small 1D conv
        net to tell, from 60 bases of primate DNA, whether an exon ends there,
        an intron ends there, or neither. Then you read its kernels back and
        find the patterns biologists found by hand. The training runs outside
        the app, on the fastest device the machine has. Download the data once
        from the project folder.

        ```bash
        .venv/bin/python run.py data splice
        ```

        ## Further reading

        - [UCI, the splice junction gene sequences](https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences)
        - [Wikipedia, RNA splicing](https://en.wikipedia.org/wiki/RNA_splicing)
        - [Alipanahi and colleagues, DeepBind](https://www.nature.com/articles/nbt.3300)
        - [Jaganathan and colleagues, SpliceAI](https://www.cell.com/cell/fulltext/S0092-8674(18)31629-5)
        """,
    ),
)

BASES = "ACGT"
VALIDATION_SIZE = 400
TEST_SIZE = 600


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


# ---------------------------------------------------------------------------
# dna1
# ---------------------------------------------------------------------------

CLEAN_SOLUTION = '''
def clean(
    sequences: list[str],
    labels: Tensor,
) -> tuple[list[str], Tensor]:
    kept: list[int] = []
    seen: set[str] = set()

    for index, sequence in enumerate(sequences):
        if set(sequence) <= set(BASES) and sequence not in seen:
            kept.append(index)

        seen.add(sequence)

    return [sequences[index] for index in kept], labels[kept]
'''

ONE_HOT_SOLUTION = '''
def one_hot(sequences: list[str]) -> Tensor:
    ids = torch.tensor([[BASES.index(base) for base in sequence] for sequence in sequences])

    return F.one_hot(ids, 4).float().transpose(1, 2)
'''

FREQUENCIES_SOLUTION = '''
def base_frequencies(
    x: Tensor,
    labels: Tensor,
    label: int,
) -> Tensor:
    return x[labels == label].mean(dim=0)
'''

SPLIT_SOLUTION = '''
def split(
    x: Tensor,
    labels: Tensor,
    validation_size: int,
    test_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    order = torch.randperm(len(x), generator=generator)
    test = order[:test_size]
    validation = order[test_size : test_size + validation_size]
    train = order[test_size + validation_size :]

    return x[train], labels[train], x[validation], labels[validation], x[test], labels[test]
'''

# What dna2 and dna3 are handed, written once so the files never drift apart.
DATA_PART = '''
BASES = "ACGT"


# clean, one_hot and split from dna1.
''' + CLEAN_SOLUTION + "\n" + ONE_HOT_SOLUTION + "\n" + SPLIT_SOLUTION


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "BASES": BASES,
    }
    exec(code, namespace)

    return namespace


DATA_REFERENCE = namespace_of(CLEAN_SOLUTION + ONE_HOT_SOLUTION + FREQUENCIES_SOLUTION + SPLIT_SOLUTION)


@functools.cache
def splice_sets() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The split every part of the project uses, seed 0, with 400 windows for validation and 600 for test."""
    sequences, labels = load_splice()
    kept, y = DATA_REFERENCE["clean"](sequences, labels)

    return DATA_REFERENCE["split"](DATA_REFERENCE["one_hot"](kept), y, VALIDATION_SIZE, TEST_SIZE, torch.Generator().manual_seed(0))


@functools.cache
def clean_windows() -> tuple[Tensor, Tensor]:
    sequences, labels = load_splice()
    kept, y = DATA_REFERENCE["clean"](sequences, labels)

    return DATA_REFERENCE["one_hot"](kept), y


DNA1_LESSON = prose(
    """
    This is Section 5's project. DNA is a string of four letters, the bases A,
    C, G and T. A gene holds stretches the cell keeps, exons, and stretches it
    cuts out, introns. Here a conv net learns to find where one meets the other.

    Each window is 60 bases of primate DNA from GenBank, gathered by
    Noordewier, Towell and Shavlik in 1991. The place in question sits between
    the 30th and the 31st base, and the label says what happens there.

    - exon to intron: an exon ends and an intron begins.
    - intron to exon: an intron ends and an exon begins.
    - neither: nothing is cut there.

    Download the data once, about 160 KB, from the project folder with
    `.venv/bin/python run.py data splice`.

    Clean it first. 15 windows hold D, N, R or S, letters for a base the
    sequencing couldn't settle. 185 more repeat an earlier window exactly, one
    of them with another label. A window in both the training and the test set
    would let the score count memory as skill. So `clean` keeps each window
    once, which leaves 2,990, with 674, 670 and 1,646 per label.

    A conv net reads channels, as an image has red, green and blue. One hot
    gives DNA four, one per base, so a window becomes a tensor of shape (4, 60).
    `F.one_hot` puts the four last, and `.transpose(1, 2)` swaps dimensions 1
    and 2.

    ```python
    torch.zeros(2, 60, 4).transpose(1, 2).shape    # torch.Size([2, 4, 60])
    ```

    Look at the data before training on it, the first step of Karpathy's
    recipe. `base_frequencies` gives the share of each base at each place, over
    one label's windows.

    | Windows | Places | Bases there |
    |---|---|---|
    | exon to intron | 31st and 32nd | G in 99.9%, then T in 99.0% |
    | intron to exon | 29th and 30th | A in 99.6%, then G in 99.7% |
    | neither | 29th to 32nd | each base in 23% to 28% |

    An intron almost always begins with GT and ends with AG. Textbooks call it
    the GU-AG rule, with U in RNA where DNA has T. The windows with neither show
    no such pattern at the middle.

    `split` keeps 600 windows for the test set and 400 for validation, and
    trains on the other 1,990. The test set waits until dna3.
    """,
)

DNA1_SOURCE = source_file(
    '''
    # dna1.py
    #
    # Section 5's project starts with the data. Download it once from the project
    # folder with
    #
    #     .venv/bin/python run.py data splice

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import SPLICE_CLASSES, load_splice

    BASES = "ACGT"

    sequences, labels = load_splice()          # 3,190 strings of 60 bases, and their labels
    start = sequences[0][:12]                  # 'CCAGCTGCATCA'
    first_class = SPLICE_CLASSES[labels[0]]    # 'exon to intron'
    hot = F.one_hot(torch.tensor([0, 2]), 4)   # tensor([[1, 0, 0, 0], [0, 0, 1, 0]])


    # TODO: Return the windows of sequences, and their labels as a tensor, that
    #   hold only the letters of BASES, keeping each window the first time it
    #   comes and dropping every repeat of it after.
    def clean(
        sequences: list[str],
        labels: Tensor,
    ) -> tuple[list[str], Tensor]:
        ...


    # TODO: Return sequences one hot, as floats of shape (count, 4, length), with
    #   one channel per letter of BASES in that order.
    def one_hot(sequences: list[str]) -> Tensor:
        ...


    # TODO: Return the share of each base at each place over the windows of x
    #   with the given label, of shape (4, length).
    def base_frequencies(
        x: Tensor,
        labels: Tensor,
        label: int,
    ) -> Tensor:
        ...


    # TODO: Finish the middle lines. The first test_size shuffled positions are
    #   the test set, the next validation_size the validation set, and the rest
    #   the training set.
    def split(
        x: Tensor,
        labels: Tensor,
        validation_size: int,
        test_size: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        order = torch.randperm(len(x), generator=generator)
        test = ...
        validation = ...
        train = ...

        return x[train], labels[train], x[validation], labels[validation], x[test], labels[test]


    kept, y = clean(sequences, labels)
    x = one_hot(kept)

    # TODO: Set after_exon to the share of each base at the 31st and 32nd places,
    #   right after the middle, of the windows where an exon ends, of shape (4, 2).
    after_exon = ...

    # TODO: Set before_exon the same way, at the 29th and 30th places, right
    #   before the middle, of the windows where an intron ends.
    before_exon = ...
    ''',
)

DNA1_SOLUTION = source_file(
    '''
    # dna1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import SPLICE_CLASSES, load_splice

    BASES = "ACGT"

    sequences, labels = load_splice()
    start = sequences[0][:12]
    first_class = SPLICE_CLASSES[labels[0]]
    hot = F.one_hot(torch.tensor([0, 2]), 4)


''' + indented(CLEAN_SOLUTION) + '''


''' + indented(ONE_HOT_SOLUTION) + '''


''' + indented(FREQUENCIES_SOLUTION) + '''


''' + indented(SPLIT_SOLUTION) + '''


    kept, y = clean(sequences, labels)
    x = one_hot(kept)

    after_exon = base_frequencies(x, y, 0)[:, 30:32]

    before_exon = base_frequencies(x, y, 1)[:, 28:30]
    ''',
)


def expect_cleaned(
    call: str,
    actual: object,
    expected: tuple[list[str], Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should give the kept windows and their labels as a pair.")

    windows, labels = actual
    expect_value(
        call=f"the windows {call} keeps",
        actual=list(windows) if isinstance(windows, (list, tuple)) else windows,
        expected=expected[0],
    )
    expect_tensor(
        call=f"the labels {call} keeps",
        actual=labels,
        expected=expected[1],
    )


def expect_floats(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    if isinstance(actual, Tensor) and not actual.is_floating_point():
        raise Mismatch(f"{call} holds {actual.dtype}, but nn.Conv1d needs floats. Call .float() on the result.")

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def expect_six_tensors(
    call: str,
    actual: object,
    expected: tuple[Tensor, ...],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != len(expected):
        raise Mismatch(f"{call} should give {len(expected)} tensors in a tuple, in the order of its return line.")

    for position, (part, wanted) in enumerate(zip(actual, expected), start=1):
        expect_tensor(
            call=f"part {position} of {call}",
            actual=part,
            expected=wanted,
        )


def small_windows(
    seed: int,
    count: int,
    length: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 4, (count, length), generator=generator)

    return F.one_hot(ids, 4).float().transpose(1, 2), torch.randint(0, 3, (count,), generator=generator)


def check_named_share(
    name: str,
    label: int,
    places: slice,
) -> object:
    def run(value: object) -> None:
        x, y = clean_windows()
        expect_tensor(
            call=name,
            actual=value,
            expected=DATA_REFERENCE["base_frequencies"](x, y, label)[:, places],
        )

    return run


def check_dna1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="clean",
        description="drops windows with other letters, and every repeat",
        inputs=(
            (["ACGT", "ACNT", "ACGT", "GGTA", "TTAA", "GGTA"], torch.tensor([0, 1, 2, 1, 2, 0])),
            (["CCRC", "AAAA", "AAAA", "TTTT", "CCRC"], torch.tensor([2, 0, 1, 1, 0])),
        ),
        reference=DATA_REFERENCE["clean"],
        expect=expect_cleaned,
    )
    check_on_inputs(
        checker=checker,
        function="one_hot",
        description="gives one channel per base, as floats",
        inputs=(
            (["ACGT", "TTGA"],),
            (["GGCAT", "CATTA", "AAAAA"],),
        ),
        reference=DATA_REFERENCE["one_hot"],
        expect=expect_floats,
    )
    check_on_inputs(
        checker=checker,
        function="base_frequencies",
        description="gives the share of each base at each place for one label",
        inputs=(
            (Described(small_windows(370, 9, 6)[0], "9 windows of 6 bases"), small_windows(370, 9, 6)[1], 1),
            (Described(small_windows(371, 12, 5)[0], "12 windows of 5 bases"), small_windows(371, 12, 5)[1], 2),
        ),
        reference=DATA_REFERENCE["base_frequencies"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="split",
        description="takes test, then validation, then training from one shuffle",
        inputs=(
            (Described(small_windows(372, 10, 4)[0], "10 windows"), torch.arange(10), 2, 3, seeded(373)),
            (Described(small_windows(374, 8, 4)[0], "8 windows"), torch.arange(8) % 3, 1, 2, seeded(375)),
        ),
        reference=DATA_REFERENCE["split"],
        expect=expect_six_tensors,
    )
    checker.check_name(
        name="after_exon",
        description="shows G then T right after an exon ends",
        run=check_named_share("after_exon", 0, slice(30, 32)),
    )
    checker.check_name(
        name="before_exon",
        description="shows A then G right before an exon begins",
        run=check_named_share("before_exon", 1, slice(28, 30)),
    )


DNA1 = Exercise(
    name="dna1",
    chapter=DNA,
    title="Project: the data",
    lesson=DNA1_LESSON,
    source=DNA1_SOURCE,
    solution=DNA1_SOLUTION,
    hints={
        "clean": "Keep a set of the windows seen so far. A window stays when set(window) <= set(BASES) and it is not "
        "in that set yet. Index labels with the list of kept positions.",
        "one_hot": "Turn each letter into its place in BASES with BASES.index, then F.one_hot(ids, 4). That gives "
        "(count, length, 4), so swap the last two dimensions with .transpose and call .float().",
        "base_frequencies": "x[labels == label] keeps one label's windows. Their mean over the first dimension is "
        "the share of each base at each place.",
        "split": "test is order[:test_size], validation the next validation_size positions, train the rest.",
        "after_exon": "Exons ending is label 0. Take base_frequencies of x and y for it, then the columns 30:32.",
        "before_exon": "Introns ending is label 1. Take the columns 28:30 of its base_frequencies.",
    },
    introduces=(),
    check=check_dna1,
    time_limit=60,
)


# ---------------------------------------------------------------------------
# dna2
# ---------------------------------------------------------------------------

BLOCK_PART = '''
# resnet1's basic block in one dimension, the same number of channels in and out.
class Block1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return torch.relu(out + x)
'''

DNA_NET_SOLUTION = '''
class DnaNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, 32, 7, padding=3),
            nn.ReLU(),
        )
        self.block = Block1d(32)
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(32 * 30, 3)

    def forward(self, x: Tensor) -> Tensor:
        features = self.pool(self.block(self.stem(x)))

        return self.head(self.drop(features.flatten(1)))
'''

EVALUATE_SOLUTION = '''
def evaluate(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    device: torch.device,
    batch_size: int = 200,
) -> float:
    model.eval()
    correct = 0

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            scores = model(x[start : start + batch_size].to(device))
            correct += (scores.argmax(dim=1).cpu() == y[start : start + batch_size]).sum().item()

    model.train()

    return correct / len(x)
'''

TRAIN_EPOCH_SOLUTION = '''
def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    device: torch.device,
    generator: torch.Generator,
    batch_size: int = 64,
) -> float:
    order = torch.randperm(len(x), generator=generator)

    for start in range(0, len(x) - batch_size + 1, batch_size):
        batch = order[start : start + batch_size]
        loss = F.cross_entropy(model(x[batch].to(device)), y[batch].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
'''

EPOCHS = 32

DNA2_MAIN = '''
    # Run by .venv/bin/python run.py train dna2, never by the app.
    def main() -> None:
        sequences, labels = load_splice()
        kept, y = clean(sequences, labels)
        train_x, train_y, validation_x, validation_y, _, _ = split(one_hot(kept), y, 400, 600, torch.Generator().manual_seed(0))
        device = best_device()
        torch.manual_seed(0)
        model = DnaNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        generator = torch.Generator().manual_seed(1)
        print(f"Training on {device} for 32 epochs.")

        for epoch in range(32):
            loss = train_epoch(model, optimizer, train_x, train_y, device, generator)

            if (epoch + 1) % 4 == 0:
                accuracy = evaluate(model, validation_x, validation_y, device)
                print(f"Epoch {epoch + 1}, loss {loss:.3f}, validation accuracy {accuracy:.1%}")

        save_checkpoint(model, "dna2")
        print("Saved the model in checkpoints/dna2.pt. Save dna2.py in the app to check it.")
'''

DNA2_LESSON = prose(
    """
    `DnaNet` reads each window with 32 kernels of 7 bases, the same kernels at
    every place, as conv4's detector did with `nn.Conv1d`. One kernel that
    learns GT finds it wherever it sits. A Linear layer over all 240 numbers
    would need its own weight for every base at every place.

    Then comes resnet1's basic block in one dimension, and a max pool that
    halves the 60 places to 30. The head flattens what is left, so it knows
    where each pattern was. That matters here, since the junction always sits
    in the middle, and pooling2 showed a global pool would throw it away.
    Dropout before the head, from dropout1, keeps it from leaning on any one
    feature.

    This step trains outside the app. Once every check above the last one
    passes, train from the project folder with
    `.venv/bin/python run.py train dna2`. That runs the file's `main()`, which
    trains for 32 epochs with AdamW and saves `checkpoints/dna2.pt`. The last
    check here reads it back.

    It trains on the fastest device the machine has, which `best_device()`
    picks. That is an NVIDIA GPU through CUDA, Apple silicon's GPU through MPS,
    or the CPU when there is neither. A model and every batch it sees must sit
    on the same device, and `.to(device)` moves either one.

    ```python
    model = model.to(device)             # every weight moves
    scores = model(windows.to(device))   # the batch moves to meet it
    scores.argmax(dim=1).cpu()           # and the answer comes back
    ```

    The shuffle comes from a generator on the CPU, so each batch is picked
    there and then moved. Measuring runs in batches with the model in eval
    mode, so BatchNorm uses its running statistics and dropout switches off.
    `model.train()` switches both back afterwards.

    Here is the final validation accuracy of `main()`'s training on MPS, with
    its seed and with four others.

    | Seed | 0 | 1 | 2 | 3 | 4 |
    |---|---|---|---|---|---|
    | Validation accuracy | 95.0% | 97.5% | 97.75% | 97.5% | 97.5% |

    One run takes about 10 seconds here. The check asks for more than 93%.
    """,
)

DNA2_SOURCE = source_file(
    '''
    # dna2.py
    #
    # Train once every check above the last one passes, from the project folder.
    #
    #     .venv/bin/python run.py train dna2

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_splice
    from tensorlings.training import best_device, save_checkpoint

    windows = torch.zeros(8, 4, 60)
    moved = windows.to(best_device())    # the same numbers, on the fastest device
    halved = nn.MaxPool1d(2)(windows)    # shape (8, 4, 30)


''' + indented(DATA_PART) + '''


''' + indented(BLOCK_PART) + '''


    # TODO: Finish DnaNet. The stem is an nn.Sequential of an nn.Conv1d from 4
    #   to 32 channels with kernels of 7 and padding 3, then a ReLU. block is
    #   Block1d(32), pool an nn.MaxPool1d(2), drop an nn.Dropout(0.3), and the
    #   head a Linear from 32 * 30 to 3. forward runs the stem, the block and
    #   the pool, flattens each window's features into one row, then runs drop
    #   and the head.
    class DnaNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = ...
            self.block = ...
            self.pool = ...
            self.drop = ...
            self.head = ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return the share of x that model labels right, as a plain number.
    #   Go through x in batches of batch_size, each moved to device, in eval
    #   mode and without recording for autograd. Then put model back in
    #   training mode.
    def evaluate(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        device: torch.device,
        batch_size: int = 200,
    ) -> float:
        ...


    # TODO: Finish the loop. For each batch, move its windows and its labels to
    #   device, take the cross entropy of the model's scores, and take a step of
    #   optimizer. It returns the last batch's loss.
    def train_epoch(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        x: Tensor,
        y: Tensor,
        device: torch.device,
        generator: torch.Generator,
        batch_size: int = 64,
    ) -> float:
        order = torch.randperm(len(x), generator=generator)

        for start in range(0, len(x) - batch_size + 1, batch_size):
            batch = order[start : start + batch_size]
            ...

        return loss.item()

''' + DNA2_MAIN,
)

DNA2_SOLUTION = source_file(
    '''
    # dna2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_splice
    from tensorlings.training import best_device, save_checkpoint

    windows = torch.zeros(8, 4, 60)
    moved = windows.to(best_device())
    halved = nn.MaxPool1d(2)(windows)


''' + indented(DATA_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(DNA_NET_SOLUTION) + '''


''' + indented(EVALUATE_SOLUTION) + '''


''' + indented(TRAIN_EPOCH_SOLUTION) + '''

''' + DNA2_MAIN,
)

REFERENCE = namespace_of(BLOCK_PART + DNA_NET_SOLUTION + EVALUATE_SOLUTION + TRAIN_EPOCH_SOLUTION)


def seeded_net(make: type) -> nn.Module:
    torch.manual_seed(0)

    return make()


def tiny_net() -> nn.Module:
    """A net with BatchNorm, so evaluate's eval mode shows, and small enough to train in a check."""
    torch.manual_seed(375)

    return nn.Sequential(
        nn.Conv1d(4, 4, 3, padding=1),
        nn.BatchNorm1d(4),
        nn.ReLU(),
        nn.AdaptiveMaxPool1d(1),
        nn.Flatten(),
        nn.Linear(4, 3),
    )


def check_dna_net(found: type) -> None:
    model = expect_module("DnaNet()", seeded_net(found))
    reference = seeded_net(REFERENCE["DnaNet"])
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    wanted = {name: tuple(value.shape) for name, value in reference.state_dict().items()}

    if shapes != wanted:
        missing = sorted(set(wanted) - set(shapes))[:3]
        raise Mismatch(
            "DnaNet() should hold the layers the TODO lists under the names stem, block, pool, drop and head. "
            f"It is missing {', '.join(missing) or 'nothing'}, or some of its layers have other shapes.",
        )

    x, _ = small_windows(376, 3, 60)
    model.eval()
    reference.eval()

    with torch.no_grad():
        expected = reference(x)

    expect_tensor(
        call="DnaNet() on 3 windows of 60 bases",
        actual=model(x),
        expected=expected,
    )

    # In training mode dropout draws a mask, and the same seed draws the same one only where the TODO puts drop.
    model.train()
    reference.train()
    torch.manual_seed(377)
    expected = reference(x).detach()
    torch.manual_seed(377)
    expect_tensor(
        call="DnaNet() in training mode, with dropout drawn from one seed",
        actual=model(x).detach(),
        expected=expected,
    )


def check_evaluate(found: object) -> None:
    model = tiny_net()
    x, y = small_windows(377, 60, 12)
    expected = REFERENCE["evaluate"](tiny_net(), x, y, torch.device("cpu"), 25)
    running_mean = model[1].running_mean.clone()
    accuracy = found(model, x, y, torch.device("cpu"), 25)

    if not torch.equal(model[1].running_mean, running_mean):
        raise Mismatch("evaluate changed BatchNorm's running mean, so the model was still in training mode while it measured.")

    if not model.training:
        raise Mismatch("evaluate left the model in eval mode. Call model.train() before returning.")

    expect_close(
        call="evaluate on 60 windows in batches of 25",
        actual=accuracy,
        expected=expected,
    )

    recorder = DeviceRecorder(
        names=("the windows",),
        output=lambda windows: windows.flatten(1)[:, :3],
    )
    expect_on_device(
        function="evaluate",
        run=lambda: found(recorder, x, y, torch.device("meta"), 25),
        recorders=[recorder],
        advice="Move each batch of windows to device. Then compare on one device, the predictions brought back "
        "with .cpu() or the labels moved to device too.",
    )


def reference_train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    generator: torch.Generator,
    batch_size: int,
) -> float:
    return REFERENCE["train_epoch"](model, optimizer, x, y, torch.device("cpu"), generator, batch_size)


def check_train_epoch(found: object) -> None:
    x, y = small_windows(378, 50, 12)
    runs = []

    for train in (found, None):
        model = tiny_net()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        generator = torch.Generator().manual_seed(379)

        if train is None:
            loss = reference_train_epoch(model, optimizer, x, y, generator, 16)
        else:
            loss = train(model, optimizer, x, y, torch.device("cpu"), generator, 16)

        runs.append((loss, model[-1].weight.detach().clone()))

    (loss, weight), (expected_loss, expected_weight) = runs
    expect_close(
        call="train_epoch on 50 windows in batches of 16",
        actual=loss,
        expected=expected_loss,
        tolerance=1e-5,
    )
    expect_tensor(
        call="the head's weights after train_epoch",
        actual=weight,
        expected=expected_weight,
    )

    recorder = DeviceRecorder(
        names=("the windows",),
        output=lambda windows: windows.flatten(1)[:, :3],
    )
    optimizer = torch.optim.SGD(recorder.parameters(), lr=0.1, momentum=0.9)
    expect_on_device(
        function="train_epoch",
        run=lambda: found(recorder, optimizer, x, y, torch.device("meta"), torch.Generator().manual_seed(379), 16),
        recorders=[recorder],
        advice="Move each batch's windows and its labels to device before the model and the loss.",
    )


TRAINED_BAR = 0.93


def trained_net(make: type) -> nn.Module:
    model = seeded_net(make)

    try:
        load_checkpoint(model, "dna2")
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None
    except RuntimeError:
        raise Mismatch(
            "The saved model doesn't fit DnaNet as it is now. Train it again from the project folder.\n\n"
            "    .venv/bin/python run.py train dna2",
        ) from None

    return model


def check_trained_model(found: type) -> None:
    model = trained_net(found)
    _, _, validation_x, validation_y, _, _ = splice_sets()
    accuracy = REFERENCE["evaluate"](model, fresh_copy(validation_x), fresh_copy(validation_y), torch.device("cpu"))

    if accuracy < TRAINED_BAR:
        raise Mismatch(f"The trained model labels {accuracy:.1%} of the validation set right, but it should pass {TRAINED_BAR:.0%}.")


def check_dna2(checker: Checker) -> None:
    checker.test_method(
        class_name="DnaNet",
        method="forward",
        description="builds the stem, the block, the pool and the head",
        run=check_dna_net,
    )
    checker.test(
        function="evaluate",
        description="measures in eval mode, in batches, then switches back",
        run=check_evaluate,
    )
    checker.test(
        function="train_epoch",
        description="moves each batch to the device and steps the optimizer",
        run=check_train_epoch,
    )
    checker.test_method(
        class_name="DnaNet",
        method="forward",
        description=f"trained, passes {TRAINED_BAR:.0%} on the validation set",
        run=check_trained_model,
    )


DNA2 = Exercise(
    name="dna2",
    chapter=DNA,
    title="Project: training on the GPU",
    lesson=DNA2_LESSON,
    source=DNA2_SOURCE,
    solution=DNA2_SOLUTION,
    hints={
        "DnaNet.forward": "Run self.stem, self.block and self.pool. Then .flatten(1) gives one row per window, and "
        "self.drop and self.head finish it.",
        "evaluate": "Call model.eval(), and inside torch.no_grad() loop over range(0, len(x), batch_size). Compare "
        "each batch's argmax, moved back with .cpu(), to its labels, and count the matches.",
        "train_epoch": "The loss is F.cross_entropy of model(x[batch].to(device)) and y[batch].to(device). Then zero "
        "the gradients, call backward and optimizer.step().",
    },
    introduces=(),
    check=check_dna2,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# dna3
# ---------------------------------------------------------------------------

PREDICTIONS_SOLUTION = '''
def predictions(
    model: nn.Module,
    x: Tensor,
    batch_size: int = 200,
) -> Tensor:
    model.eval()

    with torch.no_grad():
        return torch.cat([model(x[start : start + batch_size]).argmax(dim=1) for start in range(0, len(x), batch_size)])
'''

MOTIF_SOLUTION = '''
def kernel_motif(kernel: Tensor) -> str:
    return "".join(BASES[index] for index in kernel.argmax(dim=0).tolist())


def kernel_for(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    label: int,
) -> int:
    with torch.no_grad():
        strongest = model.stem(x).amax(dim=2)

    gap = strongest[y == label].mean(dim=0) - strongest[y != label].mean(dim=0)

    return gap.argmax().item()
'''

DNA3_LESSON = prose(
    """
    The test set gets used once, at the very end. Every choice is made by now,
    so its number is a fair guess at how the model does on DNA it has never
    seen. The run behind this project labels 96.3% of the test set right.

    Biologists score a site with a position weight matrix, one weight for every
    base at every place. That is a single Linear layer over the one hot window.
    An MLP adds a hidden layer to it. Here are both against `DnaNet`, trained
    the same way on the first windows of the training set, over three seeds,
    measured on validation.

    | Training windows | DnaNet | Position weights | MLP |
    |---|---|---|---|
    | 300 | 91.2% to 95.5% | 88.2% to 88.5% | 88.5% to 89.5% |
    | 1,000 | 95.3% to 97.0% | 93.8% to 94.3% | 92.2% to 92.7% |
    | 1,990 | 97.0% to 97.8% | 95.0% to 95.3% | 93.5% to 93.8% |

    `DnaNet` wins at every size and every seed. With all the data it makes
    about half the mistakes of the position weights. Its stem holds 928 weights,
    32 detectors of 7 bases, each shared by all 60 places. The MLP holds 31,235 and
    trails the position weights from 1,000 windows on, since it must learn each
    pattern again at every place.

    Sharing pays off a second way. Each kernel is a pattern of 7 bases you can
    read. `kernel_motif` takes the strongest base at each place, and
    `kernel_for` finds the kernel that fires most on one label beyond the rest.
    Handed `base_frequencies`, `kernel_motif` also reads the commonest bases of
    the data itself.

    | Windows | Its kernel in the run behind this project | Commonest bases around the middle |
    |---|---|---|
    | exon to intron | AGGTGAG | AGGTAAG, the 29th to 35th places |
    | intron to exon | CTCCGTT | CCCTCCAG, the 23rd to 30th places |

    The first kernel matches the data at 6 of 7 places, the GT where an intron
    begins with its neighbours. The second found the run of C and T before an
    intron ends, which biologists call the polypyrimidine tract. Nobody told
    the net either one, and each came out of training on labels alone.

    `.argmax(dim=0)` gives the strongest base's number at each place, and
    `.amax(dim=2)` the largest response over the places, as pooling2 did.
    """,
)

DNA3_SOURCE = source_file(
    '''
    # dna3.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_splice
    from tensorlings.training import load_checkpoint

    kernel = torch.tensor([[0.9, 0.1], [0.0, 0.2], [0.3, 0.1], [0.1, 0.8]])
    places = kernel.argmax(dim=0)              # tensor([0, 3]), the strongest base at each place
    strongest = torch.tensor([[1.0, 4.0, 2.0], [3.0, 0.5, 2.5]]).amax(dim=1)    # tensor([4., 3.])


''' + indented(DATA_PART) + '''


''' + indented(BLOCK_PART) + '''


    # DnaNet from dna2.
''' + indented(DNA_NET_SOLUTION) + '''


    # TODO: Return model's predicted label for every window of x, going through
    #   x in batches of batch_size, in eval mode and without recording for
    #   autograd.
    def predictions(
        model: nn.Module,
        x: Tensor,
        batch_size: int = 200,
    ) -> Tensor:
        ...


    # TODO: Return the strongest base at each place of kernel, of shape
    #   (4, places), as one string, such as "AT" for the kernel above.
    def kernel_motif(kernel: Tensor) -> str:
        ...


    # TODO: Return the number of the stem's kernel that fires most on the
    #   windows of x with label, beyond how it fires on the other windows, as a
    #   plain int. Run model.stem on x without recording for autograd, take
    #   each kernel's largest response over the places of each window, and
    #   compare its mean over the windows with label to its mean over the rest.
    def kernel_for(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        label: int,
    ) -> int:
        ...


    sequences, labels = load_splice()
    kept, y = clean(sequences, labels)
    _, _, validation_x, validation_y, test_x, test_y = split(one_hot(kept), y, 400, 600, torch.Generator().manual_seed(0))
    model = load_checkpoint(DnaNet(), "dna2")
    model.eval()

    # TODO: Set test_accuracy to the share of the test set that model labels
    #   right, as a plain number, from predictions.
    test_accuracy = ...

    # TODO: Set exon_end_motif to the motif of the stem kernel that fires most
    #   where an exon ends, label 0, found on the validation set, and
    #   exon_start_motif the same for label 1. model.stem[0].weight holds the
    #   kernels, of shape (32, 4, 7).
    exon_end_motif = ...
    exon_start_motif = ...
    ''',
)

DNA3_SOLUTION = source_file(
    '''
    # dna3.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_splice
    from tensorlings.training import load_checkpoint

    kernel = torch.tensor([[0.9, 0.1], [0.0, 0.2], [0.3, 0.1], [0.1, 0.8]])
    places = kernel.argmax(dim=0)
    strongest = torch.tensor([[1.0, 4.0, 2.0], [3.0, 0.5, 2.5]]).amax(dim=1)


''' + indented(DATA_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(DNA_NET_SOLUTION) + '''


''' + indented(PREDICTIONS_SOLUTION) + '''


''' + indented(MOTIF_SOLUTION) + '''


    sequences, labels = load_splice()
    kept, y = clean(sequences, labels)
    _, _, validation_x, validation_y, test_x, test_y = split(one_hot(kept), y, 400, 600, torch.Generator().manual_seed(0))
    model = load_checkpoint(DnaNet(), "dna2")
    model.eval()

    test_accuracy = (predictions(model, test_x) == test_y).float().mean().item()

    exon_end_motif = kernel_motif(model.stem[0].weight[kernel_for(model, validation_x, validation_y, 0)])
    exon_start_motif = kernel_motif(model.stem[0].weight[kernel_for(model, validation_x, validation_y, 1)])
    ''',
)

MOTIF_REFERENCE = namespace_of(PREDICTIONS_SOLUTION + MOTIF_SOLUTION)


def checkpoint_net() -> nn.Module:
    model = trained_net(REFERENCE["DnaNet"])
    model.eval()

    return model


class StemOnly(nn.Module):
    """A stand in with a stem and nothing else, small enough that kernel_for's answer is easy to follow."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.stem = nn.Sequential(
            nn.Conv1d(4, 5, 3, padding=1),
            nn.ReLU(),
        )


def expect_int(
    call: str,
    actual: object,
    expected: int,
) -> None:
    if isinstance(actual, Tensor):
        raise Mismatch(f"{call} is {actual}, a tensor, but it should be the plain int {expected}. Use .item().")

    expect_value(
        call=call,
        actual=actual,
        expected=expected,
    )


def check_test_accuracy(value: object) -> None:
    model = checkpoint_net()
    _, _, _, _, test_x, test_y = splice_sets()
    expected = (MOTIF_REFERENCE["predictions"](model, test_x) == test_y).float().mean().item()
    expect_close(
        call="test_accuracy",
        actual=value,
        expected=expected,
        tolerance=1e-6,
    )


def check_motif_name(
    name: str,
    label: int,
) -> object:
    def run(value: object) -> None:
        model = checkpoint_net()
        _, _, validation_x, validation_y, _, _ = splice_sets()
        number = MOTIF_REFERENCE["kernel_for"](model, validation_x, validation_y, label)
        expect_value(
            call=name,
            actual=value,
            expected=MOTIF_REFERENCE["kernel_motif"](model.stem[0].weight[number]),
        )

    return run


def check_dna3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="predictions",
        description="gives each window's predicted label, in batches",
        inputs=(
            (Described(tiny_net(), "a small net with BatchNorm"), Described(small_windows(380, 23, 12)[0], "23 windows"), 10),
        ),
        reference=lambda model, x, batch_size: MOTIF_REFERENCE["predictions"](tiny_net(), x, batch_size),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="kernel_motif",
        description="reads the strongest base at each place",
        inputs=(
            (torch.tensor([[0.9, 0.1, -0.2], [0.0, 0.2, 0.7], [0.3, 0.1, 0.1], [0.1, 0.8, 0.0]]),),
            (torch.tensor([[-1.0, 0.5, 0.2, 0.0, 0.1], [0.2, 0.1, 0.9, 0.3, 0.0], [0.4, 0.0, 0.1, 0.8, 0.0], [0.1, 0.3, 0.0, 0.2, 0.6]]),),
        ),
        reference=MOTIF_REFERENCE["kernel_motif"],
        expect=expect_value,
    )
    first_x, first_y = small_windows(381, 30, 8)
    second_x, second_y = small_windows(382, 40, 10)
    check_on_inputs(
        checker=checker,
        function="kernel_for",
        description="finds the kernel that fires most on one label",
        inputs=(
            (Described(StemOnly(383), "a stem of 5 kernels"), Described(first_x, "30 windows"), first_y, 0),
            (Described(StemOnly(384), "a stem of 5 kernels"), Described(second_x, "40 windows"), second_y, 2),
        ),
        reference=MOTIF_REFERENCE["kernel_for"],
        expect=expect_int,
    )
    checker.check_name(
        name="test_accuracy",
        description="is the share of the test set labelled right",
        run=check_test_accuracy,
    )
    checker.check_name(
        name="exon_end_motif",
        description="is the motif of the kernel that marks an exon's end",
        run=check_motif_name("exon_end_motif", 0),
    )
    checker.check_name(
        name="exon_start_motif",
        description="is the motif of the kernel that marks an exon's start",
        run=check_motif_name("exon_start_motif", 1),
    )


DNA3 = Exercise(
    name="dna3",
    chapter=DNA,
    title="Project: the test set and the motifs",
    lesson=DNA3_LESSON,
    source=DNA3_SOURCE,
    solution=DNA3_SOLUTION,
    hints={
        "predictions": "model.eval(), then inside torch.no_grad() take model(...).argmax(dim=1) of each batch and "
        "torch.cat them.",
        "kernel_motif": "kernel.argmax(dim=0) gives the strongest base's number at each place. Look each one up in "
        "BASES and join the letters.",
        "kernel_for": "strongest = model.stem(x).amax(dim=2) has one row per window and one column per kernel. Take "
        "the mean of its rows with y == label, take away the mean of the rest, and give the argmax as an int.",
        "test_accuracy": "Compare predictions(model, test_x) with test_y, take the mean of the matches as floats, "
        "then .item().",
        "exon_end_motif": "kernel_for(model, validation_x, validation_y, 0) gives a kernel's number. Hand "
        "model.stem[0].weight at that number to kernel_motif.",
        "exon_start_motif": "The same, with label 1.",
    },
    introduces=(),
    check=check_dna3,
    time_limit=60,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_data() -> dict[str, object]:
    """dna1's numbers: what clean drops, the class counts, and the shares around the middle."""
    sequences, labels = load_splice()
    odd = sum(not set(sequence) <= set(BASES) for sequence in sequences)
    kept, y = DATA_REFERENCE["clean"](sequences, labels)
    x = DATA_REFERENCE["one_hot"](kept)
    first_labels: dict[str, int] = {}
    conflicts = 0

    for sequence, label in zip(sequences, labels.tolist()):
        if sequence in first_labels and first_labels[sequence] != label:
            conflicts += 1

        first_labels.setdefault(sequence, label)

    return {
        "windows": len(sequences),
        "odd letters": odd,
        "repeats dropped": len(sequences) - odd - len(kept),
        "repeats with another label": conflicts,
        "kept": len(kept),
        "per label": y.bincount().tolist(),
        "after exon end": DATA_REFERENCE["base_frequencies"](x, y, 0)[:, 30:32],
        "before exon start": DATA_REFERENCE["base_frequencies"](x, y, 1)[:, 28:30],
        "neither, middle four": DATA_REFERENCE["base_frequencies"](x, y, 2)[:, 28:32],
        "commonest bases, 29th to 35th, exon to intron": MOTIF_REFERENCE["kernel_motif"](DATA_REFERENCE["base_frequencies"](x, y, 0)[:, 28:35]),
        "commonest bases, 23rd to 30th, intron to exon": MOTIF_REFERENCE["kernel_motif"](DATA_REFERENCE["base_frequencies"](x, y, 1)[:, 22:30]),
    }


def measure_checkpoint() -> dict[str, object]:
    """dna3's numbers from checkpoints/dna2.pt: test accuracy and the two motifs."""
    model = checkpoint_net()
    _, _, validation_x, validation_y, test_x, test_y = splice_sets()
    motifs = [
        MOTIF_REFERENCE["kernel_motif"](model.stem[0].weight[MOTIF_REFERENCE["kernel_for"](model, validation_x, validation_y, label)])
        for label in (0, 1)
    ]

    return {
        "test accuracy": (MOTIF_REFERENCE["predictions"](model, test_x) == test_y).float().mean().item(),
        "exon end motif": motifs[0],
        "exon start motif": motifs[1],
    }


def pwm() -> nn.Module:
    """A weight for every base at every place, the position weight matrix biologists score sites with."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(4 * 60, 3),
    )


def mlp() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(4 * 60, 128),
        nn.ReLU(),
        nn.Linear(128, 3),
    )


def fit(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    steps: int,
    seed: int,
) -> nn.Module:
    """dna2's training, AdamW at 1e-3 with a decay of 0.1 and batches of 64, for a set number of steps."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    generator = torch.Generator().manual_seed(seed + 1)
    step = 0
    batch_size = min(64, len(x))

    while step < steps:
        order = torch.randperm(len(x), generator=generator)

        for start in range(0, len(x) - batch_size + 1, batch_size):
            batch = order[start : start + batch_size]
            loss = F.cross_entropy(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step == steps:
                break

    return model


def measure_models(
    sizes: tuple[int, ...] = (300, 1000, 1990),
    seeds: range = range(3),
    steps: int = 1000,
) -> dict[tuple[str, int], list[float]]:
    """dna3's table: validation accuracy of DnaNet, the position weight matrix and an MLP, each trained on the first
    size windows of the training set, on the CPU with 4 threads."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    train_x, train_y, validation_x, validation_y, _, _ = splice_sets()
    results: dict[tuple[str, int], list[float]] = {}

    try:
        for size in sizes:
            for seed in seeds:
                for name, make in (("DnaNet", REFERENCE["DnaNet"]), ("position weights", pwm), ("MLP", mlp)):
                    torch.manual_seed(seed)
                    model = fit(make(), train_x[:size], train_y[:size], steps, seed)
                    accuracy = REFERENCE["evaluate"](model, validation_x, validation_y, torch.device("cpu"))
                    results.setdefault((name, size), []).append(accuracy)
    finally:
        torch.set_num_threads(threads)

    return results


def measure_weights() -> dict[str, int]:
    return {
        "DnaNet": sum(parameter.numel() for parameter in REFERENCE["DnaNet"]().parameters()),
        "DnaNet stem": sum(parameter.numel() for parameter in REFERENCE["DnaNet"]().stem.parameters()),
        "position weights": sum(parameter.numel() for parameter in pwm().parameters()),
        "MLP": sum(parameter.numel() for parameter in mlp().parameters()),
    }


def measure_training(
    device: str = "mps",
    seed: int = 0,
) -> dict[str, object]:
    """main's run on one device without saving, with its time, seeded as main seeds it when seed is 0."""
    train_x, train_y, validation_x, validation_y, _, _ = splice_sets()
    chosen = torch.device(device)
    torch.manual_seed(seed)
    model = REFERENCE["DnaNet"]().to(chosen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    generator = torch.Generator().manual_seed(seed + 1)
    accuracies = {}
    started = time.perf_counter()

    for epoch in range(EPOCHS):
        REFERENCE["train_epoch"](model, optimizer, train_x, train_y, chosen, generator)

        if (epoch + 1) % 4 == 0:
            accuracies[epoch + 1] = REFERENCE["evaluate"](model, validation_x, validation_y, chosen)

    return {"accuracy by epoch": accuracies, "seconds": time.perf_counter() - started}


def measure_seeds(
    device: str = "mps",
    seeds: range = range(5),
) -> dict[int, float]:
    """dna2's table, the final validation accuracy of main's training with each seed."""
    return {seed: measure_training(device, seed)["accuracy by epoch"][EPOCHS] for seed in seeds}


def measure_motifs(seeds: range = range(3)) -> list[tuple[str, str, float]]:
    """dna3's motifs, from DnaNet trained as main trains it on the CPU with each seed, with its validation accuracy."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    train_x, train_y, validation_x, validation_y, _, _ = splice_sets()
    found = []

    try:
        for seed in seeds:
            torch.manual_seed(seed)
            model = REFERENCE["DnaNet"]()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
            generator = torch.Generator().manual_seed(seed + 1)

            for _ in range(EPOCHS):
                REFERENCE["train_epoch"](model, optimizer, train_x, train_y, torch.device("cpu"), generator)

            model.eval()
            motifs = [
                MOTIF_REFERENCE["kernel_motif"](model.stem[0].weight[MOTIF_REFERENCE["kernel_for"](model, validation_x, validation_y, label)])
                for label in (0, 1)
            ]
            accuracy = REFERENCE["evaluate"](model, validation_x, validation_y, torch.device("cpu"))
            found.append((motifs[0], motifs[1], accuracy))
    finally:
        torch.set_num_threads(threads)

    return found
