"""64_paper_topics, Section 8's project, DeepWalk embeddings of the Cora citation graph learned with no topics, then a linear probe on them with 140 labelled papers."""

from __future__ import annotations

import textwrap
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import CORA_TOPICS, load_cora
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
    reject_missing_return,
    seeded,
    value_text,
)
from ..text import prose, source_file
from ..training import MissingCheckpoint, best_device, load_checkpoint
from .word2vec import SKIPGRAM_INIT

PAPER_TOPICS = Chapter(
    folder="64_paper_topics",
    title="Project: a map of papers",
    readme=source_file(
        """
        # Project: a map of papers

        Section 8's project. Cora is a graph of 2,708 machine learning papers
        joined by their citations. You learn a vector for every paper from
        random walks along the citations, with word2vec's skip gram and no
        topic in sight. Then a linear probe finds each paper's topic from 140
        labelled papers, and you measure what the citations bought against the
        papers' own words. The training runs outside the app, in about 10
        seconds.

        ## Further reading

        - [Perozzi and colleagues, DeepWalk](https://arxiv.org/abs/1403.6652)
        - [Grover and Leskovec, node2vec](https://arxiv.org/abs/1607.00653)
        - [Yang and colleagues, Planetoid and its split of Cora](https://arxiv.org/abs/1603.08861)
        - [Kipf and Welling, graph convolutional networks](https://arxiv.org/abs/1609.02907)
        - [Sen and colleagues, collective classification in network data](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157)
        """,
    ),
)

# Each paper's vector, the walks behind it and its training.
PAPERS = 2708
TOPICS = 7
SIZE = 64
WALKS = 10
LENGTH = 40
WINDOW = 10
NEGATIVES = 5
STEPS = 2000
BATCH = 4096

SETTINGS_PART = '''
PAPERS = 2708
TOPICS = 7
SIZE = 64
WALKS = 10
LENGTH = 40
WINDOW = 10
NEGATIVES = 5
'''

# word2vec2's noise, loss and model, the parts DeepWalk takes over whole.
SKIPGRAM_PART = '''
def noise_distribution(counts: Tensor) -> Tensor:
    weights = counts.float() ** 0.75

    return weights / weights.sum()


def sgns_loss(
    centers: Tensor,
    contexts: Tensor,
    negatives: Tensor,
) -> Tensor:
    real = F.logsigmoid((centers * contexts).sum(dim=-1))
    made_up = F.logsigmoid(-(negatives @ centers.unsqueeze(2)).squeeze(2)).sum(dim=-1)

    return -(real + made_up).mean()


''' + SKIPGRAM_INIT.strip("\n") + '''

    def forward(
        self,
        centers: Tensor,
        contexts: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        return sgns_loss(self.centers(centers), self.contexts(contexts), self.contexts(negatives))
'''

WALK_SOLUTION = '''
def neighbour_lists(
    links: Tensor,
    count: int,
) -> list[list[int]]:
    neighbours: list[list[int]] = [[] for _ in range(count)]

    for a, b in links.tolist():
        neighbours[a].append(b)
        neighbours[b].append(a)

    return neighbours


def random_walk(
    neighbours: list[list[int]],
    start: int,
    length: int,
    generator: torch.Generator,
) -> list[int]:
    walk = [start]

    while len(walk) < length:
        options = neighbours[walk[-1]]
        walk.append(options[torch.randint(len(options), (1,), generator=generator).item()])

    return walk


def walk_pairs(
    walks: Tensor,
    window: int,
) -> Tensor:
    pairs = []

    for gap in range(1, window + 1):
        before, after = walks[:, :-gap].flatten(), walks[:, gap:].flatten()
        pairs.append(torch.stack([before, after], dim=1))
        pairs.append(torch.stack([after, before], dim=1))

    return torch.cat(pairs)
'''

WALKS_GIVEN = '''
# walks_per_paper rounds, each a walk from every paper in a fresh shuffled order, as DeepWalk's paper does.
def all_walks(
    neighbours: list[list[int]],
    walks_per_paper: int,
    length: int,
    generator: torch.Generator,
) -> Tensor:
    walks = []

    for _ in range(walks_per_paper):
        for start in torch.randperm(len(neighbours), generator=generator).tolist():
            walks.append(random_walk(neighbours, start, length, generator))

    return torch.tensor(walks)
'''

STEP_SOLUTION = '''
def deepwalk_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    pairs: Tensor,
    noise: Tensor,
    batch: int,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    picked = pairs[torch.randint(0, len(pairs), (batch,), generator=generator)]
    negatives = torch.multinomial(noise, batch * NEGATIVES, replacement=True, generator=generator).view(batch, NEGATIVES)
    loss = model(picked[:, 0].to(device), picked[:, 1].to(device), negatives.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
'''

TOPICS1_MAIN = '''
    # Run by .venv/bin/python run.py train topics1, never by the app.
    def main() -> None:
        _, topics, links = load_cora()
        generator = torch.Generator().manual_seed(0)
        neighbours = neighbour_lists(links, PAPERS)
        walks = all_walks(neighbours, WALKS, LENGTH, generator)
        pairs = walk_pairs(walks, WINDOW)
        noise = noise_distribution(torch.bincount(walks.flatten(), minlength=PAPERS))
        device = best_device()
        torch.manual_seed(0)
        model = SkipGram(PAPERS, SIZE).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        print(f"{len(walks)} walks gave {len(pairs)} pairs. Training on {device}, with no topics.")

        for step in range(2000):
            loss = deepwalk_step(model, optimizer, pairs, noise, 4096, generator, device)

            if (step + 1) % 500 == 0:
                print(f"Step {step + 1}, loss {loss:.3f}")

        save_checkpoint(model, "topics1")
        print("Saved the paper vectors in checkpoints/topics1.pt. topics2 reads them back.")
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


TOPICS_REFERENCE = namespace_of(SETTINGS_PART + SKIPGRAM_PART + WALK_SOLUTION + WALKS_GIVEN + STEP_SOLUTION)


# ---------------------------------------------------------------------------
# topics1
# ---------------------------------------------------------------------------

TOPICS1_LESSON = prose(
    """
    This is Section 8's project. Cora is a graph of 2,708 machine learning
    papers joined by 5,278 citations. Each paper also has one of 7 topics,
    such as neural networks or theory. Here no topic is ever used. Each paper
    learns a vector from the citations alone, and topics2 checks what it knows.

    DeepWalk, from Perozzi and colleagues, turns a graph into sentences. A
    random walk starts at a paper and steps to a paper it cites or that cites
    it, picked at random, again and again. Each walk is a sentence and each
    paper a word. Then word2vec2's skip gram learns a vector for every paper.
    Papers that turn up near each other in walks end up with vectors that
    point the same way.

    A citation joins two papers either way round, so each paper's neighbours
    are the papers on the other end of its links.

    ```python
    walk = [0, 8, 0, 14]    # paper 0, a neighbour, back to 0, then another neighbour
    ```

    A pair is a paper and another one at most `window` steps away in the same
    walk, on either side. The pairs one step apart line up as two shifted
    copies of the walks.

    ```python
    walks = torch.tensor([[0, 8, 0]])
    walks[:, :-1], walks[:, 1:]    # (tensor([[0, 8]]), tensor([[8, 0]])), each paper and the next
    ```

    `deepwalk_step` is word2vec2's training step. It draws a batch of pairs,
    then 5 made up papers for each. A paper is drawn by how often it shows up
    in the walks, raised to the power 3/4.

    Once every check passes, train with `.venv/bin/python run.py train
    topics1`. It walks 10 times from every paper, each walk 40 papers long.
    With a window of 10, as DeepWalk's paper used, that gives 18.7 million
    pairs. Then it trains 2,000 steps of 4,096 pairs.

    The three runs behind this project took 9 to 10 seconds each on an Apple
    M5's GPU. Two on 4 of its CPU threads were faster, at 5.6 to 5.8 seconds.
    The loss fell from 4.16, word2vec2's 6 log 2, to about 1.0 within 500
    steps, and held there.
    """,
)

TOPICS1_SOURCE = source_file(
    '''
    # topics1.py
    #
    # Train once every check passes, from the project folder.
    #
    #     .venv/bin/python run.py train topics1

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_cora
    from tensorlings.training import best_device, save_checkpoint

    walks = torch.tensor([[0, 8, 0]])
    after = walks[:, 1:]    # tensor([[8, 0]]), the paper one step later
''' + indented(SETTINGS_PART) + '''


    # word2vec2's noise, loss and model.
''' + indented(SKIPGRAM_PART) + '''


    # TODO: Return count lists, each paper's neighbours. For each link (a, b),
    #   in the order links lists them, add b to a's list, then a to b's.
    def neighbour_lists(
        links: Tensor,
        count: int,
    ) -> list[list[int]]:
        ...


    # TODO: Return a walk of length papers as a list, starting at start. Each
    #   step picks one of the last paper's neighbours, the one at position
    #   torch.randint(len(options), (1,), generator=generator).
    def random_walk(
        neighbours: list[list[int]],
        start: int,
        length: int,
        generator: torch.Generator,
    ) -> list[int]:
        ...


    # TODO: Return every pair of a paper and another 1 to window steps from it
    #   in the same walk, before or after, as a tensor of shape (pairs, 2), the
    #   center first. walks has shape (count, length).
    def walk_pairs(
        walks: Tensor,
        window: int,
    ) -> Tensor:
        ...


''' + indented(WALKS_GIVEN) + '''


    # TODO: Take one training step and return the loss as a plain number. Draw
    #   batch rows of pairs with torch.randint and generator, then batch *
    #   NEGATIVES made up papers with torch.multinomial(noise, ...,
    #   replacement=True, generator=generator), as rows of NEGATIVES. Move the
    #   centers, the contexts and the made up papers to device, take model's
    #   loss of them, and step.
    def deepwalk_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        pairs: Tensor,
        noise: Tensor,
        batch: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> float:
        ...

''' + TOPICS1_MAIN,
)

TOPICS1_SOLUTION = source_file(
    '''
    # topics1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_cora
    from tensorlings.training import best_device, save_checkpoint

    walks = torch.tensor([[0, 8, 0]])
    after = walks[:, 1:]
''' + indented(SETTINGS_PART) + '''


''' + indented(SKIPGRAM_PART) + '''


''' + indented(WALK_SOLUTION) + '''


''' + indented(WALKS_GIVEN) + '''


''' + indented(STEP_SOLUTION) + '''

''' + TOPICS1_MAIN,
)

SMALL_LINKS = torch.tensor([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]])
CHAIN_LINKS = torch.tensor([[1, 0], [1, 2], [2, 3], [0, 3], [3, 5], [4, 5]])


def expect_pairs(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    """The same pairs in any order, since the training draws them at random anyway."""
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, Tensor):
        raise Mismatch(f"{call} is a {type(actual).__name__}, but it should be a tensor of shape (pairs, 2).")

    if actual.is_floating_point():
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but papers are whole numbers, as the walks are.")

    if actual.shape != expected.shape:
        raise Mismatch(
            f"{call} has shape {tuple(actual.shape)}, but it should have shape {tuple(expected.shape)}, one row per pair.",
        )

    def in_order(pairs: Tensor) -> Tensor:
        keys = pairs[:, 0] * 10_000 + pairs[:, 1]

        return pairs[keys.argsort(stable=True)]

    expect_tensor(
        call=f"the pairs of {call}, sorted",
        actual=in_order(actual),
        expected=in_order(expected),
    )


def check_deepwalk_step(found: object) -> None:
    pairs = torch.randint(0, 6, (40, 2), generator=torch.Generator().manual_seed(641))
    noise = TOPICS_REFERENCE["noise_distribution"](torch.arange(1, 7))
    results = []

    for step in (found, TOPICS_REFERENCE["deepwalk_step"]):
        torch.manual_seed(642)
        model = TOPICS_REFERENCE["SkipGram"](6, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        generator = torch.Generator().manual_seed(643)
        # Two steps, since gradients left from a first step only show in a second.
        step(model, optimizer, pairs, noise, 8, generator, torch.device("cpu"))
        loss = step(model, optimizer, pairs, noise, 8, generator, torch.device("cpu"))
        results.append((loss, model.centers.weight.detach().clone(), model.contexts.weight.detach().clone()))

    (loss, centers, contexts), (expected_loss, expected_centers, expected_contexts) = results
    expect_close(
        call="deepwalk_step's loss at the second step, on batches of 8 pairs",
        actual=loss,
        expected=expected_loss,
        tolerance=1e-5,
    )
    expect_tensor(
        call="the center table after two deepwalk_steps",
        actual=centers,
        expected=expected_centers,
    )
    # The context table starts at zero, so the centers get no gradient at the first step and a missing
    # zero_grad only shows in the contexts.
    expect_tensor(
        call="the context table after two deepwalk_steps",
        actual=contexts,
        expected=expected_contexts,
    )

    stand_in = DeviceRecorder(
        names=("the centers", "the contexts", "the made up papers"),
        output=lambda centers, contexts, negatives: centers.float().mean() + contexts.float().mean() + negatives.float().mean(),
    )
    optimizer = torch.optim.SGD(stand_in.parameters(), lr=0.1)
    expect_on_device(
        function="deepwalk_step",
        run=lambda: found(stand_in, optimizer, pairs, noise, 8, torch.Generator().manual_seed(643), torch.device("meta")),
        recorders=[stand_in],
        advice="Draw the pairs and the made up papers on the CPU, then move each of the three to device.",
    )


def check_topics1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="neighbour_lists",
        description="lists the papers at the other end of each paper's links",
        inputs=(
            (SMALL_LINKS, 5),
            (CHAIN_LINKS, 6),
        ),
        reference=TOPICS_REFERENCE["neighbour_lists"],
        expect=expect_value,
    )
    small = TOPICS_REFERENCE["neighbour_lists"](SMALL_LINKS, 5)
    chain = TOPICS_REFERENCE["neighbour_lists"](CHAIN_LINKS, 6)
    check_on_inputs(
        checker=checker,
        function="random_walk",
        description="steps to a random neighbour each time",
        inputs=(
            (Described(small, f"the neighbours {small}"), 4, 6, seeded(644)),
            (Described(chain, f"the neighbours {chain}"), 0, 9, seeded(645)),
        ),
        reference=TOPICS_REFERENCE["random_walk"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="walk_pairs",
        description="pairs each paper with the ones up to window steps away",
        inputs=(
            (torch.tensor([[0, 1, 2, 1]]), 1),
            (torch.tensor([[3, 1, 0, 2, 4], [4, 2, 2, 3, 0]]), 2),
            (torch.tensor([[5, 0, 1, 3, 2, 4]]), 4),
        ),
        reference=TOPICS_REFERENCE["walk_pairs"],
        expect=expect_pairs,
    )
    checker.test(
        function="deepwalk_step",
        description="trains on a batch of pairs and made up papers",
        run=check_deepwalk_step,
    )


TOPICS1 = Exercise(
    name="topics1",
    chapter=PAPER_TOPICS,
    title="Project: walks along the citations",
    lesson=TOPICS1_LESSON,
    source=TOPICS1_SOURCE,
    solution=TOPICS1_SOLUTION,
    hints={
        "neighbour_lists": "Make one empty list per paper. Loop over links.tolist(), and append each end of a link to "
        "the other end's list.",
        "random_walk": "Keep a list that starts as [start]. While it is shorter than length, look up the neighbours "
        "of its last paper and append the one torch.randint picks, read out with .item().",
        "walk_pairs": "For each gap from 1 to window, walks[:, :-gap] and walks[:, gap:] line up each paper with the "
        "one gap steps later. Flatten both, stack them both ways round into columns, and torch.cat every gap's rows.",
        "deepwalk_step": "The rows of pairs come first, then the made up papers, both drawn with generator, so the "
        "draws match. The loss is model(centers, contexts, made_up), each moved with .to(device). Then zero_grad, "
        "backward and step.",
    },
    introduces=(),
    check=check_topics1,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# topics2
# ---------------------------------------------------------------------------

SPLIT_GIVEN = '''
# Planetoid's 20 labelled papers per topic and 1,000 for test, with Kipf and Welling's 500 for validation, from one fixed shuffle.
def split_papers(topics: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    order = torch.randperm(len(topics), generator=torch.Generator().manual_seed(100))
    train = torch.cat([order[topics[order] == topic][:20] for topic in range(TOPICS)])
    rest = order[~torch.isin(order, train)]

    return train, rest[:500], rest[500:1500]
'''

PROBE_SOLUTION = '''
def unit_rows(vectors: Tensor) -> Tensor:
    return F.normalize(vectors, dim=1)


def train_probe(
    features: Tensor,
    topics: Tensor,
    rows: Tensor,
    decay: float,
    steps: int,
) -> nn.Linear:
    torch.manual_seed(0)
    probe = nn.Linear(features.shape[1], TOPICS)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01, weight_decay=decay)

    for _ in range(steps):
        loss = F.cross_entropy(probe(features[rows]), topics[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return probe


def accuracy(
    probe: nn.Linear,
    features: Tensor,
    topics: Tensor,
    rows: Tensor,
) -> float:
    with torch.no_grad():
        return (probe(features[rows]).argmax(dim=1) == topics[rows]).float().mean().item()
'''

TOPICS_REFERENCE.update(namespace_of(SETTINGS_PART + SKIPGRAM_PART + SPLIT_GIVEN + PROBE_SOLUTION))

LOAD_LINE = 'embeddings = load_checkpoint(SkipGram(PAPERS, SIZE), "topics1").centers.weight.detach()'

TOPICS2_LESSON = prose(
    """
    topics1's vectors never saw a topic. A linear probe, as masked2 described
    it, measures what they learned anyway. A single Linear layer from each
    paper's 64 numbers to the 7 topics trains on a few labelled papers. A
    deeper head could learn the topics by itself. A linear one can only use
    what the vectors already lay out in straight lines.

    The split follows Planetoid, from Yang and colleagues. Only 20 papers of
    each topic keep their topic, 140 in all, and 1,000 are for test. Kipf and
    Welling added 500 papers for validation, and so does this split.
    `split_papers` makes it, the same every run.

    word2vec3 compared word vectors by their direction, with the cosine. The
    probe reads directions too. Each paper's vector is scaled to length 1
    first, so a paper that turned up in many walks doesn't stand out by its
    length alone.

    ```python
    F.normalize(torch.tensor([[3.0, 4.0]]), dim=1)    # tensor([[0.6000, 0.8000]])
    ```

    The probe trains with Adam on all 140 papers at once, with a weight decay
    picked on validation, as decay2 picked one. topics3 picks it each time.
    Here it is 0.005.

    In the run behind this project, three trainings of topics1 scored 65.4%,
    66.6% and 69.6% on validation. The check asks for more than 62%. topics3
    measures what that is worth.
    """,
)

TOPICS2_SOURCE = source_file(
    '''
    # topics2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_cora
    from tensorlings.training import load_checkpoint

    lengths = F.normalize(torch.tensor([[3.0, 4.0]]), dim=1)    # tensor([[0.6000, 0.8000]]), a length of 1
''' + indented(SETTINGS_PART) + '''


    # word2vec2's model, to load topics1's vectors into.
''' + indented(SKIPGRAM_PART) + '''


''' + indented(SPLIT_GIVEN) + '''


    # TODO: Return vectors with each row scaled to a length of 1.
    def unit_rows(vectors: Tensor) -> Tensor:
        ...


    # TODO: Make a Linear from features' width to TOPICS right after
    #   torch.manual_seed(0), so every probe starts the same. Train it with Adam
    #   at a rate of 0.01 and weight decay decay, for steps steps of cross
    #   entropy on the rows of features and topics that rows picks, all at once.
    #   Return the trained Linear.
    def train_probe(
        features: Tensor,
        topics: Tensor,
        rows: Tensor,
        decay: float,
        steps: int,
    ) -> nn.Linear:
        ...


    # TODO: Return the share of the papers that rows picks whose topic probe
    #   gets right, as a plain number, without recording for autograd.
    def accuracy(
        probe: nn.Linear,
        features: Tensor,
        topics: Tensor,
        rows: Tensor,
    ) -> float:
        ...


    _, topics, _ = load_cora()
    train, validation, test = split_papers(topics)
    ''' + LOAD_LINE + '''
    vectors = unit_rows(embeddings)
    validation_score = accuracy(train_probe(vectors, topics, train, 0.005, 200), vectors, topics, validation)
    ''',
)

TOPICS2_SOLUTION = source_file(
    '''
    # topics2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_cora
    from tensorlings.training import load_checkpoint

    lengths = F.normalize(torch.tensor([[3.0, 4.0]]), dim=1)
''' + indented(SETTINGS_PART) + '''


''' + indented(SKIPGRAM_PART) + '''


''' + indented(SPLIT_GIVEN) + '''


''' + indented(PROBE_SOLUTION) + '''


    _, topics, _ = load_cora()
    train, validation, test = split_papers(topics)
    ''' + LOAD_LINE + '''
    vectors = unit_rows(embeddings)
    validation_score = accuracy(train_probe(vectors, topics, train, 0.005, 200), vectors, topics, validation)
    ''',
)


def trained_embeddings() -> Tensor:
    try:
        model = load_checkpoint(TOPICS_REFERENCE["SkipGram"](PAPERS, SIZE), "topics1")
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None

    return model.centers.weight.detach()


def reference_score(
    features: Tensor,
    topics: Tensor,
    rows: tuple[Tensor, Tensor],
    decay: float,
) -> float:
    train, held = rows
    probe = TOPICS_REFERENCE["train_probe"](features, topics, train, decay, 200)

    return TOPICS_REFERENCE["accuracy"](probe, features, topics, held)


def small_rows(
    seed: int,
    rows: int,
    columns: int,
) -> Tensor:
    return torch.randn(rows, columns, generator=torch.Generator().manual_seed(seed)) * 2 + 0.5


def check_train_probe(found: object) -> None:
    generator = torch.Generator().manual_seed(653)
    features = torch.randn(30, 5, generator=generator)
    topics = torch.randint(0, TOPICS, (30,), generator=generator)
    rows = torch.tensor([0, 3, 4, 7, 9, 12, 15, 18, 21, 22, 26, 29])

    for decay, steps in ((0.0, 15), (0.5, 25)):
        trained = found(features.clone(), topics.clone(), rows.clone(), decay, steps)

        if not isinstance(trained, nn.Linear):
            raise Mismatch(f"train_probe should return the trained nn.Linear, but it returns {value_text(trained)}.")

        expected = TOPICS_REFERENCE["train_probe"](features, topics, rows, decay, steps)
        expect_tensor(
            call=f"the probe's weights after train_probe with a weight decay of {decay} for {steps} steps",
            actual=trained.weight.detach(),
            expected=expected.weight.detach(),
        )


def check_accuracy_without_autograd(found: object) -> None:
    torch.manual_seed(657)
    probe = nn.Linear(4, TOPICS)
    recording: list[bool] = []
    probe.register_forward_pre_hook(lambda module, inputs: recording.append(torch.is_grad_enabled()))
    found(probe, small_rows(658, 20, 4), torch.randint(0, TOPICS, (20,), generator=torch.Generator().manual_seed(659)), torch.arange(20))

    if any(recording):
        raise Mismatch("accuracy recorded for autograd. Score inside torch.no_grad().")


def check_validation_score(value: object) -> None:
    _, topics, _ = load_cora()
    train, validation, _ = TOPICS_REFERENCE["split_papers"](topics)
    vectors = TOPICS_REFERENCE["unit_rows"](trained_embeddings())
    expected = reference_score(vectors, topics, (train, validation), 0.005)
    expect_close(
        call="validation_score",
        actual=value,
        expected=expected,
        tolerance=1e-6,
    )

    if expected <= PROBE_BAR:
        raise Mismatch(
            f"A probe on your topics1 vectors labels {expected:.1%} of the validation papers right, but it should pass "
            f"{PROBE_BAR:.0%}. Check topics1's functions, then train it again.",
        )


# The three trainings behind the lessons scored 65.4% to 69.6% on validation, on MPS and on the CPU alike.
PROBE_BAR = 0.62


def check_topics2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="unit_rows",
        description="scales every row to a length of 1",
        inputs=(
            (small_rows(650, 4, 3),),
            (small_rows(651, 6, 5),),
        ),
        reference=TOPICS_REFERENCE["unit_rows"],
        expect=expect_tensor,
    )
    checker.test(
        function="train_probe",
        description="trains a Linear with Adam on the rows picked",
        run=check_train_probe,
    )
    torch.manual_seed(657)
    probe = nn.Linear(4, TOPICS)
    torch.manual_seed(668)
    wider_probe = nn.Linear(5, TOPICS)
    check_on_inputs(
        checker=checker,
        function="accuracy",
        description="gives the share of the picked papers labelled right",
        inputs=(
            (Described(probe, f"a Linear from 4 to {TOPICS}"), small_rows(658, 20, 4), torch.randint(0, TOPICS, (20,), generator=torch.Generator().manual_seed(659)), torch.arange(0, 20, 2)),
            (Described(wider_probe, f"a Linear from 5 to {TOPICS}"), small_rows(669, 30, 5), torch.randint(0, TOPICS, (30,), generator=torch.Generator().manual_seed(670)), torch.arange(5, 30)),
        ),
        reference=TOPICS_REFERENCE["accuracy"],
        expect=expect_close,
    )
    checker.test(
        function="accuracy",
        description="scores without recording for autograd",
        run=check_accuracy_without_autograd,
    )
    checker.check_name(
        name="validation_score",
        description=f"the probe on your topics1 vectors, past {PROBE_BAR:.0%} on validation",
        run=check_validation_score,
    )


TOPICS2 = Exercise(
    name="topics2",
    chapter=PAPER_TOPICS,
    title="Project: a probe with 140 labels",
    lesson=TOPICS2_LESSON,
    source=TOPICS2_SOURCE,
    solution=TOPICS2_SOLUTION,
    hints={
        "unit_rows": "F.normalize scales rows to a length of 1. Name the dimension the rows run along.",
        "train_probe": "Seed first, then the Linear, then torch.optim.Adam with lr=0.01 and weight_decay=decay. "
        "Each step, the cross entropy of the probe on features[rows] against topics[rows], then zero_grad, "
        "backward, step.",
        "accuracy": "Inside torch.no_grad(), compare the argmax over dim=1 of the probe on features[rows] with "
        "topics[rows], average as floats, and read the number out with .item().",
        "validation_score": "It comes out of the three functions above and topics1's trained vectors. If it is low, "
        "check topics1, then train it again.",
    },
    introduces=(),
    check=check_topics2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# topics3
# ---------------------------------------------------------------------------

DECAYS = [0.0005, 0.005, 0.05, 0.5]
PAPER = 0

SCORES_SOLUTION = '''
def pick_decay(
    features: Tensor,
    topics: Tensor,
    train: Tensor,
    validation: Tensor,
    decays: list[float],
) -> float:
    scores = [accuracy(train_probe(features, topics, train, decay, 200), features, topics, validation) for decay in decays]

    return decays[scores.index(max(scores))]


def test_score(
    features: Tensor,
    topics: Tensor,
    split: tuple[Tensor, Tensor, Tensor],
    decays: list[float],
) -> float:
    train, validation, test = split
    decay = pick_decay(features, topics, train, validation, decays)

    return accuracy(train_probe(features, topics, train, decay, 200), features, topics, test)


def nearest_papers(
    vectors: Tensor,
    paper: int,
    count: int,
) -> Tensor:
    unit = unit_rows(vectors)
    similarity = unit @ unit[paper]
    similarity[paper] = float("-inf")

    return similarity.topk(count).indices


def neighbour_agreement(
    vectors: Tensor,
    topics: Tensor,
    count: int,
) -> float:
    unit = unit_rows(vectors)
    similarity = unit @ unit.T
    similarity.fill_diagonal_(float("-inf"))
    nearest = similarity.topk(count, dim=1).indices

    return (topics[nearest] == topics.unsqueeze(1)).float().mean().item()
'''

TOPICS_REFERENCE.update(namespace_of(SETTINGS_PART + PROBE_SOLUTION + SCORES_SOLUTION))

TABLE_LINES = '''
    split = split_papers(topics)
    random_vectors = torch.randn(PAPERS, SIZE, generator=torch.Generator().manual_seed(1))
    both = torch.cat([unit_rows(words), unit_rows(embeddings)], dim=1)
    scores = {
        "random": test_score(unit_rows(random_vectors), topics, split, DECAYS),
        "words": test_score(unit_rows(words), topics, split, DECAYS),
        "embeddings": test_score(unit_rows(embeddings), topics, split, DECAYS),
        "words and embeddings": test_score(both, topics, split, DECAYS),
    }
    agreement = {
        "random": neighbour_agreement(random_vectors, topics, 10),
        "words": neighbour_agreement(words, topics, 10),
        "embeddings": neighbour_agreement(embeddings, topics, 10),
    }
    nearest = [CORA_TOPICS[topic] for topic in topics[nearest_papers(embeddings, 0, 5)].tolist()]
'''

TOPICS3_LESSON = prose(
    """
    A probe score means little alone. Three comparisons give it weight, all
    with the same 140 labels.

    - Random vectors: a vector of 64 random numbers per paper. Whatever beats it is what the walks taught.
    - Words: each paper's own words, a 1 for each of the 1,433 words it uses. Here the probe reads the papers themselves.
    - Words and embeddings: both, each scaled to length 1, side by side.

    Each one gets its weight decay picked on validation from 0.0005, 0.005,
    0.05 and 0.5, as decay2 did. Then its test score is measured once.

    Here are the test scores of the run behind this project, for three
    trainings of topics1, on 1,000 papers.

    | Vectors | Test score |
    |---|---|
    | Random | 15.1% |
    | Words | 58.5% |
    | Embeddings | 66.6% to 69.6% |
    | Words and embeddings | 72.2% to 73.1% |

    The citations alone beat the papers' own words by 8 to 11 points, with no
    topic seen while learning them. Kipf and Welling report DeepWalk at 67.2%
    on Planetoid's own split. Side by side, words and embeddings beat either
    one, since each holds what the other lacks.

    The citations know the topics because 81% of Cora's links join two papers
    of the same topic. A walk mostly stays inside one topic. So papers of a
    topic share walks, and skip gram pulls their vectors together.

    A probe reads one direction per topic. Neighbours read the map itself.
    For every paper, take the 10 papers whose vectors point closest to its
    own, by cosine, and count how many share its topic.

    | Vectors | Nearest 10 sharing the topic |
    |---|---|
    | Random | 17.6% |
    | Words | 54.8% |
    | Embeddings | 74.4% to 74.7% |

    Random vectors share a topic only as often as topics repeat by chance.
    The embeddings' neighbours share it three times in four. Paper 0 is about
    neural networks, and so are all five of its nearest papers.
    """,
)

TOPICS3_SOURCE = source_file(
    '''
    # topics3.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import CORA_TOPICS, load_cora
    from tensorlings.training import load_checkpoint

    scores = torch.tensor([0.2, 0.9, 0.4])
    best = scores.topk(2).indices    # tensor([1, 2]), the places of the two largest
''' + indented(SETTINGS_PART) + '''
    DECAYS = [0.0005, 0.005, 0.05, 0.5]


    # word2vec2's model, to load topics1's vectors into.
''' + indented(SKIPGRAM_PART) + '''


    # topics2's split and probe.
''' + indented(SPLIT_GIVEN) + '''


''' + indented(PROBE_SOLUTION) + '''


    # TODO: Train a probe of 200 steps on train for each decay in decays, and
    #   return the decay whose probe scores best on validation, the first one
    #   on a tie.
    def pick_decay(
        features: Tensor,
        topics: Tensor,
        train: Tensor,
        validation: Tensor,
        decays: list[float],
    ) -> float:
        ...


    # TODO: Pick the decay on validation, train a probe of 200 steps on train
    #   with it, and return its accuracy on test. split is (train, validation,
    #   test).
    def test_score(
        features: Tensor,
        topics: Tensor,
        split: tuple[Tensor, Tensor, Tensor],
        decays: list[float],
    ) -> float:
        ...


    # TODO: Return the count papers whose vectors are closest to paper's by
    #   cosine, the closest first, leaving paper itself out.
    def nearest_papers(
        vectors: Tensor,
        paper: int,
        count: int,
    ) -> Tensor:
        ...


    # TODO: For every paper, take its count closest papers by cosine, itself
    #   left out. Return the share of all of them that have the same topic as
    #   the paper they are close to, as a plain number.
    def neighbour_agreement(
        vectors: Tensor,
        topics: Tensor,
        count: int,
    ) -> float:
        ...


    words, topics, _ = load_cora()
    ''' + LOAD_LINE + TABLE_LINES,
)

TOPICS3_SOLUTION = source_file(
    '''
    # topics3.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import CORA_TOPICS, load_cora
    from tensorlings.training import load_checkpoint

    scores = torch.tensor([0.2, 0.9, 0.4])
    best = scores.topk(2).indices
''' + indented(SETTINGS_PART) + '''
    DECAYS = [0.0005, 0.005, 0.05, 0.5]


''' + indented(SKIPGRAM_PART) + '''


''' + indented(SPLIT_GIVEN) + '''


''' + indented(PROBE_SOLUTION) + '''


''' + indented(SCORES_SOLUTION) + '''


    words, topics, _ = load_cora()
    ''' + LOAD_LINE + TABLE_LINES,
)


def decay_inputs(seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Features where the decay matters: 40 noisy columns and few rows to train on."""
    generator = torch.Generator().manual_seed(seed)
    topics = torch.randint(0, TOPICS, (120,), generator=generator)
    features = torch.randn(120, 40, generator=generator)
    features[:, :TOPICS] += 1.5 * F.one_hot(topics, TOPICS)
    order = torch.randperm(120, generator=generator)

    return features, topics, order[:21], order[21:81]


def check_pick_decay(found: object) -> None:
    for seed, decays in ((661, [0.0005, 0.05, 5.0]), (662, [5.0, 0.5, 0.005])):
        features, topics, train, validation = decay_inputs(seed)
        expected = TOPICS_REFERENCE["pick_decay"](features, topics, train, validation, decays)
        actual = found(features.clone(), topics.clone(), train.clone(), validation.clone(), list(decays))
        expect_close(
            call=f"pick_decay on 120 papers of 40 numbers, from the decays {decays}",
            actual=actual,
            expected=expected,
            tolerance=0.0,
        )


def check_test_score(found: object) -> None:
    for seed, decays in ((663, [0.0005, 0.05, 5.0]), (664, [0.5, 0.005])):
        features, topics, train, validation = decay_inputs(seed)
        test = torch.arange(120)[~torch.isin(torch.arange(120), torch.cat([train, validation]))]
        split = (train, validation, test)
        expected = TOPICS_REFERENCE["test_score"](features, topics, split, decays)
        actual = found(features.clone(), topics.clone(), tuple(part.clone() for part in split), list(decays))
        expect_close(
            call=f"test_score on 120 papers of 40 numbers, from the decays {decays}",
            actual=actual,
            expected=expected,
        )


def expect_places(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if isinstance(actual, Tensor) and actual.is_floating_point():
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but papers are whole numbers. topk's .indices gives them.")

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def reference_table(
    words: Tensor,
    topics: Tensor,
    embeddings: Tensor,
) -> dict[str, object]:
    namespace = dict(TOPICS_REFERENCE)
    namespace.update(
        {
            "words": words,
            "topics": topics,
            "embeddings": embeddings,
            "split_papers": TOPICS_REFERENCE["split_papers"],
            "DECAYS": DECAYS,
            "CORA_TOPICS": CORA_TOPICS,
        },
    )
    exec(textwrap.dedent(TABLE_LINES), namespace)

    return {name: namespace[name] for name in ("scores", "agreement", "nearest")}


def check_table_name(name: str) -> object:
    def run(value: object) -> None:
        words, topics, _ = load_cora()
        expected = reference_table(words, topics, trained_embeddings())[name]

        if not isinstance(value, type(expected)):
            raise Mismatch(f"{name} is {value_text(value)}, but it should be a {type(expected).__name__} like {value_text(expected)}.")

        if isinstance(expected, list):
            expect_value(
                call=name,
                actual=value,
                expected=expected,
            )
            return

        if set(value) != set(expected):
            raise Mismatch(f"{name} has the keys {sorted(value)}, but it should have {sorted(expected)}.")

        # Words tie often, since many papers share a word count, and a tie can break either way.
        tolerance = 0.01 if name == "agreement" else 1e-6

        for key in expected:
            expect_close(
                call=f'{name}["{key}"]',
                actual=value[key],
                expected=expected[key],
                tolerance=tolerance,
            )

    return run


def check_topics3(checker: Checker) -> None:
    checker.test(
        function="pick_decay",
        description="picks the decay that scores best on validation",
        run=check_pick_decay,
    )
    checker.test(
        function="test_score",
        description="picks the decay on validation, then scores test once",
        run=check_test_score,
    )
    check_on_inputs(
        checker=checker,
        function="nearest_papers",
        description="gives the closest papers by cosine, leaving the paper out",
        inputs=(
            (small_rows(671, 8, 3), 2, 3),
            (small_rows(672, 12, 5), 0, 4),
        ),
        reference=TOPICS_REFERENCE["nearest_papers"],
        expect=expect_places,
    )
    check_on_inputs(
        checker=checker,
        function="neighbour_agreement",
        description="gives the share of near papers that share a topic",
        inputs=(
            (small_rows(673, 12, 3), torch.randint(0, 3, (12,), generator=torch.Generator().manual_seed(674)), 3),
            (small_rows(675, 20, 4), torch.randint(0, 4, (20,), generator=torch.Generator().manual_seed(676)), 5),
        ),
        reference=TOPICS_REFERENCE["neighbour_agreement"],
        expect=expect_close,
    )

    for name, description in (
        ("scores", "the four test scores"),
        ("agreement", "how often the 10 nearest papers share a topic"),
        ("nearest", "the topics of paper 0's five nearest papers"),
    ):
        checker.check_name(
            name=name,
            description=description,
            run=check_table_name(name),
        )


TOPICS3 = Exercise(
    name="topics3",
    chapter=PAPER_TOPICS,
    title="Project: what the citations bought",
    lesson=TOPICS3_LESSON,
    source=TOPICS3_SOURCE,
    solution=TOPICS3_SOLUTION,
    hints={
        "pick_decay": "Score a probe of 200 steps for every decay with train_probe and accuracy on validation. Keep "
        "the scores in a list, and return the decay at the place of the largest. list.index finds the first one.",
        "test_score": "Unpack split into its three parts. pick_decay finds the decay, then one more probe with it "
        "is scored on test.",
        "nearest_papers": "Scale the rows to length 1, so a matrix product gives every cosine. Set paper's own "
        "score to minus infinity so it can't be picked, then take the topk's indices.",
        "neighbour_agreement": "Every cosine at once is the scaled rows times their transpose. Fill the diagonal with "
        "minus infinity, take the topk along each row, look up those papers' topics, and compare them with each "
        "paper's own topic.",
        "scores": "It comes out of test_score on the four sets of vectors the file builds.",
        "agreement": "It comes out of neighbour_agreement on the three sets of vectors the file builds.",
        "nearest": "It comes out of nearest_papers on the embeddings, with each paper's topic looked up by name.",
    },
    introduces=(),
    check=check_topics3,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def train_deepwalk(
    seed: int,
    device: torch.device,
) -> tuple[Tensor, list[float], float]:
    """topics1's main for one seed, with the losses every 500 steps and the seconds it took."""
    started = time.perf_counter()
    _, _, links = load_cora()
    generator = torch.Generator().manual_seed(seed)
    neighbours = TOPICS_REFERENCE["neighbour_lists"](links, PAPERS)
    walks = TOPICS_REFERENCE["all_walks"](neighbours, WALKS, LENGTH, generator)
    pairs = TOPICS_REFERENCE["walk_pairs"](walks, WINDOW)
    noise = TOPICS_REFERENCE["noise_distribution"](torch.bincount(walks.flatten(), minlength=PAPERS))
    torch.manual_seed(seed)
    model = TOPICS_REFERENCE["SkipGram"](PAPERS, SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    losses = []

    for step in range(STEPS):
        loss = TOPICS_REFERENCE["deepwalk_step"](model, optimizer, pairs, noise, BATCH, generator, device)

        if step == 0 or (step + 1) % 500 == 0:
            losses.append(loss)

    return model.centers.weight.detach().cpu(), losses, time.perf_counter() - started


def same_topic_links() -> float:
    """The share of Cora's links that join two papers of the same topic."""
    _, topics, links = load_cora()

    return (topics[links[:, 0]] == topics[links[:, 1]]).float().mean().item()


def measure_paper_topics(
    seeds: tuple[int, ...] = (0, 1, 2),
    device: str = "",
) -> dict[str, object]:
    """Every number of topics1 to topics3: three trainings on the best device, their validation scores, their test
    table and their neighbour agreement, with the CPU's threads pinned so the probes give the same numbers again."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    words, topics, links = load_cora()
    chosen = torch.device(device) if device else best_device()
    split = TOPICS_REFERENCE["split_papers"](topics)
    results: dict[str, object] = {
        "links": len(links),
        "same topic links": same_topic_links(),
        "pairs": WALKS * PAPERS * 2 * sum(LENGTH - gap for gap in range(1, WINDOW + 1)),
    }

    try:
        for seed in seeds:
            embeddings, losses, seconds = train_deepwalk(seed, chosen)
            vectors = TOPICS_REFERENCE["unit_rows"](embeddings)
            results[f"seed {seed} losses"] = losses
            results[f"seed {seed} seconds"] = seconds
            results[f"seed {seed} validation"] = reference_score(vectors, topics, (split[0], split[1]), 0.005)
            results[f"seed {seed} table"] = reference_table(words, topics, embeddings)
    finally:
        torch.set_num_threads(threads)

    return results
