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

        This is Section 8's project. Cora is a graph of 2,708 machine learning
        papers, linked by their citations. You learn a vector for every paper
        from random walks along the citations, using word2vec's skip gram. The
        topics of the papers are never used for this. Then a linear probe
        predicts each paper's topic after training on 140 labelled papers.
        Finally you measure how much the citations help, compared with the
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
# Makes walks_per_paper rounds of walks, as DeepWalk's paper does. Each round starts one walk from every paper,
# in a new random order.
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
    papers, linked by 5,278 citations. Each paper also has one of 7 topics,
    such as neural networks or theory. This exercise never uses the topics.
    Each paper gets a vector learned from the citations alone, and topics2
    checks what those vectors know about the topics.

    DeepWalk, from Perozzi and colleagues, turns a graph into sentences. A
    random walk starts at a paper. At each step it moves to a random paper
    that the current paper cites or is cited by. Each walk is treated as a
    sentence, and each paper as a word. Then word2vec2's skip gram learns a
    vector for every paper. Papers that appear near each other in walks end
    up with vectors that point in similar directions.

    A walk can follow a citation in either direction. So a paper's neighbours
    are all the papers at the other end of its links, whichever paper cites
    which.

    ```python
    walk = [0, 8, 0, 14]    # paper 0, a neighbour, back to 0, then another neighbour
    ```

    A pair is a paper and another paper at most `window` steps away from it
    in the same walk, before or after it. The pairs that are one step apart
    come from two shifted copies of the walks. `walks[:, :-1]` drops the last
    paper of each walk, and `walks[:, 1:]` drops the first. Lined up, they put
    each paper next to the paper that follows it.

    ```python
    walks = torch.tensor([[0, 8, 0]])
    walks[:, :-1], walks[:, 1:]    # (tensor([[0, 8]]), tensor([[8, 0]])), each paper and the next
    ```

    `deepwalk_step` is the training step from word2vec2. It draws a batch of
    pairs, then 5 made up papers for each pair. The made up papers are drawn
    with probability proportional to how often each paper appears in the
    walks, raised to the power 3/4.

    Once every check passes, train with `.venv/bin/python run.py train
    topics1`. It starts 10 walks from every paper, each 40 papers long. With
    a window of 10, the value DeepWalk's paper used, that gives 18.7 million
    pairs. Then it trains for 2,000 steps on batches of 4,096 pairs.

    The three training runs behind this project took 9 to 10 seconds each on
    an Apple M5's GPU. Two runs on 4 of its CPU threads were faster, at 5.6
    to 5.8 seconds. The loss fell from 4.16, which is `6 * log 2` as in
    word2vec2, to about 1.0 within 500 steps, and stayed there.
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
    after = walks[:, 1:]    # tensor([[8, 0]]), the walk without its first paper
''' + indented(SETTINGS_PART) + '''


    # The noise distribution, loss and model from word2vec2.
''' + indented(SKIPGRAM_PART) + '''


    # TODO: Return a list of count lists, where list i holds the neighbours of
    #   paper i. links has shape (links, 2), one citation (a, b) per row. Go
    #   through the links in order, and for each one append b to a's list, then
    #   a to b's list.
    #   Example: neighbour_lists(torch.tensor([[0, 1], [1, 2]]), 3) returns
    #   [[1], [0, 2], [1]].
    def neighbour_lists(
        links: Tensor,
        count: int,
    ) -> list[list[int]]:
        ...


    # TODO: Return a random walk as a list of length papers, starting at start.
    #   At each step, let options be the neighbours of the walk's last paper,
    #   and append the option at index
    #   torch.randint(len(options), (1,), generator=generator).
    def random_walk(
        neighbours: list[list[int]],
        start: int,
        length: int,
        generator: torch.Generator,
    ) -> list[int]:
        ...


    # TODO: Return every skip gram pair from walks, as a tensor of shape
    #   (pairs, 2) with the center paper first. walks has shape (count, length).
    #   A pair is a paper and another paper 1 to window steps before or after
    #   it in the same walk. The order of the pairs does not matter.
    #   Example: walk_pairs(torch.tensor([[0, 1, 2]]), 1) returns the pairs
    #   [0, 1], [1, 2], [1, 0] and [2, 1], in any order.
    def walk_pairs(
        walks: Tensor,
        window: int,
    ) -> Tensor:
        ...


''' + indented(WALKS_GIVEN) + '''


    # TODO: Take one training step and return the loss as a Python float.
    #   First pick batch rows of pairs at random indices from torch.randint,
    #   drawn with generator. Then draw batch * NEGATIVES made up papers with
    #   torch.multinomial(noise, ..., replacement=True, generator=generator),
    #   and reshape them to (batch, NEGATIVES). Move the center column, the
    #   context column and the made up papers to device, and pass them to model
    #   to get the loss. Then take one optimizer step.
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
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but paper ids are integers, as in the walks.")

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
        description="moves to a random neighbour at each step",
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
        "random_walk": "Start a list as [start]. While it is shorter than length, look up the neighbours of its last "
        "paper. Append the neighbour at the index torch.randint picks, converted to an int with .item().",
        "walk_pairs": "For each gap from 1 to window, walks[:, :-gap] and walks[:, gap:] line up each paper with the "
        "paper gap steps later. Flatten both and stack them as two columns, in both orders. Then torch.cat the rows "
        "from every gap.",
        "deepwalk_step": "Draw the rows of pairs first and the made up papers second, both with generator, so the "
        "random numbers match. The loss is model(centers, contexts, made_up), with each moved by .to(device). Then "
        "call zero_grad, backward and step.",
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
# Planetoid's split, 20 labelled papers per topic and 1,000 for test, plus Kipf and Welling's 500 for validation.
# One fixed shuffle makes it the same on every run.
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
    topics1's vectors were learned without any topics. A linear probe, as
    described in masked2, measures what they learned anyway. A single Linear
    layer maps each paper's 64 numbers to scores for the 7 topics, and it
    trains on a few labelled papers. A deeper network could learn the topics
    by itself. A linear layer can only find topics that the vectors already
    separate with flat boundaries, like lines or planes.

    The split of the papers follows Planetoid, from Yang and colleagues. Only
    20 papers of each topic are labelled for training, 140 in all, and 1,000
    papers are for test. Kipf and Welling added 500 papers for validation,
    and so does this split. `split_papers` makes the split, and it is the
    same on every run.

    word2vec3 compared word vectors by their direction, with cosine
    similarity. The probe uses directions too. Each paper's vector is first
    scaled to length 1. So a paper that appeared in many walks does not stand
    out because of its length alone.

    ```python
    F.normalize(torch.tensor([[3.0, 4.0]]), dim=1)    # tensor([[0.6000, 0.8000]])
    ```

    The probe trains with Adam on all 140 papers at once. Its weight decay is
    chosen by the validation score, as in decay2. topics3 writes the code that
    makes this choice. Here the decay is fixed at 0.005.

    In the runs behind this project, probes on three trainings of topics1
    scored 65.4%, 66.6% and 69.6% on validation. The check asks for more than
    62%. topics3 measures how good that score is by comparing it with other
    vectors.
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

    lengths = F.normalize(torch.tensor([[3.0, 4.0]]), dim=1)    # tensor([[0.6000, 0.8000]]), the row scaled to length 1
''' + indented(SETTINGS_PART) + '''


    # The model from word2vec2, needed to load topics1's vectors.
''' + indented(SKIPGRAM_PART) + '''


''' + indented(SPLIT_GIVEN) + '''


    # TODO: Return vectors, of shape (papers, size), with each row scaled to a
    #   length of 1.
    def unit_rows(vectors: Tensor) -> Tensor:
        ...


    # TODO: Train and return a linear probe. Call torch.manual_seed(0), then
    #   create an nn.Linear from features.shape[1] inputs to TOPICS outputs, so
    #   every probe starts from the same weights. Train it with Adam, with a
    #   learning rate of 0.01 and weight_decay=decay, for steps steps. Each step
    #   uses all the papers that rows picks at once. The loss is the cross
    #   entropy of the probe's scores for those papers against their topics.
    def train_probe(
        features: Tensor,
        topics: Tensor,
        rows: Tensor,
        decay: float,
        steps: int,
    ) -> nn.Linear:
        ...


    # TODO: Return the fraction of the papers picked by rows whose topic the
    #   probe predicts correctly, as a Python float. The prediction is the topic
    #   with the highest score. Compute it inside torch.no_grad(), so autograd
    #   records nothing.
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
        raise Mismatch("accuracy let autograd record the computation. Compute the score inside torch.no_grad().")


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
            f"A probe on your topics1 vectors predicts the topic of {expected:.1%} of the validation papers correctly. "
            f"It should get more than {PROBE_BAR:.0%}. Check topics1's functions, then train topics1 again.",
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
        description="trains a Linear with Adam on the picked rows",
        run=check_train_probe,
    )
    torch.manual_seed(657)
    probe = nn.Linear(4, TOPICS)
    torch.manual_seed(668)
    wider_probe = nn.Linear(5, TOPICS)
    check_on_inputs(
        checker=checker,
        function="accuracy",
        description="returns the fraction of the picked papers whose topic is predicted correctly",
        inputs=(
            (Described(probe, f"a Linear from 4 to {TOPICS}"), small_rows(658, 20, 4), torch.randint(0, TOPICS, (20,), generator=torch.Generator().manual_seed(659)), torch.arange(0, 20, 2)),
            (Described(wider_probe, f"a Linear from 5 to {TOPICS}"), small_rows(669, 30, 5), torch.randint(0, TOPICS, (30,), generator=torch.Generator().manual_seed(670)), torch.arange(5, 30)),
        ),
        reference=TOPICS_REFERENCE["accuracy"],
        expect=expect_close,
    )
    checker.test(
        function="accuracy",
        description="computes the score without autograd recording",
        run=check_accuracy_without_autograd,
    )
    checker.check_name(
        name="validation_score",
        description=f"the probe on your topics1 vectors scores more than {PROBE_BAR:.0%} on validation",
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
        "unit_rows": "F.normalize scales vectors to length 1 along the dimension you pass as dim. Each row's numbers "
        "lie along dimension 1.",
        "train_probe": "Seed first, then create the Linear, then torch.optim.Adam with lr=0.01 and "
        "weight_decay=decay. In each step, compute the cross entropy of the probe on features[rows] against "
        "topics[rows]. Then call zero_grad, backward and step.",
        "accuracy": "Inside torch.no_grad(), compare the argmax over dim=1 of the probe on features[rows] with "
        "topics[rows]. Average the matches as floats, and read the number out with .item().",
        "validation_score": "The three functions above compute it from topics1's trained vectors. If it is low, check "
        "topics1's functions, then train topics1 again.",
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
    A probe score means little on its own. Three comparisons show what it is
    worth, all trained on the same 140 labels.

    - Random vectors: 64 random numbers per paper. Whatever the embeddings score above this comes from the walks.
    - Words: 1,433 numbers per paper, with a 1 for each word the paper uses. Here the probe reads the papers' own text.
    - Words and embeddings: both vectors, each scaled to length 1, joined end to end.

    For each set of vectors, the weight decay is chosen from 0.0005, 0.005,
    0.05 and 0.5 by the validation score, as in decay2. Then the test score
    is measured once, with that decay.

    Here are the test scores on the 1,000 test papers, from the runs behind
    this project. The embeddings come from three trainings of topics1.

    | Vectors | Test score |
    |---|---|
    | Random | 15.1% |
    | Words | 58.5% |
    | Embeddings | 66.6% to 69.6% |
    | Words and embeddings | 72.2% to 73.1% |

    The embeddings, learned from the citations alone with no topics, beat the
    papers' own words by 8 to 11 points. Kipf and Welling report 67.2% for
    DeepWalk on Planetoid's own split. Words and embeddings together beat
    either one alone, because each holds information the other lacks.

    The citations carry information about topics because 81% of Cora's links
    join two papers with the same topic. So a walk mostly stays inside one
    topic. Papers of the same topic then appear in the same walks, and skip
    gram pulls their vectors together.

    A linear probe looks for one direction per topic. Another test looks at
    which papers are close to each other. For every paper, take the 10 papers
    whose vectors have the highest cosine similarity to its own. Then count
    how many of them share its topic.

    | Vectors | Nearest 10 sharing the topic |
    |---|---|
    | Random | 17.6% |
    | Words | 54.8% |
    | Embeddings | 74.4% to 74.7% |

    The neighbours from random vectors share a topic only as often as two
    random papers do by chance. The neighbours from the embeddings share it
    about three times in four. Paper 0 is about neural networks, and so are
    all five of its nearest papers.
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
    best = scores.topk(2).indices    # tensor([1, 2]), the indices of the two largest
''' + indented(SETTINGS_PART) + '''
    DECAYS = [0.0005, 0.005, 0.05, 0.5]


    # The model from word2vec2, needed to load topics1's vectors.
''' + indented(SKIPGRAM_PART) + '''


    # The split and the probe from topics2.
''' + indented(SPLIT_GIVEN) + '''


''' + indented(PROBE_SOLUTION) + '''


    # TODO: For each decay in decays, train a probe for 200 steps on the train
    #   papers and measure its accuracy on the validation papers. Return the
    #   decay with the highest accuracy. If several tie, return the first one.
    def pick_decay(
        features: Tensor,
        topics: Tensor,
        train: Tensor,
        validation: Tensor,
        decays: list[float],
    ) -> float:
        ...


    # TODO: Return a probe's accuracy on the test papers. split is a tuple
    #   (train, validation, test). Pick the decay with pick_decay, train a probe
    #   for 200 steps on train with that decay, and return its accuracy on test.
    def test_score(
        features: Tensor,
        topics: Tensor,
        split: tuple[Tensor, Tensor, Tensor],
        decays: list[float],
    ) -> float:
        ...


    # TODO: Return the indices of the count papers whose vectors have the
    #   highest cosine similarity to the vector of paper, as a tensor with the
    #   most similar first. Leave out paper itself.
    def nearest_papers(
        vectors: Tensor,
        paper: int,
        count: int,
    ) -> Tensor:
        ...


    # TODO: For every paper, find the count other papers with the highest
    #   cosine similarity to it. Over all those neighbours of all papers,
    #   return the fraction that have the same topic as the paper they are
    #   near, as a Python float.
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
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but paper indices are integers. Use the .indices of topk.")

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
        description="picks the decay on validation, then measures accuracy on test once",
        run=check_test_score,
    )
    check_on_inputs(
        checker=checker,
        function="nearest_papers",
        description="returns the most similar papers by cosine, leaving out the paper itself",
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
        description="returns the fraction of nearest papers that share a topic",
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
    title="Project: how much the citations help",
    lesson=TOPICS3_LESSON,
    source=TOPICS3_SOURCE,
    solution=TOPICS3_SOLUTION,
    hints={
        "pick_decay": "For every decay, train a probe of 200 steps with train_probe and score it with accuracy on "
        "validation. Keep the scores in a list, and return the decay at the index of the largest score. list.index "
        "finds the first one.",
        "test_score": "Unpack split into its three parts. Use pick_decay to find the decay, then train one more probe "
        "with it and score that probe on test.",
        "nearest_papers": "Scale the rows to length 1, so a matrix product gives every cosine. Set the score of paper "
        "itself to minus infinity so it cannot be picked, then take the indices from topk.",
        "neighbour_agreement": "Multiply the scaled rows by their transpose to get every cosine at once. Fill the "
        "diagonal with minus infinity and take the topk along each row. Look up those papers' topics and compare "
        "them with each paper's own topic.",
        "scores": "The file computes it with test_score on the four sets of vectors it builds.",
        "agreement": "The file computes it with neighbour_agreement on the three sets of vectors it builds.",
        "nearest": "The file computes it with nearest_papers on the embeddings, then looks up each paper's topic by "
        "name.",
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
