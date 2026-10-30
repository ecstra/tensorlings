"""60_word2vec, word vectors learned from raw text: skip gram pairs, negative sampling on tiny Shakespeare, and what the vectors put near each other."""

from __future__ import annotations

import functools
import re
import textwrap
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import load_shakespeare
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

WORD2VEC = Chapter(
    folder="60_word2vec",
    title="Word vectors",
    readme=source_file(
        """
        # Word vectors

        Section 8 is about learning representations, rows of numbers that
        capture what something is, without any labels. This chapter learns a
        row for every word of tiny Shakespeare from the words around it, with
        word2vec's skip grams and negative sampling, then looks at which words
        end up close together and which analogies hold.

        ## Further reading

        - [Mikolov and colleagues, efficient estimation of word representations](https://arxiv.org/abs/1301.3781)
        - [Mikolov and colleagues, distributed representations of words and phrases](https://arxiv.org/abs/1310.4546)
        - [Alammar, the illustrated word2vec](https://jalammar.github.io/illustrated-word2vec/)
        """,
    ),
)

MIN_COUNT = 5
WINDOW = 2
NEGATIVES = 5


def reference_words_of(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def reference_vocabulary(
    words: list[str],
    min_count: int,
) -> list[str]:
    counts = Counter(words)
    kept = [word for word in counts if counts[word] >= min_count]

    return sorted(kept, key=lambda word: (-counts[word], word))


def reference_skipgram_pairs(
    ids: list[int],
    window: int,
) -> Tensor:
    pairs = []

    for center in range(len(ids)):
        for context in range(max(0, center - window), min(len(ids), center + window + 1)):
            if context != center:
                pairs.append((ids[center], ids[context]))

    return torch.tensor(pairs, dtype=torch.long).view(-1, 2)


@functools.cache
def shakespeare_words() -> tuple[list[str], list[str], Tensor, Tensor]:
    """The vocabulary, every kept word's count, and the skip gram pairs of tiny Shakespeare, built once per run."""
    words = reference_words_of(load_shakespeare())
    vocabulary = reference_vocabulary(words, MIN_COUNT)
    index = {word: position for position, word in enumerate(vocabulary)}
    ids = [index[word] for word in words if word in index]
    counts = Counter(words)

    return words, vocabulary, torch.tensor([counts[word] for word in vocabulary]), reference_skipgram_pairs(ids, WINDOW)


# ---------------------------------------------------------------------------
# word2vec1
# ---------------------------------------------------------------------------

WORD2VEC1_LESSON = prose(
    """
    Section 8 is about learning representations, rows of numbers that capture
    what something is, without any labels. The first is word2vec, from
    Mikolov and colleagues in 2013.

    embedding1 learned rows for ids inside a task with labels. word2vec learns
    them from raw text alone. A word is known by the words around it, so
    words used in the same places should end up with similar rows.

    The text first becomes words. Here a word is a run of letters,
    lowercased, so `"Thou'rt"` gives `thou` and `rt`.

    ```python
    re.findall(r"[0-9]+", "room 101, floor 7")    # ['101', '7']
    ```

    Rare words give too few examples to learn from, so the vocabulary keeps
    the words seen at least `min_count` times, most common first. Tiny
    Shakespeare has 208,503 words, 11,455 of them different. 3,225 appear 5
    times or more, and they make up 93.4% of the text. The rest are dropped
    before pairing, so their neighbors close up.

    A **skip gram** pair is a word and one of its neighbors, up to `window`
    places away on either side. word2vec2 trains a model to tell such real
    pairs from made up ones. With a window of 1, "to be or not" gives these.

    ```text
    (to, be)  (be, to)  (be, or)  (or, be)  (or, not)  (not, or)
    ```

    With a window of 2, the kept words of tiny Shakespeare give 779,182
    pairs.
    """,
)

WORD2VEC1_SOURCE = source_file(
    '''
    # word2vec1.py

    import re
    from collections import Counter

    import torch
    from torch import Tensor

    runs = re.findall(r"[0-9]+", "room 101, floor 7")              # ['101', '7']
    counts = Counter(["to", "be", "or", "not", "to", "be"])         # counts["to"] is 2
    ordered = sorted(["pear", "fig", "apple"], key=lambda word: (len(word), word))    # ['fig', 'pear', 'apple']


    # TODO: Return the words of text, each a run of the letters a to z, after
    #   lowercasing the whole text.
    def words_of(text: str) -> list[str]:
        ...


    # TODO: Return the words seen at least min_count times, each once, the
    #   most common first, and words with the same count in alphabetical order.
    def vocabulary(
        words: list[str],
        min_count: int,
    ) -> list[str]:
        ...


    # TODO: Return every skip gram pair of ids as a tensor of shape (pairs, 2),
    #   the center first and its neighbor second. Each center pairs with every
    #   position up to window away on either side, itself left out. Go center by
    #   center, and within a center from the leftmost neighbor to the rightmost.
    def skipgram_pairs(
        ids: list[int],
        window: int,
    ) -> Tensor:
        ...
    ''',
)

WORD2VEC1_SOLUTION = source_file(
    '''
    # word2vec1.py, a solution for comparison

    import re
    from collections import Counter

    import torch
    from torch import Tensor

    runs = re.findall(r"[0-9]+", "room 101, floor 7")
    counts = Counter(["to", "be", "or", "not", "to", "be"])
    ordered = sorted(["pear", "fig", "apple"], key=lambda word: (len(word), word))


    def words_of(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())


    def vocabulary(
        words: list[str],
        min_count: int,
    ) -> list[str]:
        counted = Counter(words)
        kept = [word for word in counted if counted[word] >= min_count]

        return sorted(kept, key=lambda word: (-counted[word], word))


    def skipgram_pairs(
        ids: list[int],
        window: int,
    ) -> Tensor:
        pairs = []

        for center in range(len(ids)):
            for context in range(max(0, center - window), min(len(ids), center + window + 1)):
                if context != center:
                    pairs.append((ids[center], ids[context]))

        return torch.tensor(pairs)
    ''',
)


def expect_pairs(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    if isinstance(actual, Tensor) and actual.dtype.is_floating_point:
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but ids are whole numbers. Leave the dtype as torch.tensor gives it.")

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def check_word2vec1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="words_of",
        description="gives the lowercased runs of letters",
        inputs=(
            ("Thou'rt a GOOD friend, Horatio; 2 days!",),
            ("O Romeo, Romeo!\nwherefore art thou Romeo?",),
        ),
        reference=reference_words_of,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="vocabulary",
        description="keeps the common words, most common first",
        inputs=(
            ("to be or not to be that is the question to be".split(), 2),
            ("b a c b a c d c".split(), 1),
            ("one two two three three three".split(), 4),
        ),
        reference=reference_vocabulary,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="skipgram_pairs",
        description="pairs each id with its neighbors within the window",
        inputs=(
            ([4, 7, 2, 9], 1),
            ([3, 1, 4, 1, 5, 9], 2),
            ([8, 6], 3),
        ),
        reference=reference_skipgram_pairs,
        expect=expect_pairs,
    )


WORD2VEC1 = Exercise(
    name="word2vec1",
    chapter=WORD2VEC,
    title="Words and their neighbors",
    lesson=WORD2VEC1_LESSON,
    source=WORD2VEC1_SOURCE,
    solution=WORD2VEC1_SOLUTION,
    hints={
        "words_of": "re.findall with the pattern r\"[a-z]+\" on text.lower().",
        "vocabulary": "Count the words with Counter and keep those seen min_count times or more. Sort them with a "
        "key of (minus the count, the word).",
        "skipgram_pairs": "For each center, loop the neighbor from max(0, center - window) up to min(len(ids), "
        "center + window + 1). Skip the center itself, append each pair of ids, and pass the list to torch.tensor.",
    },
    introduces=(),
    check=check_word2vec1,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# word2vec2
# ---------------------------------------------------------------------------

SKIPGRAM_INIT = '''
class SkipGram(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        size: int,
    ) -> None:
        super().__init__()
        self.centers = nn.Embedding(vocabulary, size)
        self.contexts = nn.Embedding(vocabulary, size)
        # word2vec's own start, small random centers and contexts of zero.
        nn.init.uniform_(self.centers.weight, -0.5 / size, 0.5 / size)
        nn.init.zeros_(self.contexts.weight)
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def reference_noise_distribution(counts: Tensor) -> Tensor:
    weights = counts.float() ** 0.75

    return weights / weights.sum()


def reference_sgns_loss(
    centers: Tensor,
    contexts: Tensor,
    negatives: Tensor,
) -> Tensor:
    real = F.logsigmoid((centers * contexts).sum(dim=-1))
    made_up = F.logsigmoid(-(negatives @ centers.unsqueeze(2)).squeeze(2)).sum(dim=-1)

    return -(real + made_up).mean()


class ReferenceSkipGram(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        size: int,
    ) -> None:
        super().__init__()
        self.centers = nn.Embedding(vocabulary, size)
        self.contexts = nn.Embedding(vocabulary, size)
        nn.init.uniform_(self.centers.weight, -0.5 / size, 0.5 / size)
        nn.init.zeros_(self.contexts.weight)

    def forward(
        self,
        centers: Tensor,
        contexts: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        return reference_sgns_loss(self.centers(centers), self.contexts(contexts), self.contexts(negatives))


def train_skipgram(
    model: nn.Module,
    steps: int,
    generator: torch.Generator,
) -> list[float]:
    """The run the lessons quote, Adam at 0.01 on batches of 1,024 pairs with 5 made up ones each."""
    _, _, counts, pairs = shakespeare_words()
    noise = reference_noise_distribution(counts)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []

    for _ in range(steps):
        batch = pairs[torch.randint(0, len(pairs), (1024,), generator=generator)]
        negatives = torch.multinomial(noise, 1024 * NEGATIVES, replacement=True, generator=generator).view(1024, NEGATIVES)
        loss = model(batch[:, 0], batch[:, 1], negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


WORD2VEC2_LESSON = prose(
    """
    word2vec could predict each neighbor with a softmax over all 3,225
    words, but that scores every word for every pair. **Negative sampling**
    asks a cheaper question instead, whether a pair is real.

    The model keeps two tables, like derive2's keys and queries. A word's row
    as a center is `c`, and its row as a neighbor is `o`. Their dot product
    scores the pair, and a sigmoid turns the score into a chance it is real.
    Each real pair comes with k made up ones, the same center with words drawn
    at random. The loss for one real pair is this.

    ```text
    -log sigmoid(c · o) - sum over the k made up words n of log sigmoid(-c · n)
    ```

    `F.logsigmoid(x)` gives `log(sigmoid(x))` without the trouble of taking
    the log of a number that rounds to 0.

    ```python
    F.logsigmoid(torch.tensor([-50.0, 0.0]))    # tensor([-50.0000,  -0.6931])
    ```

    The made up words are drawn in proportion to each word's count raised to
    the power 3/4, which Mikolov and colleagues found worked best. The power
    flattens the counts. `the` is 3.02% of the words but 1.40% of the draws,
    so the rarer words get drawn more often than their counts alone would
    give.

    At the start the neighbor table is all zeros, so every score is 0. With 5
    made up words the loss is then 6 log 2, about 4.16. Training 3,000 steps of
    1,024 pairs brings it to about 2.29, in seconds on a quiet machine.

    After training, the center table holds the word vectors. word2vec3 looks
    inside them.
    """,
)

WORD2VEC2_SOURCE = source_file(
    '''
    # word2vec2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    chances = F.logsigmoid(torch.tensor([-50.0, 0.0]))     # tensor([-50.0000,  -0.6931])
    rows = torch.ones(2, 5, 3) @ torch.ones(2, 3, 1)       # shape (2, 5, 1), one dot product per made up row


    # TODO: Return the chance of drawing each word as a made up neighbor, its
    #   count raised to the power 0.75, divided by the sum of them all.
    def noise_distribution(counts: Tensor) -> Tensor:
        ...


    # TODO: Return the mean over the batch of the negative sampling loss.
    #   centers and contexts have shape (batch, size), one real pair per row,
    #   and negatives has shape (batch, k, size), the made up neighbors of each
    #   center.
    def sgns_loss(
        centers: Tensor,
        contexts: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. Look the centers up in self.centers, and the
    #   contexts and the negatives, of shape (batch, k), in self.contexts. Return
    #   sgns_loss of them.
''' + indented(SKIPGRAM_INIT) + '''

        def forward(
            self,
            centers: Tensor,
            contexts: Tensor,
            negatives: Tensor,
        ) -> Tensor:
            ...
    ''',
)

WORD2VEC2_SOLUTION = source_file(
    '''
    # word2vec2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    chances = F.logsigmoid(torch.tensor([-50.0, 0.0]))
    rows = torch.ones(2, 5, 3) @ torch.ones(2, 3, 1)


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


''' + indented(SKIPGRAM_INIT) + '''

        def forward(
            self,
            centers: Tensor,
            contexts: Tensor,
            negatives: Tensor,
        ) -> Tensor:
            return sgns_loss(self.centers(centers), self.contexts(contexts), self.contexts(negatives))
    ''',
)


def seeded_skipgram(make: type) -> nn.Module:
    torch.manual_seed(600)

    return make(12, 4)


def check_skipgram_forward(found: type) -> None:
    model = expect_module("SkipGram(12, 4)", seeded_skipgram(found))
    reference = seeded_skipgram(ReferenceSkipGram)

    # The given start leaves the neighbor table at zero, which would hide a mixed up lookup.
    with torch.no_grad():
        for table in (model.contexts.weight, reference.contexts.weight):
            table.copy_(torch.randn(12, 4, generator=torch.Generator().manual_seed(601)))

    generator = torch.Generator().manual_seed(602)
    centers = torch.randint(0, 12, (6,), generator=generator)
    contexts = torch.randint(0, 12, (6,), generator=generator)
    negatives = torch.randint(0, 12, (6, 3), generator=generator)

    with torch.no_grad():
        expected = reference(centers, contexts, negatives)

    expect_tensor(
        call="SkipGram(12, 4) on 6 pairs with 3 made up neighbors each",
        actual=model(centers, contexts, negatives),
        expected=expected,
    )


def check_skipgram_learns(found: type) -> None:
    torch.manual_seed(0)
    _, vocabulary, _, _ = shakespeare_words()
    model = found(len(vocabulary), 64)
    losses = train_skipgram(model, 400, torch.Generator().manual_seed(0))
    late = sum(losses[-100:]) / 100

    if late > 2.8:
        raise Mismatch(
            f"After 400 steps on tiny Shakespeare, SkipGram's loss averages {late:.3f} over the last 100 steps, "
            "but it should fall below 2.8 from its start of 4.16.",
        )


def check_word2vec2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="noise_distribution",
        description="draws in proportion to the counts to the power 3/4",
        inputs=(
            (torch.tensor([16, 1, 81]),),
            (torch.tensor([5, 5, 120, 7, 1]),),
        ),
        reference=reference_noise_distribution,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="sgns_loss",
        description="scores real pairs up and made up ones down",
        inputs=(
            (small_tensor(603, 3, 4) / 3, small_tensor(604, 3, 4) / 3, Described(small_tensor(605, 3, 2, 4) / 3, "2 made up rows per center")),
            (small_tensor(606, 5, 2) / 3, small_tensor(607, 5, 2) / 3, Described(small_tensor(608, 5, 4, 2) / 3, "4 made up rows per center")),
        ),
        reference=reference_sgns_loss,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="SkipGram",
        method="forward",
        description="looks up the two tables and gives the loss",
        run=check_skipgram_forward,
    )
    checker.test_method(
        class_name="SkipGram",
        method="forward",
        description="learns from the pairs of tiny Shakespeare",
        run=check_skipgram_learns,
    )


WORD2VEC2 = Exercise(
    name="word2vec2",
    chapter=WORD2VEC,
    title="Negative sampling",
    lesson=WORD2VEC2_LESSON,
    source=WORD2VEC2_SOURCE,
    solution=WORD2VEC2_SOLUTION,
    hints={
        "noise_distribution": "counts.float() ** 0.75, divided by its .sum().",
        "sgns_loss": "The real score is (centers * contexts).sum(dim=-1). The made up scores are negatives @ "
        "centers.unsqueeze(2), squeezed back to (batch, k). Take F.logsigmoid of the real ones and of minus the made "
        "up ones, sum those over k, add, and take minus the mean.",
        "SkipGram.forward": "self.centers(centers), self.contexts(contexts) and self.contexts(negatives) go to "
        "sgns_loss in that order.",
    },
    introduces=(),
    check=check_word2vec2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# word2vec3
# ---------------------------------------------------------------------------


def reference_cosine_similarity(
    a: Tensor,
    b: Tensor,
) -> Tensor:
    return F.normalize(a, dim=1) @ F.normalize(b, dim=1).T


def reference_nearest(
    vectors: Tensor,
    index: int,
    k: int,
) -> list[int]:
    similarity = reference_cosine_similarity(vectors[index : index + 1], vectors)[0]
    similarity[index] = float("-inf")

    return similarity.topk(k).indices.tolist()


def reference_analogy(
    vectors: Tensor,
    a: int,
    b: int,
    c: int,
    k: int,
) -> list[int]:
    unit = F.normalize(vectors, dim=1)
    target = unit[b] - unit[a] + unit[c]
    similarity = reference_cosine_similarity(target.unsqueeze(0), vectors)[0]
    similarity[[a, b, c]] = float("-inf")

    return similarity.topk(k).indices.tolist()


@functools.cache
def trained_vectors() -> Tensor:
    """The center table after the 3,000 steps word2vec2's lesson describes, for the tables of word2vec3."""
    torch.manual_seed(0)
    _, vocabulary, _, _ = shakespeare_words()
    model = ReferenceSkipGram(len(vocabulary), 64)
    train_skipgram(model, 3000, torch.Generator().manual_seed(0))

    return model.centers.weight.detach()


WORD2VEC3_LESSON = prose(
    """
    **Cosine similarity** measures how alike two rows are by their direction
    alone, their dot product divided by both their lengths. It is 1 for the
    same direction, 0 for unrelated ones, and -1 for opposite ones. Frequent
    words tend to get longer rows, and the plain dot product would favor them.

    ```python
    a = torch.tensor([3.0, 4.0])
    a / a.norm()    # tensor([0.6000, 0.8000]), the same direction at length 1
    ```

    One hot vectors can't say which words are alike. Every pair of different
    words has a cosine of exactly 0, so `father` is as far from `mother` as
    from `sword`. Learned rows put words used alike close together. Here are
    the nearest words in word2vec2's vectors, after 3,000 steps with rows of
    64.

    | Word | Its five nearest |
    |---|---|
    | thou | wilt, art, thyself, couldst, dost |
    | father | daughter, brother, sake, master, asking |
    | romeo | mercutio, laurence, friar, juliet, banished |
    | queen | margaret, elizabeth, destiny, oh, england |
    | night | day, league, dinner, appointed, took |

    An analogy takes the rows of `b - a + c` and looks for the nearest word to
    it, leaving out the three it started from. Grammar comes out right here.
    he is to his as she is to `her`, and thou is to thy as you is to `your`.
    The famous king minus man plus woman gives `protector`, `fair` and
    `richard`, and no `queen`.

    Mikolov and colleagues trained on a billion words of news, and tiny
    Shakespeare has 208,503. A word's neighbors come from the words the text
    puts near it, so a small text teaches less.
    """,
)

WORD2VEC3_SOURCE = source_file(
    '''
    # word2vec3.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    a = torch.tensor([[3.0, 4.0]])
    unit = F.normalize(a, dim=1)                        # tensor([[0.6000, 0.8000]]), each row at length 1
    best = torch.tensor([0.2, 0.9, 0.5]).topk(2)        # values 0.9 and 0.5, at 1 and 2


    # TODO: Return the cosine similarity of every row of a, of shape (n, size),
    #   with every row of b, of shape (m, size), as a tensor of shape (n, m).
    def cosine_similarity(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the positions of the k rows of vectors most similar to the
    #   row at index by cosine, the most similar first, the row itself left out.
    def nearest(
        vectors: Tensor,
        index: int,
        k: int,
    ) -> list[int]:
        ...


    # TODO: Return the positions of the k rows most similar by cosine to
    #   b - a + c, the most similar first, where a, b and c are positions and
    #   each of their rows is first scaled to length 1. Leave a, b and c out.
    def analogy(
        vectors: Tensor,
        a: int,
        b: int,
        c: int,
        k: int,
    ) -> list[int]:
        ...
    ''',
)

WORD2VEC3_SOLUTION = source_file(
    '''
    # word2vec3.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    a = torch.tensor([[3.0, 4.0]])
    unit = F.normalize(a, dim=1)
    best = torch.tensor([0.2, 0.9, 0.5]).topk(2)


    def cosine_similarity(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return F.normalize(a, dim=1) @ F.normalize(b, dim=1).T


    def nearest(
        vectors: Tensor,
        index: int,
        k: int,
    ) -> list[int]:
        similarity = cosine_similarity(vectors[index : index + 1], vectors)[0]
        similarity[index] = float("-inf")

        return similarity.topk(k).indices.tolist()


    def analogy(
        vectors: Tensor,
        a: int,
        b: int,
        c: int,
        k: int,
    ) -> list[int]:
        unit = F.normalize(vectors, dim=1)
        target = unit[b] - unit[a] + unit[c]
        similarity = cosine_similarity(target.unsqueeze(0), vectors)[0]
        similarity[[a, b, c]] = float("-inf")

        return similarity.topk(k).indices.tolist()
    ''',
)


def word_vectors(seed: int) -> Tensor:
    return torch.randn(12, 5, generator=torch.Generator().manual_seed(seed))


def check_word2vec3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="cosine_similarity",
        description="gives the cosine of every pair of rows",
        inputs=(
            (torch.tensor([[3.0, 4.0], [1.0, 0.0]]), torch.tensor([[6.0, 8.0], [0.0, 2.0], [-1.0, 0.0]])),
            (word_vectors(610)[:4], word_vectors(611)[:3]),
        ),
        reference=reference_cosine_similarity,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="nearest",
        description="gives the most similar rows, the row itself left out",
        inputs=(
            (Described(word_vectors(612) * torch.arange(1, 13).unsqueeze(1), "12 rows of 5 of growing lengths"), 3, 4),
            (Described(word_vectors(613), "12 rows of 5"), 0, 3),
        ),
        reference=reference_nearest,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="analogy",
        description="gives the rows nearest b - a + c, leaving a, b and c out",
        inputs=(
            (Described(word_vectors(614) * torch.arange(1, 13).unsqueeze(1), "12 rows of 5 of growing lengths"), 1, 7, 4, 3),
            (Described(word_vectors(615), "12 rows of 5"), 0, 2, 5, 4),
        ),
        reference=reference_analogy,
        expect=expect_value,
    )


WORD2VEC3 = Exercise(
    name="word2vec3",
    chapter=WORD2VEC,
    title="What the vectors learned",
    lesson=WORD2VEC3_LESSON,
    source=WORD2VEC3_SOURCE,
    solution=WORD2VEC3_SOLUTION,
    hints={
        "cosine_similarity": "Scale every row of both to length 1 with F.normalize and dim=1, then multiply a by the "
        "transpose of b.",
        "nearest": "cosine_similarity of the row, kept as a batch of one with vectors[index : index + 1], against "
        "every row. Set the row's own score to float(\"-inf\"), then .topk(k).indices.tolist().",
        "analogy": "Normalize the rows, build b - a + c from them, and score it against every row with "
        "cosine_similarity. Set a, b and c to float(\"-inf\") before .topk(k).",
    },
    introduces=(),
    check=check_word2vec3,
)
