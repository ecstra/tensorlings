"""60_word2vec, word vectors learned from raw text: skip gram pairs, negative sampling on tiny Shakespeare, and what the vectors put near each other."""

from __future__ import annotations

import functools
import itertools
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

        Section 8 is about learning representations without any labels. A
        representation is a vector, a row of numbers, that captures what
        something is. This chapter learns a vector for every word in tiny
        Shakespeare from the words around it. It uses word2vec's skip gram
        pairs and negative sampling. Then it looks at which words end up close
        together, and which analogies the vectors get right.

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
    Section 8 is about learning representations without any labels. A
    representation is a vector, a row of numbers, that captures what
    something is. The first method is word2vec, from Mikolov and colleagues
    in 2013.

    In embedding1, the model learned a row for each id while it trained on a
    task with labels. word2vec learns these rows from raw text alone. The
    idea is that the words around a word show what it means. So words that
    are used in the same places should end up with similar rows.

    First the text is split into words. Here a word is a run of the letters
    a to z, after the whole text is lowercased. So `"Thou'rt"` gives two
    words, `thou` and `rt`. `re.findall` returns every match of a pattern in
    a string. Here the pattern matches runs of digits.

    ```python
    re.findall(r"[0-9]+", "room 101, floor 7")    # ['101', '7']
    ```

    A rare word appears in too few places to learn from. So the vocabulary
    keeps only the words seen at least `min_count` times, with the most
    common first. Tiny Shakespeare has 208,503 words, and 11,455 of them are
    different. Of those, 3,225 appear 5 times or more, and together they
    make up 93.4% of the text. The other words are removed before the pairs
    are made. So the two words on either side of a removed word become
    neighbours.

    A **skip gram** pair is a word, called the center, and one of its
    neighbours up to `window` places away on either side. word2vec2 trains a
    model to tell these real pairs apart from made up pairs, whose neighbour
    is a random word. With a window of 1, "to be or not" gives these pairs.

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


    # TODO: Lowercase text, then return its words as a list of strings. A word
    #   is a run of the letters a to z, so every other character separates
    #   words.
    #   Example: words_of("Thou'rt GOOD!") returns ['thou', 'rt', 'good'].
    def words_of(text: str) -> list[str]:
        ...


    # TODO: Return the vocabulary, a list of the words that appear at least
    #   min_count times in words, each listed once. Put the most common word
    #   first, and order words with the same count alphabetically.
    #   Example: vocabulary("b a c b a c d c".split(), 2) returns ['c', 'a', 'b'].
    def vocabulary(
        words: list[str],
        min_count: int,
    ) -> list[str]:
        ...


    # TODO: Return every skip gram pair of ids as a tensor of shape (pairs, 2),
    #   with the center id in column 0 and the neighbour id in column 1. Every
    #   position is a center once. It pairs with every position up to window
    #   places away on either side, but not with itself. List the pairs center
    #   by center, and for each center from its leftmost neighbour to its
    #   rightmost.
    #   Example: skipgram_pairs([4, 7, 2], 1) returns
    #   tensor([[4, 7], [7, 4], [7, 2], [2, 7]]).
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

        return torch.tensor(pairs, dtype=torch.long).view(-1, 2)
    ''',
)


def expect_pairs(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    if isinstance(actual, Tensor) and actual.dtype.is_floating_point:
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but ids are integers. Keep the integer dtype that torch.tensor gives a list of ints.")

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def check_word2vec1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="words_of",
        description="returns the runs of letters in the lowercased text",
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
        description="keeps the words seen at least min_count times, most common first",
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
        description="pairs each id with its neighbours within the window",
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
    title="Words and their neighbours",
    lesson=WORD2VEC1_LESSON,
    source=WORD2VEC1_SOURCE,
    solution=WORD2VEC1_SOLUTION,
    hints={
        "words_of": "Call re.findall with the pattern r\"[a-z]+\" on text.lower().",
        "vocabulary": "Count the words with Counter and keep those seen min_count times or more. Sort the kept words "
        "with a key that returns (minus the count, the word), so higher counts come first and ties go alphabetically.",
        "skipgram_pairs": "For each center position, loop the neighbour position over range(max(0, center - window), "
        "min(len(ids), center + window + 1)). Skip the center itself and append each pair of ids. Then pass the list "
        "to torch.tensor.",
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
        # Start the way word2vec's own code does, with small random center rows
        # and context rows of zero.
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
    word2vec could predict each neighbour with a softmax over all 3,225
    words. But that computes a score for every word in the vocabulary, for
    every pair. **Negative sampling** asks a cheaper question instead. It asks
    whether a pair is real or made up.

    The model keeps two separate embedding tables, as derive2 did for its
    keys and queries. Every word has a row in the center table, used when the
    word is the center of a pair. It also has a row in the context table, used
    when the word is the neighbour. Call the center's row `c` and the
    neighbour's row `o`. Their dot product `c · o` is the score of the pair. A
    sigmoid turns that score into the probability that the pair is real.

    Each real pair comes with k made up pairs. A made up pair has the same
    center, with a word drawn at random as its neighbour. Call that word's row
    `n`. The loss for one real pair and its k made up pairs is this.

    ```text
    -log sigmoid(c · o) - sum over the k made up words n of log sigmoid(-c · n)
    ```

    `sigmoid(-c · n)` is the probability that a made up pair is not real. So
    the loss is small when real pairs score high and made up pairs score low.

    `F.logsigmoid(x)` computes `log(sigmoid(x))`. It stays accurate even for
    very negative `x`, where `sigmoid(x)` would round to 0 and its log would
    be minus infinity.

    ```python
    F.logsigmoid(torch.tensor([-50.0, 0.0]))    # tensor([-50.0000,  -0.6931])
    ```

    The random words are drawn with probability proportional to
    `count ** 0.75`, each word's count raised to the power 3/4. Mikolov and
    colleagues found that this power worked best. It shrinks large counts
    more than small ones, which makes the probabilities more even. For
    example, `the` is 3.23% of the kept words but only 1.40% of the draws. So
    rare words get drawn more often than their counts alone would give.

    At the start, the context table is all zeros, so every score is 0 and
    every sigmoid is 0.5. With 5 made up words, the loss is then 6 terms of
    `-log 0.5`, which is `6 * log 2`, about 4.16. Training for 3,000 steps on
    batches of 1,024 real pairs brings the loss down to about 2.29. That took
    3.1 to 3.6 seconds on 4 threads of this Mac's CPU, an Apple M5.

    After training, the rows of the center table are the word vectors.
    word2vec3 looks at what they learned.
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
    rows = torch.ones(2, 5, 3) @ torch.ones(2, 3, 1)       # shape (2, 5, 1), 5 dot products per batch entry


    # TODO: Return the probability of drawing each word as a made up neighbour.
    #   counts holds each word's count. Raise every count to the power 0.75,
    #   then divide by the sum of those powers so the probabilities add up to 1.
    #   Example: noise_distribution(torch.tensor([16, 1, 81])) returns
    #   tensor([0.2222, 0.0278, 0.7500]).
    def noise_distribution(counts: Tensor) -> Tensor:
        ...


    # TODO: Return the negative sampling loss from the lesson, averaged over the
    #   batch. centers and contexts have shape (batch, size), and row i of each
    #   is one real pair. negatives has shape (batch, k, size), and negatives[i]
    #   holds the rows of the k made up neighbours of center i.
    #   Example: with every input all zeros and k = 5, sgns_loss returns
    #   tensor(4.1589), which is 6 * log 2.
    def sgns_loss(
        centers: Tensor,
        contexts: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. centers and contexts are ids of shape (batch,), and
    #   negatives are ids of shape (batch, k). Look up the rows of centers in
    #   self.centers, and the rows of contexts and negatives in self.contexts.
    #   Return sgns_loss of those three tensors of rows.
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

    # The given start leaves the neighbour table at zero, which would hide a mixed up lookup.
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
        call="SkipGram(12, 4) on 6 pairs with 3 made up neighbours each",
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
            f"After 400 training steps on tiny Shakespeare, SkipGram's average loss over the last 100 steps is "
            f"{late:.3f}. It should fall from its start of 4.16 to below 2.8.",
        )


def check_word2vec2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="noise_distribution",
        description="gives probabilities proportional to count ** 0.75",
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
        description="computes the negative sampling loss from the lesson",
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
        description="looks up rows in the two tables and returns the loss",
        run=check_skipgram_forward,
    )
    checker.test_method(
        class_name="SkipGram",
        method="forward",
        description="lowers its loss on the pairs from tiny Shakespeare",
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
        "noise_distribution": "Raise counts.float() to the power 0.75, then divide the result by its own .sum().",
        "sgns_loss": "The real scores are (centers * contexts).sum(dim=-1), one per row. The made up scores are "
        "negatives @ centers.unsqueeze(2), squeezed back to shape (batch, k). Take F.logsigmoid of the real scores "
        "and of minus the made up scores. Sum the made up terms over k, add the real term, and return minus the mean.",
        "SkipGram.forward": "Pass self.centers(centers), self.contexts(contexts) and self.contexts(negatives) to "
        "sgns_loss, in that order.",
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


# word2vec3's vectors train longer than word2vec2's 3,000 steps, since grammar analogies keep improving to here.
VECTOR_STEPS = 10_000

# Families of word pairs that share one relation. Every ordered pair of pairs in a family is one analogy.
ANALOGY_FAMILIES: dict[str, tuple[tuple[str, str], ...]] = {
    "pronoun to possessive": (("he", "his"), ("she", "her"), ("they", "their"), ("i", "my"), ("we", "our"), ("thou", "thy"), ("you", "your")),
    "subject to object": (("he", "him"), ("she", "her"), ("they", "them"), ("i", "me"), ("we", "us"), ("thou", "thee")),
}


@functools.cache
def trained_vectors(
    seed: int = 0,
    steps: int = VECTOR_STEPS,
) -> Tensor:
    """The center table of word2vec2's model after steps of train_skipgram, for the tables of word2vec3."""
    torch.manual_seed(seed)
    _, vocabulary, _, _ = shakespeare_words()
    model = ReferenceSkipGram(len(vocabulary), 64)
    train_skipgram(model, steps, torch.Generator().manual_seed(seed))

    return model.centers.weight.detach()


WORD2VEC3_LESSON = prose(
    """
    **Cosine similarity** measures how alike two vectors are by their
    direction alone, ignoring their lengths. It is their dot product divided
    by the product of their lengths.

    ```text
    cosine(a, b) = (a · b) / (|a| * |b|)
    ```

    It is 1 when the vectors point the same way, 0 when they are at right
    angles, and -1 when they point in opposite directions.

    The vectors in this lesson come from word2vec2's model, trained for
    10,000 steps instead of 3,000. The analogies below kept improving up to
    that point. The lengths of these vectors vary a lot. The rows of the 20 most common
    words are about 1.96 long on average. The rows of the 426 words seen only
    5 times are about 7.87 long. A plain dot product grows with both lengths,
    so it would rank those long, rare rows above better matches. Scaling every
    row to length 1 first leaves only the direction to compare. `a.norm()` is
    the length of `a`, so dividing by it gives length 1.

    ```python
    a = torch.tensor([3.0, 4.0])
    a / a.norm()    # tensor([0.6000, 0.8000]), the same direction at length 1
    ```

    One hot vectors cannot show which words are alike. The one hot vectors of
    any two different words have a cosine of exactly 0, so `father` is as far
    from `mother` as from `sword`. Learned rows put words that are used in
    similar places close together. Here are the nearest words by cosine
    similarity in these vectors, which have 64 numbers per row.

    | Word | Its five nearest |
    |---|---|
    | thou | dost, art, wilt, st, hast |
    | father | son, brother, daughter, body, majesty |
    | romeo | juliet, mercutio, benvolio, friar, laurence |
    | queen | elizabeth, margaret, destiny, painted, oh |
    | night | vanity, dinner, dark, day, corruption |

    An analogy has the form "a is to b as c is to d", like "he is to his as
    she is to her". To find d, compute `b - a + c` from the rows of the three
    given words. Then find the word whose row is nearest to that, leaving out
    a, b and c themselves.

    A family is a set of word pairs that share one relation. Any two
    different pairs from a family, taken in order, make one analogy. So the
    family of 7 pairs gives `7 * 6 = 42` analogies, and the family of 6 pairs
    gives 30. The table shows how often the right answer came first, and how
    often it was among the first five. It covers three trainings with
    different random seeds.

    | Family | Its pairs | Answer first | Answer in the first five |
    |---|---|---|---|
    | pronoun to possessive | he his, she her, they their, i my, we our, thou thy, you your | 55% to 81% | 74% to 95% |
    | subject to object | he him, she her, they them, i me, we us, thou thee | 30% to 37% | 67% to 77% |

    A random guess among the 3,225 words would be right about 1 time in
    3,200. The rows learned these relations only from which words appear near
    each other. The model was never given any grammar.
    """,
)

WORD2VEC3_SOURCE = source_file(
    '''
    # word2vec3.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    a = torch.tensor([[3.0, 4.0]])
    unit = F.normalize(a, dim=1)                        # tensor([[0.6000, 0.8000]]), each row scaled to length 1
    best = torch.tensor([0.2, 0.9, 0.5]).topk(2)        # values 0.9 and 0.5, at indices 1 and 2


    # TODO: Return the cosine similarity of every row of a with every row of b.
    #   a has shape (n, size) and b has shape (m, size). The result has shape
    #   (n, m), and entry [i, j] is the cosine similarity of a[i] and b[j].
    #   Example: for a = [[3.0, 4.0]] and b = [[6.0, 8.0], [0.0, 2.0]], it
    #   returns tensor([[1.0000, 0.8000]]).
    def cosine_similarity(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the indices of the k rows of vectors with the highest cosine
    #   similarity to vectors[index], as a list of ints with the most similar
    #   first. Leave out the row at index itself.
    #   Example: for vectors = [[1., 0.], [0., 1.], [1., 1.], [1., 0.1]],
    #   nearest(vectors, 0, 2) returns [3, 2].
    def nearest(
        vectors: Tensor,
        index: int,
        k: int,
    ) -> list[int]:
        ...


    # TODO: Solve the analogy "a is to b as c is to what?". a, b and c are row
    #   indices into vectors. Scale every row of vectors to length 1, then
    #   compute row b - row a + row c from the scaled rows. Return the indices
    #   of the k rows with the highest cosine similarity to that result, as a
    #   list with the most similar first. Leave out a, b and c themselves.
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
        description="returns the cosine similarity of every pair of rows",
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
        description="returns the most similar rows, leaving out the row itself",
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
        description="returns the rows nearest to b - a + c, leaving out a, b and c",
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
        "cosine_similarity": "Scale every row of a and of b to length 1 with F.normalize and dim=1. Then matrix "
        "multiply the scaled a by the transpose of the scaled b.",
        "nearest": "Call cosine_similarity on vectors[index : index + 1], which keeps the row as a batch of one, and "
        "on all of vectors. Take row 0 of the result, set the row's own score to float(\"-inf\"), then use "
        ".topk(k).indices.tolist().",
        "analogy": "Normalize the rows, build b - a + c from the normalized rows, and score it against every row with "
        "cosine_similarity. Set the scores at a, b and c to float(\"-inf\") before calling .topk(k).",
    },
    introduces=(),
    check=check_word2vec3,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def family_shares(
    vectors: Tensor,
    pairs: tuple[tuple[str, str], ...],
) -> tuple[float, float]:
    """The share of a family's analogies whose answer comes first, and the share where it is among the first five."""
    _, vocabulary, _, _ = shakespeare_words()
    index = {word: position for position, word in enumerate(vocabulary)}
    first = within_five = total = 0

    for (a, b), (c, d) in itertools.permutations(pairs, 2):
        # Pairs that share a word, like he to his and he to him, would ask for an answer that analogy leaves out, so
        # it could never land. No family here shares one, so this skips nothing today.
        if len({a, b, c, d}) < 4:
            continue

        found = reference_analogy(vectors, index[a], index[b], index[c], 5)
        total += 1
        first += found[0] == index[d]
        within_five += index[d] in found

    return first / total, within_five / total


def measure_vectors(seeds: range = range(3)) -> dict[str, object]:
    """word2vec3's numbers: the row lengths and nearest words of seed 0's vectors, and every family's analogy shares
    on each seed, after VECTOR_STEPS steps."""
    before = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        words, vocabulary, counts, _ = shakespeare_words()
        index = {word: position for position, word in enumerate(vocabulary)}
        vectors = trained_vectors(0)
        lengths = vectors.norm(dim=1)
        results: dict[str, object] = {
            "common lengths": lengths[:20].mean().item(),
            "rare lengths": lengths[counts == MIN_COUNT].mean().item(),
            "rare count": int((counts == MIN_COUNT).sum()),
        }

        for word in ("thou", "father", "romeo", "queen", "night"):
            results[word] = [vocabulary[position] for position in reference_nearest(vectors, index[word], 5)]

        for seed in seeds:
            for family, pairs in ANALOGY_FAMILIES.items():
                results[f"{family}, seed {seed}"] = family_shares(trained_vectors(seed), pairs)

        return results
    finally:
        torch.set_num_threads(before)
