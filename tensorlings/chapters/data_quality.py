"""92_data, the text a model trains on: exact and near duplicates removed with shingles and Jaccard similarity, quality rules like Gopher's, and test text that leaked into the training set found by shared n-grams."""

from __future__ import annotations

import math
import random
import statistics

import torch
import torch.nn as nn
from torch import Tensor

from ..course import QUIZZES, Chapter, Exercise
from ..data import load_shakespeare
from ..engine import (
    Checker,
    Described,
    check_on_inputs,
    expect_close,
    expect_value,
)
from ..text import prose, source_file
from .cost import tied_pair
from .scaling import expect_relative

DATA_QUALITY = Chapter(
    folder="92_data",
    title="Training data",
    readme=source_file(
        """
        # Training data

        A model learns whatever its text holds, repeats and junk included,
        and a test set it has already read measures memory, not skill. This
        chapter removes exact and near duplicates, filters text with rules
        like those behind Gopher, and finds test text that leaked into the
        training set. The measurements run on the Shakespeare of Section 7,
        downloaded once from the project folder.

        ```bash
        .venv/bin/python run.py data shakespeare
        ```

        ## Further reading

        - [Lee and colleagues, deduplicating training data makes language models better](https://arxiv.org/abs/2107.06499)
        - [Rae and colleagues, Gopher, section A.1 on MassiveText](https://arxiv.org/abs/2112.11446)
        - [Brown and colleagues, GPT-3, section 4 and appendix C on test set contamination](https://arxiv.org/abs/2005.14165)
        - [Broder, on the resemblance and containment of documents, the paper behind MinHash](https://doi.org/10.1109/SEQUEN.1997.666900)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return "\n".join("    " + line if line else "" for line in part.strip("\n").splitlines())


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(code, namespace)

    return namespace


# ---------------------------------------------------------------------------
# data1
# ---------------------------------------------------------------------------

NORMALIZE_SOLUTION = '''
def normalize(text: str) -> str:
    kept = "".join(character for character in text.lower() if character.isalnum() or character.isspace())

    return " ".join(kept.split())
'''

EXACT_SOLUTION = '''
def drop_exact(documents: list[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []

    for document in documents:
        key = normalize(document)

        if key in seen:
            continue

        seen.add(key)
        kept.append(document)

    return kept
'''

SHINGLES_SOLUTION = '''
def shingles(
    text: str,
    n: int,
) -> set[tuple[str, ...]]:
    words = normalize(text).split()

    return {tuple(words[start : start + n]) for start in range(len(words) - n + 1)}
'''

NEAR_SOLUTION = '''
def jaccard(
    a: set[tuple[str, ...]],
    b: set[tuple[str, ...]],
) -> float:
    if not a and not b:
        return 0.0

    return len(a & b) / len(a | b)


def drop_near(
    documents: list[str],
    n: int,
    threshold: float,
) -> list[str]:
    kept: list[str] = []
    kept_shingles: list[set[tuple[str, ...]]] = []

    for document in documents:
        own = shingles(document, n)

        if any(jaccard(own, other) >= threshold for other in kept_shingles):
            continue

        kept.append(document)
        kept_shingles.append(own)

    return kept
'''

DUPLICATES_REFERENCE = namespace_of(NORMALIZE_SOLUTION + EXACT_SOLUTION + SHINGLES_SOLUTION + NEAR_SOLUTION)


def is_speech(block: str) -> bool:
    """A block between blank lines with a line after its speaker's name. Tiny Shakespeare has 125 blocks that
    hold a name alone, which would count as repeats of each other."""
    return len(block.strip().splitlines()) >= 2


def play_speeches() -> list[str]:
    """Every speech of tiny Shakespeare, each opening with its speaker."""
    return [block for block in load_shakespeare().split("\n\n") if is_speech(block)]


def word_count(text: str) -> int:
    return len(DUPLICATES_REFERENCE["normalize"](text).split())


def edit_words(
    text: str,
    every: int,
    vocabulary: list[str],
    generator: random.Random,
) -> str:
    """A near copy, one word in every words swapped for another, as an edited or reposted page would differ."""
    words = text.split(" ")

    for _ in range(max(1, len(words) // every)):
        words[generator.randrange(len(words))] = generator.choice(vocabulary)

    return " ".join(words)


def planted_documents(seed: int = 0) -> tuple[list[str], list[tuple[str, str]], int]:
    """600 speeches of 30 words or more, 60 exact copies with their case and spacing changed, and 60 near copies
    with one word in 25 swapped, shuffled together. Gives the documents, the near pairs and the exact copy count."""
    generator = random.Random(seed)
    long_speeches = [speech for speech in play_speeches() if word_count(speech) >= 30]
    originals = generator.sample(long_speeches, 600)
    vocabulary = sorted({word for speech in originals for word in speech.split()})
    exact_copies = [speech.upper().replace(" ", "  ", 3) for speech in generator.sample(originals, 60)]
    near_pairs = [(speech, edit_words(speech, 25, vocabulary, generator)) for speech in generator.sample(originals, 60)]
    documents = originals + exact_copies + [near for _, near in near_pairs]
    generator.shuffle(documents)

    return documents, near_pairs, len(exact_copies)


def measure_duplicates() -> dict[str, object]:
    """data1's numbers. The repeats in the play, then the planted copies found at each shingle length."""
    normalize = DUPLICATES_REFERENCE["normalize"]
    shingles = DUPLICATES_REFERENCE["shingles"]
    jaccard = DUPLICATES_REFERENCE["jaccard"]
    speeches = play_speeches()
    repeats = len(speeches) - len(DUPLICATES_REFERENCE["drop_exact"](speeches))
    documents, near_pairs, exact_count = planted_documents()
    after_exact = DUPLICATES_REFERENCE["drop_exact"](documents)
    results: dict[str, object] = {
        "speeches": len(speeches),
        "repeats in the play": repeats,
        "documents": len(documents),
        "exact copies planted": exact_count,
        "exact copies dropped": len(documents) - len(after_exact),
    }
    for n in (3, 5, 8, 13):
        similarities = [jaccard(shingles(original, n), shingles(near, n)) for original, near in near_pairs]
        kept = DUPLICATES_REFERENCE["drop_near"](after_exact, n, 0.8)
        kept_forms = {normalize(document) for document in kept}
        # A pair is caught when one of the two goes, whichever came first in the shuffle.
        caught = sum(not (normalize(original) in kept_forms and normalize(near) in kept_forms) for original, near in near_pairs)
        results[f"n={n}"] = {
            "near copies dropped": caught,
            "median similarity": statistics.median(similarities),
            "others dropped": len(after_exact) - len(kept) - caught,
        }

    return results


DATA1_LESSON = prose(
    """
    Text gathered from the web repeats itself. Lee and colleagues found one
    61 word sentence repeated over 60,000 times in C4, a common training set.
    Models trained on their deduplicated data gave out memorized text about
    ten times less often.

    Exact copies are the easy part. Bring every document to one plain form,
    lowercase with only letters, digits and single spaces, and keep the first
    document of each form. A Python `set` answers "seen this before?" at once.

    ```python
    "Speak, speak!".lower()              # 'speak, speak!'
    " ".join("a  b\\n c".split())          # 'a b c'
    seen = {"a b"}; "a b" in seen        # True
    ```

    The play repeats itself too. Of its 7,097 speeches, 19 match an earlier
    one in that form, short ones like `CATESBY: My lord?`.

    Near copies differ by a few words, so no plain form catches them. Cut
    each document into **shingles**, every run of n words. Then compare the
    sets with Jaccard similarity, the size of their overlap over the size of
    their union. It is 1 for the same sets and 0 for sets with nothing in
    common.

    ```python
    words = "to be or not".split()
    {tuple(words[i : i + 2]) for i in range(3)}    # {('to', 'be'), ('be', 'or'), ('or', 'not')}
    len({1, 2, 3} & {2, 3, 4}) / len({1, 2, 3} | {2, 3, 4})    # 0.5
    ```

    Here are 600 speeches of 30 words or more and 60 exact copies with their
    case and spacing changed. With them go 60 near copies, one word in 25
    swapped. The exact pass drops all 60 copies. Then a document goes when it reaches
    a similarity of 0.8 with one already kept.

    | n | Near copies dropped | Their median similarity | Other speeches dropped |
    |---|---|---|---|
    | 3 | 59 of 60 | 0.84 | 0 |
    | 5 | 17 of 60 | 0.74 | 0 |
    | 8 | 6 of 60 | 0.64 | 0 |
    | 13 | 6 of 60 | 0.47 | 0 |

    One swapped word breaks every shingle that holds it, n of them. Gopher
    used 13 word shingles on web text. Speeches of a few dozen words need
    shorter ones.

    Comparing every pair takes time that grows with the square of the
    documents. Broder's MinHash shrinks each set to a short signature. Two
    sets share their smallest hash with a chance equal to their similarity,
    so a few hundred hashes estimate it, and each comparison gets cheap.
    Skipping most pairs takes one more step, locality sensitive hashing,
    which buckets signatures by their parts so only likely pairs meet.
    """,
)

DATA1_SOURCE = source_file(
    '''
    # data1.py

    shouted = "Speak, speak!".lower()                # 'speak, speak!'
    tidy = " ".join("a  b\\n c".split())               # 'a b c'
    words = "to be or not".split()
    pairs = {tuple(words[i : i + 2]) for i in range(3)}     # {('to', 'be'), ('be', 'or'), ('or', 'not')}
    overlap = len({1, 2, 3} & {2, 3, 4}) / len({1, 2, 3} | {2, 3, 4})    # 0.5


    # TODO: Return text lowercased, with every character that is not a letter,
    #   a digit or whitespace removed, and the words joined by single spaces.
    #   str.isalnum() and str.isspace() test one character.
    def normalize(text: str) -> str:
        ...


    # TODO: Return documents without exact copies. A document is a copy when
    #   its normalized form matches an earlier document's. Keep the first of
    #   each, in their order.
    def drop_exact(documents: list[str]) -> list[str]:
        ...


    # TODO: Return the set of every run of n words of the normalized text, each
    #   a tuple of words. A text of fewer than n words has none.
    def shingles(
        text: str,
        n: int,
    ) -> set[tuple[str, ...]]:
        ...


    # TODO: Return the Jaccard similarity of a and b, the size of their
    #   overlap over the size of their union, or 0.0 when both are empty.
    def jaccard(
        a: set[tuple[str, ...]],
        b: set[tuple[str, ...]],
    ) -> float:
        ...


    # TODO: Return documents without near copies. Go through them in order and
    #   keep a document unless the similarity of its shingles of n words with
    #   a kept document's is threshold or more.
    def drop_near(
        documents: list[str],
        n: int,
        threshold: float,
    ) -> list[str]:
        ...
    ''',
)

DATA1_SOLUTION = source_file(
    '''
    # data1.py, a solution for comparison

    shouted = "Speak, speak!".lower()
    tidy = " ".join("a  b\\n c".split())
    words = "to be or not".split()
    pairs = {tuple(words[i : i + 2]) for i in range(3)}
    overlap = len({1, 2, 3} & {2, 3, 4}) / len({1, 2, 3} | {2, 3, 4})


''' + indented(NORMALIZE_SOLUTION) + '''


''' + indented(EXACT_SOLUTION) + '''


''' + indented(SHINGLES_SOLUTION) + '''


''' + indented(NEAR_SOLUTION) + '''
    ''',
)

PAGES = [
    "The tide came in.  The TIDE came in!",
    "The tide came in",
    "the tide, came in?",
    "Ships sail at dawn.",
    "ships sail at dawn",
    "Ships sail at noon.",
    "3 ships, 2 tides\tand 1 dawn",
]

# With shingles of 2 words, the fifth reaches exactly 0.75 with the third, which shows whether the threshold is
# reached or only passed. The last is near the second alone, so it stays once the second has gone.
NEAR_DOCUMENTS = [
    "a b c d e f",
    "a b c d e x",
    "q r s t",
    "a b c d y z",
    "Q R S T u",
    "m n o p",
    "a b c d e f g",
    "b c d e x y",
]


def check_data1(checker: Checker) -> None:
    normalize = DUPLICATES_REFERENCE["normalize"]
    check_on_inputs(
        checker=checker,
        function="normalize",
        description="lowercases, keeps letters, digits and single spaces",
        inputs=(
            ("Speak, SPEAK!\n  you   know",),
            ("I'll  tell\tthee 3 times...",),
            ("  -- O, Romeo! --  ",),
        ),
        reference=normalize,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="drop_exact",
        description="keeps the first of each normalized text, in order",
        inputs=(
            (PAGES,),
            (["b", "a", "B.", "c", "A!", "a"],),
        ),
        reference=DUPLICATES_REFERENCE["drop_exact"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="shingles",
        description="gives every run of n words",
        inputs=(
            ("To be, or not to be: that is the question.", 2),
            ("To be, or not to be: that is the question.", 4),
            ("Hi there", 3),
            ("Exactly three words", 3),
        ),
        reference=DUPLICATES_REFERENCE["shingles"],
        expect=expect_value,
    )
    shingles = DUPLICATES_REFERENCE["shingles"]
    check_on_inputs(
        checker=checker,
        function="jaccard",
        description="gives the overlap over the union",
        inputs=(
            (shingles("a b c d e f", 2), shingles("a b c d e x", 2)),
            (shingles("one two three", 2), shingles("four five six", 2)),
            (set(), set()),
            ({("x",), ("y",), ("z",)}, {("x",)}),
        ),
        reference=DUPLICATES_REFERENCE["jaccard"],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="drop_near",
        description="keeps a document unless it reaches threshold with a kept one",
        inputs=(
            (NEAR_DOCUMENTS, 2, 0.75),
            (NEAR_DOCUMENTS, 2, 0.6),
            (NEAR_DOCUMENTS, 3, 0.5),
        ),
        reference=DUPLICATES_REFERENCE["drop_near"],
        expect=expect_value,
    )


DATA1 = Exercise(
    name="data1",
    chapter=DATA_QUALITY,
    title="Duplicates",
    lesson=DATA1_LESSON,
    source=DATA1_SOURCE,
    solution=DATA1_SOLUTION,
    hints={
        "normalize": "Go through text.lower() a character at a time and keep the ones that pass isalnum() or "
        "isspace(). Join them into a string, then split it and join the words with one space.",
        "drop_exact": "Keep a set of the normalized forms seen so far. Skip a document whose form is in it, "
        "otherwise add its form and keep the document.",
        "shingles": "Split the normalized text into words. For every start from 0 to len(words) - n, take "
        "tuple(words[start : start + n]) into a set.",
        "jaccard": "Return 0.0 first when both are empty. Then len(a & b) over len(a | b).",
        "drop_near": "Keep the shingles of every kept document in a list. A new document goes when jaccard of its "
        "shingles with any of them is threshold or more.",
    },
    introduces=("shingle",),
    check=check_data1,
)


# ---------------------------------------------------------------------------
# data2
# ---------------------------------------------------------------------------

STOP_WORDS_PART = '''
STOP_WORDS = ("the", "be", "to", "of", "and", "that", "have", "with")
'''

QUALITY_SOLUTION = '''
def mean_word_length(text: str) -> float:
    words = text.split()

    if not words:
        return 0.0

    return sum(len(word) for word in words) / len(words)


def alphabetic_share(text: str) -> float:
    words = text.split()

    if not words:
        return 0.0

    return sum(any(character.isalpha() for character in word) for word in words) / len(words)


def stop_words(text: str) -> int:
    return len(set(normalize(text).split()) & set(STOP_WORDS))


def passes_quality(text: str) -> bool:
    count = len(text.split())

    if not 50 <= count <= 100_000:
        return False

    if not 3 <= mean_word_length(text) <= 10:
        return False

    if text.count("#") / count > 0.1 or text.count("...") / count > 0.1:
        return False

    return alphabetic_share(text) >= 0.8 and stop_words(text) >= 2
'''

LEAK_SOLUTION = '''
def leaked(
    test: list[str],
    train: list[str],
    n: int,
) -> list[int]:
    seen: set[tuple[str, ...]] = set()

    for document in train:
        seen |= shingles(document, n)

    return [position for position, document in enumerate(test) if shingles(document, n) & seen]
'''

FILTER_REFERENCE = namespace_of(NORMALIZE_SOLUTION + SHINGLES_SOLUTION + STOP_WORDS_PART + QUALITY_SOLUTION + LEAK_SOLUTION)


def rule_results(text: str) -> dict[str, bool]:
    """Each of Gopher's rules that passes_quality joins, on its own, for data2's table."""
    words = text.split()
    count = len(words)

    return {
        "50 to 100,000 words": 50 <= count <= 100_000,
        "a mean word length of 3 to 10": 3 <= FILTER_REFERENCE["mean_word_length"](text) <= 10,
        "no more than one # or ... per 10 words": count > 0 and text.count("#") / count <= 0.1 and text.count("...") / count <= 0.1,
        "80% of words with a letter": FILTER_REFERENCE["alphabetic_share"](text) >= 0.8,
        "2 of the stop words": FILTER_REFERENCE["stop_words"](text) >= 2,
    }


def split_speeches() -> tuple[list[str], list[str]]:
    """The training and test speeches, cut where gpt1 cut the text, 90% in, with test speeches of 13 words or more."""
    text = load_shakespeare()
    cut = int(0.9 * len(text))
    train = [block for block in text[:cut].split("\n\n") if is_speech(block)]
    test = [block for block in text[cut:].split("\n\n") if is_speech(block) and word_count(block) >= 13]

    return train, test


def measure_filters() -> dict[str, object]:
    """data2's numbers. What each quality rule drops from the play, then test speeches flagged at each n, with
    nothing planted and with 40 test speeches copied into the training text, one word in 25 swapped."""
    speeches = play_speeches()
    results: dict[str, object] = {"speeches": len(speeches)}
    rows = [rule_results(speech) for speech in speeches]

    for rule in rows[0]:
        results[rule] = sum(not row[rule] for row in rows) / len(rows)

    results["passes all"] = sum(FILTER_REFERENCE["passes_quality"](speech) for speech in speeches) / len(speeches)
    long_speeches = [speech for speech in speeches if len(speech.split()) >= 50]
    results["long speeches"] = len(long_speeches)
    results["long speeches passing"] = sum(FILTER_REFERENCE["passes_quality"](speech) for speech in long_speeches)

    train, test = split_speeches()
    generator = random.Random(1)
    vocabulary = sorted({word for speech in train for word in speech.split()})
    planted = generator.sample(range(len(test)), 40)
    leaky_train = list(train)

    for position in planted:
        host = generator.randrange(len(leaky_train))
        leaky_train[host] = leaky_train[host] + "\n" + edit_words(test[position], 25, vocabulary, generator)

    results["train speeches"] = len(train)
    results["test speeches"] = len(test)

    for n in (3, 5, 8, 13):
        clean = FILTER_REFERENCE["leaked"](test, train, n)
        dirty = FILTER_REFERENCE["leaked"](test, leaky_train, n)
        results[f"n={n}"] = {
            "flagged with nothing planted": len(clean),
            "planted found": len(set(dirty) & set(planted)),
        }

    return results


DATA2_LESSON = prose(
    """
    Rae and colleagues cleaned the web text behind Gopher with plain rules
    anyone can read. A document stays when all of these hold.

    - It has 50 to 100,000 words.
    - Its mean word length is 3 to 10 characters.
    - It has no more than one `#` and one `...` per 10 words.
    - At least 80% of its words hold a letter.
    - It holds at least 2 of the **stop words** the, be, to, of, and, that, have and with, the commonest words of English.

    They had more, on bullet points, lines ending in `...` and repeated
    lines. Here is what each rule above drops from the play's 7,097
    speeches, words split on whitespace.

    | Rule | Speeches it drops |
    |---|---|
    | 50 to 100,000 words | 84.8% |
    | A mean word length of 3 to 10 | 1 speech |
    | No more than one `#` or `...` per 10 words | none |
    | 80% of words with a letter | none |
    | 2 of the stop words | 47.7% |

    Only 15.2% pass them all. Yet each of the 1,080 speeches of 50 words or
    more passes every rule. The rules were written against web pages of
    keywords, tags and symbols, and only shortness trips the play.

    ```python
    "a #b c#".count("#")                          # 2
    any(c.isalpha() for c in "3rd")                # True
    ```

    A test set the model has read measures memory. GPT-3's authors flagged a
    test example when a run of 13 of its words, lowercased and without
    punctuation, also sat in the training data. For benchmarks of short
    examples they used shorter runs, down to 8, since shorter ones matched
    by chance.

    Here the training text is the first 90% of the play, as gpt1 cut it. The
    test is the 421 speeches of 13 words or more after it. Then 40 test
    speeches go into the training text, one word in 25 swapped.

    | n | Test speeches flagged, nothing planted | Planted speeches found |
    |---|---|---|
    | 3 | 377 of 421 | 40 of 40 |
    | 5 | 21 of 421 | 40 of 40 |
    | 8 | none | 40 of 40 |
    | 13 | none | 32 of 40 |

    Runs of 3 words match phrases the whole play shares, and runs of 5 still
    match by chance. Runs of 8 match nothing by chance here and find every
    copy. Runs of 13 miss 8 short copies, where one swapped word leaves no
    run of 13 whole.
    """,
)

DATA2_SOURCE = source_file(
    '''
    # data2.py

    hashes = "a #b c#".count("#")                    # 2
    has_letter = any(c.isalpha() for c in "3rd")       # True

    STOP_WORDS = ("the", "be", "to", "of", "and", "that", "have", "with")


    # normalize and shingles from data1.
''' + indented(NORMALIZE_SOLUTION) + '''


''' + indented(SHINGLES_SOLUTION) + '''


    # TODO: Return the mean length in characters of the words of text, split on
    #   whitespace, or 0.0 when it has none.
    def mean_word_length(text: str) -> float:
        ...


    # TODO: Return the share of the words of text, split on whitespace, that
    #   hold at least one letter, or 0.0 when it has none.
    def alphabetic_share(text: str) -> float:
        ...


    # TODO: Return how many different words of STOP_WORDS the normalized text
    #   holds.
    def stop_words(text: str) -> int:
        ...


    # TODO: Return True when text keeps every rule of the lesson. 50 to 100,000
    #   words, a mean word length of 3 to 10, counts of # and of ... each no
    #   more than 0.1 per word, at least 0.8 of words with a letter, and at
    #   least 2 stop words.
    def passes_quality(text: str) -> bool:
        ...


    # TODO: Return the positions of the test documents that share at least one
    #   shingle of n words with any training document, in order.
    def leaked(
        test: list[str],
        train: list[str],
        n: int,
    ) -> list[int]:
        ...
    ''',
)

DATA2_SOLUTION = source_file(
    '''
    # data2.py, a solution for comparison

    hashes = "a #b c#".count("#")
    has_letter = any(c.isalpha() for c in "3rd")

    STOP_WORDS = ("the", "be", "to", "of", "and", "that", "have", "with")


''' + indented(NORMALIZE_SOLUTION) + '''


''' + indented(SHINGLES_SOLUTION) + '''


''' + indented(QUALITY_SOLUTION) + '''


''' + indented(LEAK_SOLUTION) + '''
    ''',
)


def made_document(
    sentence: str,
    times: int,
) -> Described:
    """A sentence said times times, shown by the sentence and its word count, since the text runs long."""
    text = " ".join([sentence] * times)

    return Described(
        value=text,
        text=f'"{sentence}" {times} times, {len(text.split()):,} words',
    )


HARBOUR = "The ship sailed to the harbour with all of its crew and cargo."

# Each passes or breaks one rule, so a rule left out shows. The last five pass while sitting exactly on a
# threshold, 50 words, a mean length of 3, 2 stop words, 80% with a letter and one # per 10 words, so a rule
# written one step off shows too.
QUALITY_DOCUMENTS = (
    made_document(HARBOUR, 6),
    made_document(HARBOUR, 3),
    made_document(HARBOUR, 7693),
    made_document("buy cheap shoes online best price free shipping sale today", 6),
    made_document("#sun #sea #sand and the #surf with #friends", 8),
    made_document("Supercalifragilistic extraordinarily incomprehensibilities notwithstanding of the", 12),
    made_document("A b c d e f to the of and", 6),
    made_document("Wait for it... and then the... rest of it... with more...", 6),
    made_document("1234 5678 91011 the and of to", 8),
    made_document("The sailors rowed to the harbour with their heavy cargo", 5),
    made_document("Cat sat the dog ran far and hid big box", 6),
    made_document("Sailors rowed the heavy boats toward distant harbours with patience", 6),
    made_document("Sailors rowed 1234 heavy boats with 5678 the distant harbours", 6),
    made_document("Sailors rowed the heavy #boats with distant harbours and patience", 6),
)

LEAK_TRAIN = [
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael. Some years ago, never mind how long precisely.",
    "All happy families are alike.",
]

LEAK_TEST = [
    "It was the worst of times, and the age of wisdom.",
    "Some years ago I went to sea.",
    "Happy families are all alike, they say.",
    "Nothing here was ever seen before.",
    "Never mind how long, precisely!",
]


def check_data2(checker: Checker) -> None:
    for function, description in (
        ("mean_word_length", "gives the mean characters per word"),
        ("alphabetic_share", "gives the share of words with a letter"),
    ):
        check_on_inputs(
            checker=checker,
            function=function,
            description=description,
            inputs=(
                ("The tide, came in at 3!",),
                ("#1 ... 42 is it",),
                ("",),
            ),
            reference=FILTER_REFERENCE[function],
            expect=expect_close,
        )

    check_on_inputs(
        checker=checker,
        function="stop_words",
        description="counts the different stop words the text holds",
        inputs=(
            ("The ship and THE crew, with the cargo of gold.",),
            ("Buy shoes online today",),
            ("To be, or not to be: that is the question.",),
        ),
        reference=FILTER_REFERENCE["stop_words"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="passes_quality",
        description="keeps every rule at once",
        inputs=tuple((document,) for document in QUALITY_DOCUMENTS),
        reference=FILTER_REFERENCE["passes_quality"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="leaked",
        description="finds the test documents that share a run of n words with training",
        inputs=(
            (LEAK_TEST, LEAK_TRAIN, 3),
            (LEAK_TEST, LEAK_TRAIN, 4),
            (LEAK_TEST, LEAK_TRAIN, 2),
        ),
        reference=FILTER_REFERENCE["leaked"],
        expect=expect_value,
    )


DATA2 = Exercise(
    name="data2",
    chapter=DATA_QUALITY,
    title="Quality rules and leaked tests",
    lesson=DATA2_LESSON,
    source=DATA2_SOURCE,
    solution=DATA2_SOLUTION,
    hints={
        "mean_word_length": "Split on whitespace. Return 0.0 for no words, otherwise the sum of len(word) over "
        "the number of words.",
        "alphabetic_share": "For each word, any(character.isalpha() for character in word) is True or False, "
        "and summing them counts the Trues.",
        "stop_words": "Take the set of the normalized words, and the size of its overlap with set(STOP_WORDS).",
        "passes_quality": "Count the words once. Return False at the first rule that fails, using the three "
        "functions above, and True at the end.",
        "leaked": "Gather every training document's shingles into one set with |=. A test document leaked when "
        "its own shingles overlap that set.",
    },
    introduces=("stop word",),
    check=check_data2,
)


# ---------------------------------------------------------------------------
# quiz15
# ---------------------------------------------------------------------------

QUIZ15_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines moves
    from Section 12. If a step is unclear, the exercise in the table shows it.

    | Function | Moves | Shown in |
    |---|---|---|
    | `budget_tokens` | a model's parameters counted once each, then 6 FLOPs per parameter per token | cost1, cost2 |
    | `linear_bound` | a Linear's FLOPs and the bytes it reads and writes, then whichever takes longer | cost1, profile2 |
    | `cache_bytes_for` | a key and a value per layer for every token, times the bytes of one number | kvcache2, precision1 |
    | `roundtrip_error` | a scale per row, rounded to whole numbers and back, then the relative error | quant1, quant2 |
    | `fit_and_predict` | a straight line fitted to the logs, then the power law it gives, followed out | scaling1 |
    | `near_duplicate` | each text's word shingles, then their Jaccard similarity against a threshold | data1 |
    """,
)

QUIZ15_SOURCE = source_file(
    '''
    # quiz15.py
    #
    # No examples and no new tools. Each function combines moves from Section 12.

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor


    # TODO: Return how many tokens a training budget of flops buys for model, at
    #   6 FLOPs per parameter per token, rounded down to a whole number. Count
    #   each parameter of model once.
    def budget_tokens(
        model: nn.Module,
        flops: float,
    ) -> int:
        ...


    # TODO: Return "memory" or "compute", whichever takes longer for linear's
    #   forward pass on rows rows. Each multiply and add is 2 FLOPs. It reads
    #   its input and its weight and writes its output, bytes_per_number bytes
    #   a number. Leave the bias out. Return "compute" when the two are equal.
    def linear_bound(
        linear: nn.Linear,
        rows: int,
        bytes_per_number: int,
        flops_per_second: float,
        bytes_per_second: float,
    ) -> str:
        ...


    # TODO: Return the bytes a KV cache holds for one text of tokens tokens. Each
    #   of layers layers keeps a key and a value of width numbers for every
    #   token, each number of type dtype.
    def cache_bytes_for(
        layers: int,
        width: int,
        tokens: int,
        dtype: torch.dtype,
    ) -> int:
        ...


    # TODO: Quantize weights to bits bits with absmax and a scale per row, turn
    #   them back into floats, and return the norm of the difference over the
    #   norm of weights, as a plain number.
    def roundtrip_error(
        weights: Tensor,
        bits: int,
    ) -> float:
        ...


    # TODO: Fit loss = scale * n ** -alpha by least squares on the logs, and
    #   return the loss it gives at target, as a plain number.
    def fit_and_predict(
        n: Tensor,
        loss: Tensor,
        target: float,
    ) -> float:
        ...


    # TODO: Return whether a and b are near duplicates, which is when the
    #   Jaccard similarity of their sets of n word shingles is at least
    #   threshold. Take the words from text.lower().split(), and take the
    #   similarity as 0 when both sets are empty.
    def near_duplicate(
        a: str,
        b: str,
        n: int,
        threshold: float,
    ) -> bool:
        ...
    ''',
)

QUIZ15_PARTS = '''
def budget_tokens(
    model: nn.Module,
    flops: float,
) -> int:
    parameters = sum(parameter.numel() for parameter in model.parameters())

    return int(flops // (6 * parameters))


def linear_bound(
    linear: nn.Linear,
    rows: int,
    bytes_per_number: int,
    flops_per_second: float,
    bytes_per_second: float,
) -> str:
    inputs, outputs = linear.in_features, linear.out_features
    flops = 2 * rows * inputs * outputs
    numbers = rows * inputs + inputs * outputs + rows * outputs

    if numbers * bytes_per_number / bytes_per_second > flops / flops_per_second:
        return "memory"

    return "compute"


def cache_bytes_for(
    layers: int,
    width: int,
    tokens: int,
    dtype: torch.dtype,
) -> int:
    return 2 * layers * tokens * width * torch.zeros(0, dtype=dtype).element_size()


def roundtrip_error(
    weights: Tensor,
    bits: int,
) -> float:
    levels = 2 ** (bits - 1) - 1
    scales = weights.abs().amax(dim=1, keepdim=True) / levels
    restored = (weights / scales).round() * scales

    return ((weights - restored).norm() / weights.norm()).item()


def fit_and_predict(
    n: Tensor,
    loss: Tensor,
    target: float,
) -> float:
    with_ones = torch.stack([torch.log(n), torch.ones_like(n)], dim=1)
    slope, intercept = torch.linalg.lstsq(with_ones, torch.log(loss)).solution.tolist()

    return math.exp(intercept) * target**slope


def near_duplicate(
    a: str,
    b: str,
    n: int,
    threshold: float,
) -> bool:
    sets = []

    for text in (a, b):
        words = text.lower().split()
        sets.append({tuple(words[start : start + n]) for start in range(len(words) - n + 1)})

    first, second = sets
    union = first | second
    similarity = len(first & second) / len(union) if union else 0.0

    return similarity >= threshold
'''

QUIZ15_SOLUTION = source_file(
    '''
    # quiz15.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor


''' + indented(QUIZ15_PARTS) + '''
    ''',
)

QUIZ15_REFERENCE = namespace_of("import math\nimport torch\nimport torch.nn as nn\nfrom torch import Tensor\n" + QUIZ15_PARTS)


def seeded_layers(
    seed: int,
    *layers: nn.Module,
) -> nn.Module:
    torch.manual_seed(seed)

    return nn.Sequential(*layers)


def rows_of_scales(
    seed: int,
    scales: list[float],
    columns: int,
) -> Tensor:
    """Weights whose rows run at different sizes, so one scale for the whole tensor shows."""
    weights = torch.randn(len(scales), columns, generator=torch.Generator().manual_seed(seed))

    return weights * torch.tensor(scales).unsqueeze(1)


def power_law_points(
    seed: int,
    scale: float,
    alpha: float,
) -> tuple[Tensor, Tensor]:
    n = torch.tensor([1e3, 3e3, 1e4, 3e4, 1e5])
    noise = 1 + 0.02 * torch.randn(len(n), generator=torch.Generator().manual_seed(seed))

    return n, scale * n**-alpha * noise


def check_quiz15(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="budget_tokens",
        description="divides the FLOPs by 6 per parameter per token",
        inputs=(
            (Described(seeded_layers(990, nn.Linear(10, 5)), "nn.Linear(10, 5)"), 6000.0),
            (Described(seeded_layers(991, nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)), "a Linear from 4 to 8, a ReLU, a Linear from 8 to 2"), 1e6),
            (Described(tied_pair(), "an Embedding(10, 4) and a Linear sharing its weight"), 5000.0),
        ),
        reference=QUIZ15_REFERENCE["budget_tokens"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="linear_bound",
        description="compares the time for the FLOPs with the time for the bytes",
        inputs=(
            (Described(nn.Linear(512, 512), "nn.Linear(512, 512)"), 1, 4, 1e12, 1e11),
            (Described(nn.Linear(64, 64), "nn.Linear(64, 64)"), 30, 4, 1e12, 1e11),
            (Described(nn.Linear(256, 256), "nn.Linear(256, 256)"), 32, 4, 1e12, 1e11),
            (Described(nn.Linear(512, 512), "nn.Linear(512, 512)"), 4096, 4, 1e12, 1e11),
        ),
        reference=QUIZ15_REFERENCE["linear_bound"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="cache_bytes_for",
        description="counts a key and a value per layer per token",
        inputs=(
            (4, 128, 128, torch.float32),
            (8, 512, 1024, torch.float16),
            (2, 64, 10, torch.bfloat16),
        ),
        reference=QUIZ15_REFERENCE["cache_bytes_for"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="roundtrip_error",
        description="quantizes with a scale per row and measures what is lost",
        inputs=(
            (rows_of_scales(992, [0.1, 1.0, 5.0], 6), 8),
            (rows_of_scales(993, [2.0, 0.05, 0.5, 1.0], 5), 4),
        ),
        reference=QUIZ15_REFERENCE["roundtrip_error"],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="fit_and_predict",
        description="fits a power law on the logs and follows it out",
        inputs=(
            (*power_law_points(994, 6.0, 0.12), 1e7),
            (*power_law_points(995, 2.5, 0.3), 1e6),
        ),
        reference=QUIZ15_REFERENCE["fit_and_predict"],
        expect=lambda call, actual, expected: expect_relative(
            call=call,
            actual=actual,
            expected=expected,
            share=1e-4,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="near_duplicate",
        description="compares the shingles' Jaccard similarity with the threshold",
        inputs=(
            ("the quick brown fox jumps over the lazy dog", "The quick brown fox jumps over the lazy cat", 2, 0.7),
            ("the quick brown fox jumps over the lazy dog", "the quick brown fox jumps over the lazy cat", 3, 0.8),
            ("x y z", "x y w", 1, 0.5),
            ("to be", "or not", 3, 0.5),
        ),
        reference=QUIZ15_REFERENCE["near_duplicate"],
        expect=expect_value,
    )


QUIZ15 = Exercise(
    name="quiz15",
    chapter=QUIZZES,
    title="Systems and inference, start to finish",
    lesson=QUIZ15_LESSON,
    source=QUIZ15_SOURCE,
    solution=QUIZ15_SOLUTION,
    hints={
        "budget_tokens": "model.parameters() lists a shared tensor once. Add up numel(), then floor divide flops by 6 "
        "times that count.",
        "linear_bound": "The FLOPs are 2 * rows * in_features * out_features. The numbers moved are the input, the "
        "weight and the output added up. Compare bytes over bytes_per_second with FLOPs over flops_per_second.",
        "cache_bytes_for": "2 for the key and the value, times layers, tokens and width, times the element_size() of "
        "an empty tensor of that dtype.",
        "roundtrip_error": "Each row's largest absolute value over 2 ** (bits - 1) - 1 is its scale. Divide, round, "
        "multiply back, then compare the norms.",
        "fit_and_predict": "Stack log(n) with a column of ones and solve for log(loss) with torch.linalg.lstsq. The "
        "slope is minus alpha and the exponent of the intercept is the scale.",
        "near_duplicate": "Build a set of word tuples for each text. The similarity is the size of the overlap over "
        "the size of the union.",
    },
    introduces=(),
    check=check_quiz15,
)
