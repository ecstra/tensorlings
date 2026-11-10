"""58_bpe, text as UTF-8 bytes, byte pair encoding trained, encoded and decoded as in minbpe, and why it beats characters and words."""

from __future__ import annotations

import re
import textwrap
import time
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import QUIZZES, Chapter, Exercise
from ..data import load_shakespeare
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
)
from ..text import prose, source_file

BPE = Chapter(
    folder="58_bpe",
    title="Byte pair encoding",
    readme=source_file(
        """
        # Byte pair encoding

        This chapter is about how text becomes tokens. Text starts as UTF-8
        bytes, which give the first 256 tokens. Byte pair encoding then finds
        the most common pair of neighbouring tokens and merges it into a new
        token, again and again. You build the kind of tokenizer GPT-2 uses, in
        the style of Karpathy's minbpe. Then you measure why it works better
        than characters or words, and where it behaves strangely.

        ## Further reading

        - [Karpathy, minbpe](https://github.com/karpathy/minbpe)
        - [Karpathy, let's build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
        - [Sennrich, Haddow and Birch, rare words with subword units](https://arxiv.org/abs/1508.07909)
        - [OpenAI, tiktoken](https://github.com/openai/tiktoken)
        """,
    ),
)


def indented_part(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


# ---------------------------------------------------------------------------
# bpe1
# ---------------------------------------------------------------------------

BPE1_LESSON = prose(
    """
    A model reads tokens, so text has to be turned into token ids first. A
    computer already stores text as numbers, bytes from 0 to 255, in an
    encoding called UTF-8.

    ```python
    list("hi".encode("utf-8"))           # [104, 105]
    list("é".encode("utf-8"))            # [195, 169]
    bytes([104, 105]).decode("utf-8")    # 'hi'
    ```

    An English letter takes one byte. Other characters take two to four
    bytes. For example, an accented letter takes 2 bytes and an emoji takes 4.

    The 256 byte values make a first vocabulary of tokens. Any text can be
    written with them, so a model never meets text it has no token for.
    GPT-2's tokenizer starts from these 256 tokens, and every merge in bpe2
    and bpe3 builds on them.

    Not every list of bytes is valid UTF-8. A model might write the first
    byte of a 2 byte character and then stop. Then `.decode("utf-8")` raises
    an error. With `errors="replace"`, decode puts the character � in place
    of the broken bytes instead. minbpe's decode does this.

    ```python
    bytes([195]).decode("utf-8", errors="replace")    # '\\ufffd', the character �
    ```
    """,
)

BPE1_SOURCE = source_file(
    '''
    # bpe1.py

    ids = list("hé".encode("utf-8"))      # [104, 195, 169]
    back = bytes(ids).decode("utf-8")     # 'hé'
    sizes = [len("€".encode("utf-8"))]    # [3]


    # TODO: Return the UTF-8 bytes of text as a list of ints from 0 to 255.
    #   Example: to_bytes("hé") returns [104, 195, 169].
    def to_bytes(text: str) -> list[int]:
        ...


    # TODO: Decode the list of bytes ids as UTF-8 and return the text. Put �
    #   in place of any bytes that are not valid UTF-8.
    #   Example: from_bytes([104, 105, 226, 130]) returns 'hi�'.
    def from_bytes(ids: list[int]) -> str:
        ...


    # TODO: Return a list with the number of UTF-8 bytes each character of text
    #   takes, in order.
    #   Example: byte_lengths("aé€😀") returns [1, 2, 3, 4].
    def byte_lengths(text: str) -> list[int]:
        ...


    # TODO: Return the average number of UTF-8 bytes per character of text,
    #   which is the total number of bytes divided by the number of characters.
    #   Example: bytes_per_character("aé€😀") returns 2.5.
    def bytes_per_character(text: str) -> float:
        ...
    ''',
)

BPE1_SOLUTION = source_file(
    '''
    # bpe1.py, a solution for comparison

    ids = list("hé".encode("utf-8"))
    back = bytes(ids).decode("utf-8")
    sizes = [len("€".encode("utf-8"))]


    def to_bytes(text: str) -> list[int]:
        return list(text.encode("utf-8"))


    def from_bytes(ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


    def byte_lengths(text: str) -> list[int]:
        return [len(character.encode("utf-8")) for character in text]


    def bytes_per_character(text: str) -> float:
        return len(text.encode("utf-8")) / len(text)
    ''',
)


def check_bpe1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="to_bytes",
        description="returns the UTF-8 bytes",
        inputs=(
            ("tensor",),
            ("Ünïcödé 😀",),
        ),
        reference=lambda text: list(text.encode("utf-8")),
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="from_bytes",
        description="decodes the bytes, with � for invalid bytes",
        inputs=(
            ([104, 195, 169, 33],),
            ([240, 159, 152, 128, 32, 104, 105],),
            ([104, 105, 226, 130],),
        ),
        reference=lambda ids: bytes(ids).decode("utf-8", errors="replace"),
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="byte_lengths",
        description="returns the number of bytes in each character",
        inputs=(
            ("aé€😀",),
            ("plain",),
        ),
        reference=lambda text: [len(character.encode("utf-8")) for character in text],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="bytes_per_character",
        description="returns the average bytes per character",
        inputs=(
            ("aé€😀",),
            ("hello",),
            ("naïve café",),
        ),
        reference=lambda text: len(text.encode("utf-8")) / len(text),
        expect=expect_close,
    )


BPE1 = Exercise(
    name="bpe1",
    chapter=BPE,
    title="Text as bytes",
    lesson=BPE1_LESSON,
    source=BPE1_SOURCE,
    solution=BPE1_SOLUTION,
    hints={
        "to_bytes": "text.encode(\"utf-8\") gives the bytes, and list turns them into ints.",
        "from_bytes": "bytes(ids) turns the ints back into bytes. Decode them with errors=\"replace\".",
        "byte_lengths": "For each character of text, take the length of that character's .encode(\"utf-8\").",
        "bytes_per_character": "Divide the number of bytes in the whole text by len(text).",
    },
    introduces=(),
    check=check_bpe1,
)


# ---------------------------------------------------------------------------
# bpe2
# ---------------------------------------------------------------------------

# bpe2's pieces, which bpe3 and bpe4 are handed, written once so the files never drift apart.
PAIR_PARTS = '''
def pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    counts = {}

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def most_common_pair(ids: list[int]) -> tuple[int, int]:
    counts = pair_counts(ids)

    return max(counts, key=counts.get)


def merge(
    ids: list[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    merged = []
    position = 0

    while position < len(ids):
        if position + 1 < len(ids) and (ids[position], ids[position + 1]) == pair:
            merged.append(new_id)
            position += 2
            continue

        merged.append(ids[position])
        position += 1

    return merged
'''

BPE2_LESSON = prose(
    """
    **Byte pair encoding**, BPE, grows the vocabulary one token at a time.
    Count every pair of neighbouring tokens, take the most common pair, and
    give it a new id. Then put that id in place of the pair everywhere it
    appears. This step is called a merge. Sennrich and colleagues brought BPE
    to neural translation in 2016, and GPT-2 applies it to bytes.

    ```python
    ids = [97, 97, 97, 98]          # 'aaab'
    list(zip(ids, ids[1:]))         # [(97, 97), (97, 97), (97, 98)]
    ```

    `zip(ids, ids[1:])` goes through every pair of neighbours. Here (97, 97)
    appears twice, so it becomes token 256. Merging it turns the ids into
    `[256, 97, 98]`.

    The replacement goes from left to right, and replacements never overlap.
    In `aaa` the pair `aa` is replaced only once, at the start, and the last
    `a` stays as it was.

    Each merge makes the list of ids shorter and the vocabulary one token
    bigger. Tiny Shakespeare is the text the GPT project trains on. On its
    first 100,000 characters, the first merges are `e `, `th`, `t `, `s ` and
    `ou`. These are the most common pairs of characters in English, counting
    spaces.

    Two pairs can tie for the highest count. Then the pair that appeared
    first wins. Python's `max` over a dict does exactly that. A dict keeps its
    keys in the order they were added, and `max` returns the first key with
    the highest score.

    ```python
    scores = {"x": 2, "y": 2}
    max(scores, key=scores.get)     # 'x'
    ```
    """,
)

BPE2_SOURCE = source_file(
    '''
    # bpe2.py

    ids = [97, 97, 97, 98]
    neighbours = list(zip(ids, ids[1:]))     # [(97, 97), (97, 97), (97, 98)]
    tally = {"x": 2, "y": 3}
    best = max(tally, key=tally.get)         # 'y'


    # TODO: Count how many times each pair of neighbouring ids appears in ids.
    #   Return a dict from each pair to its count, with the pairs in the order
    #   they first appear.
    #   Example: pair_counts([1, 2, 1, 2, 3]) returns
    #   {(1, 2): 2, (2, 1): 1, (2, 3): 1}.
    def pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
        ...


    # TODO: Return the pair of neighbouring ids that appears most often in ids.
    #   If several pairs tie, return the one that appears first.
    #   Example: most_common_pair([4, 5, 6, 4, 5, 6]) returns (4, 5).
    def most_common_pair(ids: list[int]) -> tuple[int, int]:
        ...


    # TODO: Return a new list where every occurrence of pair in ids is replaced
    #   by new_id. Go from left to right, so the replacements never overlap.
    #   Example: merge([97, 97, 97, 98], (97, 97), 256) returns [256, 97, 98].
    def merge(
        ids: list[int],
        pair: tuple[int, int],
        new_id: int,
    ) -> list[int]:
        ...
    ''',
)

BPE2_SOLUTION = source_file(
    '''
    # bpe2.py, a solution for comparison

    ids = [97, 97, 97, 98]
    neighbours = list(zip(ids, ids[1:]))
    tally = {"x": 2, "y": 3}
    best = max(tally, key=tally.get)


''' + indented_part(PAIR_PARTS),
)


def bpe_namespace(parts: str) -> dict[str, object]:
    """The given pieces run once, so the checks compare against exactly the code the learner's file holds."""
    namespace: dict[str, object] = {"re": re}
    exec(parts, namespace)

    return namespace


PAIRS = bpe_namespace(PAIR_PARTS)


def check_bpe2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="pair_counts",
        description="counts every pair of neighbours",
        inputs=(
            ([1, 2, 1, 2, 3],),
            ([5, 5, 5, 5],),
            ([7],),
        ),
        reference=PAIRS["pair_counts"],
        expect=lambda call, actual, expected: expect_value(
            call=call,
            actual=list(actual.items()) if isinstance(actual, dict) else actual,
            expected=list(expected.items()),
        ),
    )
    check_on_inputs(
        checker=checker,
        function="most_common_pair",
        description="returns the most common pair, the first one on a tie",
        inputs=(
            ([1, 2, 3, 1, 2, 3, 2, 3],),
            ([4, 5, 6, 4, 5, 6],),
            ([9, 8, 7, 7, 7],),
        ),
        reference=PAIRS["most_common_pair"],
        expect=lambda call, actual, expected: expect_value(
            call=call,
            actual=tuple(actual) if isinstance(actual, (tuple, list)) else actual,
            expected=expected,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="merge",
        description="replaces the pair left to right without overlaps",
        inputs=(
            ([97, 97, 97, 98], (97, 97), 256),
            ([1, 2, 3, 1, 2], (1, 2), 300),
            ([5, 5, 5, 5, 5], (5, 5), 260),
            ([5, 6], (6, 5), 9),
        ),
        reference=PAIRS["merge"],
        expect=expect_value,
    )


BPE2 = Exercise(
    name="bpe2",
    chapter=BPE,
    title="Counting pairs and merging one",
    lesson=BPE2_LESSON,
    source=BPE2_SOURCE,
    solution=BPE2_SOLUTION,
    hints={
        "pair_counts": "Loop over zip(ids, ids[1:]) and add 1 to counts.get(pair, 0) for each pair.",
        "most_common_pair": "Call max on the dict from pair_counts(ids), with key set to that dict's .get method.",
        "merge": "Move a position along ids with a while loop. When ids[position] and the next id form pair, append "
        "new_id and move forward 2. Otherwise append ids[position] and move forward 1.",
    },
    introduces=(),
    check=check_bpe2,
)


# ---------------------------------------------------------------------------
# bpe3
# ---------------------------------------------------------------------------

# bpe3's tokenizer, which bpe4 is handed and the GPT project points back to.
TOKENIZER_PARTS = '''
def train(
    text: str,
    vocab_size: int,
) -> dict[tuple[int, int], int]:
    ids = list(text.encode("utf-8"))
    merges = {}

    for new_id in range(256, vocab_size):
        if len(ids) < 2:
            break

        pair = most_common_pair(ids)
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id

    return merges


def vocabulary(merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
    tokens = {index: bytes([index]) for index in range(256)}

    for (first, second), new_id in merges.items():
        tokens[new_id] = tokens[first] + tokens[second]

    return tokens


def encode(
    text: str,
    merges: dict[tuple[int, int], int],
) -> list[int]:
    ids = list(text.encode("utf-8"))

    while len(ids) >= 2:
        pair = min(pair_counts(ids), key=lambda pair: merges.get(pair, float("inf")))

        if pair not in merges:
            break

        ids = merge(ids, pair, merges[pair])

    return ids


def decode(
    ids: list[int],
    merges: dict[tuple[int, int], int],
) -> str:
    tokens = vocabulary(merges)

    return b"".join(tokens[index] for index in ids).decode("utf-8", errors="replace")
'''

BPE3_LESSON = prose(
    """
    Training repeats the merge step from bpe2. Start from the text's bytes.
    For each new id from 256 up to the vocabulary size, merge the most common
    pair and record it. The resulting dict, from each merged pair to its new
    id, is the whole tokenizer.

    Decoding needs the bytes that each token stands for. Token 256 might
    stand for the pair (101, 32), which is the bytes of `e `. A token built
    from two other tokens stands for their bytes joined together. So build
    this table of bytes in the order the merges were made. Then both halves
    of each pair already have their bytes when you need them.

    Encoding new text has to apply the merges in the order training made
    them. Token 300 may be built from token 270, so merge 300 cannot be
    applied before merge 270. At each step, find the pair in the ids whose
    merge came earliest. That is the pair with the smallest new id. Apply its
    merge, and stop when no pair in the ids has a merge.

    ```python
    merges = {(104, 105): 256}
    min([(105, 33), (104, 105)], key=lambda pair: merges.get(pair, float("inf")))    # (104, 105)
    ```

    A pair with no merge gets the key infinity, so `min` picks it only when no
    pair can merge. In that case `pair not in merges` is true and ends the
    loop.

    `decode(encode(text, merges), merges)` has to give back exactly `text`,
    for any text at all. This round trip is the test of a tokenizer. Because
    every token is made of bytes, it works even for text that looks nothing
    like the training text.

    This `train` is plain Python, so it is slow on big texts. Learning 768
    merges on 100,000 characters of tiny Shakespeare took 4.35 to 4.45
    seconds on this Mac's CPU, an Apple M5. tiktoken, the tokenizer library
    OpenAI's models use, encodes far faster.
    """,
)

BPE3_SOURCE = source_file(
    '''
    # bpe3.py

    merges = {(104, 105): 256}
    earliest = min([(105, 33), (104, 105)], key=lambda pair: merges.get(pair, float("inf")))    # (104, 105)
    joined = b"h" + b"i"    # b'hi'


    # The functions from bpe2.
''' + indented_part(PAIR_PARTS) + '''


    # TODO: Train a tokenizer on text and return its merges, a dict from each
    #   merged pair to its new id. Start from the UTF-8 bytes of text. For each
    #   new id from 256 up to but not including vocab_size, find the most
    #   common pair, merge it into the ids, and record it. Stop early when
    #   fewer than 2 ids are left.
    #   Example: train("aaabdaaabac", 259) returns
    #   {(97, 97): 256, (256, 97): 257, (257, 98): 258}.
    def train(
        text: str,
        vocab_size: int,
    ) -> dict[tuple[int, int], int]:
        ...


    # TODO: Return the bytes each token stands for, as a dict from each id to
    #   its bytes. Each id from 0 to 255 stands for that single byte. Each
    #   merged id stands for the bytes of its two halves joined together. Fill
    #   them in the order of merges.
    #   Example: vocabulary({(104, 105): 256})[256] is b'hi'.
    def vocabulary(merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
        ...


    # TODO: Finish the line that sets pair. It should pick the pair of
    #   neighbours in ids whose merge came earliest, which is the pair with the
    #   smallest new id in merges.
    #   Example: encode("hi!", {(104, 105): 256}) returns [256, 33].
    def encode(
        text: str,
        merges: dict[tuple[int, int], int],
    ) -> list[int]:
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            pair = ...

            if pair not in merges:
                break

            ids = merge(ids, pair, merges[pair])

        return ids


    # TODO: Return the text that ids spell. Join the bytes of every token, then
    #   decode them as UTF-8, with � in place of any bytes that are not valid.
    #   Example: decode([256, 33], {(104, 105): 256}) returns 'hi!'.
    def decode(
        ids: list[int],
        merges: dict[tuple[int, int], int],
    ) -> str:
        ...
    ''',
)

BPE3_SOLUTION = source_file(
    '''
    # bpe3.py, a solution for comparison

    merges = {(104, 105): 256}
    earliest = min([(105, 33), (104, 105)], key=lambda pair: merges.get(pair, float("inf")))
    joined = b"h" + b"i"


''' + indented_part(PAIR_PARTS) + '''


''' + indented_part(TOKENIZER_PARTS),
)

TOKENIZER = bpe_namespace(PAIR_PARTS + TOKENIZER_PARTS)

TRAINING_TEXT = "the cat sat on the mat, and the rat sat on the hat that the cat had"


def expect_vocabulary(
    call: str,
    actual: object,
    expected: dict[int, bytes],
) -> None:
    if not isinstance(actual, dict):
        raise Mismatch(f"{call} should return a dict from each id to its bytes.")

    for index, wanted in expected.items():
        if actual.get(index) != wanted:
            raise Mismatch(f"In {call}, id {index} is {actual.get(index)!r}, but it should be {wanted!r}.")

    if len(actual) != len(expected):
        raise Mismatch(f"{call} has {len(actual)} ids, but it should have {len(expected)}, which is 256 bytes plus one per merge.")


def check_round_trip(found: object) -> None:
    namespace = found.__globals__
    merges = TOKENIZER["train"](TRAINING_TEXT, 290)

    for text in ("the hat sat on the cat", "Ünïcödé, 😀 and ∑ too", ""):
        ids = namespace["encode"](text, merges)
        expect_value(
            call=f"decode(encode({text!r}, merges), merges)",
            actual=found(ids, merges),
            expected=text,
        )


def check_bpe3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="train",
        description="learns the merges, most common pair first",
        inputs=(
            ("aaabdaaabac", 259),
            ("the cat in the hat sat on the mat", 270),
            ("ab", 300),
        ),
        reference=TOKENIZER["train"],
        expect=lambda call, actual, expected: expect_value(
            call=call,
            actual=list(actual.items()) if isinstance(actual, dict) else actual,
            expected=list(expected.items()),
        ),
    )
    check_on_inputs(
        checker=checker,
        function="vocabulary",
        description="gives the bytes of every token",
        inputs=(
            ({(97, 97): 256, (256, 98): 257, (257, 32): 258},),
            ({(104, 105): 256},),
        ),
        reference=TOKENIZER["vocabulary"],
        expect=expect_vocabulary,
    )
    shakespeare_like = Described(
        TOKENIZER["train"](TRAINING_TEXT, 290),
        "the merges of train on a line about cats and hats, to 290",
    )
    check_on_inputs(
        checker=checker,
        function="encode",
        description="applies the merges in the order they were made",
        inputs=(
            ("abc", {(97, 98): 257, (98, 99): 256}),
            ("the cat sat on the hat", shakespeare_like),
            ("aaabdaaabac", {(97, 97): 256, (256, 97): 257, (257, 98): 258}),
        ),
        reference=TOKENIZER["encode"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="decode",
        description="turns the ids back into text",
        inputs=(
            ([258, 100, 258, 97, 99], {(97, 97): 256, (256, 97): 257, (257, 98): 258}),
            ([104, 105, 226, 130], {}),
        ),
        reference=TOKENIZER["decode"],
        expect=expect_value,
    )
    checker.test(
        function="decode",
        description="returns any text that encode was given",
        run=check_round_trip,
    )


BPE3 = Exercise(
    name="bpe3",
    chapter=BPE,
    title="Training, encoding and decoding",
    lesson=BPE3_LESSON,
    source=BPE3_SOURCE,
    solution=BPE3_SOLUTION,
    hints={
        "train": "Start from list(text.encode(\"utf-8\")), as encode does. Loop new_id over range(256, vocab_size), "
        "and break once len(ids) < 2. Otherwise find most_common_pair, merge it into ids, and record new_id under that "
        "pair in merges.",
        "vocabulary": "Start from {index: bytes([index]) for index in range(256)}. Then for each (first, second), "
        "new_id in merges.items(), join the bytes of first and second.",
        "encode": "Use min over pair_counts(ids). Its key looks up each pair in merges and gives float(\"inf\") for "
        "a missing pair, as in the lesson's example.",
        "decode": "Get each id's bytes from vocabulary(merges) and join them with b\"\".join. Then call "
        ".decode(\"utf-8\", errors=\"replace\").",
    },
    introduces=(),
    check=check_bpe3,
)


# ---------------------------------------------------------------------------
# bpe4
# ---------------------------------------------------------------------------

# A plain ASCII version of GPT-2's split: words and runs of digits each with the space before them,
# other symbols, and runs of whitespace.
SPLIT_PART = '''
SPLIT = r" ?[A-Za-z]+| ?[0-9]+| ?[^\\sA-Za-z0-9]+|\\s+"
'''

MEASURE_PARTS = '''
def bytes_per_token(
    text: str,
    merges: dict[tuple[int, int], int],
) -> float:
    return len(text.encode("utf-8")) / len(encode(text, merges))


def unseen_share(
    train_words: list[str],
    test_words: list[str],
) -> float:
    seen = set(train_words)

    return sum(word not in seen for word in test_words) / len(test_words)


def pieces(
    text: str,
    merges: dict[tuple[int, int], int],
) -> list[str]:
    tokens = vocabulary(merges)

    return [tokens[index].decode("utf-8", errors="replace") for index in encode(text, merges)]


def encode_chunks(
    text: str,
    merges: dict[tuple[int, int], int],
) -> list[int]:
    return [index for chunk in re.findall(SPLIT, text) for index in encode(chunk, merges)]
'''

BPE4_LESSON = prose(
    """
    Why use bytes and merges instead of characters or words? Here is a
    comparison on tiny Shakespeare. Each tokenizer learned from its first
    100,000 characters and was measured on the next 50,000. A word here is a run of letters, as
    `re.findall(r"[A-Za-z]+", text)` finds them.

    | Tokens | Vocabulary | Tokens for the 50,000 characters |
    |---|---|---|
    | bytes | 256 | 50,000 |
    | BPE, 256 merges | 512 | 24,446, 2.05 bytes each |
    | BPE, 768 merges | 1,024 | 18,643, 2.68 bytes each |
    | words | 3,290 | 9,307 words, 11.7% of them unknown |

    Bytes make the longest sequences, and a model's cost grows with the
    length. The cost of attention, which compares every pair of positions,
    grows with the square of the length. Words make the shortest sequences.
    But 11.7% of the words in the later text never appeared in the first
    part, so a word vocabulary has no id for them. All of tiny Shakespeare
    holds 13,320 different words, and real text holds far more.

    BPE sits in between. Common words become single tokens, and rare words
    split into pieces. No text is ever unknown, because the 256 byte tokens
    are always there.

    BPE has quirks. With 768 merges, `the` is one token, but `The` is two,
    `T` and `he`. And `the king` splits as `the `, `k`, `ing`. So the tokens
    of a word change with its capitalization and with the spaces around it.

    Numbers split according to which pairs of digits were common in training,
    not by place value. After training on the numbers 0 to 1999, ` 1999`
    becomes ` 1`, `99`, `9`, and ` 123456` becomes ` 1`, `23`, `45`, `6`.
    That is part of why arithmetic is hard for language models.

    GPT-2 first uses a regular expression to split text into chunks. Each
    chunk is a word, a run of digits or a run of other symbols, together with
    the space before it. Merges then happen only inside each chunk. GPT-2's
    vocabulary has 50,257 tokens, which are the 256 bytes, 50,000 merges, and
    one token that marks the end of a text. GPT-4's tokenizer, cl100k_base,
    has about 100,000 tokens and splits numbers into runs of at most 3 digits.
    """,
)

BPE4_SOURCE = source_file(
    '''
    # bpe4.py

    import re

    words = re.findall(r"[A-Za-z]+", "To be, or not to be")    # ['To', 'be', 'or', 'not', 'to', 'be']
    unique = set(words)                                         # 5 words, since To and to differ


    # The functions from bpe2 and the tokenizer from bpe3.
''' + indented_part(PAIR_PARTS) + '''


''' + indented_part(TOKENIZER_PARTS) + '''


''' + indented_part(SPLIT_PART) + '''


    # TODO: Return the average number of UTF-8 bytes per token when text is
    #   encoded with merges. That is the number of bytes in text divided by the
    #   number of ids that encode returns.
    def bytes_per_token(
        text: str,
        merges: dict[tuple[int, int], int],
    ) -> float:
        ...


    # TODO: Return the fraction of test_words that never appear in
    #   train_words. A vocabulary built from the training words would have no
    #   id for these words.
    #   Example: unseen_share(["a", "b", "c"], ["a", "d", "d", "b"]) returns 0.5.
    def unseen_share(
        train_words: list[str],
        test_words: list[str],
    ) -> float:
        ...


    # TODO: Encode text with merges and return the text of each token, in
    #   order, as a list of strings. Decode each token's bytes on its own, with
    #   � in place of bytes that are not valid UTF-8 by themselves.
    #   Example: pieces("é", {}) returns ['�', '�'].
    def pieces(
        text: str,
        merges: dict[tuple[int, int], int],
    ) -> list[str]:
        ...


    # TODO: Split text into chunks with re.findall(SPLIT, text), encode each
    #   chunk on its own, and return all the ids in one list, in order. This
    #   way no merge joins two chunks.
    def encode_chunks(
        text: str,
        merges: dict[tuple[int, int], int],
    ) -> list[int]:
        ...
    ''',
)

BPE4_SOLUTION = source_file(
    '''
    # bpe4.py, a solution for comparison

    import re

    words = re.findall(r"[A-Za-z]+", "To be, or not to be")
    unique = set(words)


''' + indented_part(PAIR_PARTS) + '''


''' + indented_part(TOKENIZER_PARTS) + '''


''' + indented_part(SPLIT_PART) + '''


''' + indented_part(MEASURE_PARTS),
)

MEASURE = bpe_namespace(PAIR_PARTS + TOKENIZER_PARTS + SPLIT_PART + MEASURE_PARTS)

CAT_MERGES = Described(
    MEASURE["train"](TRAINING_TEXT, 300),
    "the merges of train on a line about cats and hats, to 300",
)


def check_bpe4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="bytes_per_token",
        description="gives the bytes per token",
        inputs=(
            ("the cat sat on the hat", CAT_MERGES),
            ("Ünïcödé cat", CAT_MERGES),
        ),
        reference=MEASURE["bytes_per_token"],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="unseen_share",
        description="returns the fraction of test words never seen in training",
        inputs=(
            (["a", "b", "c"], ["a", "d", "d", "b"]),
            (["to", "be", "or"], ["To", "be", "to", "not"]),
        ),
        reference=MEASURE["unseen_share"],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="pieces",
        description="gives the text of each token",
        inputs=(
            ("the rat had a hat", CAT_MERGES),
            ("é", Described({(195, 169): 256}, "{(195, 169): 256}")),
            ("é", Described({}, "no merges")),
        ),
        reference=MEASURE["pieces"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="encode_chunks",
        description="encodes each chunk separately, so no merge crosses a chunk boundary",
        inputs=(
            ("the cat sat on the mat", CAT_MERGES),
            ("that rat, 1999 hats", CAT_MERGES),
        ),
        reference=MEASURE["encode_chunks"],
        expect=expect_value,
    )


BPE4 = Exercise(
    name="bpe4",
    chapter=BPE,
    title="Why bytes and merges",
    lesson=BPE4_LESSON,
    source=BPE4_SOURCE,
    solution=BPE4_SOLUTION,
    hints={
        "bytes_per_token": "Divide the length of text.encode(\"utf-8\") by the number of ids that encode(text, merges) "
        "returns.",
        "unseen_share": "Make a set of train_words, then count the test words not in it and divide by len(test_words).",
        "pieces": "vocabulary(merges) gives each id's bytes. For each id that encode returns, decode its bytes with "
        "errors=\"replace\".",
        "encode_chunks": "For each chunk of re.findall(SPLIT, text), add the ids of encode(chunk, merges) to one list.",
    },
    introduces=(),
    check=check_bpe4,
)


def measure_bpe() -> dict[str, object]:
    """The numbers of bpe2, bpe3 and bpe4. Each tokenizer learns from the first 100,000 characters of tiny
    Shakespeare and is measured on the next 50,000. Gives bpe2's first five merges, the seconds bpe3's train
    took for 768 merges on the CPU, bpe4's table and quirks, and the splits of numbers after training on the
    numbers 0 to 1999."""
    text = load_shakespeare()
    train_text, test_text = text[:100_000], text[100_000:150_000]
    merges_256 = MEASURE["train"](train_text, 512)
    start = time.time()
    merges_768 = MEASURE["train"](train_text, 1024)
    seconds = time.time() - start
    tokens = MEASURE["vocabulary"](merges_768)
    train_words = re.findall(r"[A-Za-z]+", train_text)
    test_words = re.findall(r"[A-Za-z]+", test_text)
    numbers = MEASURE["train"](" ".join(str(number) for number in range(2000)), 1024)

    return {
        "first merges": [tokens[index].decode("utf-8") for index in range(256, 261)],
        "seconds for 768 merges": seconds,
        "bytes": len(test_text.encode("utf-8")),
        "tokens with 256 merges": len(MEASURE["encode"](test_text, merges_256)),
        "tokens with 768 merges": len(MEASURE["encode"](test_text, merges_768)),
        "bytes per token with 256 merges": MEASURE["bytes_per_token"](test_text, merges_256),
        "bytes per token with 768 merges": MEASURE["bytes_per_token"](test_text, merges_768),
        "training words": len(set(train_words)),
        "test words": len(test_words),
        "unseen share": MEASURE["unseen_share"](train_words, test_words),
        "words in all": len(set(re.findall(r"[A-Za-z]+", text))),
        "quirks": {piece: MEASURE["pieces"](piece, merges_768) for piece in ("the", "The", "the king")},
        "numbers": {piece: MEASURE["pieces"](piece, numbers) for piece in (" 1999", " 123456")},
    }


# ---------------------------------------------------------------------------
# quiz9
# ---------------------------------------------------------------------------

QUIZ9_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines steps
    from the attention chapters of Section 7. If a step is unclear, the
    exercise listed in the table shows how to do it.

    | Function | Steps | Shown in |
    |---|---|---|
    | `padded_causal_attention` | the future and padding masked, scores divided by the square root of the size, a softmax, a weighted sum of values | attention2, mask1, attention5 |
    | `rope_causal_scores` | each position's query and key rotated by its angles, the scores scaled, the future set to minus infinity | rope1, attention3, attention2 |
    | `heads_padded_attention` | each head's slice of the features, padded causal attention on that slice, the heads' outputs joined back together | multihead1, this quiz |
    """,
)

QUIZ9_SOURCE = source_file(
    '''
    # quiz9.py
    #
    # No examples and no new tools. Each function combines steps from Section 7.

    import math

    import torch
    from torch import Tensor


    # TODO: Compute attention of q, k and v by hand and return the output. q,
    #   k and v each have shape (batch, steps, size). A query may not attend
    #   to a key in its future. It also may not attend to a padding key, which
    #   is a key at or past its sequence's length in lengths, of shape
    #   (batch,). Divide the scores by the square root of size, set every
    #   blocked score to minus infinity, take the softmax over the keys, and
    #   return the weighted sum of the values.
    def padded_causal_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the causal attention scores of q and k after RoPE. q and k
    #   each have shape (steps, size), and angles has shape (steps, size // 2).
    #   In row m of q and of k, rotate each adjacent pair of features (0 and 1,
    #   2 and 3, and so on) by its angle in angles[m]. A pair (a, b) rotated by
    #   phi becomes (a cos phi - b sin phi, a sin phi + b cos phi). Then divide
    #   q @ k^T by the square root of size, and set the scores of future keys
    #   to minus infinity.
    def rope_causal_scores(
        q: Tensor,
        k: Tensor,
        angles: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return padded causal attention of q, k and v, split into heads
    #   heads. q, k and v each have shape (batch, steps, size). head_size is
    #   size // heads, and head h uses the features from h * head_size up to
    #   but not including (h + 1) * head_size. Run padded_causal_attention on
    #   each head's features. Then join the heads' outputs back together along
    #   the features, in order.
    def heads_padded_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        lengths: Tensor,
        heads: int,
    ) -> Tensor:
        ...
    ''',
)

QUIZ9_SOLUTION = source_file(
    '''
    # quiz9.py, a solution for comparison

    import math

    import torch
    from torch import Tensor


    def padded_causal_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        steps = q.shape[1]
        future = torch.triu(torch.ones(steps, steps, dtype=torch.bool), diagonal=1)
        padding = torch.arange(steps) >= lengths.unsqueeze(1)
        blocked = future | padding.unsqueeze(1)
        scores = q @ k.transpose(1, 2) / math.sqrt(q.shape[-1])

        return torch.softmax(scores.masked_fill(blocked, float("-inf")), dim=-1) @ v


    def rope_causal_scores(
        q: Tensor,
        k: Tensor,
        angles: Tensor,
    ) -> Tensor:
        cos, sin = torch.cos(angles), torch.sin(angles)
        q = torch.stack([q[:, 0::2] * cos - q[:, 1::2] * sin, q[:, 0::2] * sin + q[:, 1::2] * cos], dim=-1).flatten(-2)
        k = torch.stack([k[:, 0::2] * cos - k[:, 1::2] * sin, k[:, 0::2] * sin + k[:, 1::2] * cos], dim=-1).flatten(-2)
        steps = q.shape[0]
        future = torch.triu(torch.ones(steps, steps, dtype=torch.bool), diagonal=1)

        return (q @ k.T / math.sqrt(q.shape[-1])).masked_fill(future, float("-inf"))


    def heads_padded_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        lengths: Tensor,
        heads: int,
    ) -> Tensor:
        head_size = q.shape[-1] // heads
        outputs = []

        for head in range(heads):
            start, end = head * head_size, (head + 1) * head_size
            outputs.append(padded_causal_attention(q[:, :, start:end], k[:, :, start:end], v[:, :, start:end], lengths))

        return torch.cat(outputs, dim=-1)
    ''',
)


def allowed_keys(
    steps: int,
    lengths: Tensor,
) -> Tensor:
    """True where a query may look at a key, the past and itself, inside the sequence's length."""
    past = torch.tril(torch.ones(steps, steps, dtype=torch.bool))
    inside = torch.arange(steps) < lengths.unsqueeze(1)

    return past & inside.unsqueeze(1)


def reference_padded_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lengths: Tensor,
) -> Tensor:
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed_keys(q.shape[1], lengths))


def reference_rotate_pairs(
    x: Tensor,
    angles: Tensor,
) -> Tensor:
    turned = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous()) * torch.polar(torch.ones_like(angles), angles)

    return torch.view_as_real(turned).flatten(-2)


def reference_rope_causal_scores(
    q: Tensor,
    k: Tensor,
    angles: Tensor,
) -> Tensor:
    scores = reference_rotate_pairs(q, angles) @ reference_rotate_pairs(k, angles).T / q.shape[-1] ** 0.5
    future = ~torch.tril(torch.ones(len(q), len(q), dtype=torch.bool))

    return scores.masked_fill(future, float("-inf"))


def reference_heads_padded_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lengths: Tensor,
    heads: int,
) -> Tensor:
    batch, steps, size = q.shape
    q, k, v = (part.view(batch, steps, heads, size // heads).transpose(1, 2) for part in (q, k, v))
    mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed_keys(steps, lengths).unsqueeze(1))

    return mixed.transpose(1, 2).reshape(batch, steps, size)


def quiz_tensor(
    seed: int,
    *shape: int,
) -> Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def check_quiz9(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="padded_causal_attention",
        description="masks the future and the padding, then computes attention",
        inputs=(
            (quiz_tensor(930, 2, 5, 4), quiz_tensor(931, 2, 5, 4), quiz_tensor(932, 2, 5, 3), torch.tensor([5, 3])),
            (quiz_tensor(933, 3, 6, 8), quiz_tensor(934, 3, 6, 8), quiz_tensor(935, 3, 6, 8), torch.tensor([6, 1, 4])),
        ),
        reference=reference_padded_causal_attention,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="rope_causal_scores",
        description="rotates q and k with RoPE, then scales and masks the scores",
        inputs=(
            (quiz_tensor(942, 3, 4), quiz_tensor(943, 3, 4), quiz_tensor(944, 3, 2)),
            (quiz_tensor(945, 5, 6), quiz_tensor(946, 5, 6), quiz_tensor(947, 5, 3)),
        ),
        reference=reference_rope_causal_scores,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="heads_padded_attention",
        description="runs padded causal attention in each head and joins them",
        inputs=(
            (quiz_tensor(936, 2, 5, 8), quiz_tensor(937, 2, 5, 8), quiz_tensor(938, 2, 5, 8), torch.tensor([5, 2]), 2),
            (quiz_tensor(939, 1, 4, 12), quiz_tensor(940, 1, 4, 12), quiz_tensor(941, 1, 4, 12), torch.tensor([3]), 3),
        ),
        reference=reference_heads_padded_attention,
        expect=expect_tensor,
    )


QUIZ9 = Exercise(
    name="quiz9",
    chapter=QUIZZES,
    title="Attention, start to finish",
    lesson=QUIZ9_LESSON,
    source=QUIZ9_SOURCE,
    solution=QUIZ9_SOLUTION,
    hints={
        "padded_causal_attention": "torch.triu of a square of True with diagonal=1 marks the future. "
        "torch.arange(steps) >= lengths.unsqueeze(1) marks each sequence's padded keys. Unsqueeze the padding to "
        "(batch, 1, steps) and combine the two masks with |. Then masked_fill the scaled scores.",
        "rope_causal_scores": "q[:, 0::2] holds the first feature of every pair, and q[:, 1::2] holds the second. "
        "torch.stack the two rotated halves on dim=-1, and .flatten(-2) interleaves them back into pairs. Do the same "
        "for k, then mask the scaled q @ k.T as in attention2.",
        "heads_padded_attention": "Loop over the heads, slicing q[:, :, start:end] and the same for k and v. Join "
        "the outputs with torch.cat on dim=-1.",
    },
    introduces=(),
    check=check_quiz9,
)


# ---------------------------------------------------------------------------
# quiz10
# ---------------------------------------------------------------------------

QUIZ10_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines steps
    from the transformer chapters of Section 7. `greedy_text` also uses the
    greedy text generation from Section 6. If a step is unclear, the exercise
    listed in the table shows how to do it.

    | Function | Steps | Shown in |
    |---|---|---|
    | `mixtral_layer` | each token's top k experts and the softmax of their scores, every expert a SwiGLU MLP, the outputs weighted and summed | moe1, llama2, dynamic1 |
    | `embed_patches` | an image cut into patches, each embedded with the same matrix, a class token in front and positions added | vit1 |
    | `apply_merges` | every merge applied in the order it was made, each replacing its pair from left to right | bpe3, bpe2 |
    | `greedy_text` | the highest scoring id at the last position appended each step, the new ids decoded with the merges' bytes | bidir2, greedy1, bpe3 |
    """,
)

QUIZ10_SOURCE = source_file(
    '''
    # quiz10.py
    #
    # No examples and no new tools. Each function combines steps from Sections
    # 6 and 7.

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Return the output of a Mixtral layer for x, of shape
    #   (tokens, size). The router's scores are x @ router.T. Each token gives
    #   the experts with its k largest scores the weights of a softmax over
    #   only those scores, and gives every other expert a weight of 0. Expert
    #   i is a SwiGLU MLP, (F.silu(x @ W_gate[i].T) * (x @ W_up[i].T)) @
    #   W_down[i].T. The result is the weighted sum of the experts' outputs.
    #   Running every expert is fine here.
    def mixtral_layer(
        x: Tensor,
        router: Tensor,
        W_gate: Tensor,
        W_up: Tensor,
        W_down: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # TODO: Return a ViT's input for images, of shape
    #   (batch, channels, height, width). The result has shape
    #   (batch, patches + 1, size). Cut the images into patch by patch squares,
    #   in order row by row, and flatten each patch by channel, then row, then
    #   column. Embed each flattened patch row as row @ W.T. Put class_token,
    #   of shape (size,), in front of each image's patches, and add positions
    #   to every token.
    def embed_patches(
        images: Tensor,
        patch: int,
        W: Tensor,
        class_token: Tensor,
        positions: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return ids after applying every merge in merges, in the order the
    #   dict holds them. Each merge replaces every occurrence of its pair with
    #   its new id, from left to right and without overlaps.
    #   Example: apply_merges([5, 5, 5, 5, 5], {(5, 5): 260, (260, 260): 261})
    #   returns [261, 5].
    def apply_merges(
        ids: list[int],
        merges: dict[tuple[int, int], int],
    ) -> list[int]:
        ...


    # TODO: Return the text that model writes after ids, a list of token ids.
    #   model takes ids of shape (1, steps) and returns scores of shape
    #   (1, steps, tokens). At each step, run model on all the ids so far and
    #   append the id with the highest score at the last position. Repeat
    #   until count new ids are written, without letting autograd record.
    #   Then decode only the new ids. Join the bytes of their tokens, using the
    #   tokens that merges builds, and decode them as UTF-8 with � in place of
    #   invalid bytes.
    def greedy_text(
        model: nn.Module,
        ids: list[int],
        count: int,
        merges: dict[tuple[int, int], int],
    ) -> str:
        ...
    ''',
)

QUIZ10_SOLUTION = source_file(
    '''
    # quiz10.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    def mixtral_layer(
        x: Tensor,
        router: Tensor,
        W_gate: Tensor,
        W_up: Tensor,
        W_down: Tensor,
        k: int,
    ) -> Tensor:
        top = (x @ router.T).topk(k, dim=-1)
        weights = torch.zeros(len(x), len(router)).scatter(1, top.indices, torch.softmax(top.values, dim=-1))
        outputs = torch.stack([(F.silu(x @ gate.T) * (x @ up.T)) @ down.T for gate, up, down in zip(W_gate, W_up, W_down)], dim=1)

        return (weights.unsqueeze(2) * outputs).sum(dim=1)


    def embed_patches(
        images: Tensor,
        patch: int,
        W: Tensor,
        class_token: Tensor,
        positions: Tensor,
    ) -> Tensor:
        batch, channels, height, width = images.shape
        cells = images.reshape(batch, channels, height // patch, patch, width // patch, patch)
        rows = cells.permute(0, 2, 4, 1, 3, 5).reshape(batch, -1, channels * patch * patch)
        first = class_token.expand(batch, 1, -1)

        return torch.cat([first, rows @ W.T], dim=1) + positions


    def apply_merges(
        ids: list[int],
        merges: dict[tuple[int, int], int],
    ) -> list[int]:
        for pair, new_id in merges.items():
            merged = []
            position = 0

            while position < len(ids):
                if position + 1 < len(ids) and (ids[position], ids[position + 1]) == pair:
                    merged.append(new_id)
                    position += 2
                    continue

                merged.append(ids[position])
                position += 1

            ids = merged

        return ids


    # The model reads every id so far at each step, and only the new ids become
    # text. Their bytes are joined before decoding, since one character's bytes
    # can be split across tokens.
    def greedy_text(
        model: nn.Module,
        ids: list[int],
        count: int,
        merges: dict[tuple[int, int], int],
    ) -> str:
        written = list(ids)

        with torch.no_grad():
            for _ in range(count):
                scores = model(torch.tensor([written]))
                written.append(scores[0, -1].argmax().item())

        tokens = {index: bytes([index]) for index in range(256)}

        for (first, second), new_id in merges.items():
            tokens[new_id] = tokens[first] + tokens[second]

        return b"".join(tokens[index] for index in written[len(ids) :]).decode("utf-8", errors="replace")
    ''',
)


def reference_mixtral_layer(
    x: Tensor,
    router: Tensor,
    W_gate: Tensor,
    W_up: Tensor,
    W_down: Tensor,
    k: int,
) -> Tensor:
    """Each token's chosen experts one by one, a different road to the same numbers."""
    rows = []

    for token in x:
        top = (router @ token).topk(k)
        weights = torch.softmax(top.values, dim=0)
        experts = [W_down[index] @ (F.silu(W_gate[index] @ token) * (W_up[index] @ token)) for index in top.indices.tolist()]
        rows.append(sum(weight * expert for weight, expert in zip(weights, experts)))

    return torch.stack(rows)


def reference_embed_patches(
    images: Tensor,
    patch: int,
    W: Tensor,
    class_token: Tensor,
    positions: Tensor,
) -> Tensor:
    rows = F.unfold(images, kernel_size=patch, stride=patch).transpose(1, 2) @ W.T
    first = class_token.repeat(len(images), 1, 1)

    return torch.cat([first, rows], dim=1) + positions


def reference_apply_merges(
    ids: list[int],
    merges: dict[tuple[int, int], int],
) -> list[int]:
    for pair, new_id in merges.items():
        ids = PAIRS["merge"](ids, pair, new_id)

    return ids


# é is the two bytes 195 and 169, so a script that writes them as two tokens shows whether the bytes are joined
# before decoding. 258 spells hi!, two merges deep.
WRITER_MERGES = {(195, 169): 256, (104, 105): 257, (257, 33): 258}

# (id, place) to the id written after it. From [72, 105] the script writes " ", 195, 169, 258 and 256, which spell
# " éhi!é". From [258] it writes 257, 256 and a lone 195, which spell "hié" and a �.
WRITER_SCRIPT = {
    (105, 1): 32,
    (32, 2): 195,
    (195, 3): 169,
    (169, 4): 258,
    (258, 5): 256,
    (258, 0): 257,
    (257, 1): 256,
    (256, 2): 195,
}


class ScriptedWriter(nn.Module):
    """A stand in language model whose score at each position picks the id its script puts after that position's
    id at that place, so what it writes depends on the whole sequence so far. Every other pair gets a fixed random
    byte. It notes whether autograd was recording while it ran."""

    def __init__(self) -> None:
        super().__init__()
        follows = torch.randint(0, 256, (256 + len(WRITER_MERGES), 8), generator=torch.Generator().manual_seed(976))

        for (token, place), next_id in WRITER_SCRIPT.items():
            follows[token, place] = next_id

        self.follows = follows
        self.scale = nn.Parameter(torch.ones(()))
        self.recorded = False

    def forward(self, ids: Tensor) -> Tensor:
        if ids.dim() != 2:
            raise Mismatch(
                f"model was given ids of shape {tuple(ids.shape)}, but it takes shape (1, steps). Pass the ids so "
                "far as a batch with one row.",
            )

        self.recorded = self.recorded or torch.is_grad_enabled()
        places = torch.arange(ids.shape[1]).clamp(max=self.follows.shape[1] - 1)

        return F.one_hot(self.follows[ids, places], len(self.follows)).float() * self.scale


def reference_greedy_text(
    ids: list[int],
    count: int,
) -> str:
    """The script followed by hand, and each token's bytes found by unmerging it down to single bytes."""
    unmerged = {new_id: pair for pair, new_id in WRITER_MERGES.items()}
    written = list(ids)

    for _ in range(count):
        written.append(WRITER_SCRIPT[(written[-1], len(written) - 1)])

    def spell(token: int) -> bytes:
        if token < 256:
            return bytes([token])

        first, second = unmerged[token]

        return spell(first) + spell(second)

    return b"".join(spell(token) for token in written[len(ids) :]).decode("utf-8", errors="replace")


def check_greedy_text(found: Callable[..., object]) -> None:
    for ids, count in (([72, 105], 5), ([258], 3)):
        call = f"greedy_text(a model that follows a script, {ids}, {count}, {WRITER_MERGES})"
        model = ScriptedWriter()
        actual = found(model, list(ids), count, dict(WRITER_MERGES))

        if model.recorded:
            raise Mismatch(f"{call} ran model while autograd was recording. Run the loop inside torch.no_grad().")

        expect_value(
            call=call,
            actual=actual,
            expected=reference_greedy_text(ids, count),
        )


def check_quiz10(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="mixtral_layer",
        description="weights each token's top k SwiGLU experts by a softmax over their scores",
        inputs=(
            (
                quiz_tensor(958, 5, 4),
                quiz_tensor(959, 4, 4),
                quiz_tensor(960, 4, 6, 4) / 2,
                quiz_tensor(961, 4, 6, 4) / 2,
                quiz_tensor(962, 4, 4, 6) / 2,
                2,
            ),
            (
                quiz_tensor(963, 3, 4),
                quiz_tensor(964, 3, 4),
                quiz_tensor(965, 3, 5, 4) / 2,
                quiz_tensor(966, 3, 5, 4) / 2,
                quiz_tensor(967, 3, 4, 5) / 2,
                1,
            ),
        ),
        reference=reference_mixtral_layer,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="embed_patches",
        description="cuts patches, embeds them, puts the class token first and adds positions",
        inputs=(
            (quiz_tensor(968, 2, 3, 4, 6), 2, quiz_tensor(969, 5, 12), quiz_tensor(970, 5), quiz_tensor(971, 7, 5)),
            (quiz_tensor(972, 1, 1, 6, 6), 3, quiz_tensor(973, 4, 9), quiz_tensor(974, 4), quiz_tensor(975, 5, 4)),
        ),
        reference=reference_embed_patches,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="apply_merges",
        description="applies every merge in order, each from left to right",
        inputs=(
            (list(b"aaabdaaabac"), {(97, 97): 256, (256, 97): 257, (257, 98): 258}),
            ([1, 2, 3, 1, 2, 3], {(1, 2): 300, (2, 3): 301, (300, 3): 302}),
            ([5, 5, 5, 5, 5], {(5, 5): 260, (260, 260): 261}),
        ),
        reference=reference_apply_merges,
        expect=expect_value,
    )
    checker.test(
        function="greedy_text",
        description="writes greedily from the ids so far and spells only the new ids",
        run=check_greedy_text,
    )


QUIZ10 = Exercise(
    name="quiz10",
    chapter=QUIZZES,
    title="Transformers, start to finish",
    lesson=QUIZ10_LESSON,
    source=QUIZ10_SOURCE,
    solution=QUIZ10_SOLUTION,
    hints={
        "mixtral_layer": "Take the topk of the router's scores, then scatter their softmax into "
        "torch.zeros(len(x), len(router)), as in dynamic1. Stack every expert's SwiGLU output with torch.stack on "
        "dim=1, multiply by the weights, and sum over dim=1.",
        "embed_patches": "Cut the patches with the reshape, permute and reshape from vit1. "
        "class_token.expand(batch, 1, -1) gives one class token per image. Put it in front with torch.cat on dim=1.",
        "apply_merges": "Loop over merges.items(). For each merge, move a position along ids as bpe2's merge did, "
        "building a new list.",
        "greedy_text": "Keep a list of every id so far. Each step, run model on the list as a batch with one row, as "
        "generate does in bidir2. Append the argmax at the last position, as greedy_decode does in greedy1. Then build each token's "
        "bytes as vocabulary does in bpe3, and join the new ids' bytes before decoding, as decode does.",
    },
    introduces=(),
    check=check_quiz10,
)
