"""58_bpe, text as UTF-8 bytes, byte pair encoding trained, encoded and decoded as in minbpe, and why it beats characters and words."""

from __future__ import annotations

import re
import textwrap

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_value,
)
from ..text import prose, source_file

BPE = Chapter(
    folder="58_bpe",
    title="Byte pair encoding",
    readme=source_file(
        """
        # Byte pair encoding

        How text becomes tokens. Text starts as UTF-8 bytes, the first 256
        tokens, and byte pair encoding merges the commonest pair of neighbours
        into a new token, again and again. You build the tokenizer GPT-2 uses,
        in the style of Karpathy's minbpe, then measure why it beats characters
        and words, and where it gets odd.

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
    A model reads tokens, so text has to become ids before anything else. A
    computer already keeps text as numbers, bytes from 0 to 255, in an
    encoding called UTF-8.

    ```python
    list("hi".encode("utf-8"))           # [104, 105]
    list("é".encode("utf-8"))            # [195, 169]
    bytes([104, 105]).decode("utf-8")    # 'hi'
    ```

    An English letter takes one byte. Other characters take two to four, so
    an accented letter takes 2 bytes and an emoji 4.

    The 256 byte values make a first vocabulary of tokens. Every text there is
    can be written with them, so a model never meets text it has no token for.
    GPT-2's tokenizer starts here, and every merge of bpe2 and bpe3 builds on
    these 256.

    Not every list of bytes is valid UTF-8. A model can write the first byte of
    a 2 byte character and stop, and `.decode("utf-8")` then raises an error.
    With `errors="replace"`, the character � stands in for the broken bytes
    instead. minbpe's decode does this.

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
    def to_bytes(text: str) -> list[int]:
        ...


    # TODO: Return the text that the bytes ids spell, with � standing in for any
    #   bytes that aren't valid UTF-8.
    def from_bytes(ids: list[int]) -> str:
        ...


    # TODO: Return how many bytes each character of text takes, in order.
    def byte_lengths(text: str) -> list[int]:
        ...


    # TODO: Return the number of bytes text takes per character, on average.
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
        description="gives the UTF-8 bytes",
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
        description="reads the bytes back, with � for broken ones",
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
        description="gives each character's number of bytes",
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
        description="gives the bytes per character",
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
        "byte_lengths": "For each character of text, the length of its own .encode(\"utf-8\").",
        "bytes_per_character": "The length of the whole text's bytes over len(text).",
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
    Count every pair of neighbouring tokens, take the commonest pair, and give
    it a new id. Then put that id in place of the pair everywhere it appears.
    Sennrich and colleagues brought it to neural translation in 2016, and
    GPT-2 runs it on bytes.

    ```python
    ids = [97, 97, 97, 98]          # 'aaab'
    list(zip(ids, ids[1:]))         # [(97, 97), (97, 97), (97, 98)]
    ```

    `zip(ids, ids[1:])` walks every pair of neighbours. Here (97, 97) comes up
    twice, so it becomes token 256, and the ids become `[256, 97, 98]`.

    The replacing goes left to right and never overlaps. In `aaa` the pair
    `aa` is replaced once, at the start, and the last `a` stays as it was.

    Each merge makes the ids shorter and the vocabulary one bigger. On the
    first 100,000 characters of tiny Shakespeare, the text of the GPT project,
    the first merges are `e `, `th`, `t `, `s ` and `ou`. These are the
    commonest pairs of English, spaces included.

    Two pairs can tie. Then the pair seen first wins. Python's `max` over a
    dict gives exactly that, since a dict keeps its keys in the order they
    came in.

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


    # TODO: Return how many times each pair of neighbours appears in ids, as a
    #   dict from the pair to its count, in the order the pairs first appear.
    def pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
        ...


    # TODO: Return the pair of neighbours that appears most often in ids. A tie
    #   goes to the pair that appears first.
    def most_common_pair(ids: list[int]) -> tuple[int, int]:
        ...


    # TODO: Return ids with new_id in place of every pair of neighbours equal to
    #   pair, going left to right, so the replacements never overlap.
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
        description="gives the commonest pair, the first one on a tie",
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
        "most_common_pair": "max over pair_counts(ids) with key set to that dict's .get.",
        "merge": "Walk a position along ids with a while loop. When ids[position] and the next one make pair, append "
        "new_id and step 2, otherwise append ids[position] and step 1.",
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
    Training repeats bpe2's step. Start from the text's bytes, and for each
    new id from 256 up to the vocabulary size, merge the commonest pair and
    write it down. That dict from each pair to its new id is the whole
    tokenizer.

    Decoding needs the bytes each token stands for. Token 256 might stand for
    the pair (101, 32), the bytes of `e `. A token built from other tokens
    stands for both their bytes joined. So build that table in the order the
    merges were made, and both halves always have their bytes already.

    Encoding new text has to apply the merges in the order training made them.
    Merge 300 may be built from merge 270, so it can't come first. At each step
    find the pair in the ids with the earliest merge, the smallest new id, and
    apply it. Stop when no pair has a merge.

    ```python
    merges = {(104, 105): 256}
    min([(105, 33), (104, 105)], key=lambda pair: merges.get(pair, float("inf")))    # (104, 105)
    ```

    A pair with no merge counts as infinity, so it wins only when nothing can
    merge, and `pair not in merges` then ends the loop.

    `decode(encode(text, merges), merges)` has to give back `text` exactly, for
    any text at all. That round trip is the test of a tokenizer, and the
    bytes underneath make it hold even for text unlike the training text.

    This `train` is plain Python, so it is slow on big texts. 768 merges on
    100,000 characters of tiny Shakespeare took about 25 seconds here.
    tiktoken, the tokenizer OpenAI's models use, encodes far faster.
    """,
)

BPE3_SOURCE = source_file(
    '''
    # bpe3.py

    merges = {(104, 105): 256}
    earliest = min([(105, 33), (104, 105)], key=lambda pair: merges.get(pair, float("inf")))    # (104, 105)
    joined = b"h" + b"i"    # b'hi'


    # bpe2's pieces.
''' + indented_part(PAIR_PARTS) + '''


    # TODO: Return the merges that training on text learns, a dict from each
    #   merged pair to its new id. Start from text's UTF-8 bytes, and for each
    #   new id from 256 up to vocab_size, merge the commonest pair. Stop early
    #   when fewer than 2 ids are left.
    def train(
        text: str,
        vocab_size: int,
    ) -> dict[tuple[int, int], int]:
        ...


    # TODO: Return the bytes each token stands for, a dict from each id to its
    #   bytes. Ids 0 to 255 are their own byte, and each merged id is the bytes
    #   of its pair joined, filled in the order of merges.
    def vocabulary(merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
        ...


    # TODO: Finish the line that picks pair, the pair of neighbours in ids with
    #   the earliest merge, the smallest new id in merges.
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


    # TODO: Return the text that ids spell, with � standing in for any bytes
    #   that aren't valid UTF-8.
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
        raise Mismatch(f"{call} should be a dict from each id to its bytes.")

    for index, wanted in expected.items():
        if actual.get(index) != wanted:
            raise Mismatch(f"In {call}, id {index} is {actual.get(index)!r}, but it should be {wanted!r}.")

    if len(actual) != len(expected):
        raise Mismatch(f"{call} has {len(actual)} ids, but it should have {len(expected)}, 256 bytes and one per merge.")


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
        description="learns the merges, commonest pair first",
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
        description="spells the ids back out",
        inputs=(
            ([258, 100, 258, 97, 99], {(97, 97): 256, (256, 97): 257, (257, 98): 258}),
            ([104, 105, 226, 130], {}),
        ),
        reference=TOKENIZER["decode"],
        expect=expect_value,
    )
    checker.test(
        function="decode",
        description="gives back any text that encode was given",
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
        "train": "ids = list(text.encode(\"utf-8\")). For each new_id in range(256, vocab_size), break if len(ids) < 2, "
        "then take most_common_pair, merge it into ids and record merges[pair] = new_id.",
        "vocabulary": "Start from {index: bytes([index]) for index in range(256)}. Then for each (first, second), "
        "new_id in merges.items(), join the bytes of first and second.",
        "encode": "min over pair_counts(ids), with key=lambda pair: merges.get(pair, float(\"inf\")).",
        "decode": "b\"\".join the bytes of each id from vocabulary(merges), then .decode(\"utf-8\", errors=\"replace\").",
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
    Why bytes and merges, and not characters or words? Here is tiny
    Shakespeare. Each tokenizer learned from its first 100,000 characters and
    was measured on the next 50,000.

    | Tokens | Vocabulary | Tokens for the 50,000 characters |
    |---|---|---|
    | bytes | 256 | 50,000 |
    | BPE, 256 merges | 512 | 24,446, 2.05 bytes each |
    | BPE, 768 merges | 1,024 | 18,643, 2.68 bytes each |
    | words | 3,472 | 9,090 words, 12.7% of them unknown |

    Bytes make the longest sequences, and a model's cost grows with length,
    the cost of comparing every pair of positions with its square. Words make
    the shortest, but 12.7% of the words in the later text never appeared in
    the first part. A model has no id for them. All of tiny Shakespeare holds
    14,554 different words, and real text holds far more.

    BPE sits between. Common words become single tokens, rare ones split into
    pieces, and nothing is ever unknown, since the bytes are always there.

    It has quirks. With 768 merges, `the` is one token, `The` is two, `T` and
    `he`, and `the king` splits as `the `, `k`, `ing`. A word's tokens change
    with its capital and with the spaces around it.

    Numbers split by how often pairs showed up, not by place value. Trained on
    the numbers 0 to 1999, ` 1999` becomes ` 1`, `99`, `9`, and ` 123456`
    becomes ` 1`, `23`, `45`, `6`. That is part of why arithmetic is hard for
    language models.

    GPT-2 first splits text with a regular expression into words, runs of
    digits and other symbols, each with the space before it. Merges then stay
    inside each chunk. Its vocabulary is 50,257 tokens, the 256 bytes, 50,000
    merges and one token that marks the end of a text. GPT-4's tokenizer,
    cl100k_base, has about 100,000 tokens and splits numbers into runs of at
    most 3 digits.
    """,
)

BPE4_SOURCE = source_file(
    '''
    # bpe4.py

    import re

    words = re.findall(r"[A-Za-z]+", "To be, or not to be")    # ['To', 'be', 'or', 'not', 'to', 'be']
    unique = set(words)                                         # 5 words, To and to apart


    # bpe2's pieces and bpe3's tokenizer.
''' + indented_part(PAIR_PARTS) + '''


''' + indented_part(TOKENIZER_PARTS) + '''


''' + indented_part(SPLIT_PART) + '''


    # TODO: Return the number of UTF-8 bytes of text per token that encode
    #   gives it.
    def bytes_per_token(
        text: str,
        merges: dict[tuple[int, int], int],
    ) -> float:
        ...


    # TODO: Return the share of test_words that never appear in train_words, the
    #   words a vocabulary of the training words has no id for.
    def unseen_share(
        train_words: list[str],
        test_words: list[str],
    ) -> float:
        ...


    # TODO: Return the text of each token that encode gives text, in order, with
    #   � standing in for any bytes that aren't valid UTF-8 alone.
    def pieces(
        text: str,
        merges: dict[tuple[int, int], int],
    ) -> list[str]:
        ...


    # TODO: Return the ids of text encoded one chunk at a time, the chunks that
    #   re.findall(SPLIT, text) gives, so no merge joins two chunks.
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
        description="gives the share of test words never seen in training",
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
        description="encodes each chunk alone, so no merge crosses a chunk",
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
        "bytes_per_token": "len(text.encode(\"utf-8\")) over len(encode(text, merges)).",
        "unseen_share": "Make a set of train_words, then count the test words not in it and divide by len(test_words).",
        "pieces": "vocabulary(merges) gives each id's bytes. Decode each one with errors=\"replace\", for each id that "
        "encode gives.",
        "encode_chunks": "For each chunk of re.findall(SPLIT, text), add the ids of encode(chunk, merges) to one list.",
    },
    introduces=(),
    check=check_bpe4,
)
