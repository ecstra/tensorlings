"""87_flash_attention, the online softmax that reads a row in one pass, and FlashAttention's tiling, attention a block of keys at a time with no full matrix of scores."""

from __future__ import annotations

import math
import statistics
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.overrides import TorchFunctionMode

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_parts,
    expect_tensor,
    expect_value,
    small_tensor,
    value_text,
)
from ..text import prose, source_file

FLASH_ATTENTION = Chapter(
    folder="87_flash_attention",
    title="FlashAttention",
    readme=source_file(
        """
        # FlashAttention

        Attention written plainly builds every score of every query against
        every key before its softmax. This chapter first reads a softmax's row
        in one pass, with a running largest score and total, then uses that
        to compute attention a block of keys at a time, the way FlashAttention
        does, and measures what plain PyTorch makes of it.

        ## Further reading

        - [Milakov and Gimelshein, online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
        - [Dao and colleagues, FlashAttention](https://arxiv.org/abs/2205.14135)
        - [Dao, FlashAttention-2](https://arxiv.org/abs/2307.08691)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "F": F,
        "Tensor": Tensor,
    }
    exec(code, namespace)

    return namespace


# ---------------------------------------------------------------------------
# flash1
# ---------------------------------------------------------------------------

UPDATE_SOLUTION = '''
def update(
    largest: Tensor,
    total: Tensor,
    score: Tensor,
) -> tuple[Tensor, Tensor]:
    new_largest = torch.maximum(largest, score)

    return new_largest, total * torch.exp(largest - new_largest) + torch.exp(score - new_largest)
'''

RUNNING_PART = '''
# Walks along the last dimension of scores once, a score at a time.
def running_max_and_total(scores: Tensor) -> tuple[Tensor, Tensor]:
    largest = torch.full(scores.shape[:-1], float("-inf"))
    total = torch.zeros(scores.shape[:-1])

    for j in range(scores.shape[-1]):
        largest, total = update(largest, total, scores[..., j])

    return largest, total
'''

SOFTMAX_SOLUTION = '''
def online_softmax(scores: Tensor) -> Tensor:
    largest, total = running_max_and_total(scores)

    return torch.exp(scores - largest.unsqueeze(-1)) / total.unsqueeze(-1)
'''

MERGE_SOLUTION = '''
def merge(
    largest_a: Tensor,
    total_a: Tensor,
    largest_b: Tensor,
    total_b: Tensor,
) -> tuple[Tensor, Tensor]:
    largest = torch.maximum(largest_a, largest_b)

    return largest, total_a * torch.exp(largest_a - largest) + total_b * torch.exp(largest_b - largest)
'''

ONLINE_REFERENCE = namespace_of(UPDATE_SOLUTION + RUNNING_PART + SOFTMAX_SOLUTION + MERGE_SOLUTION)

FLASH1_LESSON = prose(
    """
    softmax2 took each row's largest score away before the exp, so no exp
    could overflow. That safe softmax reads the scores three times, once for
    the largest, once for the total of exps, and once to divide.

    Milakov and Gimelshein folded the first two reads into one. Walk along
    the scores, keeping a running largest `m` and a running total `d` of exps
    measured from it. When a new score beats `m`, every exp counted so far was
    measured from too low a mark. So the total shrinks first, then the new
    score's exp joins it.

    ```text
    m_new = max(m_old, x)
    d_new = d_old * exp(m_old - m_new) + exp(x - m_new)
    ```

    Start with `m` at minus infinity and `d` at 0. The first score's shrink
    is `exp(-inf)`, which is 0. At the end, `d` is the total of `exp(x - m)`
    over the row, the same total the safe softmax finds.

    Their paper counts memory traffic per score. The safe softmax reads or
    writes each score 4 times, and the online one 3. On a GPU, where moving
    numbers costs more than the math, that ran up to 1.3 times faster.

    The same rule joins two parts of a row. Each part brings its own `m` and
    `d`, the whole row's `m` is the larger one, and each `d` shrinks to it.

    ```text
    m = max(m_a, m_b)
    d = d_a * exp(m_a - m) + d_b * exp(m_b - m)
    ```

    So a row can be read in blocks, and the blocks joined afterwards. That is
    the step flash2 builds attention on. `update` is this join with a part of
    a single score, whose total is `exp(0)`, 1.
    """,
)

FLASH1_SOURCE = source_file(
    '''
    # flash1.py

    import torch
    from torch import Tensor

    larger = torch.maximum(torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0]))    # tensor([3., 5.])
    nothing = torch.exp(torch.tensor(float("-inf")))                              # tensor(0.)


    # TODO: Return the running largest and total after one more score. Each
    #   of largest, total and score holds one number per row. The new largest
    #   is the larger of largest and score. The old total shrinks by the exp of
    #   the old largest minus the new one, then the exp of score minus the new
    #   largest joins it.
    def update(
        largest: Tensor,
        total: Tensor,
        score: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ...


''' + indented(RUNNING_PART) + '''


    # TODO: Return the softmax of scores along the last dimension. Take the
    #   largest and total from running_max_and_total, then make one more pass,
    #   the exp of each score minus its row's largest over its row's total.
    def online_softmax(scores: Tensor) -> Tensor:
        ...


    # TODO: Return the largest and the total of two parts of the same rows
    #   together. The largest is the larger of the two, and each part's total
    #   shrinks by the exp of its own largest minus that one before they add.
    def merge(
        largest_a: Tensor,
        total_a: Tensor,
        largest_b: Tensor,
        total_b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ...
    ''',
)

FLASH1_SOLUTION = source_file(
    '''
    # flash1.py, a solution for comparison

    import torch
    from torch import Tensor

    larger = torch.maximum(torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0]))
    nothing = torch.exp(torch.tensor(float("-inf")))


''' + indented(UPDATE_SOLUTION) + '''

''' + indented(RUNNING_PART) + '''


''' + indented(SOFTMAX_SOLUTION) + '''


''' + indented(MERGE_SOLUTION) + '''
    ''',
)

MINUS_INFINITY = float("-inf")


def part_stats(scores: Tensor) -> tuple[Tensor, Tensor]:
    """The largest score of each row and its total of exps, the way the safe softmax finds them."""
    largest = scores.amax(dim=-1)

    return largest, torch.exp(scores - largest.unsqueeze(-1)).sum(dim=-1)


def merge_inputs(
    scores: Tensor,
    cut: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (*part_stats(scores[..., :cut]), *part_stats(scores[..., cut:]))


def check_online_softmax(found: object) -> None:
    """A stand in for running_max_and_total records that online_softmax asks it for the largest and the total.
    A second stand in gives a largest 1 too high and a total twice too large, so an answer that works either one
    out again on its own, or calls torch.softmax, no longer follows it."""
    calls: list[Tensor] = []
    skewed = False

    def recorded(scores: Tensor) -> tuple[Tensor, Tensor]:
        calls.append(scores)
        largest, total = part_stats(scores)

        if skewed:
            return largest + 1, total * 2

        return largest, total

    original = found.__globals__.get("running_max_and_total")
    found.__globals__["running_max_and_total"] = recorded

    try:
        for scores in (small_tensor(870, 3, 6), torch.tensor([[1000.0, 1001.0, 999.0], [-2.0, 0.5, 0.0]])):
            skewed = False
            calls.clear()
            expect_tensor(
                call=f"online_softmax({value_text(scores)})",
                actual=found(scores.clone()),
                expected=torch.softmax(scores, dim=-1),
            )

            if not calls:
                raise Mismatch("online_softmax should take the largest and the total from running_max_and_total.")

            skewed = True
            largest, total = part_stats(scores)
            followed = torch.exp(scores - largest.unsqueeze(-1) - 1) / (2 * total.unsqueeze(-1))
            actual = found(scores.clone())

            if not isinstance(actual, Tensor) or not torch.allclose(actual.double(), followed.double(), atol=1e-6):
                raise Mismatch(
                    "online_softmax should use the very largest and total that running_max_and_total gives. The "
                    "check's stand in gave a largest 1 too high and a total twice too large, and the result did not "
                    "follow them. Build the softmax from those two, not from torch.softmax or a max or sum of your own.",
                )
    finally:
        if original is None:
            found.__globals__.pop("running_max_and_total", None)
        else:
            found.__globals__["running_max_and_total"] = original


def check_flash1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="update",
        description="raises the largest and shrinks the total before adding",
        inputs=(
            (torch.full((2,), MINUS_INFINITY), torch.zeros(2), torch.tensor([2.0, -1.0])),
            (torch.tensor([1.0, 3.0, 0.5]), torch.tensor([2.0, 1.5, 3.0]), torch.tensor([2.5, -1.0, 0.5])),
        ),
        reference=ONLINE_REFERENCE["update"],
        expect=expect_parts("the largest", "the total"),
    )
    checker.test(
        function="online_softmax",
        description="divides each exp by the running total, matching torch.softmax",
        run=check_online_softmax,
    )
    check_on_inputs(
        checker=checker,
        function="merge",
        description="joins two parts of a row, each total shrunk to the larger mark",
        inputs=(
            merge_inputs(small_tensor(871, 3, 7), 3),
            merge_inputs(small_tensor(872, 2, 5) * 4, 4),
            (torch.full((2,), MINUS_INFINITY), torch.zeros(2), torch.tensor([1.0, -2.0]), torch.tensor([3.0, 1.5])),
        ),
        reference=ONLINE_REFERENCE["merge"],
        expect=expect_parts("the largest", "the total"),
    )


FLASH1 = Exercise(
    name="flash1",
    chapter=FLASH_ATTENTION,
    title="The online softmax",
    lesson=FLASH1_LESSON,
    source=FLASH1_SOURCE,
    solution=FLASH1_SOLUTION,
    hints={
        "update": "torch.maximum(largest, score) is the new largest. Multiply total by torch.exp of largest minus it, "
        "then add torch.exp of score minus it.",
        "online_softmax": "Give largest and total an end dimension with .unsqueeze(-1), so they line up with each "
        "row. Then take torch.exp of scores minus largest, and divide by total.",
        "merge": "torch.maximum of the two largests first. Each total times torch.exp of its own largest minus that, "
        "added together.",
    },
    introduces=(),
    check=check_flash1,
)


# ---------------------------------------------------------------------------
# flash2
# ---------------------------------------------------------------------------

TILED_SOLUTION = '''
def tiled_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block: int,
) -> Tensor:
    scale = 1 / math.sqrt(q.shape[-1])
    outputs = []

    for start in range(0, q.shape[-2], block):
        rows = q[..., start : start + block, :]
        largest = torch.full(rows.shape[:-1], float("-inf"), device=q.device)
        total = torch.zeros(rows.shape[:-1], device=q.device)
        out = torch.zeros(*rows.shape[:-1], v.shape[-1], device=q.device)

        for key_start in range(0, k.shape[-2], block):
            keys = k[..., key_start : key_start + block, :]
            values = v[..., key_start : key_start + block, :]
            scores = rows @ keys.transpose(-2, -1) * scale
            new_largest = torch.maximum(largest, scores.amax(dim=-1))
            shrink = torch.exp(largest - new_largest)
            weights = torch.exp(scores - new_largest.unsqueeze(-1))
            total = total * shrink + weights.sum(dim=-1)
            out = out * shrink.unsqueeze(-1) + weights @ values
            largest = new_largest

        outputs.append(out / total.unsqueeze(-1))

    return torch.cat(outputs, dim=-2)
'''

COUNTS_SOLUTION = '''
def blocks_read(
    steps: int,
    block: int,
    causal: bool,
) -> int:
    blocks = math.ceil(steps / block)

    if causal:
        return blocks * (blocks + 1) // 2

    return blocks * blocks


def scores_held(
    steps: int,
    block: int,
    tiled: bool,
) -> int:
    if tiled:
        return min(block, steps) ** 2

    return steps * steps
'''

TILED_REFERENCE = namespace_of(TILED_SOLUTION + COUNTS_SOLUTION)

TILED_START = '''
    def tiled_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        block: int,
    ) -> Tensor:
        scale = 1 / math.sqrt(q.shape[-1])
        outputs = []

        for start in range(0, q.shape[-2], block):
            rows = q[..., start : start + block, :]
            largest = torch.full(rows.shape[:-1], float("-inf"), device=q.device)
            total = torch.zeros(rows.shape[:-1], device=q.device)
            out = torch.zeros(*rows.shape[:-1], v.shape[-1], device=q.device)

            for key_start in range(0, k.shape[-2], block):
                keys = k[..., key_start : key_start + block, :]
                values = v[..., key_start : key_start + block, :]
                scores = ...
                new_largest = ...
                shrink = ...
                weights = ...
                total = ...
                out = ...
                largest = new_largest

            outputs.append(out / total.unsqueeze(-1))

        return torch.cat(outputs, dim=-2)
'''

FLASH2_LESSON = prose(
    """
    attention5 computed `softmax(q kᵀ / √d) v`. Written that way, it builds
    every score first, steps by steps numbers for each head. At 4,096 steps
    that is 16.8 million, written out to memory and read back.

    FlashAttention, from Dao and colleagues, never builds them all. It cuts
    the keys and values into blocks and reads one block at a time, as flash1
    read a row in parts. Each query keeps its largest score `m`, its total of
    exps `d`, and its output so far, a sum of values weighted by exps.

    When a block raises `m`, the old total and the old output were measured
    from too low a mark, so both shrink by `exp(m_old - m_new)`. Then the
    block's exps join the total, and its exps times its values join the
    output. At the end, the output over the total is attention's output. The
    method is exact, not an estimate.

    `tiled_attention` keeps a block of queries in the outer loop and walks
    the key blocks inside it. That is FlashAttention-2's order, and the first
    paper looped the other way round.

    A causal model skips more. A block of queries never reads the key blocks
    after its own, which the mask would hide entirely, so nearly half the
    blocks go unread.

    The speed comes from where the blocks live. An A100 GPU has 40 to 80 GB of
    main memory, read at 1.5 to 2.0 TB/s. Beside each of its 108 compute units
    sit 192 KB of fast memory, read at about 19 TB/s. FlashAttention-2's
    kernel keeps a block of queries, with its `m`, `d` and output, in that
    fast memory. It writes back only the output and one number per row. The
    first paper reports GPT-2 training 3 to 3.5 times faster than Hugging
    Face's version at 1,024 tokens.

    Here is the peak memory one call added on this Mac's CPU, for 4 heads of
    size 64. Each call ran in a fresh process, over three runs. SDPA is
    `F.scaled_dot_product_attention`.

    | Steps | Softmax written out | Tiled, blocks of 1,024 | Tiled, blocks of 128 | SDPA |
    |---|---|---|---|---|
    | 2,048 | 131 MiB | 69 to 75 MiB | 5 to 6 MiB | 5 MiB |
    | 4,096 | 517 to 518 MiB | 75 to 79 MiB | 9 to 11 MiB | 7 MiB |
    | 8,192 | 2,057 to 2,058 MiB | 87 to 88 MiB | 18 to 19 MiB | 11 MiB |

    Each doubling of the steps made the written out softmax 4 times bigger,
    since it holds every score. The tiled version holds one block of scores
    at a time, so its memory grew no faster than the steps. At 8,192 steps,
    blocks of 1,024 needed 24 times less. The paper measured up to 20 times
    less than exact attention, growing linearly with the steps.

    On the CPU, SDPA runs PyTorch's own FlashAttention kernel, written in C++.
    Here are the times on 4 threads, over three runs.

    | Steps | Softmax written out | Tiled, blocks of 1,024 | SDPA |
    |---|---|---|---|
    | 4,096 | 87 to 92 ms | 109 to 118 ms | 54 to 74 ms |
    | 8,192 | 329 to 414 ms | 426 to 494 ms | 198 to 205 ms |

    The kernel beat the softmax written out in every run, in 50 to 80 percent
    of its time. It does every step for a block in one pass, while the block
    sits in the CPU's cache. Your version runs each line as its own job
    through main memory, so it took 1.2 to 1.4 times as long.

    On CUDA, `F.scaled_dot_product_attention` can call a FlashAttention-2
    kernel, so a model gets the method without any of this code.
    """,
)

FLASH2_SOURCE = source_file(
    '''
    # flash2.py

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    starts = list(range(0, 10, 4))                # [0, 4, 8], the last block holds 2
    last = torch.arange(10.0)[8 : 8 + 4]          # tensor([8., 9.]), a slice stops at the end


    # TODO: Finish the inner loop, attention of q, k and v, each of shape (...,
    #   steps, size), one block of keys at a time. The block's scores are rows
    #   times keys transposed, times scale. Raise largest to the block's largest
    #   score in each row. Shrink total and out by the exp of the old largest
    #   minus the new, then add the block's exps to total, and its exps times
    #   values to out. Never build the scores against every key at once.
''' + TILED_START.strip("\n") + '''


    # TODO: Return how many blocks of keys tiled_attention reads in all, for
    #   steps queries and keys cut into blocks of block. Every block of queries
    #   reads every block of keys. With causal, count what a causal version of
    #   tiled_attention would read, each block of queries only the key blocks up
    #   to its own, since the mask hides the ones after it.
    def blocks_read(
        steps: int,
        block: int,
        causal: bool,
    ) -> int:
        ...


    # TODO: Return the most scores one head holds at once. Written out, that
    #   is every query against every key. Tiled, it is one block of queries
    #   against one block of keys, and a block holds at most steps.
    def scores_held(
        steps: int,
        block: int,
        tiled: bool,
    ) -> int:
        ...
    ''',
)

FLASH2_SOLUTION = source_file(
    '''
    # flash2.py, a solution for comparison

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    starts = list(range(0, 10, 4))
    last = torch.arange(10.0)[8 : 8 + 4]


''' + indented(TILED_SOLUTION) + '''


''' + indented(COUNTS_SOLUTION) + '''
    ''',
)


class ScoreShapes(TorchFunctionMode):
    """Notes every tensor a call makes, so a check can see whether scores against every key were ever built."""

    def __init__(self) -> None:
        super().__init__()
        self.made: list[tuple[str, tuple[int, ...], bool]] = []

    def __torch_function__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        result = func(*args, **(kwargs or {}))

        if isinstance(result, Tensor):
            self.made.append((getattr(func, "__name__", ""), tuple(result.shape), result._base is not None))

        return result


def check_no_full_scores(found: object) -> None:
    """Scores against every key at once have the keys' count as their last size. The keys transposed have it
    too, but with the head size before it, and a slice of them is a view, which makes no new numbers."""
    generator = torch.Generator().manual_seed(873)
    q, k, v = (torch.randn(2, 2, 10, 3, generator=generator) for _ in range(3))

    with ScoreShapes() as mode:
        found(q, k, v, 4)

    for name, shape, is_view in mode.made:
        if name == "scaled_dot_product_attention":
            raise Mismatch("tiled_attention called F.scaled_dot_product_attention. Build the blocks' scores yourself.")

        if is_view or len(shape) < 2:
            continue

        if shape[-1] == 10 and shape[-2] not in (1, 3):
            raise Mismatch(
                f"tiled_attention built a tensor of shape {shape}, scores against all 10 keys at once. Take the "
                "scores one block of keys at a time.",
            )


def check_far_apart_blocks(found: object) -> None:
    """The first block of keys scores about 141 above the second. Measured from the second block's own largest,
    the first block's total would have to grow by exp(141), which overflows, so only a raised largest gets through."""
    q = torch.ones(1, 6, 2)
    k = torch.cat([torch.full((1, 3, 2), 50.0), torch.full((1, 3, 2), -50.0)], dim=1)
    v = torch.randn(1, 6, 2, generator=torch.Generator().manual_seed(878))
    expected = F.scaled_dot_product_attention(q, k, v)
    actual = found(q, k, v, 3)

    if isinstance(actual, Tensor) and actual.shape == expected.shape and not torch.isfinite(actual).all():
        raise Mismatch(
            "tiled_attention gave nan when an early block of keys scored far above a later one. Raise largest to "
            "torch.maximum of the old largest and the block's largest, so no exp overflows.",
        )

    expect_tensor(
        call="tiled_attention of q of ones, with the first 3 keys scoring far above the last 3, in blocks of 3",
        actual=actual,
        expected=expected,
    )


def attention_inputs(
    seed: int,
    shape: tuple[int, ...],
    block: int,
) -> tuple[Tensor, Tensor, Tensor, int]:
    generator = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(*shape, generator=generator) for _ in range(3))

    return q, k, v, block


def check_flash2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="tiled_attention",
        description="matches F.scaled_dot_product_attention",
        inputs=tuple(
            (
                Described(q, f"q of shape {tuple(q.shape)}"),
                Described(k, f"k of shape {tuple(k.shape)}"),
                Described(v, f"v of shape {tuple(v.shape)}"),
                block,
            )
            for q, k, v, block in (
                attention_inputs(874, (2, 2, 10, 3), 4),
                attention_inputs(875, (1, 9, 2), 3),
                attention_inputs(876, (1, 5, 3), 8),
            )
        ),
        reference=lambda q, k, v, block: F.scaled_dot_product_attention(q, k, v),
        expect=expect_tensor,
    )
    checker.test(
        function="tiled_attention",
        description="keeps the running largest, so scores far apart never overflow",
        run=check_far_apart_blocks,
    )
    checker.test(
        function="tiled_attention",
        description="never builds the scores against every key at once",
        run=check_no_full_scores,
    )
    check_on_inputs(
        checker=checker,
        function="blocks_read",
        description="counts every block, or the ones up to the diagonal",
        inputs=(
            (1024, 128, False),
            (1024, 128, True),
            (10, 4, False),
            (10, 4, True),
        ),
        reference=TILED_REFERENCE["blocks_read"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="scores_held",
        description="counts the scores of the whole matrix or of one block",
        inputs=(
            (4096, 128, False),
            (4096, 128, True),
            (100, 128, True),
        ),
        reference=TILED_REFERENCE["scores_held"],
        expect=expect_value,
    )


FLASH2 = Exercise(
    name="flash2",
    chapter=FLASH_ATTENTION,
    title="Attention a block at a time",
    lesson=FLASH2_LESSON,
    source=FLASH2_SOURCE,
    solution=FLASH2_SOLUTION,
    hints={
        "tiled_attention": "scores is rows @ keys.transpose(-2, -1) * scale. new_largest is torch.maximum of largest "
        "and scores.amax(dim=-1), shrink is torch.exp(largest - new_largest), and weights is torch.exp of scores "
        "minus new_largest with .unsqueeze(-1). total adds weights.sum(dim=-1), and out adds weights @ values, each "
        "after its shrink.",
        "blocks_read": "math.ceil(steps / block) blocks each way. Query block i reads key blocks 0 to i when causal, "
        "which adds up to blocks * (blocks + 1) // 2.",
        "scores_held": "steps * steps written out. Tiled, a block is min(block, steps) wide, squared.",
    },
    introduces=(),
    check=check_flash2,
)


def median_seconds(
    run: Callable[[], object],
    device: torch.device,
    rounds: int = 5,
    repeats: int = 5,
) -> float:
    """The middle of several timed rounds after a warmup, since other work on the machine makes single runs swing."""

    def wait() -> None:
        if device.type == "mps":
            torch.mps.synchronize()

        if device.type == "cuda":
            torch.cuda.synchronize()

    run()
    wait()
    times = []

    for _ in range(rounds):
        begin = time.perf_counter()

        for _ in range(repeats):
            run()

        wait()
        times.append((time.perf_counter() - begin) / repeats)

    return statistics.median(times)


def measure_attention_times(
    devices: tuple[str, ...] = ("cpu",),
    block: int = 1024,
    lengths: tuple[int, ...] = (4096, 8192),
    threads: int = 4,
) -> dict[tuple[str, int], tuple[float, float, float]]:
    """The times behind flash2's time table, in milliseconds, for 4 heads of size 64: scaled_dot_product_attention,
    the softmax written out, and tiled_attention with blocks of block. Times swing with the machine's load, so
    they are taken with little else running, on as many threads as the memory probe uses."""
    results = {}

    def written_out(
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        return torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1) @ v

    previous = torch.get_num_threads()
    torch.set_num_threads(threads)

    try:
        for name in devices:
            device = torch.device(name)

            for steps in lengths:
                generator = torch.Generator().manual_seed(877)
                q, k, v = (torch.randn(1, 4, steps, 64, generator=generator).to(device) for _ in range(3))

                with torch.no_grad():
                    results[(name, steps)] = tuple(
                        1000 * median_seconds(run, device, rounds=15, repeats=1)
                        for run in (
                            lambda: F.scaled_dot_product_attention(q, k, v),
                            lambda: written_out(q, k, v),
                            lambda: TILED_REFERENCE["tiled_attention"](q, k, v, block),
                        )
                    )
    finally:
        torch.set_num_threads(previous)

    return results


# Run in a fresh process for each call, so the peak it reports belongs to that call alone. The kernels are
# warmed on a tiny input first, so their own setup isn't counted.
MEMORY_PROBE = '''
import math
import resource
import sys

import torch
import torch.nn.functional as F
from torch import Tensor

TILED_SOLUTION

steps, kind = int(sys.argv[1]), sys.argv[2]
torch.set_num_threads(4)
generator = torch.Generator().manual_seed(877)
q, k, v = (torch.randn(1, 4, steps, 64, generator=generator) for _ in range(3))


def attend(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    if kind == "written out":
        return torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1) @ v

    if kind == "sdpa":
        return F.scaled_dot_product_attention(q, k, v)

    return tiled_attention(q, k, v, int(kind))


with torch.no_grad():
    attend(q[..., :8, :], k[..., :8, :], v[..., :8, :])
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    attend(q, k, v)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# macOS counts ru_maxrss in bytes and Linux in kilobytes.
print((after - before) * (1 if sys.platform == "darwin" else 1024))
'''


def measure_attention_memory(
    lengths: tuple[int, ...] = (2048, 4096, 8192),
    kinds: tuple[str, ...] = ("written out", "1024", "128", "sdpa"),
) -> dict[tuple[int, str], int]:
    """The peak memory one call adds on the CPU, in bytes, for 4 heads of size 64. A kind is the softmax written
    out, a block size for tiled_attention, or sdpa. Peak memory doesn't swing with the machine's load as time does,
    but it needs a machine with memory to spare. ru_maxrss counts only resident pages, so under memory pressure macOS
    compresses some of a big call's pages and the peak reads low. On a 16 GB Mac with another heavy session running,
    the written out softmax at 8,192 steps read 1,249 to 1,941 MiB against its 2,048 MiB of scores and weights."""
    code = MEMORY_PROBE.replace("TILED_SOLUTION", TILED_SOLUTION)
    results = {}

    for steps in lengths:
        for kind in kinds:
            run = subprocess.run(
                [sys.executable, "-I", "-c", code, str(steps), kind],
                capture_output=True,
                text=True,
                check=True,
            )
            results[(steps, kind)] = int(run.stdout.strip())

    return results
