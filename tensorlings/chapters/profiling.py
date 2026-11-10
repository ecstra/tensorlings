"""85_profiling, timing GPU work with warmup and a synchronize, torch.profiler's count of every operation, and
the memory bound against the compute bound with why fusing elementwise steps helps."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
    reject_missing_return,
    small_tensor,
)
from ..text import prose, source_file
from ..training import best_device
from .gpt import MODEL_REFERENCE

PROFILING = Chapter(
    folder="85_profiling",
    title="Profiling",
    readme=source_file(
        """
        # Profiling

        Counting tells you what a model should cost. Timing tells you what it
        actually costs, and timing work on a GPU has traps of its own. This
        chapter shows how to time work correctly, and how to break a run down
        by operation with `torch.profiler`. It also shows how to tell whether
        an operation is limited by the chip's arithmetic speed or by its
        memory speed.

        ## Further reading

        - [PyTorch, the profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
        - [Williams, Waterman and Patterson, the roofline model](https://doi.org/10.1145/1498765.1498785)
        - [Horace He, making deep learning go brrrr from first principles](https://horace.io/brrr_intro.html)
        """,
    ),
)

# The timing functions profile1 builds. Section 12's fast GPT project times its generation with them.
TIMING_PART = '''
def synchronize(
    device: torch.device,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()

    if device.type == "mps":
        torch.mps.synchronize()


def time_call(
    run: Callable[[], object],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        run()

    synchronize(device)
    seconds: list[float] = []

    for _ in range(repeats):
        start = time.perf_counter()
        run()
        synchronize(device)
        seconds.append(time.perf_counter() - start)

    return statistics.median(seconds)
'''


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "statistics": statistics,
        "time": time,
        "torch": torch,
        "F": F,
        "Tensor": Tensor,
        "Callable": Callable,
        "ProfilerActivity": ProfilerActivity,
        "profile": profile,
    }
    exec(code, namespace)

    return namespace


# ---------------------------------------------------------------------------
# profile1
# ---------------------------------------------------------------------------

PROFILE1_LESSON = prose(
    """
    Timing code on a GPU has two traps. The first is that a PyTorch call on
    the GPU returns before the work is done. The call only adds the work to a
    queue, and Python moves on to the next line while the GPU runs it.

    So if you read the clock right after the call, you measure only the time
    it took to queue the work. You have to wait until the queue is empty
    before you read the clock, and that wait is called synchronizing.
    `torch.mps.synchronize()` waits for a Mac's GPU, and
    `torch.cuda.synchronize()` waits for an Nvidia GPU. On the CPU, each call
    finishes its work before it returns, so there is nothing to wait for.

    The second trap is the first call. It sets up memory and the program the
    GPU runs for the operation, so it is far slower than later calls. A few
    warmup calls that you do not time get that setup out of the way. The
    program the GPU runs for one operation is called a kernel. That is a
    different meaning of the word from the sliding weights in conv1.

    Here are timings of a 2,048 by 2,048 matrix multiply in float32 on this
    Mac's GPU, an Apple M5. This exercise's functions timed it in three fresh
    runs, with little else running on the machine.

    | How it was timed | Milliseconds |
    |---|---|
    | After warmup, no synchronize | 0.02 |
    | After warmup, with a synchronize | 6.4 to 7.9 |
    | The first call of the run, with a synchronize | 25 to 27 |

    Timings vary from run to run, because other programs share the machine.
    So time several calls and take the median, the middle value after
    sorting. One unusually slow call barely changes the median.
    `torch.utils.benchmark` also reports the median.

    `torch.profiler` breaks a run down by operation. Inside a
    `with profile(activities=[ProfilerActivity.CPU])` block, it records every
    operation that runs.

    ```python
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        torch.ones(4, 4) @ torch.ones(4, 4)
    rows = profiler.key_averages()    # one row per kind of operation, including aten::mm
    ```

    Each row holds the operation's name in `key`, the number of times it was
    called in `count`, and its times. A key that starts with `aten::` names
    one of PyTorch's own operations.

    Here is a profile of a forward pass of the GPT that gpt3 trains, run on
    the CPU with 4 threads. Each share is the operation's fraction of the
    CPU time over 10 passes, counting only time spent in the operation
    itself. The ranges come from three runs.

    | Operation | Calls | Share of CPU time |
    |---|---|---|
    | `aten::addmm`, the Linears of the blocks | 16 | 29% to 30% |
    | `aten::gelu` | 4 | 29% to 31% |
    | attention, one kernel for all its steps | 4 | 23% to 24% |
    | `aten::copy_`, copies into new memory | 16 | 5% to 6% |

    The call counts are the same in every run. The 4 GELUs do a few FLOPs on
    each of 2.1 million numbers, while the 16 Linears do 1.6 billion FLOPs.
    Yet the GELUs took 98% to 107% as long as the Linears. With 1 thread,
    GELU took 47% of the time, and with 10 threads it took 27% to 28%.
    `torch.profiler` cannot record work on a Mac's GPU. So on MPS, its times
    would show only the CPU queueing the work.
    """,
)

PROFILE1_SOURCE = source_file(
    '''
    # profile1.py

    import statistics
    import time
    from collections.abc import Callable

    import torch
    from torch.profiler import ProfilerActivity, profile

    kind = torch.device("mps").type                      # "mps", or "cuda" or "cpu" for those devices
    middle = statistics.median([0.3, 0.1, 0.9])          # 0.3, the middle value after sorting
    start = time.perf_counter()                          # seconds since a fixed point, so subtract two readings for a duration
    rows = [("aten::mm", 3), ("aten::relu", 3)]
    relu_calls = next((count for key, count in rows if key == "aten::relu"), 0)    # 3


    # TODO: Wait until all the work queued on device is done. Call
    #   torch.cuda.synchronize() for a "cuda" device and torch.mps.synchronize()
    #   for an "mps" device. Do nothing for the CPU, which never queues work.
    def synchronize(
        device: torch.device,
    ) -> None:
        ...


    # TODO: Return the median number of seconds one call of run takes on
    #   device, as a plain float. First call run warmup times without timing
    #   it, then synchronize. Then time repeats calls. For each one, read
    #   time.perf_counter() before the call, and read it again after the call
    #   and a synchronize.
    def time_call(
        run: Callable[[], object],
        device: torch.device,
        warmup: int,
        repeats: int,
    ) -> float:
        ...


    # TODO: Return how many times run calls the operation named name, like
    #   "aten::mm", as a plain int. Call run inside a profile block with
    #   activities=[ProfilerActivity.CPU]. After the with block ends, find the
    #   row of key_averages() whose key equals name, and return its count.
    #   Return 0 when no row has that key.
    #   Example: calls_of(lambda: torch.ones(4, 4) @ torch.ones(4, 4), "aten::mm")
    #   returns 1.
    def calls_of(
        run: Callable[[], object],
        name: str,
    ) -> int:
        ...
    ''',
)

CALLS_PART = '''
def calls_of(
    run: Callable[[], object],
    name: str,
) -> int:
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        run()

    return next((row.count for row in profiler.key_averages() if row.key == name), 0)
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return "\n".join(f"    {line}" if line else "" for line in part.strip("\n").splitlines())


PROFILE1_SOLUTION = source_file(
    '''
    # profile1.py, a solution for comparison

    import statistics
    import time
    from collections.abc import Callable

    import torch
    from torch.profiler import ProfilerActivity, profile

    kind = torch.device("mps").type
    middle = statistics.median([0.3, 0.1, 0.9])
    start = time.perf_counter()
    rows = [("aten::mm", 3), ("aten::relu", 3)]
    relu_calls = next((count for key, count in rows if key == "aten::relu"), 0)


''' + indented(TIMING_PART) + '''


''' + indented(CALLS_PART) + '''
    ''',
)

PROFILE_REFERENCE = namespace_of(TIMING_PART + CALLS_PART)


class PassThrough:
    """A module with a few names swapped and every other name passed through to the real one, so a call the
    check doesn't watch still works and a wrong one fails with the check's own message."""

    def __init__(
        self,
        module: object,
        **swapped: object,
    ) -> None:
        self.module = module
        self.__dict__.update(swapped)

    def __getattr__(
        self,
        name: str,
    ) -> object:
        return getattr(self.module, name)


class ScriptedTorch:
    """Stands in for torch in the learner's file, so a check sees which backend a synchronize waits on.
    Everything else is the real torch."""

    def __init__(
        self,
        waits: list[str],
    ) -> None:
        self.cuda = SimpleNamespace(
            synchronize=lambda *arguments, **keywords: waits.append("cuda"),
            is_available=lambda: True,
        )
        self.mps = SimpleNamespace(
            synchronize=lambda *arguments, **keywords: waits.append("mps"),
        )

    def __getattr__(
        self,
        name: str,
    ) -> object:
        return getattr(torch, name)


def check_synchronize(
    found: Callable[..., object],
    device: torch.device,
    expected: list[str],
) -> None:
    waits: list[str] = []
    original = found.__globals__.get("torch")
    found.__globals__["torch"] = ScriptedTorch(waits)

    try:
        found(device)
    finally:
        found.__globals__["torch"] = original

    if waits == expected:
        return

    wanted = "nothing" if not expected else f"torch.{expected[0]}.synchronize() once"
    called = "nothing" if not waits else ", then ".join(f"torch.{backend}.synchronize()" for backend in waits)

    raise Mismatch(f'synchronize(torch.device("{device.type}")) called {called}, but it should call {wanted}.')


class ScriptedTiming:
    """A clock and a synchronize that write down every event, so a check can see the order of a timing loop.
    The clock's second reading of each pair comes the next gap after the first."""

    def __init__(
        self,
        gaps: list[float],
    ) -> None:
        self.gaps = gaps
        self.events: list[str] = []
        self.readings = 0
        self.now = 100.0

    def perf_counter(self) -> float:
        self.events.append("clock")
        pair = self.readings // 2
        self.now += self.gaps[pair % len(self.gaps)] if self.readings % 2 == 1 else 1.0
        self.readings += 1

        return self.now

    def synchronize(
        self,
        device: torch.device,
    ) -> None:
        self.events.append("sync")

    def run(self) -> None:
        self.events.append("run")


def check_time_call(
    found: Callable[..., object],
    warmup: int,
    gaps: list[float],
) -> None:
    script = ScriptedTiming(gaps)
    repeats = len(gaps)
    swapped: dict[str, object] = {
        "time": PassThrough(time, perf_counter=script.perf_counter),
        "synchronize": script.synchronize,
    }

    # A file that imported perf_counter by name reads that name, not time.perf_counter.
    if "perf_counter" in found.__globals__:
        swapped["perf_counter"] = script.perf_counter

    originals = {name: found.__globals__.get(name) for name in swapped}
    found.__globals__.update(swapped)

    try:
        actual = found(script.run, torch.device("mps"), warmup, repeats)
    finally:
        found.__globals__.update(originals)

    call = f"time_call(run, a GPU, warmup={warmup}, repeats={repeats})"
    events = script.events

    if not events:
        reject_missing_return(
            call=call,
            actual=actual,
        )

    runs = events.count("run")

    if runs != warmup + repeats:
        raise Mismatch(f"{call} called run {runs} times, but it should call it {warmup + repeats} times, warmup plus repeats.")

    clocks = [position for position, event in enumerate(events) if event == "clock"]

    if len(clocks) != 2 * repeats:
        raise Mismatch(f"{call} read time.perf_counter() {len(clocks)} times, but it should read it twice for each of the {repeats} timed calls.")

    before_timing = events[: clocks[0]]

    if before_timing.count("run") != warmup:
        raise Mismatch(f"{call} should make all {warmup} warmup calls before it first reads the clock.")

    last_warmup = max((position for position, event in enumerate(before_timing) if event == "run"), default=-1)

    if warmup and "sync" not in before_timing[last_warmup + 1 :]:
        raise Mismatch(f"{call} started the clock while the warmup's work could still be queued. Synchronize after the warmup calls.")

    for start, stop in zip(clocks[::2], clocks[1::2]):
        between = events[start + 1 : stop]

        if between.count("run") != 1:
            raise Mismatch(f"{call} should call run exactly once between the two clock readings of each timed call.")

        if between[-1] != "sync":
            raise Mismatch(f"{call} read the clock again before the GPU was done. Synchronize after each timed call, before the second reading.")

    expect_close(
        call=f"{call}, with timed calls of {', '.join(f'{gap:g}' for gap in gaps)} seconds,",
        actual=actual,
        expected=statistics.median(gaps),
    )


class ScriptedProfile:
    """Stands in for torch.profiler.profile, whose rows are scripted. The real profiler writes lines of its own
    to the terminal, which would spoil the app's screen."""

    def __init__(
        self,
        rows: list[tuple[str, int]],
        events: list[str],
    ) -> None:
        self.rows = rows
        self.events = events

    def __call__(
        self,
        *arguments: object,
        **keywords: object,
    ) -> ScriptedProfile:
        return self

    def __enter__(self) -> ScriptedProfile:
        self.events.append("enter")

        return self

    def __exit__(
        self,
        *exception: object,
    ) -> None:
        self.events.append("exit")

    def key_averages(self) -> list[SimpleNamespace]:
        self.events.append("read")

        return [SimpleNamespace(key=key, count=count) for key, count in self.rows]


def check_calls_of(
    found: Callable[..., object],
    rows: list[tuple[str, int]],
    name: str,
) -> None:
    events: list[str] = []
    scripted = ScriptedProfile(rows, events)
    original = found.__globals__.get("profile")
    found.__globals__["profile"] = scripted

    # A file can also reach the profiler as torch.profiler.profile, which would print to the screen.
    original_module_profile = torch.profiler.profile
    torch.profiler.profile = scripted

    try:
        actual = found(lambda: events.append("run"), name)
    finally:
        found.__globals__["profile"] = original
        torch.profiler.profile = original_module_profile

    call = f'calls_of(run, "{name}") on a run whose profile has rows {", ".join(f"{key} x{count}" for key, count in rows)}'

    if not events:
        reject_missing_return(
            call=call,
            actual=actual,
        )

    if "enter" not in events or "exit" not in events:
        raise Mismatch(f"{call} should call run inside a with profile(...) block.")

    enter = events.index("enter")
    leave = events.index("exit")

    if events[enter + 1 : leave].count("run") != 1:
        raise Mismatch(f"{call} should call run exactly once inside the with profile(...) block. A warmup call before it is fine.")

    if "read" not in events[leave + 1 :]:
        raise Mismatch(f"{call} should read key_averages() after the with block ends, once the profiler has stopped.")

    expected = next((count for key, count in rows if key == name), 0)

    if isinstance(actual, float):
        raise Mismatch(f"{call} is {actual!r}, a float, but it should be the plain int {expected}, the row's count.")

    expect_value(
        call=call,
        actual=actual,
        expected=expected,
    )


GPT_ROWS = [("aten::linear", 17), ("aten::addmm", 16), ("aten::mm", 1), ("aten::layer_norm", 9)]


def check_profile1(checker: Checker) -> None:
    for device, expected in (("mps", ["mps"]), ("cuda", ["cuda"]), ("cpu", [])):
        checker.test(
            function="synchronize",
            description="waits on the device's own backend, and does nothing for the CPU",
            run=lambda found, device=device, expected=expected: check_synchronize(
                found=found,
                device=torch.device(device),
                expected=expected,
            ),
        )

    for warmup, gaps in ((3, [0.5, 0.2, 0.9, 0.3, 0.4]), (0, [0.7, 0.1, 0.2]), (1, [0.25])):
        checker.test(
            function="time_call",
            description="warms up, synchronizes around each timed call and returns the median",
            run=lambda found, warmup=warmup, gaps=gaps: check_time_call(
                found=found,
                warmup=warmup,
                gaps=gaps,
            ),
        )

    for rows, name in ((GPT_ROWS, "aten::addmm"), (GPT_ROWS, "aten::layer_norm"), (GPT_ROWS, "aten::softmax")):
        checker.test(
            function="calls_of",
            description="reads one operation's count from the profile",
            run=lambda found, rows=rows, name=name: check_calls_of(
                found=found,
                rows=rows,
                name=name,
            ),
        )


PROFILE1 = Exercise(
    name="profile1",
    chapter=PROFILING,
    title="Timing done right",
    lesson=PROFILE1_LESSON,
    source=PROFILE1_SOURCE,
    solution=PROFILE1_SOLUTION,
    hints={
        "synchronize": 'Compare device.type with "cuda" and with "mps".',
        "time_call": "Make the warmup calls in a loop, then call synchronize(device). In the timed loop, read the "
        "clock, call run, synchronize, read the clock again, and keep the difference. Return statistics.median of the "
        "differences.",
        "calls_of": "Open the profile block the way the lesson's example does, and call run() inside it. After the "
        "block, loop over profiler.key_averages() and compare each row.key with name.",
    },
    introduces=(),
    check=check_profile1,
)


# ---------------------------------------------------------------------------
# profile2
# ---------------------------------------------------------------------------

PROFILE2_LESSON = prose(
    """
    An operation can be held back by two things, the chip's arithmetic or
    its memory. Every input has to be read from memory, and every output has
    to be written back. Memory moves bytes at a fixed maximum rate, called
    its bandwidth.

    The arithmetic intensity of an operation is the number of FLOPs it does
    per byte it moves. Doubling a float32 tensor does 1 FLOP per 8 bytes,
    because each number is 4 bytes read and 4 bytes written. An n by n matrix
    multiply does 2n³ FLOPs but moves only 3n² numbers, for the two inputs and
    the output. So its intensity grows with n.

    Here are both kinds of operation on this Mac's GPU, timed with profile1's
    `time_call`. Each range comes from three runs with little else running.

    | Operation, in float32 | FLOPs per byte | Speed |
    |---|---|---|
    | Doubling 67 million numbers | 0.125 | 120 to 129 GB/s moved |
    | 256 by 256 matrix multiply | 43 | 156 to 172 GFLOP/s |
    | 1,024 by 1,024 matrix multiply | 171 | 2,154 to 2,218 GFLOP/s |
    | 2,048 by 2,048 matrix multiply | 341 | 2,119 to 2,627 GFLOP/s |
    | 4,096 by 4,096 matrix multiply | 683 | 2,007 to 2,598 GFLOP/s |

    The roofline model of Williams, Waterman and Patterson turns this into a
    rule. An operation takes at least `flops / peak_flops_per_second`
    seconds, because the chip cannot compute any faster. It also takes at
    least `bytes / bandwidth` seconds, because memory cannot move bytes any
    faster. The larger of these two lower bounds is the roofline's time for
    the operation.

    Here the best rates measured, about 2.6 trillion FLOPs a second and 129
    GB a second, stand in for the peaks. The ridge point is the intensity at
    which the two bounds are equal, which is the peak FLOP rate divided by
    the bandwidth. Here that is near 20 FLOPs per byte. An operation whose
    intensity is below the ridge is limited by memory.

    The 256 by 256 multiply runs far slower than its roofline. Its 34 million
    FLOPs take less time than the fixed cost of launching a kernel and
    waiting for it to finish. Horace He calls this third limit overhead.

    Elementwise operations have an intensity far below the ridge, so their
    time is the time it takes to move their bytes. A chain of elementwise
    operations, each run as its own kernel, reads and writes the whole tensor
    at every step. Fusing the chain means running it as one kernel, which
    reads the input once and writes the result once.

    The tanh form of GELU is a good test of this.

    ```text
    gelu(x) = 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x³)))
    ```

    Written as separate PyTorch operations, the formula runs 8 operations,
    each as its own kernel. `F.gelu(x, approximate="tanh")` computes the same
    formula in one kernel.

    | GELU of 16.8 million numbers | Bytes moved | Roofline at 129 GB/s | Measured |
    |---|---|---|---|
    | 8 separate steps | 1.21 GB | 9.4 ms | 9.7 to 10.8 ms |
    | `F.gelu`, one kernel | 134 MB | 1.0 ms | 1.2 ms |

    The 8 separate steps read 10 tensors and write 8, so a whole tensor goes
    through memory 18 times. With the single kernel it goes through only 2
    times. Both measured times are within a fifth of their rooflines, so the
    difference between them comes from memory traffic. Horace He calls fusion
    the most important optimization that a deep learning compiler does.
    """,
)

PROFILE2_SOURCE = source_file(
    '''
    # profile2.py

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    intensity = 100 / 800                                # 0.125 FLOPs per byte, for 100 FLOPs on 800 bytes
    slowest = max(2.0, 3.5)                              # 3.5, the larger of two lower bounds
    root = math.sqrt(2 / math.pi)                        # 0.7978845608028654
    curve = torch.tanh(torch.tensor([0.0, 1.0]))         # tensor([0.0000, 0.7616])


    # TODO: Return the FLOPs and the bytes moved for multiplying two n by n
    #   matrices, as a tuple of two plain ints, (flops, bytes). Each number in
    #   the two inputs and the output takes bytes_per_number bytes.
    #   Example: matmul_cost(4, 4) returns (128, 192).
    def matmul_cost(
        n: int,
        bytes_per_number: int,
    ) -> tuple[int, int]:
        ...


    # TODO: Return the roofline's time for an operation in seconds, as a plain
    #   float. That is the larger of two lower bounds, flops / flops_per_second
    #   and bytes_moved / bytes_per_second.
    #   Example: roofline_seconds(100, 100, 10.0, 1000.0) returns 10.0.
    def roofline_seconds(
        flops: int,
        bytes_moved: int,
        flops_per_second: float,
        bytes_per_second: float,
    ) -> float:
        ...


    # TODO: Return "memory" when bytes_moved / bytes_per_second is larger than
    #   flops / flops_per_second, which means moving the bytes takes longer
    #   than doing the FLOPs. Return "compute" otherwise.
    def bound_by(
        flops: int,
        bytes_moved: int,
        flops_per_second: float,
        bytes_per_second: float,
    ) -> str:
        ...


    # TODO: Return the tanh form of GELU applied to x, using the formula from
    #   the lesson. Write it as separate torch operations, without F.gelu.
    #   Example: gelu_by_steps(torch.tensor([1.0])) returns about
    #   tensor([0.8412]).
    def gelu_by_steps(
        x: Tensor,
    ) -> Tensor:
        ...
    ''',
)

ROOFLINE_PART = '''
def matmul_cost(
    n: int,
    bytes_per_number: int,
) -> tuple[int, int]:
    return 2 * n**3, 3 * n**2 * bytes_per_number


def roofline_seconds(
    flops: int,
    bytes_moved: int,
    flops_per_second: float,
    bytes_per_second: float,
) -> float:
    return max(flops / flops_per_second, bytes_moved / bytes_per_second)


def bound_by(
    flops: int,
    bytes_moved: int,
    flops_per_second: float,
    bytes_per_second: float,
) -> str:
    if bytes_moved / bytes_per_second > flops / flops_per_second:
        return "memory"

    return "compute"


def gelu_by_steps(
    x: Tensor,
) -> Tensor:
    inner = math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)

    return 0.5 * x * (1 + torch.tanh(inner))
'''

PROFILE2_SOLUTION = source_file(
    '''
    # profile2.py, a solution for comparison

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    intensity = 100 / 800
    slowest = max(2.0, 3.5)
    root = math.sqrt(2 / math.pi)
    curve = torch.tanh(torch.tensor([0.0, 1.0]))


''' + indented(ROOFLINE_PART) + '''
    ''',
)

ROOFLINE_REFERENCE = namespace_of(ROOFLINE_PART)


class GeluCalled(Exception):
    pass


def refuse_gelu(
    *arguments: object,
    **keywords: object,
) -> Tensor:
    raise GeluCalled


def check_gelu_by_steps(
    found: Callable[..., object],
    x: Tensor,
) -> None:
    """The real F.gelu is swapped out for the call, so nn.GELU and torch.nn.functional.gelu are caught too, while
    every other function of F still works."""
    expected = F.gelu(x, approximate="tanh")
    original = torch.nn.functional.gelu
    torch.nn.functional.gelu = refuse_gelu

    try:
        actual = found(x.clone())
    except GeluCalled:
        raise Mismatch("gelu_by_steps called PyTorch's own GELU. Write the formula out as separate torch operations, as the lesson shows.") from None
    finally:
        torch.nn.functional.gelu = original

    expect_tensor(
        call=f"gelu_by_steps({x})",
        actual=actual,
        expected=expected,
    )


def expect_pair_of_ints(
    call: str,
    actual: object,
    expected: tuple[int, int],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if isinstance(actual, list):
        raise Mismatch(f"{call} is {actual!r}, a list, but it should be a tuple, like (flops, bytes).")

    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} is {actual!r}, but it should be a tuple of two, like (flops, bytes).")

    for part, got, wanted in zip(("FLOPs", "bytes"), actual, expected):
        if isinstance(got, float):
            raise Mismatch(
                f"The {part} of {call} is {got!r}, a float, but it should be the plain int {wanted}. Multiply whole "
                "numbers only, or turn the result into an int with int(...).",
            )

        expect_value(
            call=f"the {part} of {call}",
            actual=got,
            expected=wanted,
        )


def check_profile2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="matmul_cost",
        description="counts 2n³ FLOPs and the bytes of three n by n matrices",
        inputs=(
            (4096, 4),
            (3, 2),
        ),
        reference=ROOFLINE_REFERENCE["matmul_cost"],
        expect=expect_pair_of_ints,
    )
    big_matmul = (Described(2 * 4096**3, "2 * 4096**3"), Described(3 * 4096**2 * 4, "3 * 4096**2 * 4"))
    doubling = (Described(2**26, "2**26"), Described(2**29, "2**29"))
    small_matmul = (Described(2 * 64**3, "2 * 64**3"), Described(3 * 64**2 * 4, "3 * 64**2 * 4"))
    rates = (Described(4e12, "4e12"), Described(1e11, "1e11"))

    check_on_inputs(
        checker=checker,
        function="roofline_seconds",
        description="takes the larger of the two lower bounds",
        inputs=(
            (*big_matmul, *rates),
            (*doubling, *rates),
            (100, 100, 10.0, 1000.0),
        ),
        reference=ROOFLINE_REFERENCE["roofline_seconds"],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="bound_by",
        description="names whichever of the two bounds takes longer",
        inputs=(
            (*big_matmul, *rates),
            (*doubling, *rates),
            (*small_matmul, *rates),
        ),
        reference=ROOFLINE_REFERENCE["bound_by"],
        expect=expect_value,
    )

    for x in (small_tensor(850, 2, 5) / 2, torch.linspace(-4, 4, 9)):
        checker.test(
            function="gelu_by_steps",
            description="matches F.gelu's tanh form",
            run=lambda found, x=x: check_gelu_by_steps(
                found=found,
                x=x,
            ),
        )


PROFILE2 = Exercise(
    name="profile2",
    chapter=PROFILING,
    title="Memory bound and compute bound",
    lesson=PROFILE2_LESSON,
    source=PROFILE2_SOURCE,
    solution=PROFILE2_SOLUTION,
    hints={
        "matmul_cost": "There are n * n outputs with n terms each, and each term costs two FLOPs. Three matrices of "
        "n * n numbers move through memory.",
        "roofline_seconds": "Take the max of the two divisions.",
        "bound_by": "Compare the two times, bytes_moved divided by its rate and flops divided by its rate.",
        "gelu_by_steps": "First compute the part inside tanh, which is math.sqrt(2 / math.pi) times the sum of x and "
        "0.044715 times x cubed. Then apply torch.tanh to it and finish the formula.",
    },
    introduces=(),
    check=check_profile2,
)


# ---------------------------------------------------------------------------
# The numbers in the lessons
# ---------------------------------------------------------------------------


def measure_first_call() -> dict[str, float]:
    """profile1's matrix multiply timings in milliseconds. Run it in a fresh process, so the first call is the
    first of its kind. The lesson gives the range of three such runs."""
    device = best_device()
    synchronize = PROFILE_REFERENCE["synchronize"]
    time_call = PROFILE_REFERENCE["time_call"]
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(2048, 2048, generator=generator).to(device)
    b = torch.randn(2048, 2048, generator=generator).to(device)
    synchronize(device)

    start = time.perf_counter()
    a @ b
    synchronize(device)
    first = time.perf_counter() - start

    with_wait = time_call(lambda: a @ b, device, 5, 21)
    no_wait = time_call(lambda: a @ b, torch.device("cpu"), 5, 21)
    synchronize(device)

    return {
        "first call": 1000 * first,
        "with a synchronize": 1000 * with_wait,
        "no synchronize": 1000 * no_wait,
    }


def measure_profile(
    threads: int = 4,
    passes: int = 10,
) -> list[tuple[str, int, float]]:
    """profile1's rows, a forward pass of gpt3's GPT on the CPU, with each operation's calls per pass and its share
    of the CPU's own time over passes passes, largest first. The lesson gives the range of three runs on 4 threads.
    The shares move a lot with the thread count, so it is pinned, and a single pass swung too much on a busy machine.
    Unpinned, GELU took 47% on 1 thread and 27% to 28% on 10."""
    torch.manual_seed(0)
    model = MODEL_REFERENCE["GPT"](65, 128, 128, 4, 4)
    ids = torch.randint(0, 65, (8, 128), generator=torch.Generator().manual_seed(0))
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)

    try:
        with torch.no_grad():
            model(ids)

            with profile(activities=[ProfilerActivity.CPU]) as profiler:
                for _ in range(passes):
                    model(ids)
    finally:
        torch.set_num_threads(previous)

    rows = sorted(profiler.key_averages(), key=lambda row: row.self_cpu_time_total, reverse=True)
    total = sum(row.self_cpu_time_total for row in rows)

    return [(row.key, row.count // passes, row.self_cpu_time_total / total) for row in rows[:6]]


def measure_roofline() -> dict[str, object]:
    """profile2's numbers on the best device, rates in billions a second. The lesson gives the range of three runs,
    and its roofline uses the best rates seen. The 2,048 multiply is profile1's, so its rate shows whether
    profile1's times and this table came from an equally quiet machine."""
    device = best_device()
    time_call = PROFILE_REFERENCE["time_call"]
    matmul_cost = ROOFLINE_REFERENCE["matmul_cost"]
    gelu_by_steps = ROOFLINE_REFERENCE["gelu_by_steps"]
    generator = torch.Generator().manual_seed(0)
    results: dict[str, object] = {}

    x = torch.randn(2**26, generator=generator).to(device)
    seconds = time_call(lambda: x * 2, device, 5, 21)
    results["doubling GB/s"] = 2 * x.numel() * 4 / seconds / 1e9

    for n in (256, 1024, 2048, 4096):
        a = torch.randn(n, n, generator=generator).to(device)
        b = torch.randn(n, n, generator=generator).to(device)
        flops, bytes_moved = matmul_cost(n, 4)
        seconds = time_call(lambda: a @ b, device, 5, 21)
        results[f"matmul {n} GFLOP/s"] = flops / seconds / 1e9
        results[f"matmul {n} intensity"] = flops / bytes_moved

    x = torch.randn(2**24, generator=generator).to(device)
    results["gelu steps ms"] = 1000 * time_call(lambda: gelu_by_steps(x), device, 5, 21)
    results["gelu fused ms"] = 1000 * time_call(lambda: F.gelu(x, approximate="tanh"), device, 5, 21)

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        gelu_by_steps(x.cpu()[:16])

    steps = ("aten::pow", "aten::mul", "aten::add", "aten::tanh")
    results["gelu steps"] = sum(row.count for row in profiler.key_averages() if row.key in steps)

    return results
