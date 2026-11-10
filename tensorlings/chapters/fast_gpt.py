"""93_fast_gpt, Section 12's project: your GPT's writing timed, then made faster with the KV cache and int8 weights,
and measured again on speed, validation loss and size."""

from __future__ import annotations

import functools
import math
import statistics
import textwrap
import time
from collections.abc import Callable
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..data import load_shakespeare
from ..engine import (
    Checker,
    Mismatch,
    expect_close,
    expect_tensor,
    expect_value,
    reject_missing_return,
    value_text,
)
from ..text import prose, source_file
from ..training import load_checkpoint
from .gpt import (
    DATA_PART,
    GIVEN_MODEL,
    GPT_FORWARD,
    GPT_INIT,
    GPT_SETTINGS,
    INIT_SOLUTION,
    MODEL_REFERENCE,
    SAMPLING_PARTS,
    seeded_gpt,
    trained_gpt,
)
from .kv_cache import CACHED_MODEL_PART, GENERATE_SOLUTION
from .profiling import TIMING_PART
from .quantization import INT8_PART, VALIDATION_PART

FAST_GPT = Chapter(
    folder="93_fast_gpt",
    title="Project: a faster GPT",
    readme=source_file(
        """
        # Project: a faster GPT

        This is the project for Section 12. You measure how fast your GPT from
        gpt3 writes text. Then you add the KV cache from kvcache1 and the int8
        weights from quant3, and check whether each one changes the text the
        model writes. You measure speed, validation loss and model size before
        and after. Which change actually helps depends on the device, so the
        measuring is the point of the project.

        ## Further reading

        - [Pope and colleagues, efficiently scaling transformer inference](https://arxiv.org/abs/2211.05102)
        - [Dettmers and colleagues, LLM.int8()](https://arxiv.org/abs/2208.07339)
        - [Frantar and colleagues, GPTQ, section on practical speedups](https://arxiv.org/abs/2210.17323)
        - [PyTorch, GPT fast, the int8 step and why its kernel must be fused](https://pytorch.org/blog/accelerating-generative-ai-2/)
        - [Horace He, making deep learning go brrrr from first principles](https://horace.io/brrr_intro.html)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "statistics": statistics,
        "time": time,
        "Callable": Callable,
        "deepcopy": deepcopy,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "load_shakespeare": load_shakespeare,
    }
    exec(code, namespace)

    return namespace


IMPORTS = '''
    import math
    import statistics
    import time
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import load_checkpoint
'''

GIVEN_GPT = (
    indented(GPT_SETTINGS)
    + "\n\n\n    # The model from gpt2.\n"
    + indented(GIVEN_MODEL)
    + "\n\n\n"
    + indented(INIT_SOLUTION)
    + "\n\n\n"
    + indented(GPT_INIT)
    + "\n\n"
    + indented(GPT_FORWARD)
    + "\n\n\n    # get_batch from gpt1 and the training and validation tokens.\n"
    + indented(DATA_PART)
    + "\n\n\n    # top_k_filter, next_token and generate from gpt4.\n"
    + indented(SAMPLING_PARTS)
)

GIVEN_TIMING = "    # synchronize and time_call from profile1.\n" + indented(TIMING_PART)

GIVEN_FAST = (
    "    # kvcache1's cached model and cached_generate.\n"
    + indented(CACHED_MODEL_PART)
    + "\n\n\n"
    + indented(GENERATE_SOLUTION)
    + "\n\n\n    # quant3's int8 Linear, with per_channel_quantize from quant2 and dequantize from quant1.\n"
    + indented(INT8_PART)
)

GIVEN_VALIDATION = "    # validation_loss from gpt3.\n" + indented(VALIDATION_PART)

LOADED_MODEL = '''
    characters, _, validation_data = shakespeare_splits()
    model = load_checkpoint(GPT(len(characters), BLOCK_SIZE, SIZE, LAYERS, HEADS), "gpt3")
    prompt = torch.tensor([[characters.index(character) for character in "ROMEO:"]])
    cpu = torch.device("cpu")
'''


# ---------------------------------------------------------------------------
# fast1
# ---------------------------------------------------------------------------

SPEED_SOLUTION = '''
def tokens_per_second(
    write: Callable[..., Tensor],
    model: nn.Module,
    prompt: Tensor,
    steps: int,
    device: torch.device,
) -> float:
    model = model.to(device)
    ids = prompt.to(device)
    generator = torch.Generator(device=device).manual_seed(0)
    seconds = time_call(lambda: write(model, ids, steps, 0.8, 20, generator), device, 1, 3)

    return steps / seconds
'''

STEP_SOLUTION = '''
def step_seconds(
    model: nn.Module,
    length: int,
    device: torch.device,
) -> float:
    model = model.to(device).eval()
    ids = torch.zeros(1, length, dtype=torch.long, device=device)

    with torch.no_grad():
        return time_call(lambda: model(ids), device, 3, 10)
'''

FAST1_NAMES = '''
    plain_speed = tokens_per_second(generate, model, prompt, 100, cpu)
    short_step = step_seconds(model, 8, cpu)
    long_step = step_seconds(model, 128, cpu)
    growth = long_step / short_step
'''

FAST1_LESSON = prose(
    """
    This is the project for Section 12. Your GPT from gpt3 writes text with
    the `generate` function from gpt4. In this project you measure how fast it
    writes. Then you make it faster with the KV cache from kvcache1 and the
    int8 weights from quant3, and measure it again.

    The timing uses `time_call` from profile1. It runs a few warmup calls
    first, and it synchronizes the device after each timed call. Writing text
    also draws random samples from a random number generator. On a GPU,
    `torch.multinomial` raises an error unless that generator is on the same
    device as the model.

    ```python
    torch.Generator(device=torch.device("cpu")).device    # device(type='cpu')
    ```

    `tokens_per_second` times one call of a writing function. Then it divides
    the number of tokens written by the seconds the call took. It takes the
    writing function as an argument, so fast3 can use it to time
    `cached_generate` too.

    `step_seconds` times one forward pass of the model over a text of a given
    length. gpt4's `generate` runs one forward pass for every token it writes.
    Each pass covers the whole text so far, up to the last `block_size`
    tokens.

    Here are the timings for the trained model from gpt3 on this Mac's CPU.
    Each cell is the range over three runs on an otherwise idle machine. The
    model wrote 100 tokens after the prompt "ROMEO:".

    | Tokens per second | Step at 8 tokens | Step at 128 tokens |
    |---|---|---|
    | 1,405 to 1,418 | 0.44 to 0.47 ms | 1.05 to 1.07 ms |

    A step over 128 tokens took 2.3 to 2.4 times as long as a step over 8
    tokens. Each step runs the model over the whole text again, so writing
    gets slower as the text grows. fast2 adds the KV cache from kvcache1. With
    the cache, each step runs the model on the newest token only.

    The last lines of the file use your functions to time the model on your
    machine's CPU, writing 100 tokens like the table above. The Result tab
    shows the numbers they got.
    """,
)

FAST1_SOURCE = source_file(
    '''
    # fast1.py
''' + IMPORTS + '''
    sampler = torch.Generator(device=torch.device("cpu")).manual_seed(0)    # a random generator on the CPU
    rate = 32 / 0.04                                                        # 32 tokens in 0.04 seconds is 800.0 per second


''' + GIVEN_GPT + '''


''' + GIVEN_TIMING + '''


    # TODO: Return how many tokens per second the writing function write
    #   produces. Move model and prompt to device, and make a torch.Generator
    #   on device seeded with 0. Then time the call
    #   write(model, prompt, steps, 0.8, 20, generator) with time_call, using
    #   1 warmup call and 3 timed calls. Return steps divided by the seconds
    #   that time_call returns.
    def tokens_per_second(
        write: Callable[..., Tensor],
        model: nn.Module,
        prompt: Tensor,
        steps: int,
        device: torch.device,
    ) -> float:
        ...


    # TODO: Return how many seconds one forward pass of model takes on device.
    #   The input is a batch of one text made of length token ids that are all
    #   0, so it has shape (1, length) and dtype torch.long, and it is made on
    #   device. Move model to device and put it in eval mode. Run the pass
    #   inside torch.no_grad() so autograd records nothing, and time it with
    #   time_call, using 3 warmup calls and 10 timed calls.
    def step_seconds(
        model: nn.Module,
        length: int,
        device: torch.device,
    ) -> float:
        ...

''' + LOADED_MODEL + FAST1_NAMES + '''    ''',
)

FAST1_SOLUTION = source_file(
    '''
    # fast1.py, a solution for comparison
''' + IMPORTS + '''
    sampler = torch.Generator(device=torch.device("cpu")).manual_seed(0)
    rate = 32 / 0.04


''' + GIVEN_GPT + '''


''' + GIVEN_TIMING + '''


''' + indented(SPEED_SOLUTION) + '''


''' + indented(STEP_SOLUTION) + '''

''' + LOADED_MODEL + FAST1_NAMES + '''    ''',
)


class GeneratorRecorder:
    """Stands in for torch in the learner's file and notes the device each Generator is made on. The generator it
    hands back is a real one on the CPU, so the scripted write can read its seed. Everything else is the real torch."""

    def __init__(self) -> None:
        self.devices: list[str] = []

    def Generator(
        self,
        device: object = "cpu",
    ) -> torch.Generator:
        self.devices.append(torch.device(device).type)

        return torch.Generator()

    def __getattr__(
        self,
        name: str,
    ) -> object:
        return getattr(torch, name)


class ScriptedWrite:
    """Stands in for generate, noting where the model, the ids and the generator were when it was called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        model: nn.Module,
        ids: Tensor,
        steps: int,
        temperature: float,
        top_k: int,
        generator: torch.Generator,
    ) -> Tensor:
        self.calls.append(
            {
                "model": next(model.parameters()).device.type,
                "ids": ids.device.type,
                "steps": steps,
                "temperature": temperature,
                "top_k": top_k,
                "seed": generator.initial_seed() if isinstance(generator, torch.Generator) else None,
            },
        )

        return ids


class ScriptedTimeCall:
    """Stands in for time_call. It runs the timed function once, notes how it was called, and reports a set time."""

    def __init__(
        self,
        seconds: float,
    ) -> None:
        self.seconds = seconds
        self.calls: list[tuple[str, int, int]] = []

    def __call__(
        self,
        run: Callable[[], object],
        device: torch.device,
        warmup: int,
        repeats: int,
    ) -> float:
        self.calls.append((torch.device(device).type, warmup, repeats))
        run()

        return self.seconds


def swapped_into(
    found: Callable[..., object],
    replacements: dict[str, object],
    run: Callable[[], object],
) -> object:
    """Runs found's call with some of its module's names swapped for stand ins, and puts the originals back after."""
    globals_ = found.__globals__
    originals = {name: globals_.get(name) for name in replacements}
    globals_.update(replacements)

    try:
        return run()
    finally:
        globals_.update(originals)


def expect_one_timing(
    call: str,
    timing: ScriptedTimeCall,
    device: str,
    warmup: int,
    repeats: int,
) -> None:
    if not timing.calls:
        raise Mismatch(f"{call} never called time_call. Time the work with time_call from profile1.")

    if len(timing.calls) > 1:
        raise Mismatch(f"{call} called time_call {len(timing.calls)} times, but it should call it once.")

    wanted = (device, warmup, repeats)

    if timing.calls[0] != wanted:
        called_device, called_warmup, called_repeats = timing.calls[0]
        raise Mismatch(
            f"{call} called time_call with device {called_device}, {called_warmup} warmup calls and "
            f"{called_repeats} timed ones, but it should pass device, {warmup} and {repeats}.",
        )


def check_tokens_per_second(
    found: Callable[..., object],
    steps: int,
    seconds: float,
) -> None:
    call = f"tokens_per_second(generate, a small GPT, a prompt of 4 ids, {steps}, a GPU)"
    write = ScriptedWrite()
    timing = ScriptedTimeCall(seconds)
    recorder = GeneratorRecorder()
    model = seeded_gpt(MODEL_REFERENCE["GPT"])
    prompt = torch.tensor([[1, 2, 3, 4]])

    # The meta device holds no numbers, so it shows a missing .to(device) as main's GPU would.
    actual = swapped_into(
        found=found,
        replacements={"time_call": timing, "torch": recorder},
        run=lambda: found(write, model, prompt, steps, torch.device("meta")),
    )

    expect_one_timing(
        call=call,
        timing=timing,
        device="meta",
        warmup=1,
        repeats=3,
    )

    if len(write.calls) != 1:
        raise Mismatch(
            f"{call} called write {len(write.calls)} times for one run of time_call, but it should call it once, "
            "inside the function time_call runs.",
        )

    called = write.calls[0]

    if called["model"] != "meta":
        raise Mismatch(
            f"{call} passed write the model while it was on the CPU, where a GPU can't run it. Move model to device "
            "first.",
        )

    if called["ids"] != "meta":
        raise Mismatch(
            f"{call} passed write the prompt while it was on the CPU, where a GPU can't read it. Move prompt to "
            "device first.",
        )

    if recorder.devices != ["meta"]:
        made = ", ".join(recorder.devices) if recorder.devices else "none"
        raise Mismatch(
            f"{call} made generators on these devices, {made}, but it should make one generator on device. "
            "On a GPU, torch.multinomial raises an error when its generator is on the CPU.",
        )

    wanted = {"steps": steps, "temperature": 0.8, "top_k": 20, "seed": 0}
    got = {name: called[name] for name in wanted}

    if got != wanted:
        raise Mismatch(
            f"{call} called write with steps {got['steps']}, temperature {got['temperature']}, top_k "
            f"{got['top_k']} and a generator seeded with {got['seed']}, but it should pass {steps}, 0.8, 20 and a "
            "generator seeded with 0.",
        )

    expect_close(
        call=f"{call}, when time_call reports {seconds:g} seconds,",
        actual=actual,
        expected=steps / seconds,
    )


class StepRecorder(nn.Module):
    """Stands in for the GPT, noting the ids it gets, the device of its own weight, and the modes it runs in."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.seen: list[dict[str, object]] = []

    def forward(
        self,
        ids: Tensor,
    ) -> Tensor:
        self.seen.append(
            {
                "shape": tuple(ids.shape),
                "dtype": ids.dtype,
                "ids": ids.device.type,
                "model": self.weight.device.type,
                "grad": torch.is_grad_enabled(),
                "training": self.training,
            },
        )

        # Nothing here mixes the ids with the weight, so a model left on the CPU is noted, not crashed on.
        return ids.float()


def check_step_seconds(
    found: Callable[..., object],
    length: int,
    seconds: float,
) -> None:
    call = f"step_seconds(a model, {length}, a GPU)"
    timing = ScriptedTimeCall(seconds)
    model = StepRecorder()
    actual = swapped_into(
        found=found,
        replacements={"time_call": timing},
        run=lambda: found(model, length, torch.device("meta")),
    )

    expect_one_timing(
        call=call,
        timing=timing,
        device="meta",
        warmup=3,
        repeats=10,
    )

    if len(model.seen) != 1:
        raise Mismatch(f"{call} ran the model {len(model.seen)} times for one run of time_call, but it should run it once.")

    seen = model.seen[0]

    if seen["model"] != "meta":
        raise Mismatch(f"{call} left the model on the CPU. Move it to device first.")

    if seen["ids"] != "meta":
        raise Mismatch(f"{call} made the ids on the CPU. Make them on device, with device=device.")

    if seen["shape"] != (1, length):
        raise Mismatch(
            f"{call} ran the model on ids of shape {seen['shape']}, but they should be of shape (1, {length}), a "
            "batch of one text.",
        )

    if seen["dtype"] != torch.long:
        raise Mismatch(
            f"{call} ran the model on ids of dtype {seen['dtype']}, but they should be torch.long, integers that the "
            "embedding can look up.",
        )

    if seen["training"]:
        raise Mismatch(f"{call} ran the model in training mode. Call .eval() on it first.")

    if seen["grad"]:
        raise Mismatch(f"{call} recorded the forward pass for autograd. Time it inside torch.no_grad().")

    expect_close(
        call=f"{call}, when time_call reports {seconds:g} seconds,",
        actual=actual,
        expected=seconds,
    )


def check_positive_number(name: str) -> Callable[[object], None]:
    def check(value: object) -> None:
        reject_missing_return(
            call=name,
            actual=value,
        )

        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise Mismatch(f"{name} is {value_text(value)}, but it should be a plain number above 0.")

    return check


def check_fast1(checker: Checker) -> None:
    for steps, seconds in ((32, 0.25), (10, 0.5)):
        checker.test(
            function="tokens_per_second",
            description="times one call of write on device and divides steps by the seconds",
            run=lambda found, steps=steps, seconds=seconds: check_tokens_per_second(found, steps, seconds),
        )

    for length, seconds in ((8, 0.002), (128, 0.005)):
        checker.test(
            function="step_seconds",
            description="times one forward pass in eval mode, without autograd",
            run=lambda found, length=length, seconds=seconds: check_step_seconds(found, length, seconds),
        )

    for name, description in (
        ("plain_speed", "is how many tokens per second generate writes on your CPU"),
        ("short_step", "is the seconds of one forward pass over 8 tokens"),
        ("long_step", "is the seconds of one forward pass over 128 tokens"),
        ("growth", "is how many times longer a pass over 128 tokens takes than one over 8"),
    ):
        checker.check_name(
            name=name,
            description=description,
            run=check_positive_number(name),
        )


FAST1 = Exercise(
    name="fast1",
    chapter=FAST_GPT,
    title="Project: your GPT, timed",
    lesson=FAST1_LESSON,
    source=FAST1_SOURCE,
    solution=FAST1_SOLUTION,
    hints={
        "tokens_per_second": "Move model and prompt with .to(device), and make the generator with "
        "torch.Generator(device=device).manual_seed(0). Pass time_call a lambda that calls write. Then divide steps "
        "by the seconds that time_call returns.",
        "step_seconds": "torch.zeros(1, length, dtype=torch.long, device=device) makes the ids. Inside "
        "torch.no_grad(), call time_call on a lambda that runs model(ids).",
        "plain_speed": "It comes from tokens_per_second. Make that pass first.",
        "short_step": "It comes from step_seconds. Make that pass first.",
        "long_step": "It comes from step_seconds. Make that pass first.",
        "growth": "It comes from step_seconds. Make that pass first.",
    },
    introduces=(),
    check=check_fast1,
    time_limit=60,
)


# ---------------------------------------------------------------------------
# fast2
# ---------------------------------------------------------------------------

SETTINGS_OF_PART = '''
# Returns the five settings a GPT was made with, read off its layers. They are the vocabulary size,
# block_size, embedding size, number of blocks and number of heads.
def settings_of(model: nn.Module) -> tuple[int, int, int, int, int]:
    vocabulary, size = model.wte.weight.shape

    return vocabulary, model.block_size, size, len(model.h), model.h[0].attn.heads
'''

FAST_MODEL_SOLUTION = '''
def fast_model(
    model: nn.Module,
    int8: bool,
) -> nn.Module:
    cached = CachedGPT(*settings_of(model))
    cached.load_state_dict(model.state_dict())

    if int8:
        quantize_linears_(cached.h)

    return cached.eval()
'''

DIFFERENCE_SOLUTION = '''
def first_difference(
    a: Tensor,
    b: Tensor,
) -> int | None:
    different = (a[0] != b[0]).nonzero()

    if len(different) == 0:
        return None

    return different[0].item()
'''

FAST2_NAMES = '''
    plain_text = generate(model, prompt, 100, 0.8, 20, torch.Generator().manual_seed(0))
    cached_text = cached_generate(fast_model(model, False), prompt, 100, 0.8, 20, torch.Generator().manual_seed(0))
    int8_text = cached_generate(fast_model(model, True), prompt, 100, 0.8, 20, torch.Generator().manual_seed(0))
    cache_differs_at = first_difference(plain_text, cached_text)
    int8_differs_at = first_difference(plain_text, int8_text)
'''

FAST2_LESSON = prose(
    """
    The `CachedGPT` from kvcache1 has the same layers as the GPT from gpt2,
    with the same names. So the `state_dict` of a plain GPT loads directly
    into a `CachedGPT`. To make the `CachedGPT` you need the five settings the
    GPT was made with. The given function `settings_of` reads them off the
    GPT's layers.

    ```python
    model.state_dict()["wte.weight"].shape    # torch.Size([65, 128])
    ```

    Then `quantize_linears_` from quant3 replaces each `nn.Linear` in the
    blocks with an `Int8Linear`. The order of these two steps matters. An
    `Int8Linear` has no `weight` parameter for a state dict to load into, so
    load the weights first and quantize after. Only the blocks get quantized,
    because `lm_head` shares its weight with the embedding table `wte`.

    The cache is exact. kvcache1 checked that the cached model writes the same
    tokens as gpt4's `generate`, one by one. int8 is not exact. Rounding to
    int8 changes each weight by up to half of its row's scale, so the scores
    change a little. Once in a while, sampling then picks a different token.
    Every later token depends on the tokens before it, so from that point on
    the two texts are different.

    `first_difference` finds the first position where two texts differ. Here
    is the trained model from gpt3 writing 100 tokens after "ROMEO:", with
    five different seeds. Positions count from the start of the prompt, so the
    6 prompt tokens are positions 0 to 5.

    | Seed | Cache, first difference | int8, first difference |
    |---|---|---|
    | 0 | none | 88 |
    | 1 | none | 89 |
    | 2 | none | none |
    | 3 | none | 51 |
    | 4 | none | none |

    With the cache, the model wrote exactly what `generate` wrote for all five
    seeds. With int8, the text differed on three of the five seeds, the
    earliest at position 51. On the other two seeds, int8 wrote the same 106
    tokens as `generate`.
    """,
)

FAST2_SOURCE = source_file(
    '''
    # fast2.py
''' + IMPORTS + '''
    ids = torch.tensor([[5, 1, 7, 2]])
    (ids[0] != torch.tensor([5, 1, 9, 2])).nonzero()    # tensor([[2]]), they differ at position 2


''' + GIVEN_GPT + '''


''' + GIVEN_FAST + '''


''' + indented(SETTINGS_OF_PART) + '''


    # TODO: Return a CachedGPT with the same settings and weights as model, in
    #   eval mode. When int8 is True, also replace every nn.Linear in its
    #   blocks with Int8Linear, and leave every other layer as it is. Do not
    #   change model itself.
    def fast_model(
        model: nn.Module,
        int8: bool,
    ) -> nn.Module:
        ...


    # TODO: a and b are two texts of token ids, both of shape (1, n). Return the
    #   first position where they differ, as a plain Python int. Return None
    #   when they are the same.
    #   Example: first_difference(torch.tensor([[3, 1, 4]]),
    #   torch.tensor([[3, 1, 5]])) returns 2.
    def first_difference(
        a: Tensor,
        b: Tensor,
    ) -> int | None:
        ...

''' + LOADED_MODEL + FAST2_NAMES + '''    ''',
)

FAST2_SOLUTION = source_file(
    '''
    # fast2.py, a solution for comparison
''' + IMPORTS + '''
    ids = torch.tensor([[5, 1, 7, 2]])
    (ids[0] != torch.tensor([5, 1, 9, 2])).nonzero()


''' + GIVEN_GPT + '''


''' + GIVEN_FAST + '''


''' + indented(SETTINGS_OF_PART) + '''


''' + indented(FAST_MODEL_SOLUTION) + '''


''' + indented(DIFFERENCE_SOLUTION) + '''

''' + LOADED_MODEL + FAST2_NAMES + '''    ''',
)

GIVEN_CODE = (
    GIVEN_MODEL
    + INIT_SOLUTION
    + GPT_INIT
    + GPT_FORWARD
    + GPT_SETTINGS
    + DATA_PART
    + SAMPLING_PARTS
    + TIMING_PART
    + VALIDATION_PART
    + CACHED_MODEL_PART
    + GENERATE_SOLUTION
    + INT8_PART
    + SETTINGS_OF_PART
)


def first_layer_kinds(model: nn.Module) -> list[str]:
    return [type(module).__name__ for module in model.modules()]


def check_fast_model(
    found: Callable[..., object],
    settings: tuple[int, int, int, int, int],
    int8: bool,
) -> None:
    call = f"fast_model(a GPT made with {settings}, int8={int8})"
    model = seeded_gpt(MODEL_REFERENCE["GPT"], settings)
    untouched = seeded_gpt(MODEL_REFERENCE["GPT"], settings)
    ids = torch.randint(0, settings[0], (2, 7), generator=torch.Generator().manual_seed(931))
    actual = found(model, int8)

    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, found.__globals__["CachedGPT"]):
        raise Mismatch(f"{call} is a {type(actual).__name__}, but it should be a CachedGPT.")

    if actual.training:
        raise Mismatch(f"{call} is in training mode. Call .eval() on it before returning it.")

    if first_layer_kinds(model) != first_layer_kinds(untouched):
        raise Mismatch(f"{call} changed the layers of the model it was handed. Quantize the CachedGPT's blocks, not model's.")

    with torch.no_grad():
        if not torch.equal(model(ids), untouched(ids)):
            raise Mismatch(f"{call} changed the weights of the model it was handed. Leave model as it was.")

        expected = FAST_REFERENCE["fast_model"](model, int8)
        kinds = first_layer_kinds(actual)
        wanted = first_layer_kinds(expected)

        if kinds != wanted:
            raise Mismatch(
                f"{call} has the layers {', '.join(kinds)}, but it should have {', '.join(wanted)}. Only the Linears "
                "inside the blocks become Int8Linear, and only when int8 is True.",
            )

        if actual.lm_head.weight is not actual.wte.weight:
            raise Mismatch(f"{call} no longer shares lm_head's weight with wte. Leave lm_head as CachedGPT made it.")

        scores, _ = actual(ids)
        expect_tensor(
            call=f"{call}, run on ids of shape (2, 7), its scores",
            actual=scores,
            expected=expected(ids)[0],
        )


def expect_position(
    call: str,
    actual: object,
    expected: int | None,
) -> None:
    """A position, or None when there is none, which expect_value would take for a function not written yet."""
    if expected is not None:
        expect_value(
            call=call,
            actual=actual,
            expected=expected,
        )
        return

    if actual is not None:
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be None, since the two texts are the same.")


def check_first_difference(found: Callable[..., object]) -> None:
    pairs = (
        (torch.tensor([[3, 1, 4, 1, 5]]), torch.tensor([[3, 1, 4, 2, 5]])),
        (torch.tensor([[2, 7, 1]]), torch.tensor([[8, 7, 2]])),
        (torch.tensor([[6, 6, 6, 6, 6, 6]]), torch.tensor([[6, 6, 6, 6, 6, 0]])),
        (torch.tensor([[3, 1, 4, 1, 5]]), torch.tensor([[3, 1, 4, 1, 5]])),
    )

    for a, b in pairs:
        expect_position(
            call=f"first_difference({value_text(a)}, {value_text(b)})",
            actual=found(a, b),
            expected=FAST_REFERENCE["first_difference"](a, b),
        )


@functools.cache
def reference_differences() -> dict[str, int | None]:
    """fast2's two names on the learner's trained GPT, with the reference code."""
    model = trained_gpt(MODEL_REFERENCE["GPT"])
    characters = FAST_REFERENCE["shakespeare_splits"]()[0]
    prompt = torch.tensor([[characters.index(character) for character in "ROMEO:"]])
    texts = {}

    for name, write, made in (
        ("plain", FAST_REFERENCE["generate"], model),
        ("cache", FAST_REFERENCE["cached_generate"], FAST_REFERENCE["fast_model"](model, False)),
        ("int8", FAST_REFERENCE["cached_generate"], FAST_REFERENCE["fast_model"](model, True)),
    ):
        texts[name] = write(made, prompt, 100, 0.8, 20, torch.Generator().manual_seed(0))

    return {
        "cache_differs_at": FAST_REFERENCE["first_difference"](texts["plain"], texts["cache"]),
        "int8_differs_at": FAST_REFERENCE["first_difference"](texts["plain"], texts["int8"]),
    }


def check_difference_name(name: str) -> Callable[[object], None]:
    def check(value: object) -> None:
        expect_position(
            call=name,
            actual=value,
            expected=reference_differences()[name],
        )

    return check


def check_fast2(checker: Checker) -> None:
    for settings in ((20, 16, 32, 2, 4), (11, 12, 24, 3, 2)):
        for int8 in (False, True):
            checker.test(
                function="fast_model",
                description="loads the weights into a CachedGPT, then quantizes the blocks when asked",
                run=lambda found, settings=settings, int8=int8: check_fast_model(found, settings, int8),
            )

    checker.test(
        function="first_difference",
        description="finds the first position where two texts differ, or None",
        run=check_first_difference,
    )

    for name, description in (
        ("cache_differs_at", "is the first position where the cached text differs from generate's"),
        ("int8_differs_at", "is the first position where the int8 text differs from generate's"),
    ):
        checker.check_name(
            name=name,
            description=description,
            run=check_difference_name(name),
        )


FAST2 = Exercise(
    name="fast2",
    chapter=FAST_GPT,
    title="Project: the cache and int8, added",
    lesson=FAST2_LESSON,
    source=FAST2_SOURCE,
    solution=FAST2_SOLUTION,
    hints={
        "fast_model": "Make CachedGPT(*settings_of(model)) and load model.state_dict() into it. When int8 is True, "
        "call quantize_linears_ on its .h, which holds the blocks. Return it after calling .eval().",
        "first_difference": "Compare a[0] with b[0] using !=, then .nonzero() gives the positions where they differ. "
        "Return None when there are none. Otherwise turn the first position into a number with .item().",
        "cache_differs_at": "It comes from fast_model and first_difference. Make those pass first.",
        "int8_differs_at": "It comes from fast_model and first_difference. Make those pass first.",
    },
    introduces=(),
    check=check_fast2,
    time_limit=60,
)


# ---------------------------------------------------------------------------
# fast3
# ---------------------------------------------------------------------------

SCORES_ONLY_SOLUTION = '''
class ScoresOnly(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.block_size = model.block_size

    def forward(self, ids: Tensor) -> Tensor:
        return self.model(ids)[0]
'''

INT8_COPY_SOLUTION = '''
def int8_copy(model: nn.Module) -> nn.Module:
    copy = deepcopy(model)
    quantize_linears_(copy.h)

    return copy.eval()
'''

COMPARE_SOLUTION = '''
def compare(
    model: nn.Module,
    prompt: Tensor,
    steps: int,
    device: torch.device,
    data: Tensor,
    batches: int,
) -> dict[str, tuple[float, float, int]]:
    setups = {
        "plain": (model, generate),
        "int8": (int8_copy(model), generate),
        "cache": (fast_model(model, False), cached_generate),
        "cache and int8": (fast_model(model, True), cached_generate),
    }
    results = {}

    for name, (setup, write) in setups.items():
        speed = tokens_per_second(write, setup, prompt, steps, device)
        scored = ScoresOnly(setup) if write is cached_generate else setup
        loss = validation_loss(scored, data, batches, torch.Generator().manual_seed(1), device)
        results[name] = (speed, loss, model_bytes(setup))

    return results
'''

FAST3_NAMES = '''
    table = compare(model, prompt, 100, cpu, validation_data, 5)
'''

FAST3_IMPORTS = IMPORTS.replace(
    "    from collections.abc import Callable\n",
    "    from collections.abc import Callable\n    from copy import deepcopy\n",
)

FAST_REFERENCE = namespace_of(
    GIVEN_CODE
    + SPEED_SOLUTION
    + STEP_SOLUTION
    + FAST_MODEL_SOLUTION
    + DIFFERENCE_SOLUTION
    + SCORES_ONLY_SOLUTION
    + INT8_COPY_SOLUTION
    + COMPARE_SOLUTION,
)

FAST3_LESSON = prose(
    """
    Now you compare four setups. They are the plain model, the model with
    int8 weights, the model with the cache, and the model with both. Each
    setup gets measured by its tokens per second, its validation loss and the
    size of its weights in bytes. `compare` runs each setup through fast1's
    `tokens_per_second` and gpt3's `validation_loss`.

    Two small pieces make this work. First, gpt3's `validation_loss` expects a
    model that returns only scores, but a `CachedGPT` returns its scores and
    its caches. `ScoresOnly` wraps a `CachedGPT` and returns only the scores.
    Second, `quantize_linears_` changes a model in place. So the int8 setup
    without the cache quantizes a copy, and the plain model stays as it was.

    ```python
    from copy import deepcopy
    twin = deepcopy(nn.Linear(2, 2))    # a copy with its own weights, so changing twin leaves the original alone
    ```

    Here are the results for the trained model from gpt3 on this Mac's CPU,
    writing 100 tokens after "ROMEO:". Each speed is the range over three runs
    on an otherwise idle machine. Each loss is the average over gpt3's 20
    validation batches.

    | Setup | Tokens per second | Validation loss | Bytes |
    |---|---|---|---|
    | Plain | 1,405 to 1,418 | 1.5351 | 3,272,192 |
    | int8 | 960 to 972 | 1.5353 | 931,328 |
    | Cache | 3,700 to 3,816 | 1.5351 | 3,272,192 |
    | Cache and int8 | 1,515 to 1,535 | 1.5353 | 931,328 |

    The cache wrote about 2.7 times as fast as the plain model, with the same
    loss. int8 made the model 3.5 times smaller and changed the loss only in
    the fourth decimal place. int8 also wrote more slowly, because quant3's
    `Int8Linear` saves memory only. Before each matrix multiplication, it
    converts every weight back into a float32 matrix, which adds work.

    quant3 mentioned that fast kernels convert each integer to a float inside
    the multiplication itself. A kernel that does both steps in one pass like
    this is called fused. PyTorch has a fused int8 kernel for the CPU,
    `torch._weight_int8pack_mm`. The leading underscore marks it as private,
    which means it can change between PyTorch versions. It multiplies each
    output by its row's scale inside the same pass, so it never builds a
    float32 weight matrix. On gpt3's trained model, its scores matched the scores of
    `Int8Linear` to within 1.4e-5.

    The next table times one cached step, which writes one token after 64
    tokens of text, on 4 CPU threads. It uses three untrained GPTs. A row like
    "8 blocks of 512" means 8 blocks with an embedding size of 512. The three
    kinds of weights took turns in each round of timing, and each range covers
    six runs.

    | Model | float32 bytes | float32 | `Int8Linear` | Fused int8 |
    |---|---|---|---|---|
    | gpt3's, 4 blocks of 128 | 3.3 MB | 0.23 to 0.27 ms | 0.52 to 0.65 ms | 0.39 to 0.51 ms |
    | 8 blocks of 512 | 102 MB | 2.1 to 2.5 ms | 4.0 to 5.9 ms | 1.6 to 2.0 ms |
    | 12 blocks of 1,024 | 609 MB | 10 to 15 ms | 32 to 48 ms | 5.6 to 8.4 ms |

    Fused int8 was faster than float32 on both bigger models. It took 76 to 92
    percent of float32's time on the middle model, and 52 to 63 percent on the
    biggest. Writing one token multiplies each weight matrix by a single
    vector. With big weights, reading them from memory takes longer than the
    arithmetic, which profile2 called memory bound. So reading fewer bytes
    makes the step faster. With small weights the reading is quick, and
    converting each integer to a float costs more time than the smaller reads
    save.

    PyTorch's gpt-fast project found the same on an A100 GPU. Unfused int8
    ran slower than 16 bit weights. Fused int8 sped up a 7 billion parameter
    Llama model from 107 tokens a second to 157. GPTQ's kernel reads 3 bit
    weights and converts each one to a float only inside the multiplication.
    On a 175 billion parameter model it took 71 ms per token on one A100. With
    FP16 weights, the same model needed five A100s and took 230 ms per token.

    The last line of the file runs `compare` on your CPU, writing 100 tokens
    like the first table, and the Result tab shows your table. It scores only
    5 validation batches, to keep the check short.
    """,
)

FAST3_SOURCE = source_file(
    '''
    # fast3.py
''' + FAST3_IMPORTS + '''
    twin = deepcopy(nn.Linear(2, 2))    # a copy with its own weights, so changing twin leaves the original alone


''' + GIVEN_GPT + '''


''' + GIVEN_TIMING + '''


''' + GIVEN_VALIDATION + '''


''' + GIVEN_FAST + '''


    # tokens_per_second from fast1, and settings_of and fast_model from fast2.
''' + indented(SPEED_SOLUTION) + '''


''' + indented(SETTINGS_OF_PART) + '''


''' + indented(FAST_MODEL_SOLUTION) + '''


    # TODO: Finish ScoresOnly, a wrapper around a CachedGPT. __init__ stores
    #   model and model.block_size as attributes of self. forward runs the
    #   stored model on ids, which returns (scores, caches), and returns only
    #   the scores.
    class ScoresOnly(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            ...

        def forward(self, ids: Tensor) -> Tensor:
            ...


    # TODO: Return a copy of model in eval mode, with every nn.Linear in the
    #   copy's blocks replaced by Int8Linear. Do not change model itself.
    def int8_copy(model: nn.Module) -> nn.Module:
        ...


    # TODO: Finish compare. It measures the four setups and returns a dict from
    #   each setup's name to (speed, loss, bytes). Set speed to the tokens per
    #   second that write produces with setup, from tokens_per_second with
    #   prompt, steps and device. Set loss with validation_loss, passing
    #   scored, data, batches, a new torch.Generator seeded with 1, and device.
    def compare(
        model: nn.Module,
        prompt: Tensor,
        steps: int,
        device: torch.device,
        data: Tensor,
        batches: int,
    ) -> dict[str, tuple[float, float, int]]:
        setups = {
            "plain": (model, generate),
            "int8": (int8_copy(model), generate),
            "cache": (fast_model(model, False), cached_generate),
            "cache and int8": (fast_model(model, True), cached_generate),
        }
        results = {}

        for name, (setup, write) in setups.items():
            speed = ...
            scored = ScoresOnly(setup) if write is cached_generate else setup
            loss = ...
            results[name] = (speed, loss, model_bytes(setup))

        return results

''' + LOADED_MODEL + FAST3_NAMES + '''    ''',
)

FAST3_SOLUTION = source_file(
    '''
    # fast3.py, a solution for comparison
''' + FAST3_IMPORTS + '''
    twin = deepcopy(nn.Linear(2, 2))


''' + GIVEN_GPT + '''


''' + GIVEN_TIMING + '''


''' + GIVEN_VALIDATION + '''


''' + GIVEN_FAST + '''


''' + indented(SPEED_SOLUTION) + '''


''' + indented(SETTINGS_OF_PART) + '''


''' + indented(FAST_MODEL_SOLUTION) + '''


''' + indented(SCORES_ONLY_SOLUTION) + '''


''' + indented(INT8_COPY_SOLUTION) + '''


''' + indented(COMPARE_SOLUTION) + '''

''' + LOADED_MODEL + FAST3_NAMES + '''    ''',
)


def check_scores_only(found: type) -> None:
    for settings, seed in (((20, 16, 32, 2, 4), 941), ((11, 12, 24, 3, 2), 942)):
        cached = FAST_REFERENCE["fast_model"](seeded_gpt(MODEL_REFERENCE["GPT"], settings), False)
        ids = torch.randint(0, settings[0], (2, 5), generator=torch.Generator().manual_seed(seed))
        call = f"ScoresOnly of a CachedGPT made with {settings}"
        wrapped = found(cached)

        if getattr(wrapped, "block_size", None) != settings[1]:
            raise Mismatch(
                f"{call} has block_size {value_text(getattr(wrapped, 'block_size', None))}, but it should keep the "
                f"model's, {settings[1]}. validation_loss reads it to decide how long each text in a batch is.",
            )

        with torch.no_grad():
            scores = wrapped(ids)

            if isinstance(scores, (tuple, list)):
                raise Mismatch(
                    f"{call}, run on ids of shape (2, 5), returns a {type(scores).__name__}, but it should return "
                    "only the scores, the first of the two things a CachedGPT returns.",
                )

            expect_tensor(
                call=f"{call}, run on ids of shape (2, 5)",
                actual=scores,
                expected=cached(ids)[0],
            )


def check_int8_copy(found: Callable[..., object]) -> None:
    for settings in ((20, 16, 32, 2, 4), (11, 12, 24, 3, 2)):
        call = f"int8_copy(a GPT made with {settings})"
        model = seeded_gpt(MODEL_REFERENCE["GPT"], settings)
        before = first_layer_kinds(model)
        expected = FAST_REFERENCE["int8_copy"](seeded_gpt(MODEL_REFERENCE["GPT"], settings))
        actual = found(model)

        reject_missing_return(
            call=call,
            actual=actual,
        )

        if actual is model or first_layer_kinds(model) != before:
            raise Mismatch(f"{call} quantized the model it was handed. Quantize a deepcopy of it instead.")

        if actual.training:
            raise Mismatch(f"{call} is in training mode. Call .eval() on it before returning it.")

        kinds = first_layer_kinds(actual)
        wanted = first_layer_kinds(expected)

        if kinds != wanted:
            raise Mismatch(
                f"{call} has the layers {', '.join(kinds)}, but it should have {', '.join(wanted)}. Only the Linears "
                "inside the blocks become Int8Linear.",
            )

        ids = torch.randint(0, settings[0], (2, 6), generator=torch.Generator().manual_seed(943))

        with torch.no_grad():
            expect_tensor(
                call=f"{call}, run on ids of shape (2, 6)",
                actual=actual(ids),
                expected=expected(ids),
            )


class ScriptedSpeed:
    """Stands in for tokens_per_second with a set speed for each writing function, noting what each was handed."""

    def __init__(
        self,
        speeds: dict[str, float],
    ) -> None:
        self.speeds = speeds
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(
        self,
        write: Callable[..., Tensor],
        model: nn.Module,
        prompt: Tensor,
        steps: int,
        device: torch.device,
    ) -> float:
        self.calls.append((write.__name__, first_layer_kinds(model)))

        return self.speeds[write.__name__]


def expect_table(
    call: str,
    actual: object,
    expected: dict[str, tuple[float, float, int]],
    loss_tolerance: float,
    speeds_known: bool,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, dict) or list(actual) != list(expected):
        keys = list(actual) if isinstance(actual, dict) else value_text(actual)
        raise Mismatch(f"{call} has the keys {keys}, but it should have {list(expected)}, in that order.")

    for name, wanted in expected.items():
        got = actual[name]

        if not isinstance(got, tuple) or len(got) != 3:
            raise Mismatch(f"{call}[\"{name}\"] is {value_text(got)}, but it should be a tuple of three.")

        speed, loss, size = got

        if speeds_known:
            expect_close(
                call=f"{call}[\"{name}\"]'s tokens per second",
                actual=speed,
                expected=wanted[0],
            )
        else:
            check_positive_number(f"{call}[\"{name}\"]'s tokens per second")(speed)

        expect_close(
            call=f"{call}[\"{name}\"]'s validation loss",
            actual=loss,
            expected=wanted[1],
            tolerance=loss_tolerance,
        )
        expect_value(
            call=f"{call}[\"{name}\"]'s bytes",
            actual=size,
            expected=wanted[2],
        )


def check_compare(found: Callable[..., object]) -> None:
    settings = (20, 16, 32, 2, 4)
    data = torch.randint(0, settings[0], (400,), generator=torch.Generator().manual_seed(944))
    prompt = torch.tensor([[1, 2, 3]])
    speeds = {"generate": 100.0, "cached_generate": 300.0}
    cpu = torch.device("cpu")

    expected_speed = ScriptedSpeed(speeds)
    expected = swapped_into(
        found=FAST_REFERENCE["compare"],
        replacements={"tokens_per_second": expected_speed},
        run=lambda: FAST_REFERENCE["compare"](seeded_gpt(MODEL_REFERENCE["GPT"], settings), prompt, 5, cpu, data, 3),
    )

    call = "compare(a small GPT, a prompt of 3 ids, 5, cpu, 400 tokens, 3)"

    def scoring(
        scored: nn.Module,
        *arguments: object,
    ) -> float:
        with torch.no_grad():
            scores = scored(prompt)

        if not isinstance(scores, Tensor):
            raise Mismatch(
                f"{call} passed validation_loss a model that returns a {type(scores).__name__}, not scores. A cached "
                "setup returns its scores and its caches, so wrap it in ScoresOnly first.",
            )

        if scores.shape[:2] != prompt.shape:
            raise Mismatch(
                f"{call} passed validation_loss a model whose scores for ids of shape {tuple(prompt.shape)} have shape "
                f"{tuple(scores.shape)}. Wrap only the cached setups in ScoresOnly.",
            )

        return FAST_REFERENCE["validation_loss"](scored, *arguments)

    model = seeded_gpt(MODEL_REFERENCE["GPT"], settings)
    before = first_layer_kinds(model)
    speed = ScriptedSpeed(speeds)
    actual = swapped_into(
        found=found,
        replacements={"tokens_per_second": speed, "validation_loss": scoring},
        run=lambda: found(model, prompt, 5, cpu, data, 3),
    )

    if first_layer_kinds(model) != before:
        raise Mismatch(
            f"{call} quantized the model it was handed, so the later setups measured a quantized model instead of the "
            "plain one. Use int8_copy.",
        )

    if speed.calls != expected_speed.calls:
        timed = "; ".join(f"{write} on a {kinds[0]} with {kinds.count('Int8Linear')} Int8Linear" for write, kinds in speed.calls)
        raise Mismatch(
            f"{call} timed these, {timed or 'nothing'}. It should time generate on the plain model and on the int8 "
            "copy, then cached_generate on fast_model without int8 and with it, in that order.",
        )

    expect_table(
        call=call,
        actual=actual,
        expected=expected,
        loss_tolerance=1e-5,
        speeds_known=True,
    )


@functools.cache
def reference_table() -> dict[str, tuple[float, float, int]]:
    """fast3's table on the learner's trained GPT with the reference code, its speeds left at 0 since they can't match."""
    model = trained_gpt(MODEL_REFERENCE["GPT"])
    _, _, validation_data = FAST_REFERENCE["shakespeare_splits"]()
    cpu = torch.device("cpu")
    validation_loss = FAST_REFERENCE["validation_loss"]
    plain_loss = validation_loss(model, validation_data, 5, torch.Generator().manual_seed(1), cpu)
    int8 = FAST_REFERENCE["int8_copy"](model)
    int8_loss = validation_loss(int8, validation_data, 5, torch.Generator().manual_seed(1), cpu)
    model_bytes = FAST_REFERENCE["model_bytes"]

    return {
        "plain": (0.0, plain_loss, model_bytes(model)),
        "int8": (0.0, int8_loss, model_bytes(int8)),
        "cache": (0.0, plain_loss, model_bytes(FAST_REFERENCE["fast_model"](model, False))),
        "cache and int8": (0.0, int8_loss, model_bytes(FAST_REFERENCE["fast_model"](model, True))),
    }


def check_table(value: object) -> None:
    # The cached model reorders some arithmetic, so its loss can move in the sixth decimal.
    expect_table(
        call="table",
        actual=value,
        expected=reference_table(),
        loss_tolerance=1e-4,
        speeds_known=False,
    )


def check_fast3(checker: Checker) -> None:
    checker.test_method(
        class_name="ScoresOnly",
        method="forward",
        description="returns only the scores, and keeps the model's block_size",
        run=check_scores_only,
    )
    checker.test(
        function="int8_copy",
        description="quantizes a copy's blocks and leaves the model alone",
        run=check_int8_copy,
    )
    checker.test(
        function="compare",
        description="measures the four setups in order",
        run=check_compare,
    )
    checker.check_name(
        name="table",
        description="is the comparison table measured on your CPU",
        run=check_table,
    )


FAST3 = Exercise(
    name="fast3",
    chapter=FAST_GPT,
    title="Project: before and after",
    lesson=FAST3_LESSON,
    source=FAST3_SOURCE,
    solution=FAST3_SOLUTION,
    hints={
        "ScoresOnly.forward": "In __init__, store the model and its block_size as attributes of self. forward runs "
        "the stored model on ids and returns the first of the two things the model returns.",
        "int8_copy": "Make deepcopy(model) first. Then call quantize_linears_ on the copy's .h, and return the copy "
        "after calling .eval().",
        "compare": "tokens_per_second takes write, setup, prompt, steps and device in that order. validation_loss "
        "takes scored, data, batches, torch.Generator().manual_seed(1) and device.",
        "table": "It comes from compare. Make that pass first.",
    },
    introduces=(),
    check=check_fast3,
    time_limit=120,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def lesson_model() -> nn.Module:
    """The trained model in checkpoints/gpt3.pt, on the CPU."""
    characters = FAST_REFERENCE["shakespeare_splits"]()[0]

    return load_checkpoint(MODEL_REFERENCE["GPT"](len(characters), 128, 128, 4, 4), "gpt3").eval()


def lesson_prompt() -> Tensor:
    characters = FAST_REFERENCE["shakespeare_splits"]()[0]

    return torch.tensor([[characters.index(character) for character in "ROMEO:"]])


def measure_speeds(
    devices: tuple[str, ...] = ("cpu",),
    runs: int = 3,
) -> dict[tuple[str, str], list[float]]:
    """fast1's and fast3's tokens per second, 100 tokens after "ROMEO:", each setup made fresh for every run."""
    prompt = lesson_prompt()
    results: dict[tuple[str, str], list[float]] = {}

    for device in devices:
        for _ in range(runs):
            model = lesson_model()
            setups = {
                "plain": (model, FAST_REFERENCE["generate"]),
                "int8": (FAST_REFERENCE["int8_copy"](model), FAST_REFERENCE["generate"]),
                "cache": (FAST_REFERENCE["fast_model"](model, False), FAST_REFERENCE["cached_generate"]),
                "cache and int8": (FAST_REFERENCE["fast_model"](model, True), FAST_REFERENCE["cached_generate"]),
            }

            for name, (setup, write) in setups.items():
                speed = FAST_REFERENCE["tokens_per_second"](write, setup, prompt, 100, torch.device(device))
                results.setdefault((device, name), []).append(speed)

    return results


def measure_steps(
    devices: tuple[str, ...] = ("cpu",),
    runs: int = 3,
) -> dict[tuple[str, int], list[float]]:
    """fast1's step times in milliseconds, at 8 and 128 tokens."""
    results: dict[tuple[str, int], list[float]] = {}

    for device in devices:
        for _ in range(runs):
            model = lesson_model()

            for length in (8, 128):
                seconds = FAST_REFERENCE["step_seconds"](model, length, torch.device(device))
                results.setdefault((device, length), []).append(1000 * seconds)

    return results


def measure_differences(seeds: range = range(5)) -> list[tuple[int, int | None, int | None]]:
    """fast2's table: for each seed, where the cached and the int8 texts first leave generate's."""
    model = lesson_model()
    prompt = lesson_prompt()
    cached = FAST_REFERENCE["fast_model"](model, False)
    int8 = FAST_REFERENCE["fast_model"](model, True)
    rows = []

    for seed in seeds:
        plain = FAST_REFERENCE["generate"](model, prompt, 100, 0.8, 20, torch.Generator().manual_seed(seed))
        texts = [
            FAST_REFERENCE["cached_generate"](made, prompt, 100, 0.8, 20, torch.Generator().manual_seed(seed))
            for made in (cached, int8)
        ]
        rows.append((seed, *(FAST_REFERENCE["first_difference"](plain, text) for text in texts)))

    return rows


def measure_losses_and_bytes() -> dict[str, tuple[float, int]]:
    """fast3's validation losses over gpt3's 20 batches, and each setup's bytes."""
    model = lesson_model()
    _, _, validation_data = FAST_REFERENCE["shakespeare_splits"]()
    cpu = torch.device("cpu")
    setups = {
        "plain": model,
        "int8": FAST_REFERENCE["int8_copy"](model),
        "cache": FAST_REFERENCE["fast_model"](model, False),
        "cache and int8": FAST_REFERENCE["fast_model"](model, True),
    }
    results = {}

    for name, setup in setups.items():
        scored = setup if name in ("plain", "int8") else FAST_REFERENCE["ScoresOnly"](setup)
        loss = FAST_REFERENCE["validation_loss"](scored, validation_data, 20, torch.Generator().manual_seed(1), cpu)
        results[name] = (loss, FAST_REFERENCE["model_bytes"](setup))

    return results


class FusedInt8Linear(nn.Module):
    """quant3's int8 Linear with a scale per row, multiplied by PyTorch's fused CPU kernel. The kernel reads the
    whole numbers and scales each output inside the multiply, so the float32 weight is never built. In torch
    2.14.1 the same kernel gave wrong values on MPS for some sizes, so it runs on the CPU only."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.detach()
        scales = weight.abs().amax(dim=1) / 127
        self.register_buffer("quantized", (weight / scales.unsqueeze(1)).round().to(torch.int8))
        self.register_buffer("scales", scales)
        self.bias = linear.bias

    def forward(self, x: Tensor) -> Tensor:
        rows = x.reshape(-1, x.shape[-1])
        out = torch._weight_int8pack_mm(rows, self.quantized, self.scales).reshape(*x.shape[:-1], -1)

        if self.bias is None:
            return out

        return out + self.bias


def fuse_linears_(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, FusedInt8Linear(child))
            continue

        fuse_linears_(child)


# fast3's three untrained GPTs, as vocabulary, block size, size, blocks and heads: gpt3's own settings, a middle
# one, and a big one whose float32 weights take 609 MB.
FUSED_SIZES: tuple[tuple[int, int, int, int, int], ...] = (
    (65, 128, 128, 4, 4),
    (65, 256, 512, 8, 8),
    (65, 1024, 1024, 12, 16),
)


def measure_fused_int8(
    settings: tuple[int, int, int, int, int] = FUSED_SIZES[-1],
    threads: int = 4,
    text: int = 64,
    rounds: int = 15,
    repeats: int = 5,
) -> dict[str, float]:
    """One row of fast3's second table: one cached step of an untrained GPT made with settings on the CPU after text
    tokens, in milliseconds, for float32 weights, quant3's Int8Linear and the fused kernel, with each model's bytes.

    The three kinds take turns inside every round, and each time is the median over the rounds. Timed one kind
    after another instead, a burst of other work on the machine slowed whichever kind it landed on, and once
    made float32 look 1.7 times slower than the fused kernel when, taking turns, the two came out close."""
    torch.manual_seed(0)
    plain = FAST_REFERENCE["CachedGPT"](*settings).eval()
    int8 = deepcopy(plain)
    FAST_REFERENCE["quantize_linears_"](int8.h)
    fused = deepcopy(plain)
    fuse_linears_(fused.h)
    ids = torch.randint(0, settings[0], (1, text), generator=torch.Generator().manual_seed(884))
    results: dict[str, float] = {}

    previous = torch.get_num_threads()
    torch.set_num_threads(threads)

    try:
        with torch.no_grad():
            steps: dict[str, Callable[[], object]] = {}

            for kind, model in (("float32", plain), ("int8", int8), ("fused int8", fused)):
                _, caches = model(ids[:, :-1])
                steps[kind] = functools.partial(model, ids[:, -1:], caches)
                steps[kind]()
                results[f"{kind} bytes"] = FAST_REFERENCE["model_bytes"](model)

            times: dict[str, list[float]] = {kind: [] for kind in steps}

            for _ in range(rounds):
                for kind, step in steps.items():
                    begin = time.perf_counter()

                    for _ in range(repeats):
                        step()

                    times[kind].append(1000 * (time.perf_counter() - begin) / repeats)
    finally:
        torch.set_num_threads(previous)

    for kind, kind_times in times.items():
        results[kind] = statistics.median(kind_times)

    return results


def measure_fused_sizes(runs: int = 3) -> dict[tuple[int, int, int, int, int], list[dict[str, float]]]:
    """fast3's second table, measure_fused_int8 run runs times on each of FUSED_SIZES. A small model's step takes
    under a millisecond, so it repeats 20 times between clock reads.

    The table's six runs, two calls, had a load average of 4 to 5 on a quiet Mac, and the middle model's fused step
    beat float32 in all six. An earlier table, at a load of 5 to 7 with another session busy, ran 2 to 3 times
    slower and had float32 winning on the middle model. A third call, as a build pushed the load toward 10, ran up
    to 3 times slower and float32 won two of its three middle runs, so remeasure on a quiet machine before changing the table."""
    return {
        settings: [measure_fused_int8(settings, repeats=20 if settings[2] < 1024 else 5) for _ in range(runs)]
        for settings in FUSED_SIZES
    }


def measure_fused_gap() -> dict[str, float]:
    """fast3's claim on the trained model from gpt3, after 64 tokens: the largest difference between the fused
    kernel's scores and Int8Linear's, which hold the same whole numbers and scales, beside the largest difference
    between Int8Linear's scores and float32's. An untrained model shows no gap at all, since its blocks barely move
    the scores."""
    cached = FAST_REFERENCE["fast_model"](lesson_model(), False)
    int8 = deepcopy(cached)
    FAST_REFERENCE["quantize_linears_"](int8.h)
    fused = deepcopy(cached)
    fuse_linears_(fused.h)
    ids = torch.randint(0, 65, (1, 64), generator=torch.Generator().manual_seed(884))

    with torch.no_grad():
        plain_scores, int8_scores, fused_scores = (model(ids)[0] for model in (cached, int8, fused))

    return {
        "fused against Int8Linear": (fused_scores - int8_scores).abs().max().item(),
        "Int8Linear against float32": (int8_scores - plain_scores).abs().max().item(),
    }
