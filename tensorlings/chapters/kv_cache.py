"""88_kv_cache, generation that keeps every layer's keys and values so each new token runs alone, and what that buys in time and costs in memory."""

from __future__ import annotations

import math
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_tensor,
    expect_value,
    reject_missing_return,
    small_tensor,
)
from ..text import prose, source_file
from ..training import load_checkpoint
from .flash_attention import median_seconds
from .gpt import BLOCK_PARTS, INIT_SOLUTION, MODEL_REFERENCE, SAMPLING_PARTS, SAMPLING_REFERENCE, TRAIN_REFERENCE, seeded_gpt
from .transformer_block import SelfAttention, expect_same_layers

KV_CACHE = Chapter(
    folder="88_kv_cache",
    title="The KV cache",
    readme=source_file(
        """
        # The KV cache

        When a GPT generates text, it runs the whole text through the model
        for every new token. Yet the keys and values of the earlier positions
        never change. This chapter keeps them in a cache, so each new token
        can run through the model alone. It checks that the generated text
        stays the same, and measures how much time the cache saves and how
        much memory it costs.

        ## Further reading

        - [Pope and colleagues, efficiently scaling transformer inference](https://arxiv.org/abs/2211.05102)
        - [PyTorch, gpt-fast's KV cache, made once at full length](https://github.com/meta-pytorch/gpt-fast/blob/main/model.py)
        - [Kwon and colleagues, PagedAttention](https://arxiv.org/abs/2309.06180)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


# gpt2's MLP and gpt4's sampling, without the parts the cached model replaces.
MLP_PART = BLOCK_PARTS.split("\n\n\nclass GPT2Block")[0]
NEXT_TOKEN_PART = SAMPLING_PARTS.split("\n\n\ndef generate(")[0]

CACHED_ATTENTION_START = '''
class CachedSelfAttention(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.c_attn = nn.Linear(size, 3 * size)
        self.c_proj = nn.Linear(size, size)

    def forward(
        self,
        x: Tensor,
        cache: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, steps, size = x.shape
        q, k, v = self.c_attn(x).split(size, dim=2)
        q, k, v = (part.view(batch, steps, self.heads, size // self.heads).transpose(1, 2) for part in (q, k, v))
'''

CACHED_ATTENTION_REST = '''
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        past = k.shape[2] - steps
        mask = torch.ones(steps, k.shape[2], dtype=torch.bool, device=x.device).tril(diagonal=past)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        return self.c_proj(out.transpose(1, 2).reshape(batch, steps, size)), (k, v)
'''

CACHED_BLOCK_PART = '''
# block2's block, with the attention's cache passed through.
class CachedBlock(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(size)
        self.attn = CachedSelfAttention(size, heads)
        self.ln_2 = nn.LayerNorm(size)
        self.mlp = GPT2MLP(size)

    def forward(
        self,
        x: Tensor,
        cache: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attended, cache = self.attn(self.ln_1(x), cache)
        x = x + attended

        return x + self.mlp(self.ln_2(x)), cache
'''

# gpt2's GPT with cached blocks, the same layers under the same names, so gpt3's checkpoint loads into it.
CACHED_GPT_INIT = '''
class CachedGPT(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        block_size: int,
        size: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocabulary, size)
        self.wpe = nn.Embedding(block_size, size)
        self.h = nn.ModuleList([CachedBlock(size, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(size)
        self.lm_head = nn.Linear(size, vocabulary, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(init_weights)
        scale_residuals_(self, layers)
'''

CACHED_GPT_FORWARD_START = '''
    def forward(
        self,
        ids: Tensor,
        caches: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
'''

CACHED_GPT_FORWARD_REST = '''
        start = 0 if caches is None else caches[0][0].shape[2]
        positions = torch.arange(start, start + ids.shape[1], device=ids.device)
        x = self.wte(ids) + self.wpe(positions)
        new_caches = []

        for index, block in enumerate(self.h):
            x, cache = block(x, None if caches is None else caches[index])
            new_caches.append(cache)

        return self.lm_head(self.ln_f(x)), new_caches
'''

GENERATE_SOLUTION = '''
def cached_generate(
    model: nn.Module,
    ids: Tensor,
    steps: int,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> Tensor:
    model.eval()

    with torch.no_grad():
        scores, caches = model(ids)

        for _ in range(steps):
            token = next_token(scores[:, -1], temperature, top_k, generator)
            ids = torch.cat([ids, token], dim=1)
            scores, caches = model(token, caches)

    return ids
'''

# The cached model whole, for later chapters to build on.
CACHED_MODEL_PART = (
    CACHED_ATTENTION_START
    + CACHED_ATTENTION_REST
    + "\n"
    + CACHED_BLOCK_PART
    + "\n"
    + CACHED_GPT_INIT
    + CACHED_GPT_FORWARD_START
    + CACHED_GPT_FORWARD_REST
)


def cache_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(MLP_PART + INIT_SOLUTION + NEXT_TOKEN_PART + CACHED_MODEL_PART + GENERATE_SOLUTION, namespace)

    return namespace


CACHE_REFERENCE = cache_namespace()


# ---------------------------------------------------------------------------
# kvcache1
# ---------------------------------------------------------------------------

KVCACHE1_LESSON = prose(
    """
    gpt4's `generate` runs the whole text through the model for every new
    token, so most of the work is repeated. The causal mask stops each
    position from seeing later positions. So a position's keys and values
    never change once they are computed. gqa1 named the place that stores
    them the KV cache.

    With a cache, generating has two phases. First the prompt runs through
    the model once, as before, and every layer stores its keys and values.
    Then each new token runs through the model alone. Its query attends to
    the cached keys and values plus its own, and its key and value are added
    to the cache.

    Two things change. First, a new token's position is the number of
    positions already cached, not 0. That way `wpe` gives it the same
    position embedding as in the full run. Second, the mask must let the new
    queries see the cached positions. With `past` positions cached and
    `steps` new ones, row i of the new queries may see keys up to position
    `past + i`.

    ```python
    torch.ones(2, 5, dtype=torch.bool).tril(diagonal=3)
    # tensor([[ True,  True,  True,  True, False],
    #         [ True,  True,  True,  True,  True]])
    ```

    `F.scaled_dot_product_attention` takes this mask as `attn_mask`, where
    True means the query may attend to that key. Its `is_causal=True` option
    aligns the triangle to the top left corner instead. That is wrong once
    there are more keys than queries.

    The cache only works while the text fits in the block. Past `block_size`,
    gpt4 slides its window forward, which changes the position of every
    token. Then every cached key and value would be out of date. In this
    chapter the text stays within the block.

    On the trained model from gpt3, generating with and without the cache
    wrote the same 120 characters after `ROMEO:`, for each of 5 seeds.
    """,
)

KVCACHE1_SOURCE = source_file(
    '''
    # kvcache1.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    may_see = torch.ones(2, 5, dtype=torch.bool).tril(diagonal=3)    # row 0 may see keys 0 to 3, and row 1 all 5
    longer = torch.cat([torch.zeros(1, 2, 3, 4), torch.ones(1, 2, 1, 4)], dim=2)    # shape (1, 2, 4, 4)


    # gpt2's MLP and how it initializes the weights, and gpt4's sampling.
''' + indented(MLP_PART) + '''


''' + indented(INIT_SOLUTION) + '''


''' + indented(NEXT_TOKEN_PART) + '''


    # TODO: Finish forward. x holds only the new positions, with shape (batch,
    #   steps, size). cache is None, or a tuple of the keys and values of the
    #   earlier positions. Concatenate the cached keys and values in front of
    #   the new ones along the steps dimension, dim 2. Make a mask on x's
    #   device that lets row i of the new queries see keys up to position
    #   past + i, where past is the number of cached positions. Return the
    #   output after c_proj, and the tuple of all keys and values, cached and
    #   new.
''' + indented(CACHED_ATTENTION_START) + '''
            ...


''' + indented(CACHED_BLOCK_PART) + '''


    # TODO: Finish forward. The positions start at the number of cached
    #   positions, which is the length of the first layer's cached keys along
    #   dim 2, or at 0 when caches is None. Make them on ids's device. Run each
    #   block with that layer's own cache. Return the scores and a list of
    #   every layer's new cache.
''' + indented(CACHED_GPT_INIT) + '''

''' + indented(CACHED_GPT_FORWARD_START) + '''
            ...


    # TODO: Return ids with steps more tokens added, like gpt4's generate, but
    #   using the cache. First run the model on all of ids once. Then at each
    #   step, pick a token with next_token from the scores at the last
    #   position, append it to ids, and run the model on that token alone with
    #   the caches so far. Put the model in eval mode, and run everything
    #   inside torch.no_grad().
    def cached_generate(
        model: nn.Module,
        ids: Tensor,
        steps: int,
        temperature: float,
        top_k: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...
    ''',
)

KVCACHE1_SOLUTION = source_file(
    '''
    # kvcache1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    may_see = torch.ones(2, 5, dtype=torch.bool).tril(diagonal=3)
    longer = torch.cat([torch.zeros(1, 2, 3, 4), torch.ones(1, 2, 1, 4)], dim=2)


''' + indented(MLP_PART) + '''


''' + indented(INIT_SOLUTION) + '''


''' + indented(NEXT_TOKEN_PART) + '''


''' + indented(CACHED_ATTENTION_START + CACHED_ATTENTION_REST) + '''


''' + indented(CACHED_BLOCK_PART) + '''


''' + indented(CACHED_GPT_INIT + CACHED_GPT_FORWARD_START + CACHED_GPT_FORWARD_REST) + '''


''' + indented(GENERATE_SOLUTION) + '''
    ''',
)


def run_attention(
    attention: nn.Module,
    x: Tensor,
    cache: tuple[Tensor, Tensor] | None,
    call: str,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    result = attention(x, cache)
    reject_missing_return(
        call=call,
        actual=result,
    )

    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[1], tuple) or len(result[1]) != 2:
        raise Mismatch(f"{call} should return two things, the output and the cache, which is a tuple of keys and values.")

    return result


def check_cached_attention(found: type) -> None:
    """The prompt, then two new positions, then one, each against the same positions of a full run."""
    torch.manual_seed(0)
    attention = found(8, 2)
    torch.manual_seed(0)
    reference = SelfAttention(8, 2)
    expect_same_layers(
        call="CachedSelfAttention(8, 2)",
        model=attention,
        reference=reference,
    )
    x = small_tensor(880, 2, 7, 8) / 3

    with torch.no_grad():
        expected = reference(x)
        cache = None

        for start, end in ((0, 4), (4, 6), (6, 7)):
            call = f"CachedSelfAttention(8, 2) on positions {start} to {end - 1} of 2 rows of 7, with {start} cached"
            out, cache = run_attention(attention, x[:, start:end], cache, call)
            expect_tensor(
                call=call,
                actual=out,
                expected=expected[:, start:end],
            )

            for name, part in zip(("keys", "values"), cache):
                if not isinstance(part, Tensor) or tuple(part.shape) != (2, 2, end, 4):
                    shape = tuple(part.shape) if isinstance(part, Tensor) else type(part).__name__
                    raise Mismatch(
                        f"After {call}, the cached {name} have shape {shape}, but they should have shape "
                        f"(2, 2, {end}, 4), covering every position so far.",
                    )

    # The meta device holds no numbers, and a tensor left on the CPU beside it crashes, as it would on a GPU.
    on_meta = found(8, 2).to("meta")
    meta_cache = (torch.zeros(1, 2, 3, 4, device="meta"), torch.zeros(1, 2, 3, 4, device="meta"))

    for x, cache, when in (
        (torch.zeros(1, 3, 8, device="meta"), None, "with no cache"),
        (torch.zeros(1, 1, 8, device="meta"), meta_cache, "with 3 positions cached"),
    ):
        try:
            on_meta(x, cache)
        except RuntimeError as error:
            if "device" not in str(error):
                raise

            raise Mismatch(
                f"CachedSelfAttention.forward crashed {when}, with x on a device other than the CPU, as happens on a GPU. "
                f"Make the mask on x's device, with device=x.device.\n\nThe original message, {error}",
            ) from None


def check_cached_gpt_forward(found: type) -> None:
    """With the reference blocks swapped into the learner's module, so this tests the GPT's forward alone."""
    module_globals = found.__init__.__globals__
    original = module_globals.get("CachedBlock")
    module_globals["CachedBlock"] = CACHE_REFERENCE["CachedBlock"]

    try:
        model = seeded_gpt(found)
    finally:
        module_globals["CachedBlock"] = original

    reference = seeded_gpt(MODEL_REFERENCE["GPT"])
    expect_same_layers(
        call="CachedGPT(20, 16, 32, 2, 4)",
        model=model,
        reference=reference,
    )
    ids = torch.randint(0, 20, (2, 10), generator=torch.Generator().manual_seed(881))

    with torch.no_grad():
        expected = reference(ids)
        caches = None

        for start, end in ((0, 6), (6, 8), (8, 10)):
            call = f"CachedGPT(20, 16, 32, 2, 4) on tokens {start} to {end - 1} of 2 rows of 10, with {start} cached"
            result = model(ids[:, start:end], caches)
            reject_missing_return(
                call=call,
                actual=result,
            )

            if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[1], list):
                raise Mismatch(f"{call} should return two things, the scores and a list of every layer's cache.")

            scores, caches = result

            if len(caches) != 2:
                raise Mismatch(f"{call} returned {len(caches)} caches, but it should return one per layer, so 2.")

            expect_tensor(
                call=call,
                actual=scores,
                expected=expected[:, start:end],
            )

    check_positions_device(model)


def check_positions_device(model: nn.Module) -> None:
    """On the meta device an Embedding takes CPU positions without a word, so a hook on wpe looks at them."""
    devices: list[str] = []
    hook = model.wpe.register_forward_pre_hook(lambda module, args: devices.append(args[0].device.type))
    model.to("meta")
    ids = torch.zeros(2, 4, dtype=torch.long, device="meta")

    try:
        with torch.no_grad():
            _, caches = model(ids)
            model(ids[:, :1], caches)
    except RuntimeError as error:
        if "device" not in str(error):
            raise

        raise Mismatch(
            "CachedGPT.forward crashed with ids on a device other than the CPU, as happens on a GPU. Make every new tensor "
            f"on ids's device, with device=ids.device.\n\nThe original message, {error}",
        ) from None
    finally:
        hook.remove()

    if any(device != "meta" for device in devices):
        raise Mismatch(
            "CachedGPT.forward made the positions on the CPU while ids was on another device, which crashes on a "
            "GPU. Make them on ids's device, with device=ids.device.",
        )


def check_cached_generate(found: object) -> None:
    calls: list[tuple[bool, bool, int, bool]] = []

    def record(
        module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        caches = args[1] if len(args) > 1 else kwargs.get("caches")
        calls.append((module.training, torch.is_grad_enabled(), args[0].shape[1], caches is not None))

    for rows, prompt_length, steps, seed in ((2, 5, 9, 882), (1, 3, 13, 883)):
        prompt = torch.randint(0, 20, (rows, prompt_length), generator=torch.Generator().manual_seed(seed))
        expected = SAMPLING_REFERENCE["generate"](seeded_gpt(MODEL_REFERENCE["GPT"]), prompt, steps, 0.9, 5, torch.Generator().manual_seed(seed + 10))
        model = seeded_gpt(CACHE_REFERENCE["CachedGPT"])
        model.train()
        calls.clear()
        hook = model.register_forward_pre_hook(record, with_kwargs=True)

        try:
            actual = found(model, prompt.clone(), steps, 0.9, 5, torch.Generator().manual_seed(seed + 10))
        finally:
            hook.remove()

        if any(training for training, _, _, _ in calls):
            raise Mismatch("cached_generate ran the model in training mode. Call model.eval() first.")

        if any(recording for _, recording, _, _ in calls):
            raise Mismatch("cached_generate let autograd record the computation. Generate the tokens inside torch.no_grad().")

        if calls and calls[0][2:] != (prompt_length, False):
            raise Mismatch(f"cached_generate should first run the whole prompt of {prompt_length} tokens, with no caches.")

        if any((length, cached) != (1, True) for _, _, length, cached in calls[1:]):
            raise Mismatch("After the prompt, cached_generate should run each new token alone, with the caches so far.")

        expect_tensor(
            call=f"cached_generate of {steps} tokens after {rows} rows of {prompt_length}, against gpt4's generate",
            actual=actual,
            expected=expected,
        )


def check_kvcache1(checker: Checker) -> None:
    checker.test_method(
        class_name="CachedSelfAttention",
        method="forward",
        description="gives the same outputs a few positions at a time as all at once",
        run=check_cached_attention,
    )
    checker.test_method(
        class_name="CachedGPT",
        method="forward",
        description="starts the positions at the number of cached positions",
        run=check_cached_gpt_forward,
    )
    checker.test(
        function="cached_generate",
        description="generates the same tokens as gpt4's generate",
        run=check_cached_generate,
    )


KVCACHE1 = Exercise(
    name="kvcache1",
    chapter=KV_CACHE,
    title="Keys and values kept",
    lesson=KVCACHE1_LESSON,
    source=KVCACHE1_SOURCE,
    solution=KVCACHE1_SOLUTION,
    hints={
        "CachedSelfAttention.forward": "When cache is not None, torch.cat its keys in front of k on dim=2, and its "
        "values in front of v. past is then k.shape[2] - steps. The mask is torch.ones(steps, k.shape[2], "
        "dtype=torch.bool, device=x.device).tril(diagonal=past), passed as attn_mask.",
        "CachedGPT.forward": "start is caches[0][0].shape[2], or 0 when caches is None, and the positions run from "
        "start to start + ids.shape[1]. Loop over enumerate(self.h), passing caches[index] or None, and keep each "
        "block's cache in a list.",
        "cached_generate": "First run model on all of ids to get the scores and the caches. At each step, call "
        "next_token on scores[:, -1], torch.cat the token onto ids on dim=1, then run model on that token and the "
        "caches.",
    },
    introduces=(),
    check=check_kvcache1,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# kvcache2
# ---------------------------------------------------------------------------

COUNTS_SOLUTION = '''
def cache_bytes(caches: list[tuple[Tensor, Tensor]]) -> int:
    return sum(part.numel() * part.element_size() for pair in caches for part in pair)


def positions_run(
    prompt: int,
    steps: int,
    block_size: int,
    cached: bool,
) -> int:
    if cached:
        return prompt + steps

    return sum(min(prompt + step, block_size) for step in range(steps))
'''

COUNTS_REFERENCE: dict[str, object] = {"Tensor": Tensor}
exec(COUNTS_SOLUTION, COUNTS_REFERENCE)

KVCACHE2_LESSON = prose(
    """
    Without the cache, generating a token after a text of n tokens runs the
    model on all n tokens. With the cache, the model runs on 1 token, and that
    token's attention reads n cached keys. So the Linear layers, which do
    most of a small model's work, run on 1 position instead of n.

    `positions_run` counts this work. To generate 120 tokens after a prompt
    of 8, gpt4's `generate` runs the model on 8,100 positions in total, and
    `cached_generate` on only 128.

    Here is the time to generate one token after a text of each length, on
    this Mac with 4 threads, over three runs. The bigger GPT has 8 blocks of
    size 512 and reads up to 1,024 tokens. It is untrained, because the time
    does not depend on the values of the weights.

    | Model, device | Text | Without the cache | With it |
    |---|---|---|---|
    | Course GPT, CPU | 16 | 0.51 to 0.53 ms | 0.25 to 0.26 ms |
    | Course GPT, CPU | 128 | 1.1 to 1.3 ms | 0.25 to 0.29 ms |
    | Bigger GPT, CPU | 128 | 9.6 to 12 ms | 2.1 to 2.4 ms |
    | Bigger GPT, CPU | 1,024 | 74 to 76 ms | 3.5 to 4.2 ms |
    | Bigger GPT, MPS | 1,024 | 33 to 36 ms | 3.3 to 3.5 ms |

    The cache was faster on every row in every run, by 2 to 21 times, and its
    advantage grew with the length of the text. Without the cache, each step
    runs the model over every earlier token again, so a step costs more as
    the text grows. On the CPU without the cache, the bigger GPT's step over
    1,024 tokens took 6.5 to 7.8 times as long as its step over 128. With the
    cache, it took 1.7 to 1.8 times as long.

    The time with the cache still grew, for two reasons. Attention reads
    every cached key, and `torch.cat` copies the whole cache at each step.
    PyTorch's gpt-fast creates the cache once at its full length and writes
    each new key into its slot, which avoids that copy.

    The cache costs memory. For the course's GPT at 128 tokens, the cache
    holds 0.5 MiB. The weights take 3.1 MiB, which is the 3.27 MB that cost1
    counted. The bigger GPT's cache at 1,024 tokens is 32 MiB for one
    text. A server generating for many people keeps one cache per text. The
    shared key and value heads from gqa1 make exactly this cache smaller.

    The prompt's cache has one catch. Its keys and values are views of the
    whole output of `c_attn`, which includes the queries. So they keep 1.5
    times as many bytes in memory as `cache_bytes` counts, until the first
    `torch.cat` copies them.
    """,
)

KVCACHE2_SOURCE = source_file(
    '''
    # kvcache2.py

    import torch
    from torch import Tensor

    keys = torch.zeros(1, 4, 7, 32)
    numbers = keys.numel()             # 896, 1 * 4 * 7 * 32
    each = keys.element_size()         # 4 bytes, for a float32


    # TODO: Return how many bytes caches holds in total. caches is a list with
    #   one (keys, values) tuple per layer, as kvcache1's CachedGPT returns it.
    #   Example: for one layer whose keys and values each have shape
    #   (1, 4, 7, 32) in float32, cache_bytes returns 7168.
    def cache_bytes(caches: list[tuple[Tensor, Tensor]]) -> int:
        ...


    # TODO: Return how many positions the model runs on in total, to generate
    #   steps tokens after a prompt of prompt tokens. When cached is False,
    #   gpt4's generate runs the whole text at each step, but at most
    #   block_size positions of it. When cached is True, cached_generate runs
    #   the prompt once, then one position per step.
    #   Example: positions_run(8, 120, 128, False) returns 8100, and
    #   positions_run(8, 120, 128, True) returns 128.
    def positions_run(
        prompt: int,
        steps: int,
        block_size: int,
        cached: bool,
    ) -> int:
        ...
    ''',
)

KVCACHE2_SOLUTION = source_file(
    '''
    # kvcache2.py, a solution for comparison

    import torch
    from torch import Tensor

    keys = torch.zeros(1, 4, 7, 32)
    numbers = keys.numel()
    each = keys.element_size()


''' + indented(COUNTS_SOLUTION) + '''
    ''',
)


def zero_caches(
    layers: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> Described:
    caches = [(torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)) for _ in range(layers)]

    return Described(caches, f"{layers} layers of keys and values of shape {shape} in {str(dtype).removeprefix('torch.')}")


def check_kvcache2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="cache_bytes",
        description="adds up the bytes of every layer's keys and values",
        inputs=(
            (zero_caches(2, (1, 4, 7, 32), torch.float32),),
            (zero_caches(3, (2, 2, 5, 8), torch.float16),),
        ),
        reference=COUNTS_REFERENCE["cache_bytes"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="positions_run",
        description="counts the whole text at each step, or one position per step with the cache",
        inputs=(
            (8, 120, 128, False),
            (8, 120, 128, True),
            (100, 60, 128, False),
            (6, 10, 16, True),
        ),
        reference=COUNTS_REFERENCE["positions_run"],
        expect=expect_value,
    )


KVCACHE2 = Exercise(
    name="kvcache2",
    chapter=KV_CACHE,
    title="What the cache buys and costs",
    lesson=KVCACHE2_LESSON,
    source=KVCACHE2_SOURCE,
    solution=KVCACHE2_SOLUTION,
    hints={
        "cache_bytes": "For every pair in caches and every part of the pair, add part.numel() * part.element_size().",
        "positions_run": "With the cache, count the prompt once and then one position for each step. Without it, "
        "step s runs min(prompt + s, block_size) positions. Add that up for s from 0 to steps - 1.",
    },
    introduces=(),
    check=check_kvcache2,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_matching_samples(seeds: range = range(5)) -> list[bool]:
    """kvcache1's claim, on the trained model in checkpoints/gpt3.pt: the cached and plain ways write the same text."""
    characters, _, _ = TRAIN_REFERENCE["shakespeare_splits"]()
    plain = load_checkpoint(MODEL_REFERENCE["GPT"](len(characters), 128, 128, 4, 4), "gpt3")
    cached = load_checkpoint(CACHE_REFERENCE["CachedGPT"](len(characters), 128, 128, 4, 4), "gpt3")
    prompt = torch.tensor([[characters.index(character) for character in "ROMEO:"]])

    return [
        torch.equal(
            SAMPLING_REFERENCE["generate"](plain, prompt, 120, 0.8, 20, torch.Generator().manual_seed(seed)),
            CACHE_REFERENCE["cached_generate"](cached, prompt, 120, 0.8, 20, torch.Generator().manual_seed(seed)),
        )
        for seed in seeds
    ]


SMALL_SETTINGS = (65, 128, 128, 4, 4)
BIG_SETTINGS = (65, 1024, 512, 8, 8)


def step_milliseconds(
    settings: tuple[int, int, int, int, int],
    length: int,
    device: torch.device,
) -> tuple[float, float]:
    """The time to write one token after a text of length tokens, without the cache and with it."""
    torch.manual_seed(0)
    plain = MODEL_REFERENCE["GPT"](*settings).to(device).eval()
    torch.manual_seed(0)
    cached = CACHE_REFERENCE["CachedGPT"](*settings).to(device).eval()
    ids = torch.randint(0, settings[0], (1, length), generator=torch.Generator().manual_seed(884)).to(device)

    with torch.no_grad():
        _, caches = cached(ids[:, :-1])

        return (
            1000 * median_seconds(lambda: plain(ids), device, rounds=7, repeats=10),
            1000 * median_seconds(lambda: cached(ids[:, -1:], caches), device, rounds=7, repeats=10),
        )


def measure_step_times(threads: int = 4) -> dict[tuple[str, str, int], tuple[float, float]]:
    """The times behind kvcache2's table, in milliseconds: the course GPT on the CPU, and the bigger GPT on the
    CPU and on MPS. MPS runs only the long text. At 128 tokens its step is bound by launching kernels, and the
    cached step launches more of them, 761 calls against 647, so on a busy machine the cache won by 0.9 to 1.5
    times there."""
    results = {}
    runs = (
        ("course", SMALL_SETTINGS, "cpu", (16, 128)),
        ("bigger", BIG_SETTINGS, "cpu", (128, 1024)),
        ("bigger", BIG_SETTINGS, "mps", (1024,)),
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)

    try:
        for name, settings, device, lengths in runs:
            for length in lengths:
                results[(name, device, length)] = step_milliseconds(settings, length, torch.device(device))
    finally:
        torch.set_num_threads(previous)

    return results


def measure_memory() -> dict[str, int]:
    """kvcache2's bytes, of each model's cache over its whole block and of its weights."""
    results = {}

    for name, settings in (("course", SMALL_SETTINGS), ("bigger", BIG_SETTINGS)):
        torch.manual_seed(0)
        model = CACHE_REFERENCE["CachedGPT"](*settings).eval()
        ids = torch.zeros(1, settings[1], dtype=torch.long)

        with torch.no_grad():
            _, caches = model(ids)

        results[f"{name} cache"] = COUNTS_REFERENCE["cache_bytes"](caches)
        results[f"{name} weights"] = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    return results
