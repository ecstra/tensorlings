"""59_gpt, Section 7's project, a GPT built from your blocks, trained on tiny Shakespeare for minutes on the best device, and sampled."""

from __future__ import annotations

import contextlib
import math
import re
import textwrap
import time
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import load_shakespeare
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
    seeded,
)
from ..text import prose, source_file
from ..training import MissingCheckpoint, best_device, load_checkpoint
from .transformer_block import SELF_ATTENTION, expect_same_layers


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """Four CPU threads while measuring, so the sums add up in the same order on every run."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        yield
    finally:
        torch.set_num_threads(threads)

GPT = Chapter(
    folder="59_gpt",
    title="Project: a GPT that writes Shakespeare",
    readme=source_file(
        """
        # Project: a GPT that writes Shakespeare

        This is Section 7's project. You assemble your transformer blocks into
        a GPT and train it on a million characters of Shakespeare. Training
        takes a few minutes on the fastest device the machine has. Then you
        sample new text from the model, as in Karpathy's GPT lecture and
        nanoGPT. Download the data once from the project folder.

        ```bash
        .venv/bin/python run.py data shakespeare
        ```

        ## Further reading

        - [Karpathy, let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
        - [Karpathy, nanoGPT](https://github.com/karpathy/nanoGPT)
        - [Radford and colleagues, language models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
        """,
    ),
)


def reference_characters_of(text: str) -> list[str]:
    return sorted(set(text))


def reference_encode(
    text: str,
    characters: list[str],
) -> Tensor:
    index = {character: position for position, character in enumerate(characters)}

    return torch.tensor([index[character] for character in text])


def reference_decode(
    ids: Tensor,
    characters: list[str],
) -> str:
    return "".join(characters[index] for index in ids.tolist())


def reference_split(
    data: Tensor,
    fraction: float,
) -> tuple[Tensor, Tensor]:
    cut = int(fraction * len(data))

    return data[:cut], data[cut:]


def reference_get_batch(
    data: Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator).tolist()
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])

    return x, y


# ---------------------------------------------------------------------------
# gpt1
# ---------------------------------------------------------------------------

GPT1_LESSON = prose(
    """
    This is Section 7's project. You build a GPT and train it on tiny
    Shakespeare, a million characters of the plays, as Karpathy does in his
    GPT lecture. Download the text once, about 1 MB, from the project folder
    with `.venv/bin/python run.py data shakespeare`.

    ```python
    from tensorlings.data import load_shakespeare

    text = load_shakespeare()
    len(text)        # 1115394
    text[:14]        # 'First Citizen:'
    ```

    The tokens here are characters, 65 different ones, from the newline
    character to `z`. The BPE tokens from 58_bpe would make the sequences
    shorter. But with only a million characters of text, a small vocabulary
    keeps the model small and quick to train. nanoGPT's Shakespeare example
    makes the same choice.

    The first 90% of the text is used for training and the last 10% for
    validation. The split is a single cut, not a shuffle. Blocks of text
    overlap, so after a shuffle the same lines would end up in both the
    training and the validation data.

    A batch picks random starting points in the data. From each start, `x` is
    a block of `block_size` tokens, and `y` is the same block shifted one
    token to the right. So each position of `y` holds the character that
    comes after the same position of `x`. Every position is one training
    example, predicting the next character from all the characters before it.
    A block of 128 tokens holds 128 examples.

    ```python
    data = torch.tensor([5, 6, 7, 8, 9])
    data[1:4], data[2:5]    # (tensor([6, 7, 8]), tensor([7, 8, 9]))
    ```

    A model that knows nothing gives every character the same probability,
    1/65. Its loss is `log(65)`, which is 4.17. That is the loss training
    starts from.
    """,
)

GPT1_SOURCE = source_file(
    '''
    # gpt1.py
    #
    # Section 7's project starts with the data. Download it once from the project
    # folder with
    #
    #     .venv/bin/python run.py data shakespeare

    import torch
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    text = load_shakespeare()                       # 1,115,394 characters
    index = {letter: position for position, letter in enumerate("abc")}    # {'a': 0, 'b': 1, 'c': 2}


    # TODO: Return a sorted list of every distinct character in text, each one
    #   once.
    #   Example: characters_of("banana") returns ['a', 'b', 'n'].
    def characters_of(text: str) -> list[str]:
        ...


    # TODO: Return the ids of the characters of text as a tensor. The id of a
    #   character is its position in characters. Build a dict from character
    #   to position first, because characters.index would search the whole
    #   list for each of a million characters.
    #   Example: encode("cab", ["a", "b", "c"]) returns tensor([2, 0, 1]).
    def encode(
        text: str,
        characters: list[str],
    ) -> Tensor:
        ...


    # TODO: Return the text that ids spell, which reverses encode.
    #   Example: decode(torch.tensor([2, 0, 1]), ["a", "b", "c"]) returns 'cab'.
    def decode(
        ids: Tensor,
        characters: list[str],
    ) -> str:
        ...


    # TODO: Split data into a training part and a validation part with a
    #   single cut at int(fraction * len(data)). Return the part before the
    #   cut and the part after it.
    #   Example: split(torch.arange(10), 0.8) returns
    #   (tensor([0, 1, 2, 3, 4, 5, 6, 7]), tensor([8, 9])).
    def split(
        data: Tensor,
        fraction: float,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return x and y, each of shape (batch_size, block_size). Draw the
    #   starting points with torch.randint(0, len(data) - block_size,
    #   (batch_size,), generator=generator). Row i of x is the block_size
    #   tokens from start i, and row i of y is the same block shifted one
    #   token to the right.
    def get_batch(
        data: Tensor,
        block_size: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        ...
    ''',
)

GPT1_SOLUTION = source_file(
    '''
    # gpt1.py, a solution for comparison

    import torch
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    text = load_shakespeare()
    index = {letter: position for position, letter in enumerate("abc")}


    def characters_of(text: str) -> list[str]:
        return sorted(set(text))


    def encode(
        text: str,
        characters: list[str],
    ) -> Tensor:
        positions = {character: position for position, character in enumerate(characters)}

        return torch.tensor([positions[character] for character in text])


    def decode(
        ids: Tensor,
        characters: list[str],
    ) -> str:
        return "".join(characters[index] for index in ids.tolist())


    def split(
        data: Tensor,
        fraction: float,
    ) -> tuple[Tensor, Tensor]:
        cut = int(fraction * len(data))

        return data[:cut], data[cut:]


    def get_batch(
        data: Tensor,
        block_size: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator).tolist()
        x = torch.stack([data[start : start + block_size] for start in starts])
        y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])

        return x, y
    ''',
)


def expect_pair(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should return a tuple of two tensors.")

    for name, part, wanted in zip(("first", "second"), actual, expected):
        expect_tensor(
            call=f"the {name} tensor of {call}",
            actual=part,
            expected=wanted,
        )


SAMPLE_TEXT = "To be, or not to be: that is the question.\nWhether 'tis nobler in the mind to suffer"
SAMPLE_CHARACTERS = reference_characters_of(SAMPLE_TEXT)


def check_gpt1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="characters_of",
        description="returns each character once, sorted",
        inputs=(
            (SAMPLE_TEXT,),
            ("banana bread\n",),
        ),
        reference=reference_characters_of,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="encode",
        description="returns each character's position",
        inputs=(
            ("not to be", Described(SAMPLE_CHARACTERS, "the characters of a line of Hamlet")),
            ("tis nobler\n", Described(SAMPLE_CHARACTERS, "the characters of a line of Hamlet")),
        ),
        reference=reference_encode,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="decode",
        description="turns the ids back into text",
        inputs=(
            (reference_encode("the question", SAMPLE_CHARACTERS), Described(SAMPLE_CHARACTERS, "the characters of a line of Hamlet")),
        ),
        reference=reference_decode,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="split",
        description="makes one cut, with the training part first",
        inputs=(
            (torch.arange(20), 0.9),
            (torch.arange(7), 0.5),
        ),
        reference=reference_split,
        expect=expect_pair,
    )
    check_on_inputs(
        checker=checker,
        function="get_batch",
        description="returns blocks and the same blocks shifted one token",
        inputs=(
            (torch.arange(100, 140), 5, 3, seeded(590)),
            (torch.arange(12), 4, 6, seeded(591)),
        ),
        reference=reference_get_batch,
        expect=expect_pair,
    )


GPT1 = Exercise(
    name="gpt1",
    chapter=GPT,
    title="Project: Shakespeare as tokens",
    lesson=GPT1_LESSON,
    source=GPT1_SOURCE,
    solution=GPT1_SOLUTION,
    hints={
        "characters_of": "set(text) keeps each character once, and sorted turns it into a sorted list.",
        "encode": "Make the dict with enumerate(characters), then look every character of text up in it and pass "
        "the list to torch.tensor.",
        "decode": "Look up characters[index] for each index in ids.tolist(), and join them with \"\".join.",
        "split": "Cut at int(fraction * len(data)), and slice data before and after the cut.",
        "get_batch": "Turn the starts into a list with .tolist(). torch.stack the slices data[start : start + "
        "block_size] for x, and the slices shifted one token to the right for y.",
    },
    introduces=(),
    check=check_gpt1,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# gpt2
# ---------------------------------------------------------------------------

# block2's MLP and block, which gpt2, gpt3 and gpt4 are handed after SelfAttention.
BLOCK_PARTS = '''
class GPT2MLP(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(size, 4 * size)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * size, size)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class GPT2Block(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(size)
        self.attn = SelfAttention(size, heads)
        self.ln_2 = nn.LayerNorm(size)
        self.mlp = GPT2MLP(size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))

        return x + self.mlp(self.ln_2(x))
'''

INIT_SOLUTION = '''
def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


def scale_residuals_(
    model: nn.Module,
    layers: int,
) -> None:
    for block in model.h:
        nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * layers))
        nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * layers))
'''

GPT_INIT = '''
class GPT(nn.Module):
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
        self.h = nn.ModuleList([GPT2Block(size, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(size)
        self.lm_head = nn.Linear(size, vocabulary, bias=False)
        # Tied as in GPT-2, so the same table embeds the tokens and scores them.
        self.lm_head.weight = self.wte.weight
        self.apply(init_weights)
        scale_residuals_(self, layers)
'''

GPT_FORWARD = '''
    def forward(self, ids: Tensor) -> Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.wte(ids) + self.wpe(positions)

        for block in self.h:
            x = block(x)

        return self.lm_head(self.ln_f(x))
'''

GIVEN_MODEL = SELF_ATTENTION + "\n\n" + BLOCK_PARTS
INDENTED_GIVEN_MODEL = textwrap.indent(GIVEN_MODEL.strip("\n"), "    ")
INDENTED_INIT = textwrap.indent(INIT_SOLUTION.strip("\n"), "    ")
INDENTED_GPT_INIT = textwrap.indent(GPT_INIT.strip("\n"), "    ")
INDENTED_GPT_FORWARD = textwrap.indent(GPT_FORWARD.strip("\n"), "    ")

GPT2_LESSON = prose(
    """
    A GPT is the block from block2 repeated several times, with two embedding
    tables before the blocks and a head after them. `wte` turns each token
    into a vector, and `wpe` adds a vector for its position, as in position2.
    Then come the blocks and a final LayerNorm. Last, `lm_head` gives every
    token in the vocabulary a score for being the next token. The names are
    the ones GPT-2 uses. GPT-2's released weights load into this model once
    four of its matrices are transposed, as block2 explained.

    The head shares its weight matrix with `wte`, as in GPT-2. This is called
    tying the weights. The same row of numbers embeds a token at the input
    and scores that token at the output, which saves a whole table.

    nanoGPT draws every Linear and Embedding weight from a normal
    distribution with mean 0 and standard deviation 0.02. It sets every bias
    to 0. GPT-2's released code does the same, except that it uses a standard
    deviation of 0.01 for `wpe`.

    The **residual stream** is the main path from residual2. It is the `x`
    that every block adds its attention and MLP outputs to. As the GPT-2
    paper describes, nanoGPT starts the layers that write into the stream
    with smaller weights. Those layers are the `c_proj` of each attention and
    each MLP. Their standard deviation is divided by the square root of how
    many such layers there are, `2 * layers`.

    Here is why. The GPTs below have the size of gpt3's model, 128, and read
    128 random characters. The table shows the standard deviation of the
    stream after the last block. It is a multiple of the stream's standard
    deviation at the input, after the token and position vectors are added.
    Each cell is the range over 3 seeds.

    | Blocks | 4 | 12 | 24 | 48 |
    |---|---|---|---|---|
    | Every weight drawn with standard deviation 0.02 | 4.2 to 4.5 | 7.7 to 7.8 | 11.1 to 11.5 | 15.7 to 16.5 |
    | `c_proj` scaled down | 1.7 | 1.7 | 1.7 | 1.7 to 1.8 |

    Every block adds its output to the stream. Without the scaling, the
    stream keeps growing with depth, as residual2 measured. Here it grows
    roughly like the square root of the number of blocks. With the scaling,
    the stream's size stays the same however deep the model is, much as
    starting at zero did in residual3.

    `model.apply(fn)` calls `fn` on every module inside `model`, including
    `model` itself. That is how one function can set up the weights of every
    layer.

    ```python
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    net.apply(lambda module: print(type(module).__name__))    # Linear, ReLU, Linear, Sequential
    ```
    """,
)

GPT2_SOURCE = source_file(
    '''
    # gpt2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    layer = nn.Linear(3, 2)
    nn.init.normal_(layer.weight, mean=0.0, std=0.5)    # redraws the weight with standard deviation 0.5
    nn.init.zeros_(layer.bias)                          # sets every bias to 0


    # SelfAttention, GPT2MLP and GPT2Block from block2.
''' + INDENTED_GIVEN_MODEL + '''


    # TODO: Initialize module the way nanoGPT does, if it is an nn.Linear or
    #   an nn.Embedding. Draw its weight from a normal distribution with mean 0
    #   and standard deviation 0.02. If it is a Linear with a bias, set the bias
    #   to 0. Leave every other kind of module unchanged.
    def init_weights(module: nn.Module) -> None:
        ...


    # TODO: Redraw the c_proj weights of every block's attention and MLP from
    #   a normal distribution with mean 0 and standard deviation
    #   0.02 / sqrt(2 * layers). model.h holds the blocks.
    def scale_residuals_(
        model: nn.Module,
        layers: int,
    ) -> None:
        ...


    # TODO: Finish forward. ids has shape (batch, steps). Add wte(ids) to the
    #   wpe of the positions 0, 1, 2 and so on up to steps - 1. Run the result
    #   through every block of h in turn, then through ln_f and lm_head, and
    #   return the scores.
''' + INDENTED_GPT_INIT + '''

        def forward(self, ids: Tensor) -> Tensor:
            ...
    ''',
)

GPT2_SOLUTION = source_file(
    '''
    # gpt2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    layer = nn.Linear(3, 2)
    nn.init.normal_(layer.weight, mean=0.0, std=0.5)
    nn.init.zeros_(layer.bias)


''' + INDENTED_GIVEN_MODEL + '''


''' + INDENTED_INIT + '''


''' + INDENTED_GPT_INIT + '''

''' + INDENTED_GPT_FORWARD + '''
    ''',
)


def model_namespace() -> dict[str, object]:
    """The given blocks and the solution's GPT, run once, so the checks compare against exactly the code
    the learner's file holds."""
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(GIVEN_MODEL + INIT_SOLUTION + GPT_INIT + GPT_FORWARD, namespace)

    return namespace


MODEL_REFERENCE = model_namespace()

SMALL_GPT = (20, 16, 32, 2, 4)


def seeded_gpt(
    make: type,
    settings: tuple[int, int, int, int, int] = SMALL_GPT,
) -> nn.Module:
    torch.manual_seed(0)

    return make(*settings)


def check_init_weights(found: object) -> None:
    torch.manual_seed(592)
    linear = nn.Linear(400, 300)
    embedding = nn.Embedding(500, 200)
    plain = nn.Linear(10, 10, bias=False)

    for module in (linear, embedding, plain):
        found(module)

    for name, weight in (("a Linear's weight", linear.weight), ("an Embedding's weight", embedding.weight)):
        spread = weight.std().item()

        if abs(spread - 0.02) > 0.002 or abs(weight.mean().item()) > 0.002:
            raise Mismatch(f"After init_weights, {name} has a standard deviation of {spread:.4f}, but it should be drawn with mean 0 and standard deviation 0.02.")

        # A normal spread puts about 4.6% of its draws beyond two spreads, and an even spread of 0.02 puts none.
        tails = (weight.abs() > 0.04).float().mean().item()

        if not 0.03 < tails < 0.06:
            raise Mismatch(
                f"After init_weights, {tails:.1%} of {name} lies beyond 0.04, two standard deviations from 0, but a "
                "normal distribution puts about 4.6% there. Draw it with nn.init.normal_.",
            )

    if not torch.all(linear.bias == 0):
        raise Mismatch("After init_weights, a Linear's bias should be all 0.")

    norm = nn.LayerNorm(4)
    found(norm)

    if not (torch.all(norm.weight == 1) and torch.all(norm.bias == 0)):
        raise Mismatch("init_weights changed a LayerNorm, but it should change only nn.Linear and nn.Embedding.")


def check_scale_residuals(found: object) -> None:
    torch.manual_seed(593)
    model = MODEL_REFERENCE["GPT"](20, 16, 256, 8, 4)

    # The reference model comes already scaled, so every c_proj goes back to 0.02 and a layer left out shows.
    with torch.no_grad():
        for block in model.h:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=0.02)

    attention_before = model.h[0].attn.c_attn.weight.clone()
    found(model, 8)
    wanted = 0.02 / math.sqrt(16)

    for index, block in enumerate(model.h):
        for name, weight in (("attn.c_proj", block.attn.c_proj.weight), ("mlp.c_proj", block.mlp.c_proj.weight)):
            spread = weight.std().item()

            if abs(spread - wanted) > 0.1 * wanted:
                raise Mismatch(
                    f"After scale_residuals_ on 8 layers, block {index}'s {name} has a standard deviation of "
                    f"{spread:.5f}, but it should be 0.02 / sqrt(16), which is {wanted:.5f}.",
                )

    if not torch.equal(model.h[0].attn.c_attn.weight, attention_before):
        raise Mismatch("scale_residuals_ changed c_attn, but only the two c_proj layers of each block should be redrawn.")


def check_gpt_forward(found: type) -> None:
    model = seeded_gpt(found)
    reference = seeded_gpt(MODEL_REFERENCE["GPT"])
    expect_same_layers(
        call="GPT(20, 16, 32, 2, 4)",
        model=model,
        reference=reference,
    )

    if model.lm_head.weight is not model.wte.weight:
        raise Mismatch("lm_head's weight should be the same tensor as wte's weight, tied as the given __init__ sets it.")

    # The reference's weights go in first, so forward is checked on its own, whatever order the start drew in.
    with torch.no_grad():
        model.load_state_dict(reference.state_dict())

    ids = torch.randint(0, 20, (3, 10), generator=torch.Generator().manual_seed(594))

    with torch.no_grad():
        expected = reference(ids)

    expect_tensor(
        call="GPT(20, 16, 32, 2, 4) on 3 rows of 10 tokens",
        actual=model(ids),
        expected=expected,
    )


def check_gpt2(checker: Checker) -> None:
    checker.test(
        function="init_weights",
        description="draws Linear and Embedding weights with standard deviation 0.02, biases at 0",
        run=check_init_weights,
    )
    checker.test(
        function="scale_residuals_",
        description="divides the standard deviation of every c_proj by sqrt(2 * layers)",
        run=check_scale_residuals,
    )
    checker.test_method(
        class_name="GPT",
        method="forward",
        description="adds positions, runs the blocks, then the norm and the head",
        run=check_gpt_forward,
    )


GPT2 = Exercise(
    name="gpt2",
    chapter=GPT,
    title="Project: your blocks as a GPT",
    lesson=GPT2_LESSON,
    source=GPT2_SOURCE,
    solution=GPT2_SOLUTION,
    hints={
        "init_weights": "isinstance(module, (nn.Linear, nn.Embedding)) matches both kinds. Use nn.init.normal_ with "
        "std=0.02 for the weight, and nn.init.zeros_ for a Linear's bias when module.bias is not None.",
        "scale_residuals_": "For each block of model.h, call nn.init.normal_ on block.attn.c_proj.weight and on "
        "block.mlp.c_proj.weight, with std=0.02 / math.sqrt(2 * layers).",
        "GPT.forward": "torch.arange(ids.shape[1], device=ids.device) gives the positions, on the same device as ids. "
        "Add self.wte(ids) and self.wpe of the positions, loop over self.h, then apply self.ln_f and self.lm_head.",
    },
    introduces=(),
    check=check_gpt2,
    time_limit=30,
)


def measure_residual_growth(
    blocks: tuple[int, ...] = (4, 12, 24, 48),
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, dict[int, list[float]]]:
    """gpt2's table: GPTs of gpt3's size, 65 tokens, 128 positions, size 128 and 4 heads, from each seed, run on 8
    rows of 128 random tokens. Gives the spread of the residual stream after the last block over its spread going
    in, the token and position rows added, with c_proj scaled down as gpt2 does and with it drawn at 0.02."""
    results: dict[str, dict[int, list[float]]] = {"plain": {}, "scaled": {}}

    with pinned_threads(), torch.no_grad():
        for layers in blocks:
            for seed in seeds:
                for kind in ("plain", "scaled"):
                    torch.manual_seed(seed)
                    model = MODEL_REFERENCE["GPT"](65, 128, 128, layers, 4)

                    if kind == "plain":
                        for block in model.h:
                            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=0.02)
                            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=0.02)

                    ids = torch.randint(0, 65, (8, 128), generator=torch.Generator().manual_seed(seed + 100))
                    stream = model.wte(ids) + model.wpe(torch.arange(128))
                    spread_in = stream.std().item()

                    for block in model.h:
                        stream = block(stream)

                    results[kind].setdefault(layers, []).append(stream.std().item() / spread_in)

    return results


# ---------------------------------------------------------------------------
# gpt3
# ---------------------------------------------------------------------------

# gpt1's batches and gpt2's model, which gpt3 and gpt4 are handed.
DATA_PART = '''
def get_batch(
    data: Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator).tolist()
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])

    return x, y


def shakespeare_splits() -> tuple[list[str], Tensor, Tensor]:
    text = load_shakespeare()
    characters = sorted(set(text))
    positions = {character: position for position, character in enumerate(characters)}
    data = torch.tensor([positions[character] for character in text])
    cut = int(0.9 * len(data))

    return characters, data[:cut], data[cut:]
'''

# The size of the GPT the project trains, its checkpoint and its checks.
GPT_SETTINGS = '''
BLOCK_SIZE = 128
SIZE = 128
LAYERS = 4
HEADS = 4
'''

TRAIN_PARTS = '''
def learning_rate(
    step: int,
    max_lr: float,
    warmup: int,
    total: int,
    min_lr: float,
) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup

    progress = (step - warmup) / (total - warmup)

    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decayed = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    kept = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    groups = [{"params": decayed, "weight_decay": weight_decay}, {"params": kept, "weight_decay": 0.0}]

    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    lr: float,
    clip: float,
) -> float:
    for group in optimizer.param_groups:
        group["lr"] = lr

    scores = model(x)
    loss = F.cross_entropy(scores.flatten(0, 1), y.flatten())
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    return loss.item()


def validation_loss(
    model: nn.Module,
    data: Tensor,
    batches: int,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch(data, model.block_size, 64, generator)
            scores = model(x.to(device))
            losses.append(F.cross_entropy(scores.flatten(0, 1), y.to(device).flatten()).item())

    model.train()

    return sum(losses) / len(losses)
'''

GPT3_MAIN = '''
    # Run by .venv/bin/python run.py train gpt3, never by the app.
    def main() -> None:
        characters, train_data, validation_data = shakespeare_splits()
        device = best_device()
        torch.manual_seed(0)
        model = GPT(len(characters), BLOCK_SIZE, SIZE, LAYERS, HEADS).to(device)
        steps, warmup = 3000, 150
        optimizer = make_optimizer(model, 1e-3, 0.1)
        generator = torch.Generator().manual_seed(0)
        print(f"Training on {device} for {steps} steps.")

        for step in range(steps):
            x, y = get_batch(train_data, BLOCK_SIZE, 64, generator)
            lr = learning_rate(step, 1e-3, warmup, steps, 1e-4)
            loss = train_step(model, optimizer, x.to(device), y.to(device), lr, 1.0)

            if (step + 1) % 500 == 0:
                checked = validation_loss(model, validation_data, 20, torch.Generator().manual_seed(1), device)
                print(f"Step {step + 1}, training loss {loss:.3f}, validation loss {checked:.3f}")

        save_checkpoint(model, "gpt3")
        print("Saved the model in checkpoints/gpt3.pt. Save gpt3.py in the app to check it.")
'''

INDENTED_DATA = textwrap.indent(DATA_PART.strip("\n"), "    ")
INDENTED_SETTINGS = textwrap.indent(GPT_SETTINGS.strip("\n"), "    ")
INDENTED_TRAIN_PARTS = textwrap.indent(TRAIN_PARTS.strip("\n"), "    ")

GPT3_LESSON = prose(
    """
    This step trains the model outside the app, as dna2 and adder2 did, and
    takes a few minutes. Once every check except the last one passes, train
    from the project folder with `.venv/bin/python run.py train gpt3`.

    It trains the way nanoGPT does, combining the tools from Section 4.

    - AdamW, with weight decay on the matrices and none on the biases and norms, as adamw2 set it up.
    - A learning rate with 150 warmup steps, then a cosine decay to a tenth of its peak, as in warmup1, schedule1 and broken2.
    - Gradients clipped to a norm of 1, as clipping1 did.

    The learning rate changes every step. `optimizer.param_groups` holds the
    learning rate of each parameter group. So your training loop sets
    `group["lr"]` for every group before each step.

    ```python
    optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    optimizer.param_groups[0]["lr"] = 0.05    # the next step uses 0.05
    ```

    Here are the training runs behind this project, on 3 seeds. Seed 0 is the
    run that this file's main makes. The GPT has 4 blocks of size 128, reads
    128 characters at a time, and holds 818,048 weights.

    | Step | 0 | 500 | 1,000 | 2,000 | 3,000 |
    |---|---|---|---|---|---|
    | Validation loss | 4.16 to 4.20 | 1.98 to 2.01 | 1.71 to 1.74 | 1.56 to 1.58 | 1.52 to 1.55 |

    A run took 3.8 to 6.4 minutes on this Mac's GPU, an Apple M5, through
    MPS. The last check asks for a validation loss below 1.7. nanoGPT's own
    Shakespeare model, 6 blocks of size 384, reaches about 1.47 on a GPU.
    """,
)

GPT3_SOURCE = source_file(
    '''
    # gpt3.py
    #
    # Once every check except the last one passes, train from the project folder.
    #
    #     .venv/bin/python run.py train gpt3

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import best_device, save_checkpoint

    halfway = 0.5 * (1 + math.cos(math.pi * 0.5))    # 0.5, the cosine factor halfway through


''' + INDENTED_SETTINGS + '''


    # The model from gpt2.
''' + INDENTED_GIVEN_MODEL + '''


''' + INDENTED_INIT + '''


''' + INDENTED_GPT_INIT + '''

''' + INDENTED_GPT_FORWARD + '''


    # get_batch from gpt1, and the training and validation tokens.
''' + INDENTED_DATA + '''


    # TODO: Return the learning rate for step, counted from 0. During the
    #   first warmup steps it rises in a straight line, reaching max_lr at step
    #   warmup - 1. After that it follows half a cosine curve, from max_lr at
    #   step warmup down to min_lr at step total.
    #   Example: learning_rate(0, 1e-3, 10, 100, 1e-4) returns 0.0001.
    def learning_rate(
        step: int,
        max_lr: float,
        warmup: int,
        total: int,
        min_lr: float,
    ) -> float:
        ...


    # TODO: Return a torch.optim.AdamW optimizer with learning rate lr and
    #   betas (0.9, 0.95). Use two parameter groups. Parameters with two or more
    #   dimensions get weight_decay, and the rest get a weight decay of 0.
    def make_optimizer(
        model: nn.Module,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        ...


    # TODO: Set the learning rate of every parameter group to lr. Compute the
    #   cross entropy between model's scores on x and the targets y. Then zero
    #   the old gradients, backpropagate, clip the gradients to a total norm of
    #   clip, take an optimizer step, and return the loss as a Python float.
    def train_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        x: Tensor,
        y: Tensor,
        lr: float,
        clip: float,
    ) -> float:
        ...


    # TODO: Return the model's average loss on data, averaged over the number
    #   of batches given by batches. Draw each batch with get_batch, as 64
    #   blocks of model.block_size tokens, using generator, and move it to
    #   device. Run in eval mode without letting autograd record, then put
    #   model back in training mode.
    def validation_loss(
        model: nn.Module,
        data: Tensor,
        batches: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> float:
        ...

''' + GPT3_MAIN,
)

GPT3_SOLUTION = source_file(
    '''
    # gpt3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import best_device, save_checkpoint

    halfway = 0.5 * (1 + math.cos(math.pi * 0.5))


''' + INDENTED_SETTINGS + '''


''' + INDENTED_GIVEN_MODEL + '''


''' + INDENTED_INIT + '''


''' + INDENTED_GPT_INIT + '''

''' + INDENTED_GPT_FORWARD + '''


''' + INDENTED_DATA + '''


''' + INDENTED_TRAIN_PARTS + '''

''' + GPT3_MAIN,
)


def train_namespace() -> dict[str, object]:
    namespace = dict(MODEL_REFERENCE)
    namespace["load_shakespeare"] = load_shakespeare
    exec(GPT_SETTINGS + DATA_PART + TRAIN_PARTS, namespace)

    return namespace


TRAIN_REFERENCE = train_namespace()


def check_learning_rate(found: object) -> None:
    for step, max_lr, warmup, total, min_lr in ((0, 1e-3, 10, 100, 1e-4), (9, 1e-3, 10, 100, 1e-4), (55, 1e-3, 10, 100, 1e-4), (100, 1e-3, 10, 100, 1e-4), (30, 0.5, 20, 60, 0.0)):
        expect_close(
            call=f"learning_rate({step}, {max_lr}, {warmup}, {total}, {min_lr})",
            actual=found(step, max_lr, warmup, total, min_lr),
            expected=TRAIN_REFERENCE["learning_rate"](step, max_lr, warmup, total, min_lr),
            tolerance=1e-9,
        )


def check_make_optimizer(found: object) -> None:
    model = seeded_gpt(MODEL_REFERENCE["GPT"])
    optimizer = found(model, 0.003, 0.1)

    if not isinstance(optimizer, torch.optim.AdamW):
        raise Mismatch(f"make_optimizer should return a torch.optim.AdamW, but it returned {type(optimizer).__name__}.")

    decay_of = {id(parameter): group["weight_decay"] for group in optimizer.param_groups for parameter in group["params"]}

    for name, parameter in model.named_parameters():
        wanted = 0.1 if parameter.dim() >= 2 else 0.0

        if decay_of.get(id(parameter)) != wanted:
            raise Mismatch(f"{name} should have a weight decay of {wanted}, but it has {decay_of.get(id(parameter))}.")

    for group in optimizer.param_groups:
        if group["lr"] != 0.003 or tuple(group["betas"]) != (0.9, 0.95):
            raise Mismatch("Every parameter group should have the learning rate lr and betas (0.9, 0.95).")


def check_train_step(found: object) -> None:
    """Plain SGD, since Adam's first step comes out the same size however the gradients were clipped. Its two
    groups catch a rate set on the first group alone, and the two steps catch gradients never zeroed."""
    data = torch.randint(0, 20, (400,), generator=torch.Generator().manual_seed(595))
    batches = [reference_get_batch(data, 16, 4, torch.Generator().manual_seed(seed)) for seed in (596, 597)]
    results = []

    for step in (found, TRAIN_REFERENCE["train_step"]):
        model = seeded_gpt(MODEL_REFERENCE["GPT"])
        matrices = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
        others = [parameter for parameter in model.parameters() if parameter.dim() < 2]
        optimizer = torch.optim.SGD([{"params": matrices}, {"params": others}], lr=1.0)
        losses = [step(model, optimizer, x, y, 0.5, 0.01) for x, y in batches]
        rates = [group["lr"] for group in optimizer.param_groups]
        results.append((losses, model.wte.weight.detach().clone(), model.ln_f.bias.detach().clone(), rates))

    (losses, weight, bias, rates), (expected_losses, expected_weight, expected_bias, _) = results

    if rates != [0.5, 0.5]:
        raise Mismatch(f"After train_step with a learning rate of 0.5, the two groups' learning rates are {rates}. Set group[\"lr\"] for every group.")

    for number, (loss, expected_loss) in enumerate(zip(losses, expected_losses), start=1):
        expect_close(
            call=f"train_step's loss on step {number}, 4 blocks of 16",
            actual=loss,
            expected=expected_loss,
            tolerance=1e-5,
        )

    expect_tensor(
        call="wte's weights after two train_steps with SGD, clipped to 0.01",
        actual=weight,
        expected=expected_weight,
    )
    expect_tensor(
        call="ln_f's bias, in the second group, after two train_steps",
        actual=bias,
        expected=expected_bias,
    )


def check_validation_loss(found: object) -> None:
    data = torch.randint(0, 20, (300,), generator=torch.Generator().manual_seed(597))
    model = seeded_gpt(MODEL_REFERENCE["GPT"])
    expected = TRAIN_REFERENCE["validation_loss"](model, data, 3, torch.Generator().manual_seed(598), torch.device("cpu"))
    # This small GPT has no dropout, so its outputs can't show the mode. A hook records it on every call.
    modes: list[tuple[bool, bool]] = []
    hook = model.register_forward_pre_hook(lambda module, inputs: modes.append((module.training, torch.is_grad_enabled())))

    try:
        actual = found(model, data, 3, torch.Generator().manual_seed(598), torch.device("cpu"))
    finally:
        hook.remove()

    if any(training for training, _ in modes):
        raise Mismatch("validation_loss ran the model in training mode. Call model.eval() first.")

    if any(recording for _, recording in modes):
        raise Mismatch("validation_loss recorded for autograd. Run the batches inside torch.no_grad().")

    if not model.training:
        raise Mismatch("validation_loss left the model in eval mode. Call model.train() before returning.")

    expect_close(
        call="validation_loss over 3 batches",
        actual=actual,
        expected=expected,
        tolerance=1e-5,
    )

    recorder = DeviceRecorder(
        names=("the batch x",),
        output=lambda ids: ids.unsqueeze(-1).expand(-1, -1, 20).float(),
    )
    recorder.block_size = model.block_size
    expect_on_device(
        function="validation_loss",
        run=lambda: found(recorder, data, 3, torch.Generator().manual_seed(598), torch.device("meta")),
        recorders=[recorder],
        advice="Move each batch's x and y to device before the model and F.cross_entropy.",
    )


def trained_gpt(found: type) -> nn.Module:
    model = found(65, 128, 128, 4, 4)

    try:
        load_checkpoint(model, "gpt3")
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None
    except RuntimeError:
        raise Mismatch(
            "The saved model doesn't fit GPT as it is now. Train it again from the project folder.\n\n"
            "    .venv/bin/python run.py train gpt3",
        ) from None

    return model


def check_trained_gpt(found: type) -> None:
    model = trained_gpt(found)
    _, _, validation_data = TRAIN_REFERENCE["shakespeare_splits"]()
    loss = TRAIN_REFERENCE["validation_loss"](model, validation_data, 20, torch.Generator().manual_seed(1), torch.device("cpu"))

    if loss >= 1.7:
        raise Mismatch(f"The trained model's validation loss is {loss:.3f}, but it should be below 1.7.")


def check_gpt3(checker: Checker) -> None:
    checker.test(
        function="learning_rate",
        description="rises in a straight line, then follows a cosine down",
        run=check_learning_rate,
    )
    checker.test(
        function="make_optimizer",
        description="applies weight decay to the matrices and nothing else",
        run=check_make_optimizer,
    )
    checker.test(
        function="train_step",
        description="sets the learning rate, clips the gradients and steps",
        run=check_train_step,
    )
    checker.test(
        function="validation_loss",
        description="averages the loss over batches in eval mode",
        run=check_validation_loss,
    )
    checker.test_method(
        class_name="GPT",
        method="forward",
        description="after training, reaches a validation loss below 1.7",
        run=check_trained_gpt,
    )


GPT3 = Exercise(
    name="gpt3",
    chapter=GPT,
    title="Project: training the GPT",
    lesson=GPT3_LESSON,
    source=GPT3_SOURCE,
    solution=GPT3_SOLUTION,
    hints={
        "learning_rate": "Below warmup, multiply max_lr by (step + 1) / warmup. After that, progress is how far step "
        "is past warmup, as a fraction of total - warmup. The rate is min_lr plus half of max_lr - min_lr, times "
        "1 + math.cos(math.pi * progress).",
        "make_optimizer": "Split model.parameters() by parameter.dim() >= 2 into two groups, each a dict with "
        "\"params\" and \"weight_decay\", and pass the list to torch.optim.AdamW with betas=(0.9, 0.95).",
        "train_step": "Loop over optimizer.param_groups to set each group's \"lr\". For the loss, pass "
        "scores.flatten(0, 1) and y.flatten() to F.cross_entropy. Then call zero_grad, backward, clip_grad_norm_ and "
        "step.",
        "validation_loss": "Call model.eval(). Then inside torch.no_grad(), draw each batch with get_batch(data, "
        "model.block_size, 64, generator), move it to device, and keep each loss.",
    },
    introduces=(),
    check=check_gpt3,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


def measure_gpt3(
    seeds: tuple[int, ...] = (0, 1, 2),
    device: torch.device | None = None,
) -> dict[int, tuple[dict[int, float], float]]:
    """gpt3's curve: main's run from each seed, the validation loss at step 0 and every 500 steps, and the
    seconds each run took. Seed 0 is main itself. Nothing is saved, so the checkpoint stays as it is."""
    device = device or best_device()
    characters, train_data, validation_data = TRAIN_REFERENCE["shakespeare_splits"]()
    results: dict[int, tuple[dict[int, float], float]] = {}

    with pinned_threads():
        for seed in seeds:
            start = time.time()
            torch.manual_seed(seed)
            model = TRAIN_REFERENCE["GPT"](len(characters), 128, 128, 4, 4).to(device)
            steps, warmup = 3000, 150
            optimizer = TRAIN_REFERENCE["make_optimizer"](model, 1e-3, 0.1)
            generator = torch.Generator().manual_seed(seed)
            curve = {0: TRAIN_REFERENCE["validation_loss"](model, validation_data, 20, torch.Generator().manual_seed(1), device)}

            for step in range(steps):
                x, y = TRAIN_REFERENCE["get_batch"](train_data, 128, 64, generator)
                lr = TRAIN_REFERENCE["learning_rate"](step, 1e-3, warmup, steps, 1e-4)
                TRAIN_REFERENCE["train_step"](model, optimizer, x.to(device), y.to(device), lr, 1.0)

                if (step + 1) % 500 == 0:
                    curve[step + 1] = TRAIN_REFERENCE["validation_loss"](
                        model,
                        validation_data,
                        20,
                        torch.Generator().manual_seed(1),
                        device,
                    )

            results[seed] = (curve, time.time() - start)

    return results


# ---------------------------------------------------------------------------
# gpt4
# ---------------------------------------------------------------------------

SAMPLING_PARTS = '''
def top_k_filter(
    scores: Tensor,
    k: int,
) -> Tensor:
    kth = scores.topk(k, dim=-1).values[..., -1:]

    return scores.masked_fill(scores < kth, float("-inf"))


def next_token(
    scores: Tensor,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> Tensor:
    chances = torch.softmax(top_k_filter(scores / temperature, top_k), dim=-1)

    return torch.multinomial(chances, 1, generator=generator)


def generate(
    model: nn.Module,
    ids: Tensor,
    steps: int,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> Tensor:
    model.eval()

    with torch.no_grad():
        for _ in range(steps):
            scores = model(ids[:, -model.block_size :])[:, -1]
            ids = torch.cat([ids, next_token(scores, temperature, top_k, generator)], dim=1)

    return ids
'''

INDENTED_SAMPLING = textwrap.indent(SAMPLING_PARTS.strip("\n"), "    ")

GPT4_LESSON = prose(
    """
    A trained GPT gives a score to every possible next character. To write
    text, pick one character, add it to the end, and run the model again, as
    decoder1 did. The model can read only `block_size` characters at once, so
    each time it reads the last `block_size` characters.

    names3 sampled the next character with `torch.multinomial`. Two settings
    change which characters it picks.

    - The temperature divides the scores before the softmax. Below 1 it concentrates the probability on the likeliest characters, and above 1 it spreads it out.
    - Top k keeps only the k likeliest characters and sets every other score to minus infinity, so an unlikely character can never be picked.

    ```python
    scores = torch.tensor([2.0, 1.0, 0.5, -1.0])
    scores.topk(2).values    # tensor([2., 1.]), the two largest, largest first
    ```

    Here is the trained model from gpt3 writing after `ROMEO:`, with top k of
    20, at three temperatures. Only the first lines of each sample are shown.

    ```text
    0.5   The moon that setter of thee hath a dishonour steed
          And given me and his honour of life,
    1.0   Thrember it wise? for I am rather, in the strange
          of thy hounty here hath got seemed me and him
    1.5   Por mhose hasts:
          Lestop thee hand, iddretcs froMech upon his:
    ```

    At 0.5 the words are real and the lines are predictable. At 1.5 the
    spelling falls apart. At 1.0 with all 65 characters allowed, unlikely
    characters get picked, as in `kiss solemeRoH`. The model ranked that `k`
    26th and that `H` 24th, so top k of 20 would have removed both.

    Here is the same model over 5 seeds, writing 2,000 characters each time.
    The table gives the fraction of written words that appear in the training
    text. It also gives the number of distinct words divided by the number of
    words written, which falls when the text repeats itself.

    | Temperature, top k | Real words | Distinct words per word |
    |---|---|---|
    | 0.5, 20 | 95.0% to 99.3% | 0.41 to 0.46 |
    | 0.8, 20 | 91.9% to 95.2% | 0.52 to 0.56 |
    | 1.0, 20 | 88.7% to 90.8% | 0.57 to 0.60 |
    | 1.0, all 65 | 83.3% to 89.6% | 0.62 to 0.67 |
    | 1.5, 20 | 66.0% to 70.6% | 0.74 to 0.79 |

    A lower temperature writes more real words but repeats itself more. Top k
    removes the unlikely picks, which raises the worst seed at 1.0 from 83.3%
    to 88.7% real words. So choose the temperature for the balance you want,
    and keep a top k.

    With 818,048 weights and 3.8 to 6.4 minutes of training, the model learned
    spelling, line breaks and what a speaker's name looks like. GPT-2's
    largest model is the same kind of stack with about 1,900 times as many
    weights, trained on 40 GB of web text.
    """,
)

GPT4_SOURCE = source_file(
    '''
    # gpt4.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import load_checkpoint

    largest_two = torch.tensor([2.0, 1.0, 0.5, -1.0]).topk(2).values    # tensor([2., 1.])


''' + INDENTED_SETTINGS + '''


    # The model from gpt2.
''' + INDENTED_GIVEN_MODEL + '''


''' + INDENTED_INIT + '''


''' + INDENTED_GPT_INIT + '''

''' + INDENTED_GPT_FORWARD + '''


    # get_batch from gpt1, and the training and validation tokens.
''' + INDENTED_DATA + '''


    # TODO: Return a copy of scores where every entry that is not among the k
    #   largest of its row is set to minus infinity. Then only the k likeliest
    #   tokens can be picked.
    #   Example: top_k_filter(torch.tensor([[2.0, -1.0, 0.5, 3.0]]), 2) returns
    #   tensor([[2., -inf, -inf, 3.]]).
    def top_k_filter(
        scores: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # TODO: Sample one token id for each row of scores, which has shape
    #   (batch, tokens). Divide the scores by temperature, keep only the top_k
    #   largest with top_k_filter, and take the softmax. Then draw one sample
    #   per row with torch.multinomial and generator. The result has shape
    #   (batch, 1).
    def next_token(
        scores: Tensor,
        temperature: float,
        top_k: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return ids with steps more tokens appended after it. ids has shape
    #   (batch, length). At each step, run model on the last model.block_size
    #   tokens, take the scores at the last position, and append the token that
    #   next_token picks. Use eval mode, and do not let autograd record.
    def generate(
        model: nn.Module,
        ids: Tensor,
        steps: int,
        temperature: float,
        top_k: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    characters, _, _ = shakespeare_splits()
    model = load_checkpoint(GPT(len(characters), BLOCK_SIZE, SIZE, LAYERS, HEADS), "gpt3")
    prompt = torch.tensor([[characters.index(character) for character in "ROMEO:"]])

    # TODO: Set sample to the text your model writes when it generates 200
    #   tokens after the prompt, with a temperature of 0.8 and top_k of 20. Use
    #   a generator seeded with 0. Turn the ids into text with characters,
    #   including the prompt.
    sample = ...
    ''',
)

GPT4_SOLUTION = source_file(
    '''
    # gpt4.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import load_checkpoint

    largest_two = torch.tensor([2.0, 1.0, 0.5, -1.0]).topk(2).values


''' + INDENTED_SETTINGS + '''


''' + INDENTED_GIVEN_MODEL + '''


''' + INDENTED_INIT + '''


''' + INDENTED_GPT_INIT + '''

''' + INDENTED_GPT_FORWARD + '''


''' + INDENTED_DATA + '''


''' + INDENTED_SAMPLING + '''


    characters, _, _ = shakespeare_splits()
    model = load_checkpoint(GPT(len(characters), BLOCK_SIZE, SIZE, LAYERS, HEADS), "gpt3")
    prompt = torch.tensor([[characters.index(character) for character in "ROMEO:"]])

    written = generate(model, prompt, 200, 0.8, 20, torch.Generator().manual_seed(0))
    sample = "".join(characters[index] for index in written[0].tolist())
    ''',
)


def sampling_namespace() -> dict[str, object]:
    namespace = dict(TRAIN_REFERENCE)
    exec(SAMPLING_PARTS, namespace)

    return namespace


SAMPLING_REFERENCE = sampling_namespace()


def reference_sample(
    model: nn.Module,
    characters: list[str],
    prompt_text: str,
    steps: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> str:
    prompt = torch.tensor([[characters.index(character) for character in prompt_text]])
    written = SAMPLING_REFERENCE["generate"](model, prompt, steps, temperature, top_k, torch.Generator().manual_seed(seed))

    return "".join(characters[index] for index in written[0].tolist())


def check_generate(found: object) -> None:
    model = seeded_gpt(MODEL_REFERENCE["GPT"])
    ids = torch.randint(0, 20, (2, 20), generator=torch.Generator().manual_seed(599))
    expected = SAMPLING_REFERENCE["generate"](model, ids, 6, 0.9, 5, torch.Generator().manual_seed(600))
    model.train()
    # This small GPT has no dropout, so its outputs can't show the mode. A hook records it on every call.
    modes: list[tuple[bool, bool]] = []
    hook = model.register_forward_pre_hook(lambda module, inputs: modes.append((module.training, torch.is_grad_enabled())))

    try:
        actual = found(model, ids, 6, 0.9, 5, torch.Generator().manual_seed(600))
    finally:
        hook.remove()

    if any(training for training, _ in modes):
        raise Mismatch("generate ran the model in training mode. Call model.eval() first.")

    if any(recording for _, recording in modes):
        raise Mismatch("generate recorded for autograd. Generate the tokens inside torch.no_grad().")

    expect_tensor(
        call="generate of 6 tokens after 2 rows of 20, longer than the block of 16",
        actual=actual,
        expected=expected,
    )


def check_sample(value: object) -> None:
    characters, _, _ = TRAIN_REFERENCE["shakespeare_splits"]()
    model = trained_gpt(MODEL_REFERENCE["GPT"])
    expect_value(
        call="sample",
        actual=value,
        expected=reference_sample(model, characters, "ROMEO:", 200, 0.8, 20, 0),
    )


def check_gpt4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="top_k_filter",
        description="keeps the k largest of each row",
        inputs=(
            (torch.tensor([[2.0, -1.0, 0.5, 3.0], [0.0, 1.0, 4.0, -2.0]]), 2),
            (torch.tensor([[1.0, 5.0, 2.0, 0.0, 3.0]]), 3),
        ),
        reference=SAMPLING_REFERENCE["top_k_filter"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="next_token",
        description="samples from the softmax of the scaled, filtered scores",
        inputs=(
            (torch.tensor([[2.0, -1.0, 0.5, 3.0], [0.0, 1.0, 4.0, -2.0], [1.0, 1.0, 1.0, 1.0]]), 0.7, 3, seeded(601)),
            (torch.tensor([[1.0, 5.0, 2.0, 0.0, 3.0]]), 1.5, 2, seeded(602)),
        ),
        reference=SAMPLING_REFERENCE["next_token"],
        expect=expect_tensor,
    )
    checker.test(
        function="generate",
        description="appends one token at a time, reading the last block",
        run=check_generate,
    )
    checker.check_name(
        name="sample",
        description="is the text your model writes after ROMEO:",
        run=check_sample,
    )


GPT4 = Exercise(
    name="gpt4",
    chapter=GPT,
    title="Project: the GPT writes",
    lesson=GPT4_LESSON,
    source=GPT4_SOURCE,
    solution=GPT4_SOLUTION,
    hints={
        "top_k_filter": "scores.topk(k, dim=-1).values[..., -1:] is the smallest of the k largest entries of each row, "
        "kept as a column. Use masked_fill to set the scores below it to float(\"-inf\").",
        "next_token": "Pass scores / temperature through top_k_filter, then take torch.softmax over dim=-1. Draw from "
        "the result with torch.multinomial, 1 sample, and generator=generator.",
        "generate": "Call model.eval() and work inside torch.no_grad(). Each step, run model on the last "
        "model.block_size ids and keep the scores at the last position. Then attach next_token's pick to ids with "
        "torch.cat on dim=1.",
        "sample": "Call generate with model, prompt, 200 steps, temperature 0.8, top_k 20 and "
        "torch.Generator().manual_seed(0). Then join characters[index] for each index in the first row of the result.",
    },
    introduces=(),
    check=check_gpt4,
    time_limit=120,
)


def measure_samples(seed: int = 0) -> tuple[dict[tuple[float, int], str], list[tuple[str, int]]]:
    """gpt4's samples: 200 characters after ROMEO: from the gpt3 checkpoint, at each temperature and top k of the
    lesson. Also gives each character of the 1.0 sample with all 65 allowed that the model ranked below 20th."""
    characters, _, _ = TRAIN_REFERENCE["shakespeare_splits"]()
    model = trained_gpt(MODEL_REFERENCE["GPT"]).eval()
    samples = {}

    with pinned_threads():
        for temperature, top_k in ((0.5, 20), (1.0, 20), (1.5, 20), (1.0, 65)):
            samples[(temperature, top_k)] = reference_sample(model, characters, "ROMEO:", 200, temperature, top_k, seed)

        ids = torch.tensor([[characters.index(character) for character in samples[(1.0, 65)]]])
        cut = []

        with torch.no_grad():
            for step in range(len("ROMEO:"), ids.shape[1]):
                scores = model(ids[:, max(0, step - model.block_size) : step])[0, -1]
                rank = (scores > scores[ids[0, step]]).sum().item() + 1

                if rank > 20:
                    cut.append((characters[ids[0, step]], rank))

    return samples, cut


def measure_sampling(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    length: int = 2000,
) -> dict[tuple[float, int], list[tuple[float, float]]]:
    """gpt4's table: length characters after ROMEO: from the gpt3 checkpoint, from each seed, at each temperature
    and top k. Gives the share of the written words, runs of letters, that appear in the training text, and the
    distinct words, ignoring case, per word written."""
    characters, _, _ = TRAIN_REFERENCE["shakespeare_splits"]()
    model = trained_gpt(MODEL_REFERENCE["GPT"]).eval()
    text = load_shakespeare()
    known = set(re.findall(r"[A-Za-z]+", text[: int(0.9 * len(text))]))
    prompt = torch.tensor([[characters.index(character) for character in "ROMEO:"]])
    results: dict[tuple[float, int], list[tuple[float, float]]] = {}

    with pinned_threads():
        for temperature, top_k in ((0.5, 20), (0.8, 20), (1.0, 20), (1.0, 65), (1.5, 20)):
            for seed in seeds:
                written = SAMPLING_REFERENCE["generate"](model, prompt, length, temperature, top_k, torch.Generator().manual_seed(seed))
                words = re.findall(r"[A-Za-z]+", "".join(characters[index] for index in written[0, len("ROMEO:") :].tolist()))
                real = sum(word in known for word in words) / len(words)
                distinct = len({word.lower() for word in words}) / len(words)
                results.setdefault((temperature, top_k), []).append((real, distinct))

    return results
