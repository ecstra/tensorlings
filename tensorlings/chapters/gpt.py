"""59_gpt, Section 7's project, a GPT built from your blocks, trained on tiny Shakespeare for minutes on the best device, and sampled."""

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
    seeded,
)
from ..text import prose, source_file
from .transformer_block import SELF_ATTENTION, expect_same_layers

GPT = Chapter(
    folder="59_gpt",
    title="Project: a GPT that writes Shakespeare",
    readme=source_file(
        """
        # Project: a GPT that writes Shakespeare

        Section 7's project. You assemble your transformer blocks into a GPT,
        train it on a million characters of Shakespeare for minutes on the
        fastest device the machine has, and sample new text from it, as in
        Karpathy's GPT lecture and nanoGPT. Download the data once from the
        project folder.

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
    GPT lecture. Download it once, about 1 MB, from the project folder with
    `.venv/bin/python run.py data shakespeare`.

    ```python
    from tensorlings.data import load_shakespeare

    text = load_shakespeare()
    len(text)        # 1115394
    text[:14]        # 'First Citizen:'
    ```

    The tokens here are characters, 65 of them, from the newline to `z`.
    58_bpe's tokens would make sequences shorter, but on a million characters
    a small vocabulary keeps the model small and quick to train. nanoGPT's
    Shakespeare example makes the same choice.

    The first 90% of the text is for training and the last 10% for validation.
    The split is one cut, not a shuffle, because overlapping stretches of text
    would put the same lines on both sides.

    A batch takes random starting points. From each, `x` is a block of
    `block_size` tokens, and `y` is the same block moved one token along. So
    every position is one example, predicting the next character from all the
    ones before it, and a block of 128 holds 128 of them.

    ```python
    data = torch.tensor([5, 6, 7, 8, 9])
    data[1:4], data[2:5]    # (tensor([6, 7, 8]), tensor([7, 8, 9]))
    ```

    A model that knows nothing gives every character the same chance, a loss
    of `log(65)`, which is 4.17. That is the number training starts from.
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


    # TODO: Return every character that appears in text, once each, sorted.
    def characters_of(text: str) -> list[str]:
        ...


    # TODO: Return the id of every character of text, its position in
    #   characters, as a tensor. Build a dict from character to position first,
    #   since characters.index would search the list a million times.
    def encode(
        text: str,
        characters: list[str],
    ) -> Tensor:
        ...


    # TODO: Return the text that ids spell.
    def decode(
        ids: Tensor,
        characters: list[str],
    ) -> str:
        ...


    # TODO: Return the first fraction of data for training and the rest for
    #   validation, cut at int(fraction * len(data)).
    def split(
        data: Tensor,
        fraction: float,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return x and y, each of shape (batch_size, block_size). Draw the
    #   starts with torch.randint(0, len(data) - block_size, (batch_size,),
    #   generator=generator). x holds the block at each start, and y the block
    #   one token further along.
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
        raise Mismatch(f"{call} should give two tensors in a tuple.")

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
        description="gives each character once, sorted",
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
        description="gives each character's position",
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
        description="spells the ids back out",
        inputs=(
            (reference_encode("the question", SAMPLE_CHARACTERS), Described(SAMPLE_CHARACTERS, "the characters of a line of Hamlet")),
        ),
        reference=reference_decode,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="split",
        description="cuts once, training first",
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
        description="gives blocks and the same blocks one token along",
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
        "decode": "characters[index] for each index of ids.tolist(), joined with \"\".join.",
        "split": "Cut at int(fraction * len(data)), and slice data before and after the cut.",
        "get_batch": "Turn the starts into a list with .tolist(). torch.stack the slices data[start : start + "
        "block_size] for x, and the slices one further along for y.",
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
        # Tied, as in GPT-2: the same table reads tokens in and scores them out.
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
    A GPT is block2's block repeated, with two tables in front and a head
    behind. `wte` turns each token into a row of numbers and `wpe` adds a row
    for its position, as position2 did. The blocks follow, then a last
    LayerNorm, and `lm_head` scores every token of the vocabulary as the next
    one. The names are GPT-2's own, so its released weights would load into
    this model.

    The head shares its weight with `wte`, as in GPT-2. A token's row reads it
    in and also scores it on the way out, which saves a whole table.

    GPT-2 starts every weight from a normal spread of 0.02 and every bias at
    0. Then it shrinks the layers that write into the residual stream, the
    `c_proj` of each attention and each MLP. It divides them by the square
    root of how many there are. Here is why, the spread of the residual stream
    after the last block, as a multiple of its spread going in.

    | Blocks | 4 | 12 | 24 | 48 |
    |---|---|---|---|---|
    | All weights at 0.02 | 6.1 | 11.1 | 16.5 | 24.0 |
    | `c_proj` scaled down | 2.2 | 2.3 | 2.3 | 2.3 |

    Every block adds its output to the stream. Without the scaling the stream
    grows like the square root of the depth, the growth residual2 measured. With it, the stream's size stays the same however deep the
    model, much as residual3's fix did.

    `model.apply(fn)` calls `fn` on every module inside `model`, which is how
    one function sets up every layer.

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
    nn.init.normal_(layer.weight, mean=0.0, std=0.5)    # a fresh normal spread of 0.5
    nn.init.zeros_(layer.bias)                          # every bias 0


    # SelfAttention, GPT2MLP and GPT2Block from block2.
''' + INDENTED_GIVEN_MODEL + '''


    # TODO: Start module as GPT-2 does, when it is an nn.Linear or an
    #   nn.Embedding. Its weight is drawn from a normal spread of 0.02 around 0,
    #   and a Linear's bias, when it has one, is set to 0.
    def init_weights(module: nn.Module) -> None:
        ...


    # TODO: Draw the c_proj weight of every block's attention and MLP again,
    #   from a normal spread of 0.02 divided by the square root of 2 * layers.
    #   model.h holds the blocks.
    def scale_residuals_(
        model: nn.Module,
        layers: int,
    ) -> None:
        ...


    # TODO: Finish forward. Add wte of ids to wpe of their positions, from 0 up,
    #   run every block of h in turn, then ln_f and lm_head.
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
            raise Mismatch(f"After init_weights, {name} has a spread of {spread:.4f}, but it should be drawn with a spread of 0.02 around 0.")

    if not torch.all(linear.bias == 0):
        raise Mismatch("After init_weights, a Linear's bias should be all 0.")

    norm = nn.LayerNorm(4)
    found(norm)

    if not (torch.all(norm.weight == 1) and torch.all(norm.bias == 0)):
        raise Mismatch("init_weights changed a LayerNorm, but it should touch only nn.Linear and nn.Embedding.")


def check_scale_residuals(found: object) -> None:
    torch.manual_seed(593)
    model = MODEL_REFERENCE["GPT"](20, 16, 256, 8, 4)
    attention_before = model.h[0].attn.c_attn.weight.clone()
    found(model, 8)
    wanted = 0.02 / math.sqrt(16)

    for index, block in enumerate(model.h):
        for name, weight in (("attn.c_proj", block.attn.c_proj.weight), ("mlp.c_proj", block.mlp.c_proj.weight)):
            spread = weight.std().item()

            if abs(spread - wanted) > 0.1 * wanted:
                raise Mismatch(
                    f"After scale_residuals_ on 8 layers, block {index}'s {name} has a spread of {spread:.5f}, "
                    f"but it should be 0.02 over the square root of 16, {wanted:.5f}.",
                )

    if not torch.equal(model.h[0].attn.c_attn.weight, attention_before):
        raise Mismatch("scale_residuals_ changed c_attn, but only the two c_proj layers of each block are scaled.")


def check_gpt_forward(found: type) -> None:
    model = seeded_gpt(found)
    reference = seeded_gpt(MODEL_REFERENCE["GPT"])
    expect_same_layers(
        call="GPT(20, 16, 32, 2, 4)",
        model=model,
        reference=reference,
    )

    if model.lm_head.weight is not model.wte.weight:
        raise Mismatch("lm_head's weight should be wte's weight itself, tied, as the given __init__ sets it.")

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
        description="draws Linear and Embedding weights with a spread of 0.02, biases at 0",
        run=check_init_weights,
    )
    checker.test(
        function="scale_residuals_",
        description="shrinks every c_proj by the square root of twice the layers",
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
        "init_weights": "isinstance(module, (nn.Linear, nn.Embedding)) picks both kinds. nn.init.normal_ with std=0.02 "
        "for the weight, and nn.init.zeros_ for a Linear's bias when module.bias is not None.",
        "scale_residuals_": "For each block of model.h, nn.init.normal_ block.attn.c_proj.weight and "
        "block.mlp.c_proj.weight with std=0.02 / math.sqrt(2 * layers).",
        "GPT.forward": "torch.arange(ids.shape[1]) gives the positions. Add self.wte(ids) and self.wpe of them, loop "
        "over self.h, then self.ln_f and self.lm_head.",
    },
    introduces=(),
    check=check_gpt2,
    time_limit=30,
)
