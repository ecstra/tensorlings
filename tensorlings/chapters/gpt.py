"""59_gpt, Section 7's project, a GPT built from your blocks, trained on tiny Shakespeare for minutes on the best device, and sampled."""

from __future__ import annotations

import math
import textwrap

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
from ..training import MissingCheckpoint, load_checkpoint
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

    nanoGPT starts every weight from a normal spread of 0.02 and every bias at
    0. GPT-2's released code does the same, except that it draws `wpe` at
    0.01. Then, as the GPT-2 paper describes, nanoGPT shrinks the layers that
    write into the residual stream, the `c_proj` of each attention and each
    MLP. It divides them by the square root of how many there are. Here is
    why, the spread of the residual stream after the last block, as a
    multiple of its spread going in.

    | Blocks | 4 | 12 | 24 | 48 |
    |---|---|---|---|---|
    | All weights at 0.02 | 6.1 | 11.1 | 16.5 | 24.0 |
    | `c_proj` scaled down | 2.2 | 2.3 | 2.3 | 2.3 |

    Every block adds its output to the stream. Without the scaling the stream
    grows like the square root of the depth, the growth residual2 measured.
    With it, the stream's size stays the same however deep the model, much as
    residual3's fix did.

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


    # TODO: Start module as nanoGPT does, when it is an nn.Linear or an
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
    This step trains for minutes outside the app, as dna2 and adder2 did.
    Once the checks above the last one pass, train from the project folder
    with `.venv/bin/python run.py train gpt3`.

    It trains the way nanoGPT does, with Section 4's tools together.

    - AdamW, with weight decay on the matrices and none on the biases and norms, as broken2 set it up.
    - A rate that warms up for 150 steps, then follows a cosine down to a tenth of its peak, as in warmup1 and schedule2.
    - Gradients clipped to a norm of 1, as clipping1 did.

    The rate changes every step. `optimizer.param_groups` holds the rate of
    each group, so setting `group["lr"]` before the step is how a loop of your
    own drives it.

    ```python
    optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    optimizer.param_groups[0]["lr"] = 0.05    # the next step uses 0.05
    ```

    Here is the run behind this project on MPS. The GPT has 4 blocks of size
    128, reads 128 characters at a time, and holds 818,048 weights.

    | Step | 0 | 500 | 1,000 | 2,000 | 3,000 |
    |---|---|---|---|---|---|
    | Validation loss | 4.19 | 2.02 | 1.73 | 1.57 | 1.54 |

    It took about 5 minutes on this Mac's GPU. The check asks for a
    validation loss below 1.7. nanoGPT's own Shakespeare
    model, 6 blocks of size 384, reaches about 1.47 on a GPU.
    """,
)

GPT3_SOURCE = source_file(
    '''
    # gpt3.py
    #
    # Train once the checks above the last one pass, from the project folder.
    #
    #     .venv/bin/python run.py train gpt3

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import best_device, save_checkpoint

    halfway = 0.5 * (1 + math.cos(math.pi * 0.5))    # 0.5, a cosine halfway down


''' + INDENTED_SETTINGS + '''


    # The model from gpt2.
''' + INDENTED_GIVEN_MODEL + '''


''' + INDENTED_INIT + '''


''' + INDENTED_GPT_INIT + '''

''' + INDENTED_GPT_FORWARD + '''


    # get_batch from gpt1, and the training and validation tokens.
''' + INDENTED_DATA + '''


    # TODO: Return the rate for step, counted from 0. For the first warmup steps
    #   it climbs in a line to max_lr, reaching it at step warmup - 1. After
    #   that it follows half a cosine from max_lr down to min_lr at step total.
    def learning_rate(
        step: int,
        max_lr: float,
        warmup: int,
        total: int,
        min_lr: float,
    ) -> float:
        ...


    # TODO: Return AdamW at lr with betas (0.9, 0.95), with weight_decay on the
    #   parameters of two or more dimensions and none on the rest.
    def make_optimizer(
        model: nn.Module,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        ...


    # TODO: Set every group's rate to lr and take the cross entropy of model's
    #   scores on x against y. Zero the old gradients, backpropagate, clip the
    #   gradients to a norm of clip, step, and return the loss as a number.
    def train_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        x: Tensor,
        y: Tensor,
        lr: float,
        clip: float,
    ) -> float:
        ...


    # TODO: Return the average loss over batches batches of 64 blocks from data,
    #   each model.block_size tokens long, drawn with get_batch and generator
    #   and moved to device. Run in eval mode without recording for autograd,
    #   then put model back in training mode.
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
        raise Mismatch(f"make_optimizer should give torch.optim.AdamW, but it gave {type(optimizer).__name__}.")

    decay_of = {id(parameter): group["weight_decay"] for group in optimizer.param_groups for parameter in group["params"]}

    for name, parameter in model.named_parameters():
        wanted = 0.1 if parameter.dim() >= 2 else 0.0

        if decay_of.get(id(parameter)) != wanted:
            raise Mismatch(f"{name} should have a weight decay of {wanted}, but it has {decay_of.get(id(parameter))}.")

    for group in optimizer.param_groups:
        if group["lr"] != 0.003 or tuple(group["betas"]) != (0.9, 0.95):
            raise Mismatch("Every group should have the rate lr and betas of (0.9, 0.95).")


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
        raise Mismatch(f"After train_step at a rate of 0.5, the two groups' rates are {rates}. Set group[\"lr\"] for every group.")

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
        description="warms up in a line, then follows a cosine down",
        run=check_learning_rate,
    )
    checker.test(
        function="make_optimizer",
        description="decays the matrices and nothing else",
        run=check_make_optimizer,
    )
    checker.test(
        function="train_step",
        description="sets the rate, clips and steps",
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
        description="trained, gets below a validation loss of 1.7",
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
        "learning_rate": "Below warmup, max_lr * (step + 1) / warmup. After it, progress = (step - warmup) / (total - "
        "warmup), and min_lr plus half of max_lr - min_lr times 1 + math.cos(math.pi * progress).",
        "make_optimizer": "Split model.parameters() by parameter.dim() >= 2 into two groups, each a dict with "
        "\"params\" and \"weight_decay\", and pass the list to torch.optim.AdamW with betas=(0.9, 0.95).",
        "train_step": "Loop over optimizer.param_groups to set \"lr\". scores.flatten(0, 1) against y.flatten() for "
        "the loss, then zero_grad, backward, clip_grad_norm_ and step.",
        "validation_loss": "model.eval(), then inside torch.no_grad() draw each batch with get_batch(data, "
        "model.block_size, 64, generator), move it to device, and keep each loss.",
    },
    introduces=(),
    check=check_gpt3,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


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
    A trained GPT gives scores for the next character. Writing text means
    picking one, adding it to the end, and asking again, as decoder1 did. The
    block only holds `block_size` characters, so the model reads the last
    ones each time.

    names3 sampled with `torch.multinomial`. Two knobs change what it picks.

    - The temperature divides the scores before the softmax. Below 1 it sharpens the chances toward the likeliest characters, and above 1 it flattens them.
    - Top k keeps only the k likeliest characters and sets every other score to minus infinity, so a rare mistake can never be picked.

    ```python
    scores = torch.tensor([2.0, 1.0, 0.5, -1.0])
    scores.topk(2).values    # tensor([2., 1.]), the two largest, largest first
    ```

    Here is the trained model from gpt3, writing after `ROMEO:` with top k
    of 20, the first lines of each.

    ```text
    0.5   The moon that setter of thee hath a dishonour steed
          And given me and his honour of life,
    1.0   Thrember it wise? for I am rather, in the strange
          of thy hounty here hath got seemed me and him
    1.5   Por mhose hasts:
          Lestop thee hand, iddretcs froMech upon his:
    ```

    At 0.5 the words are real and the lines play safe. At 1.5 the spelling
    falls apart. At 1.0 with every character allowed, rare picks slip through,
    like `kiss solemeRoH`, which top k would have cut.

    In 5 minutes and 818,048 weights the model learned spelling, line breaks
    and the look of a speaker's name. GPT-2's largest model is the same kind of stack with about 1,800 times as
    many weights, trained on 40 GB of web text.
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


    # TODO: Return scores with every entry below the k largest of its row set to
    #   minus infinity, so only the k likeliest can be picked.
    def top_k_filter(
        scores: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # TODO: Return one token per row of scores, of shape (batch, 1). Divide the
    #   scores by temperature, keep the top_k, take the softmax, and draw with
    #   torch.multinomial and generator.
    def next_token(
        scores: Tensor,
        temperature: float,
        top_k: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return ids with steps more tokens written after it. Each step, run
    #   model on the last model.block_size tokens, take the scores at the last
    #   position, and add next_token's pick. Use eval mode and record nothing
    #   for autograd.
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

    # TODO: Set sample to the text of 200 tokens your model writes after the
    #   prompt, at a temperature of 0.8 with top_k 20, drawn with a generator
    #   seeded with 0. Spell the ids out with characters, prompt included.
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
        raise Mismatch("generate recorded for autograd. Write the tokens inside torch.no_grad().")

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
        description="draws from the sharpened, filtered chances",
        inputs=(
            (torch.tensor([[2.0, -1.0, 0.5, 3.0], [0.0, 1.0, 4.0, -2.0], [1.0, 1.0, 1.0, 1.0]]), 0.7, 3, seeded(601)),
            (torch.tensor([[1.0, 5.0, 2.0, 0.0, 3.0]]), 1.5, 2, seeded(602)),
        ),
        reference=SAMPLING_REFERENCE["next_token"],
        expect=expect_tensor,
    )
    checker.test(
        function="generate",
        description="writes token after token from the last block",
        run=check_generate,
    )
    checker.check_name(
        name="sample",
        description="is your model writing after ROMEO:",
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
        "top_k_filter": "scores.topk(k, dim=-1).values[..., -1:] is the k-th largest of each row, kept as a column. "
        "masked_fill the scores below it with float(\"-inf\").",
        "next_token": "torch.softmax of top_k_filter(scores / temperature, top_k) over dim=-1, then "
        "torch.multinomial of it with 1 draw and generator=generator.",
        "generate": "model.eval() and torch.no_grad(). Each step, scores = model(ids[:, -model.block_size :])[:, -1], "
        "then torch.cat next_token's pick onto ids on dim=1.",
        "sample": "generate(model, prompt, 200, 0.8, 20, torch.Generator().manual_seed(0)), then join "
        "characters[index] for each index of its first row.",
    },
    introduces=(),
    check=check_gpt4,
    time_limit=120,
)
