"""89_quantization, weights stored as 8 bit whole numbers with absmax scales, one per tensor or one per row, and your GPT in int8."""

from __future__ import annotations

import functools
import textwrap
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..data import load_shakespeare
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
    reject_missing_return,
    value_text,
)
from ..text import prose, source_file
from .gpt import (
    DATA_PART,
    GPT_SETTINGS,
    INDENTED_DATA,
    INDENTED_GIVEN_MODEL,
    INDENTED_GPT_FORWARD,
    INDENTED_GPT_INIT,
    INDENTED_INIT,
    INDENTED_SETTINGS,
    MODEL_REFERENCE,
    TRAIN_PARTS,
    seeded_gpt,
    trained_gpt,
)

QUANTIZATION = Chapter(
    folder="89_quantization",
    title="Quantization",
    readme=source_file(
        """
        # Quantization

        A trained model's weights don't need 32 bits each. This chapter
        stores them as 8 bit whole numbers with a scale, measures what that
        costs, gives each row its own scale so one large weight can't spoil
        the rest, and runs your GPT from gpt3 in int8.

        ## Further reading

        - [Dettmers and colleagues, LLM.int8()](https://arxiv.org/abs/2208.07339)
        - [Jacob and colleagues, quantization for integer only inference](https://arxiv.org/abs/1712.05877)
        - [PyTorch, register_buffer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)
        """,
    ),
)

ABSMAX_PART = '''
def absmax_quantize(
    weights: Tensor,
    bits: int,
) -> tuple[Tensor, Tensor]:
    levels = 2 ** (bits - 1) - 1
    scale = weights.abs().max() / levels
    quantized = (weights / scale).round().to(torch.int8)

    return quantized, scale
'''

DEQUANTIZE_PART = '''
def dequantize(
    quantized: Tensor,
    scale: Tensor,
) -> Tensor:
    return quantized.float() * scale
'''

ERROR_PART = '''
def largest_error(
    weights: Tensor,
    bits: int,
) -> float:
    quantized, scale = absmax_quantize(weights, bits)

    return (dequantize(quantized, scale) - weights).abs().max().item()
'''

PER_CHANNEL_PART = '''
def per_channel_quantize(
    weights: Tensor,
    bits: int,
) -> tuple[Tensor, Tensor]:
    levels = 2 ** (bits - 1) - 1
    scales = weights.abs().amax(dim=1, keepdim=True) / levels
    quantized = (weights / scales).round().to(torch.int8)

    return quantized, scales
'''

MEASURES_PART = '''
def relative_error(
    weights: Tensor,
    approximation: Tensor,
) -> float:
    return ((weights - approximation).norm() / weights.norm()).item()


def zero_share(quantized: Tensor) -> float:
    return (quantized == 0).float().mean().item()
'''

INT8_LINEAR_PART = '''
class Int8Linear(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.detach()
        quantized, scales = per_channel_quantize(weight, 8)
        self.register_buffer("quantized", quantized)
        self.register_buffer("scales", scales)
        self.bias = linear.bias

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, dequantize(self.quantized, self.scales), self.bias)


def quantize_linears_(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8Linear(child))
            continue

        quantize_linears_(child)


def model_bytes(model: nn.Module) -> int:
    tensors = [*model.parameters(), *model.buffers()]

    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
'''

# The int8 Linear with what it needs, for a later chapter to hand its learner.
INT8_PART = PER_CHANNEL_PART + DEQUANTIZE_PART + INT8_LINEAR_PART

# gpt3's validation_loss alone, without its training loop.
VALIDATION_PART = TRAIN_PARTS[TRAIN_PARTS.index("def validation_loss") :]


def indented(part: str) -> str:
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(code, namespace)

    return namespace


QUANT_REFERENCE = namespace_of(ABSMAX_PART + DEQUANTIZE_PART + ERROR_PART + PER_CHANNEL_PART + MEASURES_PART + INT8_LINEAR_PART)


def model_reference() -> dict[str, object]:
    namespace = dict(MODEL_REFERENCE)
    namespace["load_shakespeare"] = load_shakespeare
    exec(GPT_SETTINGS + DATA_PART + VALIDATION_PART, namespace)

    return namespace


GPT_REFERENCE = model_reference()


def seeded_weights(
    seed: int,
    rows: int,
    columns: int,
) -> Tensor:
    """Random weights whose rows differ in size, as a trained Linear's do."""
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.rand(rows, 1, generator=generator) * 2 + 0.1

    return torch.randn(rows, columns, generator=generator) * sizes


def expect_quantized(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise Mismatch(f"{call} should give two things, the whole numbers and the scale, but it gives {value_text(actual)}.")

    quantized, scale = actual

    if isinstance(quantized, Tensor) and quantized.dtype != torch.int8:
        raise Mismatch(
            f"The whole numbers from {call} are {quantized.dtype}, but they should be torch.int8, one byte each. "
            "Turn them into it with .to(torch.int8) after rounding.",
        )

    # The scale first, since a scale of the wrong shape explains every whole number that is off.
    expect_tensor(
        call=f"the scale from {call}",
        actual=scale,
        expected=expected[1],
    )
    expect_tensor(
        call=f"the whole numbers from {call}",
        actual=quantized,
        expected=expected[0],
    )


# ---------------------------------------------------------------------------
# quant1
# ---------------------------------------------------------------------------

QUANT1_LESSON = prose(
    """
    A float32 number takes 4 bytes. gpt3's model holds 818,048 of them, about
    3.3 MB. **Quantization** stores each weight in fewer bits. Here that is a
    whole number from -127 to 127 in one byte, which PyTorch calls
    `torch.int8`.

    Absmax quantization uses one scale for the whole tensor, its largest size
    over 127. Each weight over the scale, rounded, is its whole number, and
    the largest weight lands on 127 or -127. Times the scale, a whole number
    gives its weight back, off by at most half a scale.

    ```python
    weights = torch.tensor([0.5, -1.27, 0.031])
    scale = weights.abs().max() / 127    # tensor(0.0100)
    (weights / scale).round()            # tensor([  50., -127.,    3.])
    ```

    The last weight comes back as 0.03, off by 0.001. Fewer bits give fewer
    levels, 2 ** (bits - 1) - 1 on each side of 0. 4 bits give 7, and 2 bits
    only 1.

    Here is block 0's `c_fc` weight from the run behind gpt3, 65,536 numbers
    whose largest size is 0.121. The error column is the norm of all the
    errors over the norm of the weights, the size clipping1 measured.

    | Bits | Levels each side | Largest error | Error | Rounded to 0 |
    |---|---|---|---|---|
    | 8 | 127 | 0.00048 | 1.2% | 1.8% |
    | 4 | 7 | 0.0087 | 22% | 31% |
    | 2 | 1 | 0.061 | 98% | 99% |

    At 8 bits the largest error is half the scale, 0.121 / 127 / 2. At 2 bits
    nearly every weight rounds to 0, since most sit far below the largest.

    int8 holds -128 too. Absmax leaves it out, so the levels sit evenly around
    0 and 0 stays exactly 0.
    """,
)

QUANT1_SOURCE = source_file(
    '''
    # quant1.py

    import torch
    from torch import Tensor

    scale = torch.tensor([0.5, -1.27, 0.031]).abs().max() / 127    # tensor(0.0100)
    whole = torch.tensor([1.6, -2.4]).round().to(torch.int8)       # tensor([ 2, -2], dtype=torch.int8)


    # TODO: Return weights as whole numbers stored as torch.int8, and the scale,
    #   a tensor of one number. The scale is the largest size in weights over
    #   levels. Each weight divided by the scale and rounded is its whole number.
    def absmax_quantize(
        weights: Tensor,
        bits: int,
    ) -> tuple[Tensor, Tensor]:
        levels = 2 ** (bits - 1) - 1
        ...


    # TODO: Return the floats the whole numbers stand for, each one times scale.
    def dequantize(
        quantized: Tensor,
        scale: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return, as a plain number, the largest size of the difference between
    #   weights and their round trip through absmax_quantize and dequantize.
    def largest_error(
        weights: Tensor,
        bits: int,
    ) -> float:
        ...
    ''',
)

QUANT1_SOLUTION = source_file(
    '''
    # quant1.py, a solution for comparison

    import torch
    from torch import Tensor

    scale = torch.tensor([0.5, -1.27, 0.031]).abs().max() / 127
    whole = torch.tensor([1.6, -2.4]).round().to(torch.int8)


''' + indented(ABSMAX_PART) + '''


''' + indented(DEQUANTIZE_PART) + '''


''' + indented(ERROR_PART) + '''
    ''',
)


def check_quant1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="absmax_quantize",
        description="rounds each weight to a whole number of steps of one scale",
        inputs=(
            (seeded_weights(890, 2, 4), 8),
            (seeded_weights(891, 3, 3), 4),
            (-seeded_weights(892, 1, 5).abs(), 8),
        ),
        reference=QUANT_REFERENCE["absmax_quantize"],
        expect=expect_quantized,
    )
    check_on_inputs(
        checker=checker,
        function="dequantize",
        description="gives back the floats the whole numbers stand for",
        inputs=(
            (torch.tensor([50, -127, 3], dtype=torch.int8), torch.tensor(0.01)),
            (torch.tensor([[7, -2], [0, 5]], dtype=torch.int8), torch.tensor(0.3)),
        ),
        reference=QUANT_REFERENCE["dequantize"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="largest_error",
        description="gives the largest error of the round trip",
        inputs=(
            (seeded_weights(893, 4, 6), 8),
            (seeded_weights(894, 3, 5), 4),
        ),
        reference=QUANT_REFERENCE["largest_error"],
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-7,
        ),
    )


QUANT1 = Exercise(
    name="quant1",
    chapter=QUANTIZATION,
    title="Absmax int8",
    lesson=QUANT1_LESSON,
    source=QUANT1_SOURCE,
    solution=QUANT1_SOLUTION,
    hints={
        "absmax_quantize": "The scale is weights.abs().max() / levels. Divide weights by it, .round(), then "
        ".to(torch.int8), and return both.",
        "dequantize": "Turn quantized into floats with .float(), then multiply by scale.",
        "largest_error": "Quantize, dequantize, subtract weights, then take .abs().max().item().",
    },
    introduces=(),
    check=check_quant1,
)


# ---------------------------------------------------------------------------
# quant2
# ---------------------------------------------------------------------------

QUANT2_LESSON = prose(
    """
    One scale per tensor lets its largest weight set the step for all the
    rest. A Linear's weight has a row per output, and in gpt3's blocks a row's
    largest size runs from 0.032 to 0.27. Giving each row its own scale, **per
    channel**, keeps a small row's step small.

    ```python
    rows = torch.tensor([[0.1, -0.2], [10.0, 5.0]])
    rows.abs().amax(dim=1, keepdim=True)    # tensor([[ 0.2000], [10.0000]])
    ```

    `keepdim=True` keeps the scales a column of shape (2, 1), which divides
    each row by its own. Here is the average over the 16 Linear weights of
    gpt3's blocks.

    | Scales | Bits | Error | Rounded to 0 |
    |---|---|---|---|
    | One per tensor | 8 | 1.25% | 1.9% |
    | One per row | 8 | 0.71% | 1.1% |
    | One per tensor | 4 | 22.7% | 33.7% |
    | One per row | 4 | 12.9% | 18.7% |

    An **outlier** is one number far larger than the rest. Below, one weight
    of block 0's `c_fc` was set to 20 times its largest. The error is on the
    other 65,535 weights.

    | Weights | Scales | Error | Rounded to 0 |
    |---|---|---|---|
    | No outlier | One per tensor | 1.21% | 1.8% |
    | Outlier | One per tensor | 24.3% | 33.7% |
    | Outlier | One per row | 1.29% | 1.0% |

    One scale lets the outlier spoil every weight. With one per row only its
    own row suffers, and the other rows stay at 0.65%.

    Dettmers and colleagues found outliers in large language models, in the
    hidden states that go into each Linear. From around 6.7B weights, a few
    dimensions, at most 7, grow up to 20 times larger than the rest. Plain
    int8 then rounds many small values to 0, and the models get worse. Their
    LLM.int8() keeps those dimensions in 16 bits and multiplies more than 99.9%
    of the values in 8 bits.
    """,
)

QUANT2_SOURCE = source_file(
    '''
    # quant2.py

    import torch
    from torch import Tensor

    rows = torch.tensor([[0.1, -0.2], [10.0, 5.0]])
    largest_per_row = rows.abs().amax(dim=1, keepdim=True)    # tensor([[ 0.2000], [10.0000]])
    length = torch.tensor([3.0, 4.0]).norm()                  # tensor(5.)


    # absmax_quantize and dequantize from quant1.
''' + indented(ABSMAX_PART) + '''


''' + indented(DEQUANTIZE_PART) + '''


    # TODO: Return weights as torch.int8 whole numbers with one scale per row,
    #   and the scales as a column of shape (rows, 1). A row's scale is its
    #   largest size over levels, so each row is divided by its own.
    def per_channel_quantize(
        weights: Tensor,
        bits: int,
    ) -> tuple[Tensor, Tensor]:
        levels = 2 ** (bits - 1) - 1
        ...


    # TODO: Return, as a plain number, the norm of weights - approximation over
    #   the norm of weights.
    def relative_error(
        weights: Tensor,
        approximation: Tensor,
    ) -> float:
        ...


    # TODO: Return, as a plain number, the share of the whole numbers that are 0.
    def zero_share(quantized: Tensor) -> float:
        ...
    ''',
)

QUANT2_SOLUTION = source_file(
    '''
    # quant2.py, a solution for comparison

    import torch
    from torch import Tensor

    rows = torch.tensor([[0.1, -0.2], [10.0, 5.0]])
    largest_per_row = rows.abs().amax(dim=1, keepdim=True)
    length = torch.tensor([3.0, 4.0]).norm()


''' + indented(ABSMAX_PART) + '''


''' + indented(DEQUANTIZE_PART) + '''


''' + indented(PER_CHANNEL_PART) + '''


''' + indented(MEASURES_PART) + '''
    ''',
)


def close_number(
    call: str,
    actual: object,
    expected: float,
) -> None:
    expect_close(
        call=call,
        actual=actual,
        expected=expected,
        tolerance=1e-6,
    )


def check_quant2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="per_channel_quantize",
        description="gives each row its own scale",
        inputs=(
            (seeded_weights(895, 3, 5), 8),
            (seeded_weights(896, 4, 2), 4),
        ),
        reference=QUANT_REFERENCE["per_channel_quantize"],
        expect=expect_quantized,
    )
    check_on_inputs(
        checker=checker,
        function="relative_error",
        description="gives the error's norm over the weights' norm",
        inputs=(
            (torch.tensor([[3.0, 0.0], [0.0, 4.0]]), torch.tensor([[3.0, 1.0], [0.0, 4.0]])),
            (seeded_weights(897, 3, 4), seeded_weights(898, 3, 4)),
        ),
        reference=QUANT_REFERENCE["relative_error"],
        expect=close_number,
    )
    check_on_inputs(
        checker=checker,
        function="zero_share",
        description="gives the share of whole numbers rounded to 0",
        inputs=(
            (torch.tensor([0, 3, 0, -1, 0], dtype=torch.int8),),
            (torch.tensor([[0, 0, 5], [1, 7, -2]], dtype=torch.int8),),
        ),
        reference=QUANT_REFERENCE["zero_share"],
        expect=close_number,
    )


QUANT2 = Exercise(
    name="quant2",
    chapter=QUANTIZATION,
    title="A scale per row",
    lesson=QUANT2_LESSON,
    source=QUANT2_SOURCE,
    solution=QUANT2_SOLUTION,
    hints={
        "per_channel_quantize": "weights.abs().amax(dim=1, keepdim=True) / levels gives the column of scales. "
        "Divide, round and turn into torch.int8 as absmax_quantize does.",
        "relative_error": "(weights - approximation).norm() divided by weights.norm(), then .item().",
        "zero_share": "Compare quantized with 0, turn the result into floats, and take .mean().item().",
    },
    introduces=(),
    check=check_quant2,
)


# ---------------------------------------------------------------------------
# quant3
# ---------------------------------------------------------------------------

QUANT3_LESSON = prose(
    """
    Now the whole model. `Int8Linear` takes a trained `nn.Linear` and keeps its
    weight as int8 whole numbers with a scale per row. Both are **buffers**,
    tensors a module keeps that training never changes. `.to(device)` and
    `state_dict()` take buffers along, as they take the parameters.

    ```python
    counter = nn.Module()
    counter.register_buffer("counts", torch.zeros(2))
    dict(counter.named_buffers())    # {'counts': tensor([0., 0.])}
    ```

    Its forward turns the whole numbers back into a float weight and runs
    `F.linear`, the function behind every `nn.Linear`. That saves memory, not
    time, since the multiply still runs in float. Fast int8 libraries multiply
    the whole numbers themselves, with kernels this chapter leaves out.

    `module.named_children()` gives a module's direct children with their
    names, and `setattr(module, name, new)` swaps one. A function that calls
    itself on every other child reaches every depth.

    Only the blocks get quantized. `lm_head` shares its weight with `wte`, and
    that table stays in float32 for both. `model.parameters()` lists a shared
    tensor once.

    Here is the GPT from the run behind gpt3, on the 20 validation batches its
    check used.

    | Blocks' weights | Bytes | Validation loss |
    |---|---|---|
    | float32 | 3,272,192 | 1.5351 |
    | int8, a scale per row | 931,328 | 1.5353 |
    | int8, one scale per tensor | 912,960 | 1.5356 |
    | 4 bits, a scale per row | 931,328 | 1.5833 |
    | 2 bits, a scale per row | 931,328 | 4.0665 |

    int8 made the model 3.5 times smaller and moved its loss in the fourth
    decimal. 4 bits still works, stored one to a byte here, and 2 bits breaks
    the model. Two 4 bit numbers packed in each byte would bring it near 0.54
    MB.
    """,
)

QUANT3_MEASURES = '''
    characters, _, validation_data = shakespeare_splits()
    model = load_checkpoint(GPT(len(characters), BLOCK_SIZE, SIZE, LAYERS, HEADS), "gpt3")
    cpu = torch.device("cpu")
    float_bytes = model_bytes(model)
    float_loss = validation_loss(model, validation_data, 20, torch.Generator().manual_seed(1), cpu)

    for block in model.h:
        quantize_linears_(block)
'''

QUANT3_SOURCE = source_file(
    '''
    # quant3.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import load_checkpoint

    float_size = torch.zeros(3).element_size()                       # 4 bytes each
    int8_size = torch.zeros(3, dtype=torch.int8).element_size()      # 1 byte each


''' + INDENTED_SETTINGS + '''


    # The model from gpt2.
''' + INDENTED_GIVEN_MODEL + '''


''' + INDENTED_INIT + '''


''' + INDENTED_GPT_INIT + '''

''' + INDENTED_GPT_FORWARD + '''


    # get_batch from gpt1, the training and validation tokens, and validation_loss from gpt3.
''' + INDENTED_DATA + '''


''' + indented(VALIDATION_PART) + '''


    # per_channel_quantize from quant2 and dequantize from quant1.
''' + indented(PER_CHANNEL_PART) + '''


''' + indented(DEQUANTIZE_PART) + '''


    # TODO: Finish Int8Linear. __init__ quantizes weight per channel to 8 bits
    #   and keeps the whole numbers and the scales as buffers named quantized
    #   and scales. forward runs F.linear on x with the dequantized weight and
    #   the bias.
    class Int8Linear(nn.Module):
        def __init__(self, linear: nn.Linear) -> None:
            super().__init__()
            weight = linear.weight.detach()
            ...
            self.bias = linear.bias

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Replace every nn.Linear inside module, at any depth, with an
    #   Int8Linear made from it. Leave every other layer as it is.
    def quantize_linears_(module: nn.Module) -> None:
        ...


    # TODO: Return how many bytes model's parameters and buffers take, as a
    #   plain number. A tensor shared by two layers counts once.
    def model_bytes(model: nn.Module) -> int:
        ...

''' + QUANT3_MEASURES + '''
    # TODO: Set int8_bytes and int8_loss, the size and the validation loss of the
    #   quantized model, measured as float_bytes and float_loss were.
    int8_bytes = ...
    int8_loss = ...
    ''',
)

QUANT3_SOLUTION = source_file(
    '''
    # quant3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import load_checkpoint

    float_size = torch.zeros(3).element_size()
    int8_size = torch.zeros(3, dtype=torch.int8).element_size()


''' + INDENTED_SETTINGS + '''


''' + INDENTED_GIVEN_MODEL + '''


''' + INDENTED_INIT + '''


''' + INDENTED_GPT_INIT + '''

''' + INDENTED_GPT_FORWARD + '''


''' + INDENTED_DATA + '''


''' + indented(VALIDATION_PART) + '''


''' + indented(PER_CHANNEL_PART) + '''


''' + indented(DEQUANTIZE_PART) + '''


''' + indented(INT8_LINEAR_PART) + '''

''' + QUANT3_MEASURES + '''
    int8_bytes = model_bytes(model)
    int8_loss = validation_loss(model, validation_data, 20, torch.Generator().manual_seed(1), cpu)
    ''',
)


def quantized_buffers(
    module: nn.Module,
    call: str,
) -> tuple[Tensor, Tensor]:
    buffers = dict(module.named_buffers())

    for name in ("quantized", "scales"):
        if name in buffers:
            continue

        if isinstance(getattr(module, name, None), Tensor):
            raise Mismatch(
                f"{call} keeps {name} as a plain attribute, not a buffer, so .to(device) and state_dict() would "
                f"leave it out. Keep it with self.register_buffer(\"{name}\", ...).",
            )

        raise Mismatch(f"{call} has no buffer named {name}. Keep it with self.register_buffer(\"{name}\", ...).")

    return buffers["quantized"], buffers["scales"]


def seeded_linear(
    seed: int,
    inputs: int,
    outputs: int,
    bias: bool,
) -> nn.Linear:
    torch.manual_seed(seed)

    return nn.Linear(inputs, outputs, bias=bias)


def check_int8_init(found: type) -> None:
    for seed, inputs, outputs, bias in ((899, 6, 4, True), (900, 3, 5, False)):
        call = f"Int8Linear of an nn.Linear({inputs}, {outputs}{'' if bias else ', bias=False'})"
        linear = seeded_linear(seed, inputs, outputs, bias)
        layer = found(linear)
        quantized, scales = quantized_buffers(layer, call)
        expected = QUANT_REFERENCE["per_channel_quantize"](linear.weight.detach(), 8)
        expect_quantized(
            call=call,
            actual=(quantized, scales),
            expected=expected,
        )


def check_int8_forward(found: type) -> None:
    for seed, inputs, outputs, bias in ((901, 6, 4, True), (902, 3, 5, False)):
        linear = seeded_linear(seed, inputs, outputs, bias)
        x = torch.randn(2, 3, inputs, generator=torch.Generator().manual_seed(seed))
        expected = QUANT_REFERENCE["Int8Linear"](linear)(x)
        expect_tensor(
            call=f"Int8Linear of an nn.Linear({inputs}, {outputs}{'' if bias else ', bias=False'}) on x of shape (2, 3, {inputs})",
            actual=found(linear)(x),
            expected=expected,
        )


def small_block() -> nn.Module:
    torch.manual_seed(903)

    return MODEL_REFERENCE["GPT2Block"](16, 4)


def swap_with_reference(
    found: Callable[[nn.Module], None],
    module: nn.Module,
    described: str,
) -> None:
    """quantize_linears_ runs with the reference Int8Linear in place of the learner's, so a mistake there shows in its own row."""
    linears = [name for name, child in module.named_modules() if isinstance(child, nn.Linear)]
    globals_ = found.__globals__
    learners = globals_.get("Int8Linear")
    globals_["Int8Linear"] = QUANT_REFERENCE["Int8Linear"]

    try:
        found(module)
    finally:
        globals_["Int8Linear"] = learners

    left = [name for name, child in module.named_modules() if isinstance(child, nn.Linear)]

    if left and left == linears:
        raise Mismatch(
            f"quantize_linears_ of {described} left every nn.Linear in place. Swap each one with setattr(module, name, "
            "new). Giving the loop's name a new value changes nothing inside module.",
        )

    if left:
        raise Mismatch(
            f"After quantize_linears_ of {described}, the layers named {', '.join(left)} are still nn.Linear. Call "
            "quantize_linears_ on every child that isn't a Linear, so it reaches every depth.",
        )


def check_quantize_linears(found: object) -> None:
    nested = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.ReLU(), nn.Linear(4, 4)))
    swap_with_reference(
        found=found,
        module=nested,
        described="a Sequential holding a Linear and a Sequential with a Linear inside",
    )

    block = small_block()
    expected = small_block()
    QUANT_REFERENCE["quantize_linears_"](expected)
    swap_with_reference(
        found=found,
        module=block,
        described="a GPT2Block",
    )
    kinds = [type(module).__name__ for module in block.modules()]
    wanted = [type(module).__name__ for module in expected.modules()]

    if kinds != wanted:
        raise Mismatch(
            f"After quantize_linears_ of a GPT2Block, its layers are {', '.join(kinds)}, but they should be "
            f"{', '.join(wanted)}. Swap only the nn.Linear layers.",
        )

    x = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(904))
    expect_tensor(
        call="a GPT2Block after quantize_linears_, on x of shape (2, 5, 16)",
        actual=block(x),
        expected=expected(x),
    )


def check_model_bytes(found: object) -> None:
    model = seeded_gpt(MODEL_REFERENCE["GPT"])
    expect_value(
        call="model_bytes of a small GPT, whose lm_head shares wte's weight",
        actual=found(model),
        expected=QUANT_REFERENCE["model_bytes"](model),
    )

    for block in model.h:
        QUANT_REFERENCE["quantize_linears_"](block)

    expect_value(
        call="model_bytes of that GPT with its blocks in int8",
        actual=found(model),
        expected=QUANT_REFERENCE["model_bytes"](model),
    )


@functools.cache
def reference_measures() -> dict[str, float]:
    """quant3's four names on the learner's trained GPT, with the reference code."""
    model = trained_gpt(MODEL_REFERENCE["GPT"])
    _, _, validation_data = GPT_REFERENCE["shakespeare_splits"]()
    cpu = torch.device("cpu")
    validation_loss = GPT_REFERENCE["validation_loss"]
    measures = {
        "float_bytes": QUANT_REFERENCE["model_bytes"](model),
        "float_loss": validation_loss(model, validation_data, 20, torch.Generator().manual_seed(1), cpu),
    }

    for block in model.h:
        QUANT_REFERENCE["quantize_linears_"](block)

    measures["int8_bytes"] = QUANT_REFERENCE["model_bytes"](model)
    measures["int8_loss"] = validation_loss(model, validation_data, 20, torch.Generator().manual_seed(1), cpu)

    return measures


def check_measure(name: str) -> Callable[[object], None]:
    def check(value: object) -> None:
        if value is None and name.endswith("_bytes"):
            raise Mismatch(f"{name} is None, since model_bytes gives None. Write model_bytes first.")

        expected = reference_measures()[name]

        if isinstance(expected, int):
            expect_value(
                call=name,
                actual=value,
                expected=expected,
            )
            return

        expect_close(
            call=name,
            actual=value,
            expected=expected,
            tolerance=1e-4,
        )

    return check


def check_quant3(checker: Checker) -> None:
    checker.test_method(
        class_name="Int8Linear",
        method="__init__",
        description="keeps the weight as int8 with a scale per row",
        run=check_int8_init,
    )
    checker.test_method(
        class_name="Int8Linear",
        method="forward",
        description="runs the dequantized weight",
        run=check_int8_forward,
    )
    checker.test(
        function="quantize_linears_",
        description="swaps every Linear for an Int8Linear",
        run=check_quantize_linears,
    )
    checker.test(
        function="model_bytes",
        description="counts the bytes of parameters and buffers",
        run=check_model_bytes,
    )

    for name, description in (
        ("float_bytes", "is your GPT's size in float32"),
        ("float_loss", "is its validation loss in float32"),
        ("int8_bytes", "is its size with the blocks in int8"),
        ("int8_loss", "is its validation loss in int8"),
    ):
        checker.check_name(
            name=name,
            description=description,
            run=check_measure(name),
        )


QUANT3 = Exercise(
    name="quant3",
    chapter=QUANTIZATION,
    title="Your GPT in int8",
    lesson=QUANT3_LESSON,
    source=QUANT3_SOURCE,
    solution=QUANT3_SOLUTION,
    hints={
        "Int8Linear.__init__": "per_channel_quantize(weight, 8) gives both, then self.register_buffer(\"quantized\", "
        "...) and the same for scales.",
        "Int8Linear.forward": "F.linear takes x, a weight and a bias. dequantize(self.quantized, self.scales) is the "
        "weight.",
        "quantize_linears_": "Loop over module.named_children(). An nn.Linear gets setattr(module, name, "
        "Int8Linear(child)), and any other child gets quantize_linears_ called on it.",
        "model_bytes": "model.parameters() lists a shared tensor once. Add numel() * element_size() over it and "
        "over model.buffers().",
        "int8_bytes": "model_bytes(model), now that the blocks are quantized.",
        "int8_loss": "validation_loss with the same data, 20 batches, a generator seeded with 1 and cpu.",
    },
    introduces=(),
    check=check_quant3,
    time_limit=120,
)


# ---------------------------------------------------------------------------
# The measurements behind the lessons
# ---------------------------------------------------------------------------


class MeasuredLinear(nn.Module):
    """A Linear stored as whole numbers with any of the chapter's scales, for the tables. Not handed to the learner."""

    def __init__(
        self,
        linear: nn.Linear,
        quantize: Callable[[Tensor, int], tuple[Tensor, Tensor]],
        bits: int,
    ) -> None:
        super().__init__()
        quantized, scales = quantize(linear.weight.detach(), bits)
        self.register_buffer("quantized", quantized)
        self.register_buffer("scales", scales.reshape(-1, 1) if scales.dim() else scales)
        self.bias = linear.bias

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, QUANT_REFERENCE["dequantize"](self.quantized, self.scales), self.bias)


def swap_linears_(
    module: nn.Module,
    quantize: Callable[[Tensor, int], tuple[Tensor, Tensor]],
    bits: int,
) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, MeasuredLinear(child, quantize, bits))
            continue

        swap_linears_(child, quantize, bits)


def measure_quantization() -> dict[str, object]:
    """Every number in the three lessons, from the reference gpt3 checkpoint on the CPU, under a minute. Not run by any check."""
    absmax = QUANT_REFERENCE["absmax_quantize"]
    per_channel = QUANT_REFERENCE["per_channel_quantize"]
    dequantize = QUANT_REFERENCE["dequantize"]
    relative_error = QUANT_REFERENCE["relative_error"]
    zero_share = QUANT_REFERENCE["zero_share"]
    model = trained_gpt(MODEL_REFERENCE["GPT"])
    results: dict[str, object] = {}

    first = model.h[0].mlp.c_fc.weight.detach()
    results["c_fc largest size"] = first.abs().max().item()

    for bits in (8, 4, 2):
        quantized, scale = absmax(first, bits)
        results[f"c_fc {bits} bits"] = (
            QUANT_REFERENCE["largest_error"](first, bits),
            relative_error(first, dequantize(quantized, scale)),
            zero_share(quantized),
        )

    weights = [module.weight.detach() for module in model.h.modules() if isinstance(module, nn.Linear)]
    results["block weights"] = sum(weight.numel() for weight in weights)
    sizes = [weight.abs().amax(dim=1) for weight in weights]
    results["row largest sizes"] = (min(size.min().item() for size in sizes), max(size.max().item() for size in sizes))

    for name, quantize in (("per tensor", absmax), ("per channel", per_channel)):
        for bits in (8, 4):
            errors = torch.tensor([relative_error(weight, dequantize(*quantize(weight, bits))) for weight in weights])
            zeros = sum(zero_share(quantize(weight, bits)[0]) * weight.numel() for weight in weights) / results["block weights"]
            results[f"blocks {name} {bits} bits"] = (errors.mean().item(), zeros)

    # One weight of the first row set to 20 times the largest. The errors leave that weight out, and the last
    # one leaves its whole row out.
    outlier = first.clone()
    outlier[0, 0] = 20 * first.abs().max()
    others = torch.ones_like(outlier, dtype=torch.bool)
    others[0, 0] = False
    other_rows = torch.ones_like(outlier, dtype=torch.bool)
    other_rows[0] = False

    for name, tested, quantize in (
        ("no outlier, per tensor", first, absmax),
        ("outlier, per tensor", outlier, absmax),
        ("outlier, per channel", outlier, per_channel),
    ):
        quantized, scales = quantize(tested, 8)
        approximation = dequantize(quantized, scales)
        results[name] = (
            relative_error(tested[others], approximation[others]),
            zero_share(quantized[others]),
            relative_error(tested[other_rows], approximation[other_rows]),
        )

    _, _, validation_data = GPT_REFERENCE["shakespeare_splits"]()
    cpu = torch.device("cpu")

    for name, quantize, bits in (
        ("float32", None, 32),
        ("int8 per channel", per_channel, 8),
        ("int8 per tensor", absmax, 8),
        ("int4 per channel", per_channel, 4),
        ("int2 per channel", per_channel, 2),
    ):
        model = trained_gpt(MODEL_REFERENCE["GPT"])

        if quantize is not None:
            for block in model.h:
                swap_linears_(block, quantize, bits)

        loss = GPT_REFERENCE["validation_loss"](model, validation_data, 20, torch.Generator().manual_seed(1), cpu)
        results[f"model {name}"] = (QUANT_REFERENCE["model_bytes"](model), loss)

    return results
