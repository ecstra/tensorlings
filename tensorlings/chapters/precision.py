"""86_precision, float32, float16 and bfloat16 with their range and their precision, autocast, and loss scaling
by hand for float16."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import load_shakespeare
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
)
from ..text import prose, source_file
from ..training import load_checkpoint
from .gpt import MODEL_REFERENCE
from .profiling import PassThrough

PRECISION = Chapter(
    folder="86_precision",
    title="Precision",
    readme=source_file(
        """
        # Precision

        Every number in the course so far has been a float32. A 16 bit float
        uses half as many bits, so it halves the memory and the bytes moved,
        and GPUs multiply 16 bit floats faster. This chapter compares
        float32, float16 and bfloat16. Then it trains in float16 the way
        mixed precision training does, with autocast and loss scaling.

        ## Further reading

        - [Micikevicius and colleagues, mixed precision training](https://arxiv.org/abs/1710.03740)
        - [PyTorch, automatic mixed precision](https://docs.pytorch.org/docs/2.14/amp.html)
        - [PyTorch, torch.finfo](https://docs.pytorch.org/docs/2.14/type_info.html)
        """,
    ),
)

DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(code, namespace)

    return namespace


def dtype_text(dtype: torch.dtype) -> Described:
    return Described(dtype, str(dtype))


# ---------------------------------------------------------------------------
# precision1
# ---------------------------------------------------------------------------

PRECISION1_LESSON = prose(
    """
    A float32 takes 32 bits, 1 for the sign, 8 for the exponent and 23 for
    the digits. float16 and bfloat16 each take 16 bits, so they halve the
    memory and the bytes moved. They split those 16 bits in different ways.

    | Type | Exponent bits | Digit bits | Largest | Step after 1 |
    |---|---|---|---|---|
    | float32 | 8 | 23 | 3.4e38 | 1.2e-7 |
    | float16 | 5 | 10 | 65,504 | 0.00098 |
    | bfloat16 | 8 | 7 | 3.4e38 | 0.0078 |

    The exponent bits set the range, which is how large a number can be. The
    digit bits set the precision, which is how close together neighboring
    numbers are. float16 keeps more digits than bfloat16, but it cannot hold
    anything above 65,504. bfloat16 has the same range as float32, with fewer
    digits. `torch.finfo(dtype)` holds these facts about a type. Its `max` is
    the largest number, and its `eps` is the step from 1 to the next larger
    number the type can hold.

    ```python
    torch.finfo(torch.float16).max              # 65504.0
    torch.tensor(70000.0).to(torch.float16)     # tensor(inf, dtype=torch.float16)
    torch.tensor(1 / 3).to(torch.bfloat16)      # tensor(0.3340, dtype=torch.bfloat16)
    ```

    A number that falls between two numbers the type can hold rounds to the
    nearer one. So its relative error, the error divided by the number's
    size, is at most half of `eps`. On a million draws from a normal
    distribution, the worst relative error was 0.00049 in float16 and 0.0039
    in bfloat16. That bound holds only above 6.1e-5, the smallest float16
    number with full precision. Below it, float16 keeps fewer digits, and a
    draw of 4.3e-6 came out 0.67% off.

    Low precision hurts most when you add up many small numbers. Once the
    running total is large, the gap from the total to the next number the
    type can hold is large too. If the small number is less than half that
    gap, the sum rounds back to the old total, so adding it changes nothing.
    Here is 0.0001 added 10,000 times, which should give 1.

    | Type | Total |
    |---|---|
    | float32 | 1.0000535 |
    | float16 | 0.25 |
    | bfloat16 | 0.03125 |

    float16 gets stuck at 0.25, because there the gap to the next float16
    number is 0.000244, and half of that is more than 0.0001. Micikevicius
    and colleagues found that some models trained in float16 were less
    accurate unless their dot products added up in float32. The GPUs they
    used can multiply in float16 and keep the running sums in float32.
    """,
)

PRECISION1_SOURCE = source_file(
    '''
    # precision1.py

    import torch

    biggest = torch.finfo(torch.float16).max             # 65504.0
    rounded = torch.tensor(1 / 3).to(torch.float16).float()    # tensor(0.3333), but the stored value is 0.333251953125
    error = ((rounded - 1 / 3) / (1 / 3)).abs()          # tensor(0.0002), the error divided by the number's size
    total = torch.zeros((), dtype=torch.float16) + torch.tensor(0.5, dtype=torch.float16)    # adds in float16


    # TODO: Return the largest number that dtype can hold, as a plain float.
    #   Example: largest(torch.float16) returns 65504.0.
    def largest(
        dtype: torch.dtype,
    ) -> float:
        ...


    # TODO: Return the gap between 1 and the next larger number that dtype can
    #   hold, as a plain float.
    #   Example: step_after_one(torch.float16) returns 0.0009765625.
    def step_after_one(
        dtype: torch.dtype,
    ) -> float:
        ...


    # TODO: Return the largest relative error from rounding x to dtype, as a
    #   plain float. x is a float32 tensor with no zeros. Round with
    #   .to(dtype) and convert back with .float(). Then divide each entry's
    #   error by the absolute value of that entry, and take the largest.
    #   Example: worst_relative_error(torch.tensor([1 / 3]), torch.float16)
    #   returns about 0.000244.
    def worst_relative_error(
        x: torch.Tensor,
        dtype: torch.dtype,
    ) -> float:
        ...


    # TODO: Start a total at 0 and add value to it count times, one addition at
    #   a time. Keep both the total and value as tensors of dtype. Return the
    #   final total as a plain float.
    #   Example: running_sum(0.0001, 10000, torch.float16) returns 0.25.
    def running_sum(
        value: float,
        count: int,
        dtype: torch.dtype,
    ) -> float:
        ...
    ''',
)

PRECISION1_SOLUTION = source_file(
    '''
    # precision1.py, a solution for comparison

    import torch

    biggest = torch.finfo(torch.float16).max
    rounded = torch.tensor(1 / 3).to(torch.float16).float()
    error = ((rounded - 1 / 3) / (1 / 3)).abs()
    total = torch.zeros((), dtype=torch.float16) + torch.tensor(0.5, dtype=torch.float16)


    def largest(
        dtype: torch.dtype,
    ) -> float:
        return torch.finfo(dtype).max


    def step_after_one(
        dtype: torch.dtype,
    ) -> float:
        return torch.finfo(dtype).eps


    def worst_relative_error(
        x: torch.Tensor,
        dtype: torch.dtype,
    ) -> float:
        return ((x.to(dtype).float() - x).abs() / x.abs()).max().item()


    def running_sum(
        value: float,
        count: int,
        dtype: torch.dtype,
    ) -> float:
        total = torch.zeros((), dtype=dtype)
        step = torch.tensor(value, dtype=dtype)

        for _ in range(count):
            total = total + step

        return total.item()
    ''',
)

PRECISION1_REFERENCE = namespace_of(PRECISION1_SOLUTION)


def check_precision1(checker: Checker) -> None:
    for function in ("largest", "step_after_one"):
        check_on_inputs(
            checker=checker,
            function=function,
            description="reads torch.finfo",
            inputs=tuple((dtype_text(dtype),) for dtype in DTYPES),
            reference=PRECISION1_REFERENCE[function],
            expect=expect_value,
        )

    check_on_inputs(
        checker=checker,
        function="worst_relative_error",
        description="divides each rounding error by the size of its own entry",
        inputs=(
            (torch.tensor([1 / 3, 1000.1, -0.007, 2.0]), dtype_text(torch.float16)),
            (torch.tensor([1 / 3, 1000.1, -0.007, 2.0]), dtype_text(torch.bfloat16)),
            (torch.tensor([3.14159, 1e-3, -12345.6]), dtype_text(torch.float16)),
        ),
        reference=PRECISION1_REFERENCE["worst_relative_error"],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="running_sum",
        description="adds one value at a time in the given type",
        inputs=(
            (0.0001, 10000, dtype_text(torch.float16)),
            (0.01, 300, dtype_text(torch.bfloat16)),
            (0.1, 50, dtype_text(torch.float32)),
        ),
        reference=PRECISION1_REFERENCE["running_sum"],
        expect=expect_close,
    )


PRECISION1 = Exercise(
    name="precision1",
    chapter=PRECISION,
    title="Three kinds of float",
    lesson=PRECISION1_LESSON,
    source=PRECISION1_SOURCE,
    solution=PRECISION1_SOLUTION,
    hints={
        "largest": "torch.finfo(dtype) holds this number in its max attribute.",
        "step_after_one": "torch.finfo(dtype) holds this number in its eps attribute.",
        "worst_relative_error": "The error is x.to(dtype).float() minus x. Divide its absolute value by x.abs(), take "
        "the max, then call .item().",
        "running_sum": "Start the total as torch.zeros((), dtype=dtype), and turn value into a tensor of dtype. Then "
        "add in a loop, and finish with .item().",
    },
    introduces=(),
    check=check_precision1,
    time_limit=20,
)


# ---------------------------------------------------------------------------
# precision2
# ---------------------------------------------------------------------------

PRECISION2_LESSON = prose(
    """
    Mixed precision training runs each operation in float16 where that is
    safe, and in float32 where it is not.
    `torch.autocast(device_type, dtype=...)` makes that choice for you.
    Inside a `with torch.autocast(...)` block, each operation runs in the
    type that PyTorch's own list assigns to it. Here are the types this Mac
    gives, with PyTorch 2.14.1.

    | Inside autocast to float16 | On MPS | On the CPU |
    |---|---|---|
    | A Linear's output | float16 | float16 |
    | `torch.softmax` | float32 | float16 |
    | `F.cross_entropy` | float32 | float32 |
    | The Linear's weight | float32 | float32 |

    The weights stay in float32. Micikevicius and colleagues call this the
    master copy of the weights. They keep it in float32 because an update, a
    gradient times the learning rate, is often too small to change a weight
    stored in float16.

    The backward pass has a problem of its own. Gradients are often tiny,
    and the smallest float16 number above 0 is about 6e-8. A gradient smaller
    than half of that becomes 0. Loss scaling fixes this by multiplying the
    loss by a large number, called the scale, before backward. Then every
    gradient comes out that many times larger. Dividing the gradients by the
    scale afterwards gives back the true gradients for the optimizer step.

    If the scale is too large, some gradients go past 65,504 and become inf.
    When that happens, the optimizer step is skipped and the scale is halved.
    After a run of good steps in a row, the scale is doubled again.
    `torch.amp.GradScaler` does exactly this. It starts at a scale of 65,536
    and doubles it after 2,000 good steps.

    Large models average each step's loss over millions of tokens, 3.2
    million for GPT-3. So the gradient from each token comes out that many
    times smaller. The table below imitates this with gpt3's trained GPT, on
    one batch of 4,096 characters of Shakespeare, on the CPU. It divides the
    loss by 64 and by 1,024, which is what a batch 64 or 1,024 times bigger
    would do. The gradients come from a backward pass under autocast to
    float16, and are divided by the scale afterwards. They are compared with
    the gradients from float32.

    | Loss divided by | Scale | Lost to 0 at the blocks' outputs | Error in the weights' gradients |
    |---|---|---|---|
    | 1 | 1 | 0.12% | 0.47% |
    | 64 | 1 | 3.9% | 16% |
    | 1,024 | 1 | 30% | 138% |
    | 1, 64 or 1,024 | 65,536 | none | 0.37% to 0.38% |

    The third column is the share of the gradients at the outputs of the
    blocks that became 0. The last column is how far the weight gradients
    are from float32's, relative to their size.

    Without scaling, the smaller the loss, the more of its gradients float16
    rounds to 0. When the loss is divided by 1,024, the error in the weight
    gradients is larger than the gradients themselves. A scale of 65,536
    keeps every gradient from becoming 0. Micikevicius and colleagues found
    that training their SSD object detector in float16 diverged until they
    scaled the loss by 8.

    A scale that is too large fails in the other direction. At a scale of
    16,777,216, 69% of the gradients became inf or nan. Autocast to bfloat16
    lost no gradients to 0 at any of the three losses, because bfloat16 has
    float32's range. But its coarser rounding left the weight gradients
    about 2.8% off in each case.
    """,
)

PRECISION2_SOURCE = source_file(
    '''
    # precision2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    tiny = torch.tensor([1e-8, 1e-3]).to(torch.float16).float()    # tensor([0.0000, 0.0010]), so 1e-8 became 0
    kept = (torch.tensor([1e-8, 1e-3]) * 1024).to(torch.float16).float() / 1024    # tensor([1.0012e-08, 1.0004e-03])
    finite = bool(torch.isfinite(torch.tensor([1.0, float("inf")])).all())    # False
    halved, doubled = 65536.0 * 0.5, 65536.0 * 2         # (32768.0, 131072.0)


    # TODO: Take the nonzero entries of gradients, multiply them by scale and
    #   convert them to dtype. Return the fraction of them that became 0, as a
    #   plain float.
    #   Example: lost_to_zero(torch.tensor([1e-8, 1e-3]), 1.0, torch.float16)
    #   returns 0.5.
    def lost_to_zero(
        gradients: Tensor,
        scale: float,
        dtype: torch.dtype,
    ) -> float:
        ...


    # TODO: Divide the gradient of every parameter of model by scale, in place.
    #   Skip parameters whose grad is None. Return True when every gradient is
    #   finite afterwards, and False when any gradient holds an inf or a nan.
    def unscale_(
        model: nn.Module,
        scale: float,
    ) -> bool:
        ...


    # TODO: Return the next scale and the new count of good steps in a row, as
    #   a tuple (scale, good_steps). After a step that found an inf, halve the
    #   scale and reset the count to 0. After a good step, add 1 to the count.
    #   If the count then reaches growth_interval, double the scale and reset
    #   the count to 0.
    #   Example: update_scale(8.0, False, 2, 3) returns (16.0, 0).
    def update_scale(
        scale: float,
        found_inf: bool,
        good_steps: int,
        growth_interval: int,
    ) -> tuple[float, int]:
        ...


    # TODO: Take one training step of model on inputs x and labels y. Inside
    #   torch.autocast for x's device type with dtype=torch.float16, compute
    #   the loss, F.cross_entropy of model(x) and y. Then call backward on the
    #   loss times scale, and unscale_ the gradients. Step the optimizer only
    #   if the gradients are finite, and zero the gradients either way. Return
    #   True if the optimizer stepped and False if it did not.
    def mixed_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        x: Tensor,
        y: Tensor,
        scale: float,
    ) -> bool:
        ...
    ''',
)

SCALING_PART = '''
def lost_to_zero(
    gradients: Tensor,
    scale: float,
    dtype: torch.dtype,
) -> float:
    nonzero = gradients[gradients != 0]
    lost = (nonzero * scale).to(dtype) == 0

    return lost.float().mean().item()


def unscale_(
    model: nn.Module,
    scale: float,
) -> bool:
    finite = True

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        parameter.grad.div_(scale)
        finite = finite and bool(torch.isfinite(parameter.grad).all())

    return finite


def update_scale(
    scale: float,
    found_inf: bool,
    good_steps: int,
    growth_interval: int,
) -> tuple[float, int]:
    if found_inf:
        return scale * 0.5, 0

    if good_steps + 1 == growth_interval:
        return scale * 2, 0

    return scale, good_steps + 1


def mixed_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    scale: float,
) -> bool:
    with torch.autocast(x.device.type, dtype=torch.float16):
        loss = F.cross_entropy(model(x), y)

    (loss * scale).backward()
    finite = unscale_(model, scale)

    if finite:
        optimizer.step()

    optimizer.zero_grad()

    return finite
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return "\n".join(f"    {line}" if line else "" for line in part.strip("\n").splitlines())


PRECISION2_SOLUTION = source_file(
    '''
    # precision2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    tiny = torch.tensor([1e-8, 1e-3]).to(torch.float16).float()
    kept = (torch.tensor([1e-8, 1e-3]) * 1024).to(torch.float16).float() / 1024
    finite = bool(torch.isfinite(torch.tensor([1.0, float("inf")])).all())
    halved, doubled = 65536.0 * 0.5, 65536.0 * 2


''' + indented(SCALING_PART) + '''
    ''',
)

SCALING_REFERENCE = namespace_of("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch import Tensor\n" + SCALING_PART)


def seeded_classifier(seed: int) -> nn.Module:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 3))


def model_with_gradients(
    seed: int,
    poison: float | None,
) -> nn.Module:
    """A model whose gradients are set by hand, its last bias left with none, one entry set to poison when given."""
    model = seeded_classifier(seed)
    generator = torch.Generator().manual_seed(seed)

    for parameter in list(model.parameters())[:-1]:
        parameter.grad = torch.randn(parameter.shape, generator=generator) * 64

    if poison is not None:
        model[0].weight.grad[2, 3] = poison

    return model


def check_unscale(
    found: Callable[..., object],
    seed: int,
    scale: float,
    poison: float | None,
) -> None:
    model = model_with_gradients(seed, poison)
    expected = model_with_gradients(seed, poison)
    finite = SCALING_REFERENCE["unscale_"](expected, scale)
    actual = found(model, scale)
    held = "are all finite" if poison is None else f"hold {'a nan' if poison != poison else 'an inf'}"
    call = f"unscale_ of a model whose gradients {held}, at a scale of {scale:g}"

    expect_value(
        call=call,
        actual=actual,
        expected=finite,
    )

    for (name, parameter), wanted in zip(model.named_parameters(), expected.parameters()):
        if wanted.grad is None:
            if parameter.grad is not None:
                raise Mismatch(f"{call} gave {name} a gradient, but it had none. Skip a parameter whose grad is None.")

            continue

        got, wanted_grad = parameter.grad, wanted.grad

        # A nan stays a nan after the division and never equals itself, so matching nans compare as equal.
        if isinstance(got, Tensor) and got.shape == wanted_grad.shape and torch.equal(got.isnan(), wanted_grad.isnan()):
            got, wanted_grad = got.nan_to_num(nan=0.0), wanted_grad.nan_to_num(nan=0.0)

        expect_tensor(
            call=f"the gradient of {name} after {call}",
            actual=got,
            expected=wanted_grad,
        )


def classifier_batch(seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)

    return torch.randn(8, 6, generator=generator), torch.randint(0, 3, (8,), generator=generator)


def check_mixed_step(
    found: Callable[..., object],
    scales: tuple[float, ...],
) -> None:
    models = [seeded_classifier(860), seeded_classifier(860)]
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.5) for model in models]
    steps = (found, SCALING_REFERENCE["mixed_step"])
    call = f"two calls of mixed_step at scales of {' and '.join(f'{scale:g}' for scale in scales)}"

    for number, scale in enumerate(scales):
        x, y = classifier_batch(861 + number)

        for step, model, optimizer in zip(steps, models, optimizers):
            stepped = step(model, optimizer, x, y, scale)

            if step is found:
                learner_stepped = stepped
            else:
                expect_value(
                    call=f"call {number + 1} of {call}",
                    actual=learner_stepped,
                    expected=stepped,
                )

    for (name, parameter), wanted in zip(models[0].named_parameters(), models[1].parameters()):
        if parameter.grad is not None and parameter.grad.abs().sum() > 0:
            raise Mismatch(f"After {call}, {name} still holds a gradient. Zero the gradients at the end of each step.")

        expect_tensor(
            call=f"{name} after {call}",
            actual=parameter.detach(),
            expected=wanted.detach(),
        )


class AutocastCalled(Exception):
    pass


def check_autocast_device(found: Callable[..., object]) -> None:
    """mixed_step runs on the meta device, so an autocast written for "cpu" shows. The stand in for autocast stops
    the step there, since meta tensors hold no numbers to step on."""
    device_types: list[str] = []

    def record(
        device_type: str,
        *arguments: object,
        **keywords: object,
    ) -> None:
        device_types.append(device_type)

        raise AutocastCalled

    model = seeded_classifier(860).to("meta")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    x = torch.zeros(8, 6, device="meta")
    y = torch.zeros(8, dtype=torch.long, device="meta")
    original = found.__globals__.get("torch")
    found.__globals__["torch"] = PassThrough(torch, autocast=record, amp=PassThrough(torch.amp, autocast=record))

    try:
        found(model, optimizer, x, y, 1024.0)
    except AutocastCalled:
        pass
    except Exception:
        if device_types:
            raise
    finally:
        found.__globals__["torch"] = original

    if not device_types:
        raise Mismatch("mixed_step should run the forward pass inside torch.autocast, for x's device type.")

    if device_types[0] != "meta":
        raise Mismatch(
            f'mixed_step called torch.autocast for "{device_types[0]}", but x was on another device. Use '
            "x.device.type, so the step runs in float16 on a GPU too.",
        )


def check_loss_in_autocast(found: Callable[..., object]) -> None:
    """F.cross_entropy is swapped on its module, so F.cross_entropy, torch.nn.functional.cross_entropy and
    nn.CrossEntropyLoss are all seen. Outside autocast it would run on the float16 scores in float16."""
    inside: list[bool] = []
    original = torch.nn.functional.cross_entropy

    def recorded(
        *arguments: object,
        **keywords: object,
    ) -> Tensor:
        inside.append(torch.is_autocast_enabled("cpu"))

        return original(*arguments, **keywords)

    model = seeded_classifier(860)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    x, y = classifier_batch(861)
    torch.nn.functional.cross_entropy = recorded

    try:
        found(model, optimizer, x, y, 1024.0)
    finally:
        torch.nn.functional.cross_entropy = original

    if inside and not all(inside):
        raise Mismatch(
            "mixed_step computed F.cross_entropy outside the autocast block, so it ran in float16 on the float16 "
            "scores. Compute it inside the block, where autocast runs it in float32.",
        )


def check_precision2(checker: Checker) -> None:
    generator = torch.Generator().manual_seed(862)
    gradients = torch.randn(200, generator=generator) * torch.logspace(-10, -2, 200)
    gradients[::17] = 0

    check_on_inputs(
        checker=checker,
        function="lost_to_zero",
        description="returns the fraction of nonzero gradients that round to 0",
        inputs=(
            (Described(gradients, "200 gradients from 1e-10 to 1e-2, some of them 0"), 1.0, dtype_text(torch.float16)),
            (Described(gradients, "200 gradients from 1e-10 to 1e-2, some of them 0"), 1024.0, dtype_text(torch.float16)),
            (Described(gradients, "200 gradients from 1e-10 to 1e-2, some of them 0"), 1.0, dtype_text(torch.bfloat16)),
        ),
        reference=SCALING_REFERENCE["lost_to_zero"],
        expect=expect_close,
    )

    for seed, scale, poison in ((863, 64.0, None), (864, 1024.0, float("inf")), (866, 32.0, float("nan")), (865, 8.0, None)):
        checker.test(
            function="unscale_",
            description="divides every gradient by the scale and returns whether all are finite",
            run=lambda found, seed=seed, scale=scale, poison=poison: check_unscale(
                found=found,
                seed=seed,
                scale=scale,
                poison=poison,
            ),
        )

    check_on_inputs(
        checker=checker,
        function="update_scale",
        description="halves after an inf and doubles after a run of good steps",
        inputs=(
            (65536.0, True, 7, 2000),
            (1024.0, False, 1999, 2000),
            (1024.0, False, 5, 2000),
            (8.0, False, 2, 3),
            (8.0, True, 2, 3),
        ),
        reference=SCALING_REFERENCE["update_scale"],
        expect=expect_parts("the scale", "the count of good steps"),
    )

    for scales in ((1024.0, 1024.0), (2.0**40, 256.0)):
        checker.test(
            function="mixed_step",
            description="steps on unscaled gradients, skips a step with an inf, and zeroes the gradients",
            run=lambda found, scales=scales: check_mixed_step(
                found=found,
                scales=scales,
            ),
        )

    checker.test(
        function="mixed_step",
        description="runs autocast for x's own device type",
        run=check_autocast_device,
    )
    checker.test(
        function="mixed_step",
        description="computes the loss inside autocast",
        run=check_loss_in_autocast,
    )


PRECISION2 = Exercise(
    name="precision2",
    chapter=PRECISION,
    title="Autocast and loss scaling",
    lesson=PRECISION2_LESSON,
    source=PRECISION2_SOURCE,
    solution=PRECISION2_SOLUTION,
    hints={
        "lost_to_zero": "Keep the entries of gradients that are not 0. Multiply them by scale, convert them to dtype, "
        "and find the fraction that equals 0.",
        "unscale_": "Loop over model.parameters(), skip a grad of None, and divide the rest in place with .div_. "
        "torch.isfinite(...).all() tells you whether one gradient is all finite.",
        "update_scale": "There are three cases. The step found an inf. The step was good and completes "
        "growth_interval good steps. The step was good and the count is still short.",
        "mixed_step": "Only the forward pass and the loss go inside the with block. Backward on loss * scale comes "
        "after it, then unscale_, then optimizer.step() when finite. Last, clear the gradients with the optimizer's "
        "zero_grad.",
    },
    introduces=(),
    check=check_precision2,
    imports_first=OPTIMIZER_IMPORTS,
    time_limit=20,
)


# ---------------------------------------------------------------------------
# The numbers in the lessons
# ---------------------------------------------------------------------------


def measure_rounding() -> dict[str, float]:
    """precision1's numbers, on the CPU."""
    x = torch.randn(1_000_000, generator=torch.Generator().manual_seed(0))
    x = x[x != 0]
    full_precision = x[x.abs() >= torch.finfo(torch.float16).smallest_normal]
    worst = PRECISION1_REFERENCE["worst_relative_error"]
    running_sum = PRECISION1_REFERENCE["running_sum"]
    rounded = x.to(torch.float16).float()
    worst_draw = ((rounded - x).abs() / x.abs()).argmax()
    results: dict[str, float] = {
        "float16 worst draw": x[worst_draw].item(),
        "float16 worst over all": worst(x, torch.float16),
    }

    for dtype in DTYPES:
        results[f"{dtype} worst relative error"] = worst(full_precision, dtype)
        results[f"{dtype} running sum"] = running_sum(0.0001, 10000, dtype)

    return results


def measure_autocast() -> dict[str, dict[str, str]]:
    """precision2's table of the types autocast gives, on each device this machine has."""
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])
    table: dict[str, dict[str, str]] = {}

    for device in devices:
        torch.manual_seed(0)
        linear = nn.Linear(8, 4).to(device)
        x = torch.randn(3, 8, device=device)

        with torch.autocast(device, dtype=torch.float16):
            output = linear(x)
            table[device] = {
                "linear": str(output.dtype),
                "softmax": str(torch.softmax(output, dim=-1).dtype),
                "cross_entropy": str(F.cross_entropy(output, torch.zeros(3, dtype=torch.long, device=device)).dtype),
                "weight": str(linear.weight.dtype),
            }

    return table


def gradients_of(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    dtype: torch.dtype,
    scale: float,
    divisor: float,
) -> tuple[Tensor, Tensor]:
    """Every weight gradient of model on one batch, and every gradient at a block's output, from a backward pass
    under autocast to dtype. The loss is divided by divisor, as if averaged over that many times the batch's
    tokens, then multiplied by scale. Both come back with the scale and the divisor taken off again."""
    model.zero_grad()
    at_outputs: list[Tensor] = []
    handles = [
        block.register_full_backward_hook(
            lambda module, grad_input, grad_output: at_outputs.append(grad_output[0].detach().float().clone()),
        )
        for block in model.h
    ]

    with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())

    (loss * scale / divisor).backward()

    for handle in handles:
        handle.remove()

    factor = scale / divisor
    weights = torch.cat([parameter.grad.flatten() for parameter in model.parameters()]) / factor

    return weights, torch.cat([gradient.flatten() for gradient in at_outputs]) / factor


def measure_scales() -> dict[str, float]:
    """precision2's table: gpt3's trained GPT from checkpoints/gpt3.pt, on one batch of 32 sequences of
    Shakespeare, on the CPU, with 4 threads. Each lost share is of the gradients at the blocks' outputs that
    float32 gives as nonzero, and each error is the weight gradients' distance from float32's, over their size."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        model = load_checkpoint(MODEL_REFERENCE["GPT"](65, 128, 128, 4, 4), "gpt3")
        text = load_shakespeare()
        characters = sorted(set(text))
        positions = {character: position for position, character in enumerate(characters)}
        data = torch.tensor([positions[character] for character in text[:100_000]])
        starts = torch.randint(0, len(data) - 129, (32,), generator=torch.Generator().manual_seed(0))
        x = torch.stack([data[start : start + 128] for start in starts])
        y = torch.stack([data[start + 1 : start + 129] for start in starts])
        results: dict[str, float] = {}

        for divisor in (1.0, 64.0, 1024.0):
            exact_weights, exact_outputs = gradients_of(model, x, y, torch.float32, 1.0, divisor)
            nonzero = exact_outputs != 0
            runs = [(torch.float16, 1.0), (torch.float16, 2.0**16), (torch.bfloat16, 1.0)]

            if divisor == 1.0:
                runs.append((torch.float16, 2.0**24))

            for dtype, scale in runs:
                weights, outputs = gradients_of(model, x, y, dtype, scale, divisor)
                name = f"{dtype} over {divisor:g} at {scale:g}"
                results[f"{name} lost"] = ((outputs == 0) & nonzero).float().sum().item() / nonzero.sum().item()
                results[f"{name} error"] = ((weights - exact_weights).norm() / exact_weights.norm()).item()
                results[f"{name} not finite"] = (~torch.isfinite(weights)).float().mean().item()

        return results
    finally:
        torch.set_num_threads(threads)
