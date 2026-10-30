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
from .gpt import MODEL_REFERENCE

PRECISION = Chapter(
    folder="86_precision",
    title="Precision",
    readme=source_file(
        """
        # Precision

        Every number so far has been a float32. Half as many bits halve the
        memory and the traffic, and GPUs multiply them faster. This chapter
        compares float32, float16 and bfloat16, then trains in float16 the
        way mixed precision does, with autocast and loss scaling.

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
    the digits. float16 and bfloat16 take 16 bits, so they halve the memory
    and the traffic, and they split those bits in different ways.

    | Type | Exponent bits | Digit bits | Largest | Step after 1 |
    |---|---|---|---|---|
    | float32 | 8 | 23 | 3.4e38 | 1.2e-7 |
    | float16 | 5 | 10 | 65,504 | 0.00098 |
    | bfloat16 | 8 | 7 | 3.4e38 | 0.0078 |

    The exponent sets the range and the digits set the precision. float16
    keeps more digits but runs out past 65,504. bfloat16 keeps float32's
    range with fewer digits. `torch.finfo(dtype)` holds these numbers, its
    `max` the largest and its `eps` the step from 1 to the next number up.

    ```python
    torch.finfo(torch.float16).max              # 65504.0
    torch.tensor(70000.0).to(torch.float16)     # tensor(inf, dtype=torch.float16)
    torch.tensor(1 / 3).to(torch.bfloat16)      # tensor(0.3340, dtype=torch.bfloat16)
    ```

    A number between two steps rounds to the nearer one, so its relative
    error is at most half the step. On a million draws from a normal
    distribution, the worst was 0.00049 in float16 and 0.0039 in bfloat16.
    That holds above float16's smallest full precision number, 6.1e-5.
    Below it float16 keeps fewer digits, and a draw of 4.3e-6 came out 0.67%
    off.

    Low precision hurts most when adding many small numbers. Once a total
    is large, a small number is less than half its step, and adding it
    changes nothing. Here is 0.0001 added 10,000 times, which should give 1.

    | Type | Total |
    |---|---|
    | float32 | 1.0000535 |
    | float16 | 0.25 |
    | bfloat16 | 0.03125 |

    float16 stalls at 0.25, where its step grows to 0.000244. Micikevicius
    and colleagues found some float16 models fell short unless dot products
    added up in float32. The GPUs they used can multiply in float16 and keep
    those sums in float32.
    """,
)

PRECISION1_SOURCE = source_file(
    '''
    # precision1.py

    import torch

    biggest = torch.finfo(torch.float16).max             # 65504.0
    rounded = torch.tensor(1 / 3).to(torch.float16).float()    # tensor(0.3333), 0.333251953125 underneath
    error = ((rounded - 1 / 3) / (1 / 3)).abs()          # tensor(0.0002), the error as a share of the number
    total = torch.zeros((), dtype=torch.float16) + torch.tensor(0.5, dtype=torch.float16)    # adds in float16


    # TODO: Return the largest number dtype holds, as a plain float.
    def largest(
        dtype: torch.dtype,
    ) -> float:
        ...


    # TODO: Return the step from 1 to the next number dtype holds, as a plain
    #   float.
    def step_after_one(
        dtype: torch.dtype,
    ) -> float:
        ...


    # TODO: Return the worst relative error of rounding x, a float32 tensor
    #   with no zeros, to dtype, as a plain float. Round with .to(dtype), come
    #   back with .float(), and divide each error by the size of its number.
    def worst_relative_error(
        x: torch.Tensor,
        dtype: torch.dtype,
    ) -> float:
        ...


    # TODO: Return the total of value added count times, one at a time, with
    #   the total and value both kept in dtype, as a plain float.
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
        description="divides each rounding error by its own number",
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
        description="adds one at a time in the type given",
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
        "largest": "torch.finfo(dtype) holds it, as its max.",
        "step_after_one": "torch.finfo(dtype) holds it, as its eps.",
        "worst_relative_error": "The error is x.to(dtype).float() minus x. Divide its size by x.abs(), take the largest, "
        "then .item().",
        "running_sum": "Make the total torch.zeros((), dtype=dtype) and value a tensor of dtype, then add in a loop and "
        "end with .item().",
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
    Mixed precision runs each operation in float16 where that is safe and
    in float32 where it is not. `torch.autocast(device_type, dtype=...)`
    makes the choice. Inside it, each operation runs in the type PyTorch
    lists for it. Here is what this Mac gives, with Torch 2.14.1.

    | Inside autocast to float16 | On MPS | On the CPU |
    |---|---|---|
    | A Linear's output | float16 | float16 |
    | `torch.softmax` | float32 | float16 |
    | `F.cross_entropy` | float32 | float32 |
    | The Linear's weight | float32 | float32 |

    The weights stay float32. Micikevicius and colleagues keep this master
    copy since an update, a gradient times the learning rate, is often too
    small for float16.

    Backward has a problem of its own. Gradients are often tiny, and
    float16's smallest number above 0 is about 6e-8. A gradient under half
    of that becomes 0. Loss scaling multiplies
    the loss by a large scale before backward, so every gradient comes out
    that much larger. Dividing the gradients by the scale afterwards gives
    the true ones back, for the optimizer step.

    Too large a scale overflows past 65,504 into inf. Then the step is
    skipped and the scale halved. After a run of good steps, the scale
    doubles again. That is `torch.amp.GradScaler`, which starts at 65,536
    and doubles after 2,000 good steps.

    Here is the GPT that gpt3 trains, untrained, on one batch of
    Shakespeare. Its gradients come from a backward pass under autocast to
    float16 at each scale, unscaled after, against the float32 ones.

    | Scale | 1 | 256 | 65,536 | 16,777,216 |
    |---|---|---|---|---|
    | Gradients lost to 0 | 0.024% | 0.001% | none | none |
    | Gradients turned inf or nan | none | none | none | 14.5% |

    65,536 keeps every gradient, and a scale 256 times larger breaks one in
    seven. This small GPT loses few gradients even unscaled. Micikevicius and
    colleagues' SSD detector was less lucky. It diverged in float16 until
    they scaled the loss by 8. Under autocast to bfloat16, with float32's
    range, no gradient was lost at a scale of 1.
    """,
)

PRECISION2_SOURCE = source_file(
    '''
    # precision2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    tiny = torch.tensor([1e-8, 1e-3]).to(torch.float16).float()    # tensor([0.0000, 0.0010]), 1e-8 is lost
    kept = (torch.tensor([1e-8, 1e-3]) * 1024).to(torch.float16).float() / 1024    # tensor([1.0012e-08, 1.0004e-03])
    finite = bool(torch.isfinite(torch.tensor([1.0, float("inf")])).all())    # False
    halved, doubled = 65536.0 * 0.5, 65536.0 * 2         # (32768.0, 131072.0)


    # TODO: Return the share of the nonzero entries of gradients that become 0
    #   once multiplied by scale and turned into dtype, as a plain float.
    def lost_to_zero(
        gradients: Tensor,
        scale: float,
        dtype: torch.dtype,
    ) -> float:
        ...


    # TODO: Divide the gradient of every parameter of model that has one by
    #   scale, in place. Return True when every gradient is finite afterwards,
    #   and False when any holds an inf or a nan.
    def unscale_(
        model: nn.Module,
        scale: float,
    ) -> bool:
        ...


    # TODO: Return the next scale and the count of good steps in a row, as a
    #   pair. After a step with an inf, halve the scale and start the count
    #   again at 0. After a good one, add 1 to the count, and once it reaches
    #   growth_interval, double the scale and start the count again.
    def update_scale(
        scale: float,
        found_inf: bool,
        good_steps: int,
        growth_interval: int,
    ) -> tuple[float, int]:
        ...


    # TODO: Take one training step of model on x and labels y, and return
    #   whether it stepped. Inside torch.autocast for x's device type with
    #   dtype=torch.float16, work out F.cross_entropy of model(x) and y. Then
    #   call backward on the loss times scale, unscale_ the gradients, step
    #   the optimizer only when they are finite, and zero the gradients.
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
    poisoned: bool,
) -> nn.Module:
    """A model whose gradients are set by hand, its last bias left with none, one entry an inf when poisoned."""
    model = seeded_classifier(seed)
    generator = torch.Generator().manual_seed(seed)

    for parameter in list(model.parameters())[:-1]:
        parameter.grad = torch.randn(parameter.shape, generator=generator) * 64

    if poisoned:
        model[0].weight.grad[2, 3] = float("inf")

    return model


def check_unscale(
    found: Callable[..., object],
    seed: int,
    scale: float,
    poisoned: bool,
) -> None:
    model = model_with_gradients(seed, poisoned)
    expected = model_with_gradients(seed, poisoned)
    finite = SCALING_REFERENCE["unscale_"](expected, scale)
    actual = found(model, scale)
    call = f"unscale_ of a model whose gradients {'hold an inf' if poisoned else 'are all finite'}, at a scale of {scale:g}"

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

        expect_tensor(
            call=f"the gradient of {name} after {call}",
            actual=parameter.grad,
            expected=wanted.grad,
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


def check_precision2(checker: Checker) -> None:
    generator = torch.Generator().manual_seed(862)
    gradients = torch.randn(200, generator=generator) * torch.logspace(-10, -2, 200)
    gradients[::17] = 0

    check_on_inputs(
        checker=checker,
        function="lost_to_zero",
        description="gives the share of nonzero gradients that round to 0",
        inputs=(
            (Described(gradients, "200 gradients from 1e-10 to 1e-2, some of them 0"), 1.0, dtype_text(torch.float16)),
            (Described(gradients, "200 gradients from 1e-10 to 1e-2, some of them 0"), 1024.0, dtype_text(torch.float16)),
            (Described(gradients, "200 gradients from 1e-10 to 1e-2, some of them 0"), 1.0, dtype_text(torch.bfloat16)),
        ),
        reference=SCALING_REFERENCE["lost_to_zero"],
        expect=expect_close,
    )

    for seed, scale, poisoned in ((863, 64.0, False), (864, 1024.0, True), (865, 8.0, False)):
        checker.test(
            function="unscale_",
            description="divides every gradient by the scale and says whether all are finite",
            run=lambda found, seed=seed, scale=scale, poisoned=poisoned: check_unscale(
                found=found,
                seed=seed,
                scale=scale,
                poisoned=poisoned,
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


PRECISION2 = Exercise(
    name="precision2",
    chapter=PRECISION,
    title="Autocast and loss scaling",
    lesson=PRECISION2_LESSON,
    source=PRECISION2_SOURCE,
    solution=PRECISION2_SOLUTION,
    hints={
        "lost_to_zero": "Keep the entries of gradients that are not 0. Multiply them by scale, turn them into dtype, "
        "and find the share that equals 0.",
        "unscale_": "Loop over model.parameters(), skip a grad of None, and divide the rest in place with .div_. "
        "torch.isfinite(...).all() says whether one is finite.",
        "update_scale": "Three cases, an inf, a good step that completes growth_interval, and any other good step.",
        "mixed_step": "Only the forward pass and the loss go inside the with block. Backward on loss * scale comes "
        "after it, then unscale_, then optimizer.step() when finite, then optimizer.zero_grad().",
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
) -> Tensor:
    """Every gradient of model on one batch, from a backward pass under autocast to dtype at scale, unscaled."""
    model.zero_grad()

    with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())

    (loss * scale).backward()

    return torch.cat([parameter.grad.flatten() for parameter in model.parameters()]) / scale


def measure_scales() -> dict[str, float]:
    """precision2's table, the GPT gpt3 trains, untrained, on one batch of 32 sequences of Shakespeare, on the
    CPU. Each share is of the gradients float32 gives as nonzero."""
    torch.manual_seed(0)
    model = MODEL_REFERENCE["GPT"](65, 128, 128, 4, 4)
    text = load_shakespeare()
    characters = sorted(set(text))
    positions = {character: position for position, character in enumerate(characters)}
    data = torch.tensor([positions[character] for character in text[:100_000]])
    starts = torch.randint(0, len(data) - 129, (32,), generator=torch.Generator().manual_seed(0))
    x = torch.stack([data[start : start + 128] for start in starts])
    y = torch.stack([data[start + 1 : start + 129] for start in starts])

    exact = gradients_of(model, x, y, torch.float32, 1.0)
    nonzero = exact != 0
    results: dict[str, float] = {"nonzero share": nonzero.float().mean().item()}

    for dtype, scales in ((torch.float16, (1.0, 2.0**8, 2.0**16, 2.0**24)), (torch.bfloat16, (1.0,))):
        for scale in scales:
            mixed = gradients_of(model, x, y, dtype, scale)
            results[f"{dtype} lost at {scale:g}"] = ((mixed == 0) & nonzero).float().sum().item() / nonzero.sum().item()
            results[f"{dtype} not finite at {scale:g}"] = (~torch.isfinite(mixed)).float().mean().item()

    return results
