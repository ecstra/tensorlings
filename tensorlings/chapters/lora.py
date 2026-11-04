"""77_lora, Section 11, LoRA from its paper, a frozen model with a small low rank update that starts at zero."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
)
from ..text import prose, source_file
from ..training import best_device
from .finetuning import (
    DATA_PART,
    FINE_TUNE_PARTS,
    FINE_TUNE_REFERENCE,
    GPT_SETTINGS,
    LOAD_PART,
    MODEL_PART,
    VALIDATION_PART,
    indented,
    namespace_of,
    persuasion_training,
    reference_texts,
    reference_trained_gpt,
    small_gpt,
)

LORA = Chapter(
    folder="77_lora",
    title="LoRA",
    readme=source_file(
        """
        # LoRA

        Fine tuning every weight of a model costs a whole copy of it for each
        task. LoRA freezes the model and trains a small update of low rank
        beside each chosen layer, which starts at zero and folds back into the
        layer once trained. You build it from its paper, put it on your GPT,
        and measure when it matches fine tuning everything.

        ## Further reading

        - [Hu and colleagues, LoRA, low rank adaptation of large language models](https://arxiv.org/abs/2106.09685)
        - [Microsoft's loralib, the paper's own code](https://github.com/microsoft/LoRA)
        - [Aghajanyan and colleagues, intrinsic dimensionality explains the effectiveness of language model fine tuning](https://arxiv.org/abs/2012.13255)
        - [Biderman and colleagues, LoRA learns less and forgets less](https://arxiv.org/abs/2405.09673)
        - [Schulman and Thinking Machines, LoRA without regret](https://thinkingmachines.ai/blog/lora/)
        """,
    ),
)

# The LoRA layer, and add_lora, which puts it on the attention of a GPT. 83_post_training is handed both.
LORA_PART = '''
class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
) -> nn.Module:
    model.requires_grad_(False)

    for block in model.h:
        block.attn.c_attn = LoRALinear(block.attn.c_attn, rank, alpha)

    return model
'''

# What lora2 adds, the update's own weights to save, and the update folded back into its layer.
MERGE_PART = '''
def lora_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value for name, value in model.state_dict().items() if name.endswith((".A", ".B"))}


def merge(layer: LoRALinear) -> nn.Linear:
    merged = nn.Linear(layer.base.in_features, layer.base.out_features, bias=layer.base.bias is not None)

    with torch.no_grad():
        merged.weight.copy_(layer.base.weight + layer.scale * layer.B @ layer.A)

        if layer.base.bias is not None:
            merged.bias.copy_(layer.base.bias)

    return merged


def merge_lora(model: nn.Module) -> nn.Module:
    for block in model.h:
        block.attn.c_attn = merge(block.attn.c_attn)

    return model
'''

GIVEN_PARTS = GPT_SETTINGS + MODEL_PART + DATA_PART + VALIDATION_PART + LOAD_PART + FINE_TUNE_PARTS

LORA_REFERENCE = namespace_of(GIVEN_PARTS + LORA_PART + MERGE_PART)


def lora_layer(
    seed: int,
    inputs: int,
    outputs: int,
    rank: int,
    alpha: float,
    bias: bool = True,
) -> nn.Module:
    """A reference LoRA layer whose B is drawn too, so its update is not zero and shows in every output."""
    torch.manual_seed(seed)
    layer = LORA_REFERENCE["LoRALinear"](nn.Linear(inputs, outputs, bias=bias), rank, alpha)

    with torch.no_grad():
        layer.B.normal_(0.0, 0.5)

    return layer


def lora_gpt(
    seed: int,
    rank: int,
) -> nn.Module:
    """A small GPT with LoRA on its attention, every B drawn too."""
    model = LORA_REFERENCE["add_lora"](small_gpt(), rank, 4.0)
    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for block in model.h:
            block.attn.c_attn.B.copy_(torch.randn(block.attn.c_attn.B.shape, generator=generator) * 0.1)

    return model


def small_ids(seed: int) -> Tensor:
    return torch.randint(0, 20, (3, 10), generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# lora1
# ---------------------------------------------------------------------------

LORA1_LESSON = prose(
    """
    finetune2 trained the last block, a quarter of the model. LoRA, from Hu
    and colleagues in 2021, trains far less. It freezes every trained weight
    and learns a small update beside the layers it picks.

    A Linear computes `W x`. LoRA keeps `W` frozen and adds `B A x`. `A` maps
    the input down to `rank` numbers and `B` maps them back up. So `B A` is a
    matrix as big as `W`.

    A matrix's **rank** is how many independent directions its outputs can
    point in. Every output of `B A` passes through `rank` numbers first, so
    its rank is at most `rank`.

    ```text
    out = W x + (alpha / rank) * B A x
    ```

    `B` starts at zero, so the update starts at zero, and the model starts
    exactly as trained. The paper draws `A` from a Gaussian. Here its spread
    is 1 over the square root of its inputs. Then each number of `A x` has
    about the spread of the numbers of `x`.

    ```python
    A, B = torch.ones(2, 3), torch.zeros(4, 2)
    B @ A @ torch.ones(3)    # tensor([0., 0., 0., 0.]), the update starts at zero
    ```

    A zero `B` has a side effect. `A`'s gradient comes back through `B`, so on
    the first step only `B` moves. From the second step on, both train.

    The scale `alpha / rank` cuts the need to retune the rate when the rank
    changes, as the paper says. It sets `alpha` to the first rank it tries
    and never tunes it. Here `alpha` is 4.

    The paper adapts the attention weights only and freezes the MLPs. For
    GPT-3, with a fixed number of weights to spend, splitting them between
    the query and value weights beat spending them all on one. Its own table
    shows all four attention weights at a smaller rank doing as well or
    better. GPT-2's attention makes q, k and v with one Linear, `c_attn`, so
    `add_lora` wraps that one in each block, which adapts all three.

    Freeze the whole model first, then wrap. The new `A` and `B` are made
    after the freeze, so they are the only weights that train.
    """,
)

LORA1_SOURCE = source_file(
    '''
    # lora1.py
    #
    # The last check needs your trained model from gpt3, in checkpoints/gpt3.pt.

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.training import load_checkpoint

    A, B = torch.ones(2, 3), torch.zeros(4, 2)
    update = B @ A @ torch.ones(3)                                   # tensor([0., 0., 0., 0.])
    weights = nn.Parameter(torch.randn(2, 8) / math.sqrt(8))         # a spread of about 0.35


''' + indented(GPT_SETTINGS) + '''


    # The model from gpt2, and your trained one from gpt3.
''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


    # TODO: Finish __init__. Keep base as self.base and freeze it. Make self.A,
    #   rank rows of base.in_features numbers, drawn with torch.randn and divided
    #   by the square root of base.in_features. Make self.B, base.out_features
    #   rows of rank zeros. Both are nn.Parameter. self.scale is alpha / rank.
    class LoRALinear(nn.Module):
        def __init__(
            self,
            base: nn.Linear,
            rank: int,
            alpha: float,
        ) -> None:
            super().__init__()
            ...

        # TODO: Return base of x plus scale times x through A then B.
        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Freeze every parameter of model, then replace each block's
    #   attn.c_attn with a LoRALinear around it. Return model.
    def add_lora(
        model: nn.Module,
        rank: int,
        alpha: float,
    ) -> nn.Module:
        ...
    ''',
)

LORA1_SOLUTION = source_file(
    '''
    # lora1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.training import load_checkpoint

    A, B = torch.ones(2, 3), torch.zeros(4, 2)
    update = B @ A @ torch.ones(3)
    weights = nn.Parameter(torch.randn(2, 8) / math.sqrt(8))


''' + indented(GPT_SETTINGS) + '''


''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


''' + indented(LORA_PART) + '''
    ''',
)


def check_lora_init(found: type) -> None:
    for inputs, outputs, rank, alpha in ((64, 48, 8, 4.0), (128, 96, 2, 6.0)):
        torch.manual_seed(780)
        base = nn.Linear(inputs, outputs)
        layer = found(base, rank, alpha)
        call = f"LoRALinear(nn.Linear({inputs}, {outputs}), {rank}, {alpha})"

        if getattr(layer, "base", None) is not base:
            raise Mismatch(f"{call} should keep the Linear it was given as self.base.")

        if base.weight.requires_grad or base.bias.requires_grad:
            raise Mismatch(f"{call} left base training. Freeze it with base.requires_grad_(False).")

        A, B = getattr(layer, "A", None), getattr(layer, "B", None)

        if not isinstance(A, nn.Parameter) or not isinstance(B, nn.Parameter):
            raise Mismatch(f"{call} should make self.A and self.B as nn.Parameter, so they train.")

        if tuple(A.shape) != (rank, inputs) or tuple(B.shape) != (outputs, rank):
            raise Mismatch(
                f"In {call}, A should be {rank} rows of {inputs} and B {outputs} rows of {rank}, but A is "
                f"{tuple(A.shape)} and B is {tuple(B.shape)}.",
            )

        if torch.any(B != 0):
            raise Mismatch(f"In {call}, B should start as zeros, so the update starts at zero.")

        wanted = 1 / math.sqrt(inputs)
        spread = A.detach().std().item()

        if abs(spread - wanted) > 0.12 * wanted or abs(A.detach().mean().item()) > 0.12 * wanted:
            raise Mismatch(f"In {call}, A has a spread of {spread:.4f}, but it should be drawn from a normal spread of {wanted:.4f}.")

        expect_close(
            call=f"{call}.scale",
            actual=getattr(layer, "scale", None),
            expected=alpha / rank,
        )


def check_lora_forward(found: type) -> None:
    for inputs, outputs, rank, alpha, seed in ((6, 4, 2, 4.0, 781), (5, 7, 3, 1.5, 782)):
        torch.manual_seed(seed)
        layer = found(nn.Linear(inputs, outputs), rank, alpha)
        x = torch.randn(2, 3, inputs, generator=torch.Generator().manual_seed(seed))

        with torch.no_grad():
            start = layer(x)
            expected_start = layer.base(x)
            layer.A.copy_(torch.randn(layer.A.shape, generator=torch.Generator().manual_seed(seed + 100)))
            layer.B.copy_(torch.randn(layer.B.shape, generator=torch.Generator().manual_seed(seed + 200)))
            later = layer(x)
            expected_later = layer.base(x) + (x @ layer.A.T @ layer.B.T) * (alpha / rank)

        call = f"LoRALinear(nn.Linear({inputs}, {outputs}), {rank}, {alpha})"
        expect_tensor(
            call=f"{call} on x of shape (2, 3, {inputs}), as made",
            actual=start,
            expected=expected_start,
        )
        expect_tensor(
            call=f"{call} on x of shape (2, 3, {inputs}), with A and B drawn",
            actual=later,
            expected=expected_later,
        )


def check_first_gradients(found: type) -> None:
    torch.manual_seed(783)
    layer = found(nn.Linear(6, 4), 2, 4.0)
    x = torch.randn(5, 6, generator=torch.Generator().manual_seed(784))
    layer(x).pow(2).sum().backward()

    if layer.base.weight.grad is not None:
        raise Mismatch("The frozen base got a gradient. Freeze it in __init__.")

    if layer.B.grad is None or torch.all(layer.B.grad == 0):
        raise Mismatch("On the first backward pass B should get a gradient, but it got none.")

    if layer.A.grad is not None and torch.any(layer.A.grad != 0):
        raise Mismatch("On the first backward pass A's gradient should be all 0, since it comes back through B, which starts at 0.")


def expect_only_training(
    model: nn.Module,
    wanted: list[str],
    call: str,
) -> None:
    """Names the parameters that train but shouldn't, and the ones that should but don't, so the culprit shows."""
    training = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    extra = sorted(training - set(wanted))
    missing = sorted(set(wanted) - training)

    if extra:
        raise Mismatch(
            f"After {call}, only the new A and B should train, but {', '.join(extra[:4])} train too. Freeze the "
            "whole model with model.requires_grad_(False) before wrapping, and wrap only the Linears the task names.",
        )

    if missing:
        raise Mismatch(
            f"After {call}, every new A and B should train, but {', '.join(missing[:4])} are frozen. Freeze the model "
            "before wrapping, so A and B are made after the freeze.",
        )


def check_add_lora(found: object) -> None:
    for rank in (2, 3):
        model = small_gpt()
        ids = small_ids(785)

        with torch.no_grad():
            expected = model(ids)

        returned = found(model, rank, 4.0)

        if returned is not model:
            raise Mismatch("add_lora should change model in place and return it.")

        for index, block in enumerate(model.h):
            if type(block.attn.c_attn).__name__ != "LoRALinear":
                raise Mismatch(f"After add_lora, block {index}'s attn.c_attn should be a LoRALinear.")

        expect_only_training(
            model=model,
            wanted=[f"h.{index}.attn.c_attn.{name}" for index in range(len(model.h)) for name in ("A", "B")],
            call="add_lora",
        )

        with torch.no_grad():
            actual = model(ids)

        expect_tensor(
            call=f"a small GPT after add_lora(model, {rank}, 4.0), on 3 rows of 10 tokens",
            actual=actual,
            expected=expected,
        )


def check_add_lora_trained(found: object) -> None:
    model = reference_trained_gpt()
    _, _, shakespeare_validation = FINE_TUNE_REFERENCE["shakespeare_splits"]()
    ids = shakespeare_validation[:256].view(2, 128)

    with torch.no_grad():
        expected = model(ids)
        torch.manual_seed(786)
        actual = found(model, 4, 4.0)(ids)

    expect_tensor(
        call="your gpt3 model after add_lora(model, 4, 4.0), on 2 blocks of Shakespeare",
        actual=actual,
        expected=expected,
    )


def check_lora1(checker: Checker) -> None:
    checker.test_method(
        class_name="LoRALinear",
        method="__init__",
        description="freezes base and makes A drawn and B at zero",
        run=check_lora_init,
    )
    checker.test_method(
        class_name="LoRALinear",
        method="forward",
        description="adds the scaled update B A x to base of x",
        run=check_lora_forward,
    )
    checker.test_method(
        class_name="LoRALinear",
        method="forward",
        description="gives A no gradient on the first step",
        run=check_first_gradients,
    )
    checker.test(
        function="add_lora",
        description="wraps every c_attn and leaves only A and B training",
        run=check_add_lora,
    )
    checker.test(
        function="add_lora",
        description="on your gpt3 model, changes nothing at the start",
        run=check_add_lora_trained,
    )


LORA1 = Exercise(
    name="lora1",
    chapter=LORA,
    title="A low rank update that starts at zero",
    lesson=LORA1_LESSON,
    source=LORA1_SOURCE,
    solution=LORA1_SOLUTION,
    hints={
        "LoRALinear.__init__": "base.requires_grad_(False) freezes it. torch.randn(rank, base.in_features) / "
        "math.sqrt(base.in_features) and torch.zeros(base.out_features, rank), each wrapped in nn.Parameter.",
        "LoRALinear.forward": "self.base(x) plus x @ self.A.T @ self.B.T, times self.scale.",
        "add_lora": "First freeze the whole model by passing False to its requires_grad_. Then for each block of "
        "model.h, set block.attn.c_attn to "
        "LoRALinear(block.attn.c_attn, rank, alpha).",
    },
    introduces=(),
    check=check_lora1,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# lora2
# ---------------------------------------------------------------------------

LORA2_LESSON = prose(
    """
    LoRA saves in three places. Here they are for rank 4 on your GPT's
    `c_attn`, against fine tuning every weight.

    | | Everything | LoRA, rank 4 |
    |---|---|---|
    | Weights that train | 818,048 | 8,192 |
    | Gradients, and AdamW's two numbers per weight | 2,454,144 | 24,576 |
    | The saved fine tune | 3.3 MB, a whole model | 36 KB, A and B alone |

    8,192 weights are 1% of the model. Each task keeps only its own `A` and
    `B`, and the frozen model is shared. For GPT-3 the paper's checkpoint
    shrank from 350 GB to 35 MB, roughly 10,000 times.

    `state_dict()` names every weight by where it sits, so the update's
    weights are the names that end in `.A` or `.B`.

    Once trained, the update folds back into its layer. The merged weight is
    `W + (alpha / rank) * B A`, and the bias stays as it was. The merged
    Linear gives the same outputs as the LoRA layer, and the model is gpt2's
    GPT again. The paper notes the same, no extra time at inference.

    To change task, take one update back out of `W` and add another. The
    paper calls this switching tasks by swapping only the LoRA weights.

    Writing into a weight that trains needs `torch.no_grad()`, since autograd
    refuses an in place change to a weight it tracks.

    ```python
    layer = nn.Linear(2, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2))    # the weight is now the identity
    ```
    """,
)

LORA2_SOURCE = source_file(
    '''
    # lora2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.training import load_checkpoint

    names = ["h.0.attn.c_attn.A", "h.0.attn.c_attn.base.weight"]
    picked = [name for name in names if name.endswith((".A", ".B"))]    # ['h.0.attn.c_attn.A']
    plain = nn.Linear(3, 2, bias=False)                                  # plain.bias is None


''' + indented(GPT_SETTINGS) + '''


    # The model from gpt2, and your trained one from gpt3.
''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


    # LoRALinear and add_lora from lora1.
''' + indented(LORA_PART) + '''


    # TODO: Return the entries of model's state_dict whose names end in .A or .B.
    def lora_state(model: nn.Module) -> dict[str, Tensor]:
        ...


    # TODO: Return a new nn.Linear that gives the same outputs as layer. Its
    #   weight is base's weight plus scale times B @ A, and its bias is base's,
    #   or none when base has none. Write them in without recording for autograd,
    #   and leave layer as it was.
    def merge(layer: LoRALinear) -> nn.Linear:
        ...


    # TODO: Replace each block's attn.c_attn, a LoRALinear, with merge of it.
    #   Return model.
    def merge_lora(model: nn.Module) -> nn.Module:
        ...
    ''',
)

LORA2_SOLUTION = source_file(
    '''
    # lora2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.training import load_checkpoint

    names = ["h.0.attn.c_attn.A", "h.0.attn.c_attn.base.weight"]
    picked = [name for name in names if name.endswith((".A", ".B"))]
    plain = nn.Linear(3, 2, bias=False)


''' + indented(GPT_SETTINGS) + '''


''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


''' + indented(LORA_PART) + '''


''' + indented(MERGE_PART) + '''
    ''',
)


def expect_state(
    call: str,
    actual: object,
    expected: dict[str, Tensor],
) -> None:
    if not isinstance(actual, dict) or sorted(actual) != sorted(expected):
        names = sorted(actual)[:4] if isinstance(actual, dict) else actual
        raise Mismatch(f"{call} should hold exactly {', '.join(sorted(expected))}, but it holds {names}.")

    for name, value in expected.items():
        expect_tensor(
            call=f"{call}[{name!r}]",
            actual=actual[name],
            expected=value,
        )


def check_merge(found: object) -> None:
    for inputs, outputs, rank, alpha, bias, seed in ((6, 4, 2, 4.0, True, 787), (5, 9, 3, 1.5, False, 788)):
        layer = lora_layer(seed, inputs, outputs, rank, alpha, bias)
        call = f"merge(LoRALinear(nn.Linear({inputs}, {outputs}, bias={bias}), {rank}, {alpha}) with B drawn)"
        x = torch.randn(2, 3, inputs, generator=torch.Generator().manual_seed(seed))
        base_before = layer.base.weight.detach().clone()

        # Taken before the call, so a merge that changes layer can't change what it is compared with.
        with torch.no_grad():
            expected = layer(x)

        merged = found(layer)

        if type(merged) is not nn.Linear:
            raise Mismatch(f"{call} should be an nn.Linear, but it is {type(merged).__name__}.")

        if not torch.equal(layer.base.weight.detach(), base_before):
            raise Mismatch(f"{call} changed base's weight. Leave layer as it was and write the sum into the new Linear.")

        if (merged.bias is None) != (layer.base.bias is None):
            raise Mismatch(f"{call} should have a bias exactly when base has one.")

        with torch.no_grad():
            actual = merged(x)

        expect_tensor(
            call=f"{call} on x of shape (2, 3, {inputs})",
            actual=actual,
            expected=expected,
        )


def check_merge_lora(found: object) -> None:
    for rank, seed in ((2, 789), (3, 790)):
        model = lora_gpt(seed, rank)
        ids = small_ids(seed)

        with torch.no_grad():
            expected = model(ids)

        merged = found(model)

        if merged is None:
            raise Mismatch("merge_lora should return the model.")

        plain_names = sorted(small_gpt().state_dict())

        if sorted(merged.state_dict()) != plain_names:
            raise Mismatch("After merge_lora the model should hold gpt2's GPT weights alone, with every A and B folded in.")

        with torch.no_grad():
            actual = merged(ids)

        expect_tensor(
            call=f"a small GPT with LoRA of rank {rank}, B drawn, after merge_lora, on 3 rows of 10 tokens",
            actual=actual,
            expected=expected,
        )


def check_lora2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="lora_state",
        description="keeps the A and B of every LoRA layer, nothing else",
        inputs=(
            (Described(lora_gpt(791, 2), "a small GPT with LoRA of rank 2"),),
            (Described(lora_gpt(792, 3), "a small GPT with LoRA of rank 3"),),
        ),
        reference=LORA_REFERENCE["lora_state"],
        expect=expect_state,
    )
    checker.test(
        function="merge",
        description="folds the scaled B A into a plain Linear with the same outputs",
        run=check_merge,
    )
    checker.test(
        function="merge_lora",
        description="leaves gpt2's GPT, giving the same outputs",
        run=check_merge_lora,
    )


LORA2 = Exercise(
    name="lora2",
    chapter=LORA,
    title="Saving the update and merging it back",
    lesson=LORA2_LESSON,
    source=LORA2_SOURCE,
    solution=LORA2_SOLUTION,
    hints={
        "lora_state": "A dict comprehension over model.state_dict().items(), keeping the names that end with "
        "\".A\" or \".B\". endswith takes a tuple.",
        "merge": "Make nn.Linear(layer.base.in_features, layer.base.out_features, bias=layer.base.bias is not None). "
        "Inside torch.no_grad(), copy_ the merged weight in, and base's bias when there is one.",
        "merge_lora": "For each block of model.h, set block.attn.c_attn to merge(block.attn.c_attn).",
    },
    introduces=(),
    check=check_lora2,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# lora3
# ---------------------------------------------------------------------------

EVERYWHERE_SOLUTION = '''
def add_lora_everywhere(
    model: nn.Module,
    rank: int,
    alpha: float,
) -> nn.Module:
    model.requires_grad_(False)

    for block in model.h:
        block.attn.c_attn = LoRALinear(block.attn.c_attn, rank, alpha)
        block.attn.c_proj = LoRALinear(block.attn.c_proj, rank, alpha)
        block.mlp.c_fc = LoRALinear(block.mlp.c_fc, rank, alpha)
        block.mlp.c_proj = LoRALinear(block.mlp.c_proj, rank, alpha)

    return model
'''

RANK_SOLUTION = '''
def best_rank_k(
    delta: Tensor,
    k: int,
) -> Tensor:
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)

    return u[:, :k] @ torch.diag(s[:k]) @ vh[:k]


def update_energy(
    delta: Tensor,
    k: int,
) -> float:
    squares = torch.linalg.svdvals(delta) ** 2

    return (squares[:k].sum() / squares.sum()).item()
'''

LORA_REFERENCE.update(namespace_of(GIVEN_PARTS + LORA_PART + EVERYWHERE_SOLUTION + RANK_SOLUTION))


def measure_lora() -> None:
    """The Persuasion numbers in lora3's tables, 1,000 steps on the best device from the trained gpt3 model, each rate
    held fixed as finetune1 held it. Each run fine tunes 300 steps and then 700 more, a fresh AdamW for each, and
    scores at the end. Takes 5 to 9 minutes on MPS."""
    device = best_device()
    texts = reference_texts()
    train_data = persuasion_training()
    before = LORA_REFERENCE["trained_gpt"]()
    runs = [("everything at 0.0001", lambda: LORA_REFERENCE["trained_gpt"](), 1e-4)]
    runs += [
        (f"LoRA on every Linear, rank {rank}", lambda rank=rank: LORA_REFERENCE["add_lora_everywhere"](LORA_REFERENCE["trained_gpt"](), rank, 4.0), 3e-3)
        for rank in (4, 16)
    ]

    for label, make, lr in runs:
        torch.manual_seed(0)
        model = make().to(device)
        generator = torch.Generator().manual_seed(0)

        for steps in (300, 700):
            LORA_REFERENCE["fine_tune"](model, train_data, steps, lr, generator, device)

        print(label, FINE_TUNE_REFERENCE["trainable_count"](model), LORA_REFERENCE["losses_on"](model, texts, device))

        if label.startswith("everything"):
            model.cpu()

            for index in range(len(model.h)):
                delta = model.h[index].attn.c_attn.weight.detach() - before.h[index].attn.c_attn.weight.detach()
                print(f"block {index}", [round(LORA_REFERENCE["update_energy"](delta, k), 3) for k in (1, 4, 16)])

    noise = torch.randn(384, 128, generator=torch.Generator().manual_seed(0))
    print("random", [round(LORA_REFERENCE["update_energy"](noise, k), 3) for k in (1, 4, 16)])


def pig_latin_run(
    lora: bool,
    seed: int,
    decay: bool = True,
    steps: int = 1000,
) -> tuple[float, list[list[float]]]:
    """Post training's job, from the trained gpt3 model: 83_post_training's pig latin, every demonstration right,
    batches of 64, AdamW with no decay, on the best device. LoRA is rank 4 on every Linear at 0.003, and fine tuning
    everything runs at 0.0001. With decay the rate falls in a line to 0 over the run, and without it the rate stays
    fixed. Gives the share of all 500 held out words answered right, and for fine tuning everything, each block's
    c_attn update energy at the top 1, 4 and 16 directions."""
    from .post_training import POST_REFERENCE, task_setup

    device = best_device()
    characters, positions, train_words, test_words = task_setup()
    torch.manual_seed(seed)
    model = LORA_REFERENCE["trained_gpt"]()
    before = [block.attn.c_attn.weight.detach().clone() for block in model.h]

    if lora:
        LORA_REFERENCE["add_lora_everywhere"](model, 4, 4.0)

    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    lr = 3e-3 if lora else 1e-4
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)

    for step in range(steps):
        if decay:
            optimizer.param_groups[0]["lr"] = lr * (1 - step / steps)

        picked = [train_words[place] for place in torch.randint(0, len(train_words), (64,), generator=generator).tolist()]
        rows = [POST_REFERENCE["chat_example"](word, POST_REFERENCE["pig_latin"](word), positions) for word in picked]
        x, y = POST_REFERENCE["pad_examples"](rows, positions["\n"])
        POST_REFERENCE["sft_step"](model, optimizer, x, y, device)

    right, _ = POST_REFERENCE["scores"](model, test_words, positions, characters, device)
    energies = []

    if not lora:
        model.cpu()
        energies = [
            [LORA_REFERENCE["update_energy"](block.attn.c_attn.weight.detach() - old, k) for k in (1, 4, 16)]
            for block, old in zip(model.h, before)
        ]

    return right, energies


def measure_pig_latin(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, list[float]]:
    """lora3's first table, LoRA against fine tuning everything on post training's task, over seeds, with the rate
    falling to 0 and with it held fixed, and the update energies of the decayed fine tune of everything. Threads
    pinned to 4. 12 runs of 2 to 4 minutes each on MPS, about 45 minutes on a busy laptop."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    results: dict[str, list[float]] = {}

    try:
        for seed in seeds:
            for decay in (True, False):
                schedule = "falling" if decay else "fixed"

                for lora in (False, True):
                    name = f"{'LoRA' if lora else 'everything'}, rate {schedule}"
                    right, energies = pig_latin_run(lora, seed, decay)
                    results.setdefault(name, []).append(right)
                    shown = [[round(share, 3) for share in block] for block in energies]
                    print(name, seed, right, shown if decay and not lora else "", flush=True)
    finally:
        torch.set_num_threads(threads)

    return results


LORA3_LESSON = prose(
    """
    lora2 counted what LoRA saves. Here is what it costs on post training,
    the job it is made for. Post training teaches a trained model a task or a
    format from examples, called demonstrations. The task is
    83_post_training's pig latin. A word that starts with a vowel adds `way`,
    so `apple` becomes `appleway`. Any other moves its leading consonants to
    the end and adds `ay`, so `string` becomes `ingstray`. Each run started
    from your gpt3 model and took 1,000 steps of 64 right demonstrations on
    MPS, the rate falling in a line to 0.

    | Trained | Weights | Rate | Right of 500 held out words, 3 seeds |
    |---|---|---|---|
    | everything | 818,048 | 0.0001 | 97.8%, 98%, 97% |
    | LoRA on every Linear, rank 4 | 32,768 | 0.003 | 98%, 99.2%, 99% |

    With 4% of the weights, LoRA matches fine tuning everything. Thinking
    Machines saw the same on small and medium post training sets, with LoRA
    on every layer. Its best rate was about 10 times higher, here 30. Held
    fixed, that rate cost LoRA, 95% to 98% against everything's 98.2% to 99.2%.

    Learning a whole new text is a different job. Here is Persuasion from
    finetune1 after 1,000 steps at the same rates, held fixed as there.

    | Trained | Weights | Persuasion's loss |
    |---|---|---|
    | nothing | 0 | 1.744 |
    | everything | 818,048 | 1.260 |
    | LoRA on every Linear, rank 4 | 32,768 | 1.334 |
    | LoRA on every Linear, rank 16 | 131,072 | 1.279 |

    Biderman and colleagues found full fine tuning ahead on large code and
    math sets too. Thinking Machines' rule joins the two. LoRA matches while
    its weights can hold what the data teaches. So use LoRA to teach a format
    or a skill, and fine tune everything to learn a large new text.

    A matrix's **singular values** say how far it stretches along each of
    its directions, largest first. Their squares add up to the sum of every
    entry squared. Keeping the top k gives the closest matrix of rank k. The
    **singular value decomposition**, SVD, writes a matrix as
    `u @ torch.diag(s) @ vh`. `s` holds the singular values, `u`'s columns
    the output directions and `vh`'s rows the input ones.

    ```python
    torch.linalg.svdvals(torch.tensor([[3.0, 0.0], [0.0, 1.0]]))    # tensor([3.0000, 1.0000])
    ```

    `update_energy` gives the share of an update its top k directions hold.
    Here it is for each block's 384 by 128 `c_attn` change from fine tuning
    everything, beside a random matrix.

    | Matrix | Top 4, pig latin | Top 4, Persuasion | Top 16, pig latin | Top 16, Persuasion |
    |---|---|---|---|---|
    | block 0's update | 50% to 51% | 31% | 84% to 85% | 73% |
    | block 1's update | 51% to 56% | 53% | 86% to 88% | 84% |
    | block 2's update | 48% to 52% | 28% | 83% to 85% | 69% |
    | block 3's update | 43% to 45% | 17% | 77% to 78% | 48% |
    | random | 7% | 7% | 26% | 26% |

    Both sit in far fewer directions than random ones. A change that few
    directions hold has a low **intrinsic rank**, LoRA's bet. The post
    training update sits in fewer still, so a low rank catches it. On large
    sets, Biderman and colleagues found full fine tuning's rank 10 to 100
    times LoRA's.
    """,
)

LORA3_SOURCE = source_file(
    '''
    # lora3.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.training import load_checkpoint

    values = torch.linalg.svdvals(torch.tensor([[3.0, 0.0], [0.0, 1.0]]))    # tensor([3.0000, 1.0000])
    u, s, vh = torch.linalg.svd(torch.eye(2), full_matrices=False)          # torch.eye(2) is u @ torch.diag(s) @ vh


''' + indented(GPT_SETTINGS) + '''


    # The model from gpt2, and your trained one from gpt3.
''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


    # LoRALinear and add_lora from lora1.
''' + indented(LORA_PART) + '''


    # TODO: Freeze every parameter of model, then wrap each of the four Linears
    #   of every block, attn.c_attn, attn.c_proj, mlp.c_fc and mlp.c_proj, once
    #   in a LoRALinear. Return model.
    def add_lora_everywhere(
        model: nn.Module,
        rank: int,
        alpha: float,
    ) -> nn.Module:
        ...


    # TODO: Return the closest matrix of rank k to delta. Take its SVD with
    #   full_matrices=False and keep the first k columns of u, values of s and
    #   rows of vh.
    def best_rank_k(
        delta: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # TODO: Return the share of delta's singular values squared that its top k
    #   hold, as a plain number.
    def update_energy(
        delta: Tensor,
        k: int,
    ) -> float:
        ...
    ''',
)

LORA3_SOLUTION = source_file(
    '''
    # lora3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.training import load_checkpoint

    values = torch.linalg.svdvals(torch.tensor([[3.0, 0.0], [0.0, 1.0]]))
    u, s, vh = torch.linalg.svd(torch.eye(2), full_matrices=False)


''' + indented(GPT_SETTINGS) + '''


''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


''' + indented(LORA_PART) + '''


''' + indented(EVERYWHERE_SOLUTION) + '''


''' + indented(RANK_SOLUTION) + '''
    ''',
)


LINEAR_NAMES = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


def linears_of(block: nn.Module) -> dict[str, nn.Module]:
    return {name: block.get_submodule(name) for name in LINEAR_NAMES}


def check_wrapped(
    layer: nn.Module,
    original: nn.Linear,
    where: str,
    rank: int,
) -> None:
    if type(layer).__name__ != "LoRALinear":
        raise Mismatch(f"After add_lora_everywhere, {where} should be a LoRALinear, but it is {type(layer).__name__}.")

    if layer.base is not original:
        raise Mismatch(f"After add_lora_everywhere, {where} should wrap the Linear that sat there, once.")

    if tuple(layer.A.shape) != (rank, original.in_features):
        raise Mismatch(
            f"After add_lora_everywhere(model, {rank}, 4.0), {where}'s A has shape {tuple(layer.A.shape)}, but it "
            f"should be ({rank}, {original.in_features}), rank rows.",
        )

    if torch.any(layer.B != 0):
        raise Mismatch(f"After add_lora_everywhere, {where}'s B should start at zero.")

    expect_close(
        call=f"{where}'s scale after add_lora_everywhere(model, {rank}, 4.0)",
        actual=layer.scale,
        expected=4.0 / rank,
    )


def check_add_lora_everywhere(found: object) -> None:
    """By structure, so the four Linears may be wrapped in any order. Each wraps the Linear that sat there, at the
    given rank and scale, with B at zero and A drawn with lora1's spread."""
    for rank in (1, 3):
        model = small_gpt()
        ids = small_ids(793)
        originals = [linears_of(block) for block in model.h]

        with torch.no_grad():
            expected = model(ids)

        torch.manual_seed(794)
        returned = found(model, rank, 4.0)

        if returned is not model:
            raise Mismatch("add_lora_everywhere should change model in place and return it.")

        scaled_draws = []

        for index, block in enumerate(model.h):
            for name, layer in linears_of(block).items():
                original = originals[index][name]
                check_wrapped(
                    layer=layer,
                    original=original,
                    where=f"block {index}'s {name}",
                    rank=rank,
                )
                scaled_draws.append(layer.A.detach().flatten() * math.sqrt(original.in_features))

        expect_only_training(
            model=model,
            wanted=[f"h.{index}.{name}.{part}" for index in range(len(model.h)) for name in LINEAR_NAMES for part in ("A", "B")],
            call="add_lora_everywhere",
        )

        # Each A times the square root of its inputs should look like plain normal draws, pooled over every layer.
        scaled = torch.cat(scaled_draws)

        if abs(scaled.std().item() - 1) > 0.15 or abs(scaled.mean().item()) > 0.15:
            raise Mismatch("After add_lora_everywhere, the A's aren't drawn as lora1 drew them, normal draws divided by the square root of the inputs.")

        with torch.no_grad():
            actual = model(ids)

        expect_tensor(
            call=f"a small GPT after add_lora_everywhere(model, {rank}, 4.0), on 3 rows of 10 tokens",
            actual=actual,
            expected=expected,
        )


def random_matrix(
    seed: int,
    rows: int,
    columns: int,
) -> Tensor:
    return torch.randn(rows, columns, generator=torch.Generator().manual_seed(seed))


def low_rank_plus_noise(seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)

    return torch.randn(9, 2, generator=generator) @ torch.randn(2, 6, generator=generator) + 0.1 * torch.randn(9, 6, generator=generator)


def check_lora3(checker: Checker) -> None:
    checker.test(
        function="add_lora_everywhere",
        description="wraps the four Linears of each block and leaves only A and B training",
        run=check_add_lora_everywhere,
    )
    check_on_inputs(
        checker=checker,
        function="best_rank_k",
        description="keeps the top k directions of the SVD",
        inputs=(
            (Described(low_rank_plus_noise(795), "a 9 by 6 matrix near rank 2"), 2),
            (Described(random_matrix(796, 5, 8), "a random 5 by 8 matrix"), 3),
            (Described(random_matrix(797, 4, 4), "a random 4 by 4 matrix"), 4),
        ),
        reference=LORA_REFERENCE["best_rank_k"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="update_energy",
        description="gives the share the top k singular values squared hold",
        inputs=(
            (Described(low_rank_plus_noise(798), "a 9 by 6 matrix near rank 2"), 2),
            (Described(random_matrix(799, 5, 8), "a random 5 by 8 matrix"), 1),
            (Described(random_matrix(800, 6, 3), "a random 6 by 3 matrix"), 3),
        ),
        reference=LORA_REFERENCE["update_energy"],
        expect=expect_close,
    )


LORA3 = Exercise(
    name="lora3",
    chapter=LORA,
    title="When a low rank is enough",
    lesson=LORA3_LESSON,
    source=LORA3_SOURCE,
    solution=LORA3_SOLUTION,
    hints={
        "add_lora_everywhere": "model.requires_grad_(False), then for each block of model.h set each of the four "
        "Linears to a LoRALinear around itself, as add_lora did for c_attn.",
        "best_rank_k": "torch.linalg.svd(delta, full_matrices=False) gives u, s and vh. Multiply u[:, :k], "
        "torch.diag(s[:k]) and vh[:k].",
        "update_energy": "Square torch.linalg.svdvals(delta). Divide the sum of the first k by the sum of them all, "
        "then .item().",
    },
    introduces=(),
    check=check_lora3,
    time_limit=30,
)
