"""21_normalization, BatchNorm, LayerNorm and RMSNorm built and matched, and what they do for a deep MLP."""

from __future__ import annotations

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
    expect_parts,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .initialization import bell_curve, expect_numbers
from .linear_layers import expect_module, expect_parameter

NORMALIZATION = Chapter(
    folder="21_normalization",
    title="Normalization",
    readme=source_file(
        """
        # Normalization

        A normalization layer resets the spread of the numbers inside a model
        at every step, so a deep model trains from a poor start and with larger
        steps. This chapter builds BatchNorm with its running statistics,
        LayerNorm and RMSNorm, matches each with PyTorch's, and measures what
        they do for a deep MLP.

        ## Further reading

        - [Ioffe and Szegedy, batch normalization](https://arxiv.org/abs/1502.03167)
        - [Ba, Kiros and Hinton, layer normalization](https://arxiv.org/abs/1607.06450)
        - [Zhang and Sennrich, root mean square layer normalization](https://arxiv.org/abs/1910.07467)
        - [Santurkar and others, how does batch normalization help optimization](https://arxiv.org/abs/1805.11604)
        - [Karpathy, building makemore part 3, activations and gradients](https://www.youtube.com/watch?v=P6sfmUTpUmc)
        """,
    ),
)


def with_random_affine(
    module: nn.Module,
    seed: int,
) -> nn.Module:
    """A norm layer whose scale and shift are not 1 and 0, so a check sees a forward that ignores them."""
    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))

    return module


def copy_affine(
    source: nn.Module,
    target: nn.Module,
    call: str,
    size: int,
    names: tuple[str, ...],
) -> None:
    with torch.no_grad():
        for name in names:
            expect_parameter(target, call, name, (size,)).copy_(getattr(source, name))


def check_ones_and_zeros(
    module: nn.Module,
    call: str,
    size: int,
    names: tuple[str, ...],
) -> None:
    starts = {"weight": 1.0, "bias": 0.0}

    for name in names:
        parameter = expect_parameter(module, call, name, (size,)).detach()

        if not torch.equal(parameter, torch.full((size,), starts[name])):
            raise Mismatch(f"The {name} of {call} should start as all {starts[name]:g}, but it holds {parameter}.")


# ---------------------------------------------------------------------------
# batchnorm1
# ---------------------------------------------------------------------------

BATCHNORM1_LESSON = prose(
    """
    The initialization chapter picked starting weights that keep the spread
    steady. The weights move as the model trains, and the spread moves with
    them. **Batch normalization**, BatchNorm, from Ioffe and Szegedy in 2015,
    resets it at every step instead.

    For each feature, it takes the mean and the variance over the examples in
    the batch. It subtracts the mean and divides by the square root of the
    variance. Every feature then has mean 0 and spread 1 within that batch.
    That spread divides by n too, so `.std(dim=0, unbiased=False)` of the
    result shows 1. Plain `.std(dim=0)` divides by `n - 1` and shows more,
    1.41 on 2 rows.

    A small `eps`, 0.00001 in PyTorch, is added to the variance, so a feature
    that never changes doesn't divide by 0. `torch.sqrt` takes the square
    root of every number in a tensor.

    Spread 1 everywhere might not be what the next layer needs. So each
    feature then gets a learned scale and shift, starting at 1 and 0.
    PyTorch calls them `weight` and `bias`, each of shape `(features,)`.

    ```python
    x = torch.tensor([[1.0, 10.0],
                      [3.0, 30.0]])
    x.var(dim=0, unbiased=False)    # tensor([  1., 100.]), the paper's variance, dividing by n
    ```

    `nn.BatchNorm1d(features)` does all this in training mode, on inputs of
    shape `(examples, features)`.

    The paper said it helps by keeping each layer's inputs steady while the
    layers before them change. Later work by Santurkar and others doubted
    that, and pointed to a smoother loss that tolerates larger steps. The
    effect is real either way, and norm1 measures it.
    """,
)

BATCHNORM1_SOURCE = source_file(
    '''
    # batchnorm1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.tensor([[1.0, 10.0],
                      [3.0, 30.0]])
    feature_means = x.mean(dim=0)                    # tensor([ 2., 20.])
    feature_variances = x.var(dim=0, unbiased=False)  # tensor([  1., 100.])


    # TODO: Return the mean and the variance of each feature of x over the
    #   examples, the variance dividing by n.
    def batch_stats(x: Tensor) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Finish the last line, so this gives every feature of x mean 0 and
    #   spread 1 over the batch, with eps added to the variance.
    def batch_normalize(
        x: Tensor,
        eps: float,
    ) -> Tensor:
        mean, variance = batch_stats(x)

        return ...


    # TODO: Finish both methods. weight and bias are parameters of shape
    #   (features,), starting at 1 and 0. forward normalizes x over the batch
    #   with self.eps, then scales by weight and shifts by bias.
    class MyBatchNorm(nn.Module):
        def __init__(
            self,
            features: int,
            eps: float = 1e-5,
        ) -> None:
            super().__init__()
            self.eps = eps
            self.weight = ...
            self.bias = ...

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

BATCHNORM1_SOLUTION = source_file(
    '''
    # batchnorm1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.tensor([[1.0, 10.0],
                      [3.0, 30.0]])
    feature_means = x.mean(dim=0)
    feature_variances = x.var(dim=0, unbiased=False)


    def batch_stats(x: Tensor) -> tuple[Tensor, Tensor]:
        return x.mean(dim=0), x.var(dim=0, unbiased=False)


    def batch_normalize(
        x: Tensor,
        eps: float,
    ) -> Tensor:
        mean, variance = batch_stats(x)

        return (x - mean) / torch.sqrt(variance + eps)


    class MyBatchNorm(nn.Module):
        def __init__(
            self,
            features: int,
            eps: float = 1e-5,
        ) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(features))
            self.bias = nn.Parameter(torch.zeros(features))

        def forward(self, x: Tensor) -> Tensor:
            return batch_normalize(x, self.eps) * self.weight + self.bias
    ''',
)


def check_my_batch_norm(found: type) -> None:
    for features in (3, 5):
        call = f"MyBatchNorm({features})"
        module = expect_module(call, found(features))
        check_ones_and_zeros(module, call, features, ("weight", "bias"))

        reference = with_random_affine(nn.BatchNorm1d(features), 290 + features)
        copy_affine(reference, module, call, features, ("weight", "bias"))
        x = small_tensor(292 + features, 6, features)
        expect_tensor(
            call=f"{call}, with a weight and a bias that aren't 1 and 0, on x of shape (6, {features})",
            actual=module(x),
            expected=reference(x),
        )


def check_batchnorm1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="batch_stats",
        description="gives each feature's mean and variance over the batch",
        inputs=(
            (small_tensor(285, 4, 3),),
            (small_tensor(286, 6, 2),),
        ),
        reference=lambda x: (x.mean(dim=0), x.var(dim=0, unbiased=False)),
        expect=expect_parts("the means", "the variances"),
    )
    check_on_inputs(
        checker=checker,
        function="batch_normalize",
        description="gives each feature mean 0 and spread 1 over the batch",
        inputs=(
            (small_tensor(287, 4, 3), 1e-5),
            (small_tensor(288, 5, 2), 0.1),
        ),
        reference=lambda x, eps: (x - x.mean(dim=0)) / torch.sqrt(x.var(dim=0, unbiased=False) + eps),
        expect=expect_tensor,
    )
    checker.test(
        function="MyBatchNorm",
        description="matches nn.BatchNorm1d in training mode",
        run=check_my_batch_norm,
    )


BATCHNORM1 = Exercise(
    name="batchnorm1",
    chapter=NORMALIZATION,
    title="Batch normalization",
    lesson=BATCHNORM1_LESSON,
    source=BATCHNORM1_SOURCE,
    solution=BATCHNORM1_SOLUTION,
    hints={
        "batch_stats": "Both go down the rows, dim=0. The variance needs unbiased=False to divide by n.",
        "batch_normalize": "Take the mean away from x, then divide by torch.sqrt of the variance plus eps.",
        "MyBatchNorm": "weight is nn.Parameter(torch.ones(features)) and bias the same with torch.zeros. forward "
        "multiplies batch_normalize(x, self.eps) by self.weight and adds self.bias.",
    },
    introduces=(),
    check=check_batchnorm1,
)


# ---------------------------------------------------------------------------
# batchnorm2
# ---------------------------------------------------------------------------

BATCHNORM2_LESSON = prose(
    """
    Batch statistics are a problem once the model is used for real. One
    example alone has no batch to average over. Even in a batch, an answer
    shouldn't change with whatever else happens to be in it.

    So while training, BatchNorm also keeps running averages of each feature's
    mean and variance. After every batch it moves them a fraction `momentum`
    of the way toward the batch's values, 0.1 in PyTorch. In eval mode it
    normalizes with these instead of the batch's.

    ```python
    running = torch.tensor([0.0, 1.0])
    0.9 * running + 0.1 * torch.tensor([2.0, 3.0])    # tensor([0.2000, 1.2000]), a tenth of the way
    ```

    PyTorch's running variance takes the batch variance dividing by `n - 1`,
    which is what `x.var(dim=0)` gives. The batch itself is still normalized
    with the variance dividing by `n`.

    The running averages are not parameters, since no gradient trains them. A
    module stores them with `self.register_buffer(name, tensor)`. A buffer
    moves and saves with the model like a parameter does.

    The running averages lag while the weights still change fast, so early on
    a model can do worse in eval mode. Forgetting `model.eval()` is worse.
    Every prediction then uses its own batch's statistics, and changes the
    running averages too.
    """,
)

BATCHNORM2_SOURCE = source_file(
    '''
    # batchnorm2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    running = torch.tensor([0.0, 1.0])
    batch_mean = torch.tensor([2.0, 3.0])

    # One move of a running average, a tenth of the way toward the batch.
    running = 0.9 * running + 0.1 * batch_mean       # tensor([0.2000, 1.2000])


    # TODO: Return running moved a fraction momentum of the way toward
    #   batch_value.
    def update_running(
        running: Tensor,
        batch_value: Tensor,
        momentum: float,
    ) -> Tensor:
        ...


    # TODO: Finish the three lines with ... in forward. In training mode it
    #   normalizes x with the batch's statistics, the variance dividing by n.
    #   It also updates the running ones, and the running variance takes the
    #   batch variance dividing by n - 1. In eval mode it uses the running ones.
    class MyBatchNorm(nn.Module):
        def __init__(
            self,
            features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
        ) -> None:
            super().__init__()
            self.eps = eps
            self.momentum = momentum
            self.weight = nn.Parameter(torch.ones(features))
            self.bias = nn.Parameter(torch.zeros(features))
            self.register_buffer("running_mean", torch.zeros(features))
            self.register_buffer("running_var", torch.ones(features))

        def forward(self, x: Tensor) -> Tensor:
            if self.training:
                mean = x.mean(dim=0)
                variance = x.var(dim=0, unbiased=False)

                with torch.no_grad():
                    self.running_mean = ...
                    self.running_var = ...
            else:
                mean, variance = ...

            return (x - mean) / torch.sqrt(variance + self.eps) * self.weight + self.bias


    # TODO: This gives a trained model's answers, but it leaves the model in
    #   training mode. Fix it.
    def predict(
        model: nn.Module,
        x: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            return model(x)
    ''',
)

BATCHNORM2_SOLUTION = source_file(
    '''
    # batchnorm2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    running = torch.tensor([0.0, 1.0])
    batch_mean = torch.tensor([2.0, 3.0])

    running = 0.9 * running + 0.1 * batch_mean


    def update_running(
        running: Tensor,
        batch_value: Tensor,
        momentum: float,
    ) -> Tensor:
        return (1 - momentum) * running + momentum * batch_value


    class MyBatchNorm(nn.Module):
        def __init__(
            self,
            features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
        ) -> None:
            super().__init__()
            self.eps = eps
            self.momentum = momentum
            self.weight = nn.Parameter(torch.ones(features))
            self.bias = nn.Parameter(torch.zeros(features))
            self.register_buffer("running_mean", torch.zeros(features))
            self.register_buffer("running_var", torch.ones(features))

        def forward(self, x: Tensor) -> Tensor:
            if self.training:
                mean = x.mean(dim=0)
                variance = x.var(dim=0, unbiased=False)

                with torch.no_grad():
                    self.running_mean = update_running(self.running_mean, mean, self.momentum)
                    self.running_var = update_running(self.running_var, x.var(dim=0), self.momentum)
            else:
                mean, variance = self.running_mean, self.running_var

            return (x - mean) / torch.sqrt(variance + self.eps) * self.weight + self.bias


    # model.eval() makes BatchNorm use its running statistics and leave them alone.
    def predict(
        model: nn.Module,
        x: Tensor,
    ) -> Tensor:
        model.eval()

        with torch.no_grad():
            return model(x)
    ''',
)


def check_running_batch_norm(found: type) -> None:
    features = 3
    call = f"MyBatchNorm({features})"
    module = expect_module(call, found(features))
    reference = with_random_affine(nn.BatchNorm1d(features), 300)
    copy_affine(reference, module, call, features, ("weight", "bias"))

    for step in range(3):
        x = small_tensor(301 + step, 5, features) + step
        expect_tensor(
            call=f"{call} in training mode, on batch {step + 1}",
            actual=module(x),
            expected=reference(x),
        )

        for name in ("running_mean", "running_var"):
            expect_tensor(
                call=f"{name} of {call} after batch {step + 1}",
                actual=getattr(module, name),
                expected=getattr(reference, name),
            )

    module.eval()
    reference.eval()
    x = small_tensor(305, 4, features)
    expect_tensor(
        call=f"{call} in eval mode after 3 batches of training",
        actual=module(x),
        expected=reference(x),
    )


def trained_bn_model(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 3))

    with torch.no_grad():
        for step in range(5):
            model(small_tensor(seed + step, 16, 4) + 1)

    return model


def check_predict(found: object) -> None:
    for seed in (310, 320):
        model = trained_bn_model(seed)
        reference = trained_bn_model(seed)
        before = model[1].running_mean.clone()
        x = small_tensor(seed + 9, 6, 4)
        call = f"predict on a model with BatchNorm, passed in still in training mode, and x of shape (6, 4)"
        actual = found(model, x)
        reference.eval()

        with torch.no_grad():
            expected = reference(x)

        if not torch.equal(model[1].running_mean, before):
            raise Mismatch(f"{call} changed the model's running mean, so the model was still in training mode. Call model.eval() first.")

        expect_tensor(
            call=call,
            actual=actual,
            expected=expected,
        )


def check_batchnorm2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="update_running",
        description="moves the running value a fraction momentum toward the batch's",
        inputs=(
            (torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0]), 0.1),
            (small_tensor(306, 4), small_tensor(307, 4), 0.25),
        ),
        reference=lambda running, batch_value, momentum: (1 - momentum) * running + momentum * batch_value,
        expect=expect_tensor,
    )
    checker.test(
        function="MyBatchNorm",
        description="keeps running statistics like nn.BatchNorm1d and uses them in eval mode",
        run=check_running_batch_norm,
    )
    checker.test(
        function="predict",
        description="answers in eval mode",
        run=check_predict,
    )


BATCHNORM2 = Exercise(
    name="batchnorm2",
    chapter=NORMALIZATION,
    title="Running statistics",
    lesson=BATCHNORM2_LESSON,
    source=BATCHNORM2_SOURCE,
    solution=BATCHNORM2_SOLUTION,
    hints={
        "update_running": "Follow the line that moves running in the file, with momentum in place of 0.1.",
        "MyBatchNorm": "Each running value is update_running of itself, the batch's value and self.momentum. The "
        "variance one uses x.var(dim=0). In eval mode, mean and variance are self.running_mean and self.running_var.",
        "predict": "Call model.eval() before using the model.",
    },
    introduces=(),
    check=check_batchnorm2,
)


# ---------------------------------------------------------------------------
# layernorm1
# ---------------------------------------------------------------------------

LAYERNORM1_LESSON = prose(
    """
    BatchNorm ties each example's answer to the rest of its batch. **Layer
    normalization**, LayerNorm, from Ba, Kiros and Hinton in 2016, normalizes
    each example over its own features instead.

    The mean and the variance are taken along the last dimension, one pair per
    example. So training and real use do the same thing, there are no running
    statistics, and a batch of 1 works. It keeps the learned scale and shift
    per feature, and divides the variance by n.

    `dim=-1` is the last dimension, counted from the end like a Python list
    index. `keepdim=True` keeps a reduced dimension with size 1, so one mean
    per row lines up against the row's features by broadcasting.

    ```python
    x = torch.tensor([[1.0, 3.0],
                      [10.0, 30.0]])
    x.mean(dim=-1, keepdim=True)    # tensor([[ 2.], [20.]]), shape (2, 1)
    ```

    `nn.LayerNorm(size)` matches this, with `weight` and `bias` of shape
    `(size,)` and an `eps` of 0.00001. Language models use LayerNorm or its
    cousin in the next exercise, since their examples are sequences of any
    length.

    `torch.allclose(a, b)` says whether two tensors hold the same numbers, up
    to rounding.
    """,
)

LAYERNORM1_SOURCE = source_file(
    '''
    # layernorm1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.tensor([[1.0, 3.0],
                      [10.0, 30.0]])
    row_means = x.mean(dim=-1, keepdim=True)         # tensor([[ 2.], [20.]])


    # TODO: Return x with each row given mean 0 and spread 1 over its own
    #   features, the variance dividing by n, with eps added to it.
    def layer_normalize(
        x: Tensor,
        eps: float,
    ) -> Tensor:
        ...


    # TODO: Finish both methods. weight and bias are parameters of shape (size,),
    #   starting at 1 and 0. forward normalizes each row with self.eps, then
    #   scales by weight and shifts by bias.
    class MyLayerNorm(nn.Module):
        def __init__(
            self,
            size: int,
            eps: float = 1e-5,
        ) -> None:
            super().__init__()
            self.eps = eps
            self.weight = ...
            self.bias = ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    same = torch.allclose(torch.ones(2), torch.ones(2))    # True


    # TODO: Return whether norm gives the first two rows of x the same answer
    #   on their own as it does within all of x.
    def alone_matches(
        norm: nn.Module,
        x: Tensor,
    ) -> bool:
        ...
    ''',
)

LAYERNORM1_SOLUTION = source_file(
    '''
    # layernorm1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.tensor([[1.0, 3.0],
                      [10.0, 30.0]])
    row_means = x.mean(dim=-1, keepdim=True)


    def layer_normalize(
        x: Tensor,
        eps: float,
    ) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        return (x - mean) / torch.sqrt(variance + eps)


    class MyLayerNorm(nn.Module):
        def __init__(
            self,
            size: int,
            eps: float = 1e-5,
        ) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(size))
            self.bias = nn.Parameter(torch.zeros(size))

        def forward(self, x: Tensor) -> Tensor:
            return layer_normalize(x, self.eps) * self.weight + self.bias


    same = torch.allclose(torch.ones(2), torch.ones(2))


    def alone_matches(
        norm: nn.Module,
        x: Tensor,
    ) -> bool:
        return torch.allclose(norm(x[:2]), norm(x)[:2])
    ''',
)


def check_my_layer_norm(found: type) -> None:
    for size in (4, 6):
        call = f"MyLayerNorm({size})"
        module = expect_module(call, found(size))
        check_ones_and_zeros(module, call, size, ("weight", "bias"))

        reference = with_random_affine(nn.LayerNorm(size), 330 + size)
        copy_affine(reference, module, call, size, ("weight", "bias"))
        x = small_tensor(332 + size, 3, size)
        expect_tensor(
            call=f"{call}, with a weight and a bias that aren't 1 and 0, on x of shape (3, {size})",
            actual=module(x),
            expected=reference(x),
        )


def eval_batch_norm(size: int) -> nn.Module:
    norm = nn.BatchNorm1d(size)
    norm.eval()

    return norm


def check_layernorm1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="layer_normalize",
        description="gives each row mean 0 and spread 1 over its features",
        inputs=(
            (small_tensor(325, 3, 4), 1e-5),
            (small_tensor(326, 2, 3, 5), 0.1),
        ),
        reference=lambda x, eps: F.layer_norm(x, (x.shape[-1],), eps=eps),
        expect=expect_tensor,
    )
    checker.test(
        function="MyLayerNorm",
        description="matches nn.LayerNorm",
        run=check_my_layer_norm,
    )
    x = small_tensor(327, 6, 4)
    check_on_inputs(
        checker=checker,
        function="alone_matches",
        description="says whether rows get the same answer alone as in the batch",
        inputs=(
            (Described(nn.LayerNorm(4), "nn.LayerNorm(4)"), x),
            (Described(nn.BatchNorm1d(4), "nn.BatchNorm1d(4) in training mode"), x),
            (Described(eval_batch_norm(4), "nn.BatchNorm1d(4) in eval mode"), x),
        ),
        reference=lambda norm, x: torch.allclose(norm(x[:2]), norm(x)[:2]),
        expect=expect_value,
    )


LAYERNORM1 = Exercise(
    name="layernorm1",
    chapter=NORMALIZATION,
    title="Layer normalization",
    lesson=LAYERNORM1_LESSON,
    source=LAYERNORM1_SOURCE,
    solution=LAYERNORM1_SOLUTION,
    hints={
        "layer_normalize": "Take the mean and the variance along dim=-1 with keepdim=True, the variance with "
        "unbiased=False. Then normalize as batchnorm1 did.",
        "MyLayerNorm": "The parameters start like MyBatchNorm's. forward multiplies layer_normalize(x, self.eps) by "
        "self.weight and adds self.bias.",
        "alone_matches": "Compare norm(x[:2]) with the first two rows of norm(x), using torch.allclose.",
    },
    introduces=(),
    check=check_layernorm1,
)


# ---------------------------------------------------------------------------
# rmsnorm1
# ---------------------------------------------------------------------------

RMSNORM1_LESSON = prose(
    """
    **RMSNorm**, from Zhang and Sennrich in 2019, drops LayerNorm's mean. It
    divides each row by its root mean square, the square root of the mean of
    its squares, with `eps` inside the root. Then a learned `weight` scales
    each feature, and there is no shift.

    The authors found that taking away the mean did little, and leaving it out
    made their models run 7% to 64% faster. LLaMA and most language models
    since use RMSNorm.

    ```python
    x = torch.tensor([[3.0, 4.0]])
    (x ** 2).mean(dim=-1, keepdim=True)    # tensor([[12.5000]]), the mean square
    ```

    `.sqrt()` takes the square root of every number in a tensor.

    `nn.RMSNorm(size, eps=1e-6)` matches this. Given no `eps`, PyTorch picks a
    tiny one of its own, so pass it.

    When a row already has mean 0, its variance is its mean square. So
    LayerNorm is RMSNorm on the row with its mean taken away, before the scale
    and the shift.
    """,
)

RMSNORM1_SOURCE = source_file(
    '''
    # rmsnorm1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.tensor([[3.0, 4.0]])
    mean_square = (x ** 2).mean(dim=-1, keepdim=True)    # tensor([[12.5000]])


    # TODO: Return the root mean square of each row of x, kept as a column.
    def root_mean_square(x: Tensor) -> Tensor:
        ...


    # TODO: Finish both methods. weight is a parameter of shape (size,) starting
    #   at 1. forward divides each row by its root mean square, with self.eps
    #   added inside the root, then scales by weight.
    class MyRMSNorm(nn.Module):
        def __init__(
            self,
            size: int,
            eps: float = 1e-6,
        ) -> None:
            super().__init__()
            self.eps = eps
            self.weight = ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    layer_normed = F.layer_norm(x, (2,), eps=1e-5)    # tensor([[-1.0000,  1.0000]]), up to eps


    # TODO: Return what F.layer_norm gives with no scale or shift. Take each
    #   row's mean away, then divide by the root mean square of that centered
    #   row, with eps inside the root.
    def layer_norm_by_rms(
        x: Tensor,
        eps: float,
    ) -> Tensor:
        ...
    ''',
)

RMSNORM1_SOLUTION = source_file(
    '''
    # rmsnorm1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.tensor([[3.0, 4.0]])
    mean_square = (x ** 2).mean(dim=-1, keepdim=True)


    def root_mean_square(x: Tensor) -> Tensor:
        return (x ** 2).mean(dim=-1, keepdim=True).sqrt()


    class MyRMSNorm(nn.Module):
        def __init__(
            self,
            size: int,
            eps: float = 1e-6,
        ) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(size))

        def forward(self, x: Tensor) -> Tensor:
            return x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


    layer_normed = F.layer_norm(x, (2,), eps=1e-5)


    # A row with mean 0 has its variance as its mean square.
    def layer_norm_by_rms(
        x: Tensor,
        eps: float,
    ) -> Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)

        return centered / torch.sqrt((centered ** 2).mean(dim=-1, keepdim=True) + eps)
    ''',
)


def check_my_rms_norm(found: type) -> None:
    for size in (4, 6):
        call = f"MyRMSNorm({size})"
        module = expect_module(call, found(size))
        check_ones_and_zeros(module, call, size, ("weight",))

        reference = with_random_affine(nn.RMSNorm(size, eps=1e-6), 340 + size)
        copy_affine(reference, module, call, size, ("weight",))
        x = small_tensor(342 + size, 3, size)
        expect_tensor(
            call=f"{call}, with a weight that isn't 1, on x of shape (3, {size})",
            actual=module(x),
            expected=reference(x),
        )


def check_rmsnorm1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="root_mean_square",
        description="gives each row's root mean square as a column",
        inputs=(
            (torch.tensor([[3.0, 4.0]]),),
            (small_tensor(345, 3, 4),),
        ),
        reference=lambda x: (x ** 2).mean(dim=-1, keepdim=True).sqrt(),
        expect=expect_tensor,
    )
    checker.test(
        function="MyRMSNorm",
        description="matches nn.RMSNorm",
        run=check_my_rms_norm,
    )
    check_on_inputs(
        checker=checker,
        function="layer_norm_by_rms",
        description="matches F.layer_norm by centering, then dividing by the root mean square",
        inputs=(
            (small_tensor(346, 3, 4), 1e-5),
            (small_tensor(347, 2, 5), 0.1),
        ),
        reference=lambda x, eps: F.layer_norm(x, (x.shape[-1],), eps=eps),
        expect=expect_tensor,
    )


RMSNORM1 = Exercise(
    name="rmsnorm1",
    chapter=NORMALIZATION,
    title="RMSNorm",
    lesson=RMSNORM1_LESSON,
    source=RMSNORM1_SOURCE,
    solution=RMSNORM1_SOLUTION,
    hints={
        "root_mean_square": "Square x, take the mean along dim=-1 with keepdim=True, then the square root with "
        ".sqrt().",
        "MyRMSNorm": "weight is nn.Parameter(torch.ones(size)). forward divides x by torch.sqrt of the mean square "
        "plus self.eps, then multiplies by self.weight.",
        "layer_norm_by_rms": "Take each row's mean away first, with keepdim=True. Then divide the result by its own "
        "root mean square, with eps added before the square root.",
    },
    introduces=(),
    check=check_rmsnorm1,
)


# ---------------------------------------------------------------------------
# norm1
# ---------------------------------------------------------------------------

NORM1_LESSON = prose(
    """
    Here MLPs with 8 hidden layers of 64 learned MNIST from 10,000 images for
    3 epochs. Each hidden layer was Linear, then the norm, then ReLU. The table
    gives the validation accuracy at the best of the learning rates 0.03, 0.1
    and 0.3, over three seeds.

    | Normalization | PyTorch's default start | Kaiming start |
    |---|---|---|
    | none | 11% | 84% to 91% |
    | BatchNorm | 90% to 94% | 89% to 93% |
    | LayerNorm | 70% to 84% | 89% to 91% |
    | RMSNorm | 81% to 86% | 84% to 89% |

    With PyTorch's default start the plain MLP never learned, as init3 found.
    Every norm rescued it. From the Kaiming start the gains were smaller, and
    RMSNorm gained nothing at all. BatchNorm did best. From the default start, LayerNorm and RMSNorm fell
    apart at 0.3 in every run, so a norm doesn't make every step size safe. In
    one run with 16 hidden layers, every version did worse than with 8, which
    the next chapter fixes.

    Which one, then?

    - BatchNorm: models of images. It needs `model.eval()`, and goes wrong when batches hold only a few examples.
    - LayerNorm: sequences, as in language models. Each example stands alone, so training and use match.
    - RMSNorm: the same places as LayerNorm, a little cheaper. Most recent language models use it.

    A Linear layer right before BatchNorm doesn't need its bias, since the
    mean subtraction takes it away. `nn.Linear(n, m, bias=False)` leaves it
    out. `nn.Identity()` is a module that hands its input back unchanged.

    A loop over an `nn.Sequential` gives its layers one at a time, in order.
    """,
)

NORM1_SOURCE = source_file(
    '''
    # norm1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    nothing = nn.Identity()
    unchanged = nothing(torch.ones(2))               # tensor([1., 1.])


    # TODO: Return a norm layer for size features, nn.BatchNorm1d for "batch",
    #   nn.LayerNorm for "layer", nn.RMSNorm for "rms", and nn.Identity for
    #   "none".
    def norm_layer(
        kind: str,
        size: int,
    ) -> nn.Module:
        ...


    # TODO: Finish the line in the loop, so each hidden layer is a Linear, then
    #   norm_layer(kind, outputs), then a ReLU. sizes holds every layer's size.
    def normed_mlp(
        sizes: list[int],
        kind: str,
    ) -> nn.Sequential:
        layers = []

        for inputs, outputs in zip(sizes[:-2], sizes[1:-1]):
            layers.extend(...)

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        return nn.Sequential(*layers)


    # TODO: Return the spread of x after each ReLU in model, as a list, passing x
    #   through the model's layers one at a time without recording for autograd.
    def spreads_after_relu(
        model: nn.Sequential,
        x: Tensor,
    ) -> list[float]:
        ...
    ''',
)

NORM1_SOLUTION = source_file(
    '''
    # norm1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    nothing = nn.Identity()
    unchanged = nothing(torch.ones(2))


    def norm_layer(
        kind: str,
        size: int,
    ) -> nn.Module:
        if kind == "batch":
            return nn.BatchNorm1d(size)

        if kind == "layer":
            return nn.LayerNorm(size)

        if kind == "rms":
            return nn.RMSNorm(size)

        return nn.Identity()


    def normed_mlp(
        sizes: list[int],
        kind: str,
    ) -> nn.Sequential:
        layers = []

        for inputs, outputs in zip(sizes[:-2], sizes[1:-1]):
            layers.extend([nn.Linear(inputs, outputs), norm_layer(kind, outputs), nn.ReLU()])

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        return nn.Sequential(*layers)


    def spreads_after_relu(
        model: nn.Sequential,
        x: Tensor,
    ) -> list[float]:
        spreads = []

        with torch.no_grad():
            for layer in model:
                x = layer(x)

                if isinstance(layer, nn.ReLU):
                    spreads.append(x.std().item())

        return spreads
    ''',
)


def norm_summary(module: nn.Module) -> tuple[str, object]:
    if isinstance(module, nn.Linear):
        return "Linear", (module.in_features, module.out_features)

    if isinstance(module, nn.BatchNorm1d):
        return "BatchNorm1d", module.num_features

    if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        return type(module).__name__, tuple(module.normalized_shape)[0]

    return type(module).__name__, None


def summary_text(summary: tuple[str, object]) -> str:
    name, detail = summary

    if detail is None:
        return name

    return f"{name}({detail})" if isinstance(detail, int) else f"{name}{detail}"


def reference_norm_layer(
    kind: str,
    size: int,
) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(size)

    if kind == "layer":
        return nn.LayerNorm(size)

    if kind == "rms":
        return nn.RMSNorm(size)

    return nn.Identity()


def reference_normed_mlp(
    sizes: list[int],
    kind: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []

    for inputs, outputs in zip(sizes[:-2], sizes[1:-1]):
        layers.extend([nn.Linear(inputs, outputs), reference_norm_layer(kind, outputs), nn.ReLU()])

    layers.append(nn.Linear(sizes[-2], sizes[-1]))

    return nn.Sequential(*layers)


def expect_norm(
    call: str,
    actual: object,
    expected: nn.Module,
) -> None:
    module = expect_module(call, actual)

    if norm_summary(module) != norm_summary(expected):
        raise Mismatch(f"{call} is {summary_text(norm_summary(module))}, but it should be {summary_text(norm_summary(expected))}.")


def expect_normed_mlp(
    call: str,
    actual: object,
    expected: nn.Sequential,
) -> None:
    model = expect_module(call, actual)

    if not isinstance(model, nn.Sequential):
        raise Mismatch(f"{call} is a {type(model).__name__}. It should be an nn.Sequential.")

    got = [norm_summary(layer) for layer in model]
    wanted = [norm_summary(layer) for layer in expected]

    if got != wanted:
        raise Mismatch(
            f"{call} should hold {', '.join(summary_text(part) for part in wanted)}, "
            f"but it holds {', '.join(summary_text(part) for part in got)}.",
        )


def reference_spreads_after_relu(
    model: nn.Sequential,
    x: Tensor,
) -> list[float]:
    spreads = []

    with torch.no_grad():
        for layer in model:
            x = layer(x)

            if isinstance(layer, nn.ReLU):
                spreads.append(x.std().item())

    return spreads


def seeded_normed_mlp(
    seed: int,
    sizes: list[int],
    kind: str,
) -> Described:
    torch.manual_seed(seed)

    return Described(reference_normed_mlp(sizes, kind), f"an MLP with sizes {sizes} and {kind} norms")


def check_norm1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="norm_layer",
        description="gives the norm layer for each kind",
        inputs=(
            ("batch", 8),
            ("layer", 16),
            ("rms", 4),
            ("none", 5),
        ),
        reference=reference_norm_layer,
        expect=expect_norm,
    )
    check_on_inputs(
        checker=checker,
        function="normed_mlp",
        description="puts the norm between each hidden Linear and its ReLU",
        inputs=(
            ([784, 64, 64, 10], "batch"),
            ([20, 32, 4], "rms"),
            ([5, 8, 8, 8, 2], "none"),
        ),
        reference=reference_normed_mlp,
        expect=expect_normed_mlp,
    )
    x = Described(bell_curve(350, 200, 30), "x of shape (200, 30) from the bell curve")
    check_on_inputs(
        checker=checker,
        function="spreads_after_relu",
        description="gives the spread after every ReLU",
        inputs=(
            (seeded_normed_mlp(351, [30, 64, 64, 64, 10], "none"), x),
            (seeded_normed_mlp(352, [30, 64, 64, 64, 10], "layer"), x),
        ),
        reference=reference_spreads_after_relu,
        expect=expect_numbers,
    )


NORM1 = Exercise(
    name="norm1",
    chapter=NORMALIZATION,
    title="Why and which",
    lesson=NORM1_LESSON,
    source=NORM1_SOURCE,
    solution=NORM1_SOLUTION,
    hints={
        "norm_layer": "One if per kind, each returning the module called with size, and nn.Identity() when no kind "
        "matched.",
        "normed_mlp": "Each hidden layer adds three modules, nn.Linear(inputs, outputs), norm_layer(kind, outputs) "
        "and nn.ReLU().",
        "spreads_after_relu": "Inside a with torch.no_grad() block, loop over model, set x to layer(x), and append "
        "x.std().item() whenever isinstance says the layer is an nn.ReLU.",
    },
    introduces=(),
    check=check_norm1,
)
