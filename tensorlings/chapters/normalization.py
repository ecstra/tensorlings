"""21_normalization, BatchNorm, LayerNorm and RMSNorm built and matched, and what they do for a deep MLP."""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Iterator

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
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .initialization import bell_curve, expect_numbers
from .linear_layers import expect_module, expect_parameter
from .mnist import mnist_sets

NORMALIZATION = Chapter(
    folder="21_normalization",
    title="Normalization",
    readme=source_file(
        """
        # Normalization

        A normalization layer brings the numbers inside a model back to a
        standard scale at every step. This lets a deep model train even from
        poor starting weights, and with larger steps. In this chapter you build
        BatchNorm with its running statistics, then LayerNorm and RMSNorm. You
        check each one against PyTorch's version, and you measure what they do
        for a deep MLP.

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


def expect_spreads(
    call: str,
    actual: object,
    expected: list[float],
) -> None:
    """expect_numbers within 1e-3 of each spread. On the few thousand numbers a check passes, a spread that divides by
    n, as batchnorm1's did, lands that close to plain .std(), which divides by n - 1, so either passes."""
    if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
        expect_numbers(
            call=call,
            actual=actual,
            expected=expected,
        )
        return

    for position, (got, wanted) in enumerate(zip(actual, expected)):
        expect_close(
            call=f"number {position + 1} of {call}",
            actual=got,
            expected=wanted,
            tolerance=1e-3 * abs(wanted),
        )


# ---------------------------------------------------------------------------
# batchnorm1
# ---------------------------------------------------------------------------

BATCHNORM1_LESSON = prose(
    """
    The initialization chapter picked starting weights that keep the standard
    deviation of the numbers steady from layer to layer. But the weights
    change as the model trains, and the standard deviation changes with them.
    **Batch normalization**, or BatchNorm, from Ioffe and Szegedy in 2015,
    fixes the standard deviation again at every step.

    For each feature, BatchNorm computes the mean and the variance over the
    examples in the batch. It subtracts the mean and divides by the square
    root of the variance. Then every feature has mean 0 and standard deviation
    1 within that batch. BatchNorm's variance divides the sum of squared
    differences by `n`, the number of examples. So
    `.std(dim=0, unbiased=False)`, which also divides by `n`, shows 1 for the
    result. Plain `.std(dim=0)` divides by `n - 1` instead, so it shows more,
    1.41 on 2 rows.

    A small number `eps`, 0.00001 in PyTorch, is added to the variance before
    the square root. Without it, a feature with the same value in every
    example would have variance 0, and BatchNorm would divide by 0.
    `torch.sqrt` takes the square root of every number in a tensor.

    Mean 0 and standard deviation 1 for every feature might not be what the
    next layer needs. So BatchNorm then multiplies each feature by a learned
    scale and adds a learned shift. They start at 1 and 0, so at first they
    change nothing. PyTorch calls them `weight` and `bias`, each of shape
    `(features,)`.

    ```python
    x = torch.tensor([[1.0, 10.0],
                      [3.0, 30.0]])
    x.var(dim=0, unbiased=False)    # tensor([  1., 100.]), the paper's variance, dividing by n
    ```

    `nn.BatchNorm1d(features)` does all this in training mode, on inputs of
    shape `(examples, features)`.

    The paper said BatchNorm helps by keeping each layer's inputs steady while
    the layers before it change. Later work by Santurkar and others doubted
    that explanation. They argued that BatchNorm makes the loss smoother, so
    training can take larger steps. Either way the effect is real. norm1
    measures it as faster training and larger usable learning rates.
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
    #   examples, as two tensors of shape (features,). x has shape
    #   (examples, features), and the variance divides by n, the number of
    #   examples. Example: batch_stats(x) returns the same two tensors as
    #   feature_means and feature_variances above.
    def batch_stats(x: Tensor) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Finish the last line, so every feature of x has mean 0 and standard
    #   deviation 1 over the batch. Subtract the mean, then divide by the square
    #   root of the variance plus eps. Example: batch_normalize(x, 1e-5) returns
    #   about tensor([[-1., -1.], [ 1.,  1.]]).
    def batch_normalize(
        x: Tensor,
        eps: float,
    ) -> Tensor:
        mean, variance = batch_stats(x)

        return ...


    # TODO: Finish both methods. weight and bias are parameters of shape
    #   (features,), with weight starting as all 1s and bias as all 0s. forward
    #   normalizes x over the batch with self.eps, then multiplies by weight and
    #   adds bias.
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
    # eps of 1e-5 barely moves the answer, so a large one shows where self.eps goes.
    call = "MyBatchNorm(3, eps=0.1)"
    module = expect_module(call, found(3, eps=0.1))
    reference = nn.BatchNorm1d(3, eps=0.1)
    x = small_tensor(296, 6, 3)
    expect_tensor(
        call=f"{call} on x of shape (6, 3), which should add self.eps to the variance inside the root",
        actual=module(x),
        expected=reference(x),
    )

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
        description="gives each feature mean 0 and standard deviation 1 over the batch",
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
        "batch_stats": "Compute both down the rows, with dim=0. The variance needs unbiased=False to divide by n.",
        "batch_normalize": "Subtract the mean from x, then divide by torch.sqrt of the variance plus eps.",
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
    Batch statistics cause a problem once the model is used to make
    predictions. A single example has no batch to average over. And even in a
    batch, the prediction for one example should not depend on which other
    examples happen to be in it.

    So while training, BatchNorm also keeps a running average of each
    feature's mean and variance. After every batch it moves each running
    average a fraction `momentum` of the way toward the batch's value.
    PyTorch's `momentum` is 0.1. In eval mode, BatchNorm normalizes with these
    running averages instead of the batch's statistics.

    ```python
    running = torch.tensor([0.0, 1.0])
    0.9 * running + 0.1 * torch.tensor([2.0, 3.0])    # tensor([0.2000, 1.2000]), a tenth of the way
    ```

    For the running variance, PyTorch uses the batch variance that divides by
    `n - 1`, which is what `x.var(dim=0)` gives. The batch itself is still
    normalized with the variance that divides by `n`.

    The running averages are not parameters, because no gradient trains them.
    A module stores them with `self.register_buffer(name, tensor)` instead.
    Such a tensor is called a buffer. When you move the model to another
    device or save it, its buffers go with it, like its parameters do.

    Early in training the weights still change fast, and the running averages
    lag behind them. So a model can do worse in eval mode early on. Forgetting
    `model.eval()` is worse. Then every prediction uses the statistics of its
    own batch, and also changes the running averages.
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

    # One update of a running average. It moves a tenth of the way toward the
    # batch's mean.
    running = 0.9 * running + 0.1 * batch_mean       # tensor([0.2000, 1.2000])


    # TODO: Return running moved a fraction momentum of the way toward
    #   batch_value, as the line above does with momentum 0.1. Example:
    #   update_running(torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0]), 0.1)
    #   returns tensor([0.2000, 1.2000]).
    def update_running(
        running: Tensor,
        batch_value: Tensor,
        momentum: float,
    ) -> Tensor:
        ...


    # TODO: Finish the three lines with ... in forward. In training mode,
    #   forward normalizes x with the batch's mean and variance, where the
    #   variance divides by n. It also updates running_mean and running_var
    #   with update_running and self.momentum. running_var is updated with the
    #   batch variance that divides by n - 1. In eval mode, forward normalizes
    #   with running_mean and running_var instead.
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


    # TODO: This returns a trained model's outputs on x, but it does not switch
    #   the model out of training mode. Fix it.
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


    # model.eval() makes BatchNorm use its running statistics and stop updating them.
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
        description="runs the model in eval mode",
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
        "update_running": "Copy the pattern of the line that updates running in the file, using momentum instead of "
        "0.1 and 1 - momentum instead of 0.9.",
        "MyBatchNorm": "Set each running value to update_running of itself, the batch's value and self.momentum. For "
        "running_var, the batch's value is x.var(dim=0). In eval mode, mean and variance are self.running_mean and "
        "self.running_var.",
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
    With BatchNorm, the output for each example depends on the other examples
    in its batch. **Layer normalization**, or LayerNorm, from Ba, Kiros and
    Hinton in 2016, avoids that. It normalizes each example over its own
    features instead.

    LayerNorm computes the mean and the variance along the last dimension, so
    there is one mean and one variance per example. This means training and
    prediction do the same thing. There are no running statistics, and a
    batch of 1 example works. LayerNorm keeps BatchNorm's learned scale and
    shift per feature. Its variance also divides by `n`, which here is the
    number of features.

    `dim=-1` means the last dimension, counted from the end like a negative
    Python list index. `keepdim=True` keeps the reduced dimension with size 1.
    So the mean of each row has shape `(rows, 1)`, and broadcasting subtracts
    it from every feature of that row.

    ```python
    x = torch.tensor([[1.0, 3.0],
                      [10.0, 30.0]])
    x.mean(dim=-1, keepdim=True)    # tensor([[ 2.], [20.]]), shape (2, 1)
    ```

    `nn.LayerNorm(size)` does the same, with `weight` and `bias` of shape
    `(size,)` and an `eps` of 0.00001. Language models use LayerNorm, or the
    similar RMSNorm from the next exercise, because their examples are
    sequences of any length.

    `torch.allclose(a, b)` returns `True` when two tensors hold the same
    numbers, up to small rounding differences.
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


    # TODO: Return x normalized over its last dimension, so each row has mean 0
    #   and standard deviation 1 over its own features. The variance divides by
    #   n, and eps is added to it before the square root. x can have any number
    #   of dimensions. Example: layer_normalize(x, 1e-5) returns about
    #   tensor([[-1.,  1.], [-1.,  1.]]).
    def layer_normalize(
        x: Tensor,
        eps: float,
    ) -> Tensor:
        ...


    # TODO: Finish both methods. weight and bias are parameters of shape (size,),
    #   with weight starting as all 1s and bias as all 0s. forward normalizes
    #   each row with self.eps, then multiplies by weight and adds bias.
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


    # TODO: Return whether norm gives the first two rows of x the same output
    #   when it runs on those two rows alone as when it runs on all of x.
    #   Example: for nn.LayerNorm(4) and x of shape (6, 4), it returns True.
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
    # eps of 1e-5 barely moves the answer, so a large one shows where self.eps goes.
    call = "MyLayerNorm(4, eps=0.1)"
    module = expect_module(call, found(4, eps=0.1))
    reference = nn.LayerNorm(4, eps=0.1)
    x = small_tensor(338, 3, 4)
    expect_tensor(
        call=f"{call} on x of shape (3, 4), which should add self.eps to the variance inside the root",
        actual=module(x),
        expected=reference(x),
    )

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
        description="gives each row mean 0 and standard deviation 1 over its features",
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
        description="says whether rows get the same output alone as in the batch",
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
        "layer_normalize": "Compute the mean and the variance along dim=-1 with keepdim=True, and give the variance "
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
    **RMSNorm**, from Zhang and Sennrich in 2019, is LayerNorm without the
    mean. It does not subtract the mean of each row. Instead it divides each
    row by its root mean square, which is the square root of the mean of the
    row's squares. `eps` is added inside the square root. Then a learned
    `weight` scales each feature. The paper also kept a learned shift.
    PyTorch's `nn.RMSNorm` and the LLaMA models leave the shift out, and so
    does this exercise.

    The authors found that subtracting the mean made little difference to how
    well their models trained. Leaving it out cut their models' running time
    by 7% to 64%. LLaMA and most language models since then use RMSNorm.

    ```python
    x = torch.tensor([[3.0, 4.0]])
    (x ** 2).mean(dim=-1, keepdim=True)    # tensor([[12.5000]]), the mean square
    ```

    `.sqrt()` takes the square root of every number in a tensor.

    `nn.RMSNorm(size, eps=1e-6)` does the same. If you give it no `eps`,
    PyTorch picks its own tiny value, so pass `eps` to match.

    Written the way you write them here and in layernorm1, RMSNorm does less
    work than LayerNorm. Over six timings on 4096 rows of 1024 numbers,
    MyRMSNorm's forward ran 3.6 to 7.7 times as fast as MyLayerNorm's. The
    forward and backward passes together ran 1.4 to 1.9 times as fast.

    When a row already has mean 0, its variance equals its mean square. So
    LayerNorm, before its scale and shift, is the same as RMSNorm applied to
    the row after subtracting its mean. `F.layer_norm(x, (size,), eps=eps)` is
    LayerNorm with no scale or shift, over a last dimension of length `size`.
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


    # TODO: Return the root mean square of each row of x, the square root of the
    #   mean of the row's squares. Keep the reduced dimension, so x of shape
    #   (rows, size) gives a column of shape (rows, 1).
    #   Example: root_mean_square(x) returns tensor([[3.5355]]).
    def root_mean_square(x: Tensor) -> Tensor:
        ...


    # TODO: Finish both methods. weight is a parameter of shape (size,) that
    #   starts as all 1s. forward divides each row by its root mean square, with
    #   self.eps added to the mean square inside the square root. Then it
    #   multiplies by weight.
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


    # TODO: Return the same result as F.layer_norm(x, (x.shape[-1],), eps=eps),
    #   computed with a root mean square. Subtract each row's mean, then divide
    #   the result by its own root mean square, with eps added inside the
    #   square root.
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


    # For a row with mean 0, the variance equals the mean square.
    def layer_norm_by_rms(
        x: Tensor,
        eps: float,
    ) -> Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)

        return centered / torch.sqrt((centered ** 2).mean(dim=-1, keepdim=True) + eps)
    ''',
)


def check_my_rms_norm(found: type) -> None:
    # eps of 1e-6 barely moves the answer, so a large one shows where self.eps goes.
    call = "MyRMSNorm(4, eps=0.1)"
    module = expect_module(call, found(4, eps=0.1))
    reference = nn.RMSNorm(4, eps=0.1)
    x = small_tensor(348, 3, 4)
    expect_tensor(
        call=f"{call} on x of shape (3, 4), which should add self.eps to the mean square inside the root",
        actual=module(x),
        expected=reference(x),
    )

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
        "layer_norm_by_rms": "Subtract each row's mean first, computed with keepdim=True. Then divide the result by "
        "its own root mean square, with eps added before the square root.",
    },
    introduces=(),
    check=check_rmsnorm1,
)


# ---------------------------------------------------------------------------
# norm1
# ---------------------------------------------------------------------------

NORM1_LESSON = prose(
    """
    In this experiment, MLPs with 8 hidden layers of 64 neurons learned MNIST
    from PyTorch's default starting weights, on three seeds. Each hidden layer
    was a Linear layer, then the norm, then a ReLU. Each cell is the
    validation accuracy. For the first two columns, the models trained on
    50,000 images for 2 epochs in batches of 128. Each norm used the learning
    rate from 0.03, 0.1, 0.3 and 1.0 that trained best for it. That was 0.3
    for BatchNorm, 0.03 for LayerNorm and 0.1 for RMSNorm. For the last
    column, the models trained on 10,000 images for 1 epoch in batches of 4,
    at a learning rate of 0.01.

    | Normalization | Batch 128, after 100 steps | Batch 128, after 2 epochs | Batch 4, after 1 epoch |
    |---|---|---|---|
    | none | 11% | 11% | 11% |
    | BatchNorm | 90% to 92% | 96.5% to 96.8% | 53% to 56% |
    | LayerNorm | 66% to 84% | 94.6% to 95.4% | 76% to 89% |
    | RMSNorm | 66% to 70% | 94.6% to 95.0% | 82% to 86% |

    The MLP without a norm never learned, as init3 found, and every norm fixed
    that. Each norm also showed the strength its paper claims.

    - BatchNorm: the fastest, past 90% in 100 steps, and stable at large learning rates.
    - LayerNorm and RMSNorm: independent of the batch, since each example is normalized on its own.

    BatchNorm reached 95.6% or more at every learning rate from 0.03 to 1.0.
    Starting from Kaiming's weights, the MLP without a norm turned `nan` at a
    learning rate of 1.0. BatchNorm still reached 96% at 3.0. In batches of 4,
    BatchNorm's statistics come from only 4 examples, and its accuracy fell to
    about half. LayerNorm and RMSNorm kept most of their accuracy there. They
    train best at small learning rates, 0.03 to 0.1 here.

    You can see why the norms help in the standard deviation of the numbers
    after each ReLU, which `spreads_after_relu` measures. The next table shows
    it after the first and the eighth ReLU, before any training, on 1000
    images.

    | Normalization | After the first ReLU | After the eighth |
    |---|---|---|
    | none | 0.10 to 0.13 | 0.040 to 0.045 |
    | BatchNorm | 0.58 | 0.59 to 0.60 |
    | LayerNorm | 0.57 to 0.60 | 0.54 to 0.62 |
    | RMSNorm | 0.55 to 0.60 | 0.53 to 0.59 |

    Without a norm the standard deviation shrinks layer after layer, so the
    last layers get almost no signal from the input. With any of the three
    norms it stays steady at every depth.

    So which norm should you use?

    - BatchNorm: models of images. It needs `model.eval()`, and it works badly when batches hold only a few examples.
    - LayerNorm: sequences, as in language models. Each example is normalized on its own, so training and prediction do the same thing.
    - RMSNorm: the same places as LayerNorm, with less arithmetic, since it skips the mean and the shift. Most recent language models use it.

    A Linear layer right before BatchNorm does not need its bias, because
    subtracting the mean removes it. `nn.Linear(n, m, bias=False)` makes a
    Linear layer with no bias. `nn.Identity()` is a module that returns its
    input unchanged.

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


    # TODO: Return a norm layer for size features, chosen by kind. Return
    #   nn.BatchNorm1d for "batch", nn.LayerNorm for "layer", nn.RMSNorm with
    #   eps=1e-6 for "rms", and nn.Identity for "none".
    def norm_layer(
        kind: str,
        size: int,
    ) -> nn.Module:
        ...


    # TODO: Finish the line in the loop, so each hidden layer is a Linear layer,
    #   then norm_layer(kind, outputs), then a ReLU. sizes lists the number of
    #   inputs, each hidden width, then the number of outputs.
    def normed_mlp(
        sizes: list[int],
        kind: str,
    ) -> nn.Sequential:
        layers = []

        for inputs, outputs in zip(sizes[:-2], sizes[1:-1]):
            layers.extend(...)

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        return nn.Sequential(*layers)


    # TODO: Pass x through the layers of model one at a time, and return a list
    #   with the standard deviation of x after each ReLU. Autograd must not
    #   record anything while this runs.
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
            return nn.RMSNorm(size, eps=1e-6)

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

    if isinstance(module, nn.LayerNorm):
        return "LayerNorm", tuple(module.normalized_shape)[0]

    if isinstance(module, nn.RMSNorm):
        return "RMSNorm", f"({tuple(module.normalized_shape)[0]}, eps={module.eps})"

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
        return nn.RMSNorm(size, eps=1e-6)

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
        description="gives the standard deviation after every ReLU",
        inputs=(
            (seeded_normed_mlp(351, [30, 64, 64, 64, 10], "none"), x),
            (seeded_normed_mlp(352, [30, 64, 64, 64, 10], "layer"), x),
        ),
        reference=reference_spreads_after_relu,
        expect=expect_spreads,
    )


NORM1 = Exercise(
    name="norm1",
    chapter=NORMALIZATION,
    title="Why and which",
    lesson=NORM1_LESSON,
    source=NORM1_SOURCE,
    solution=NORM1_SOLUTION,
    hints={
        "norm_layer": "Use one if per kind, each returning the module built with size, adding eps=1e-6 for "
        "nn.RMSNorm. Return nn.Identity() when no kind matched.",
        "normed_mlp": "Each hidden layer adds three modules, nn.Linear(inputs, outputs), norm_layer(kind, outputs) "
        "and nn.ReLU().",
        "spreads_after_relu": "Inside a with torch.no_grad() block, loop over model, set x to layer(x), and append "
        "x.std().item() whenever isinstance says the layer is an nn.ReLU.",
    },
    introduces=(),
    check=check_norm1,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

MEASURE_THREADS = 4
NORM_KINDS = ("none", "batch", "layer", "rms")


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(MEASURE_THREADS)

    try:
        yield
    finally:
        torch.set_num_threads(threads)


def validation_accuracy(model: nn.Module) -> float:
    sets = mnist_sets()
    was_training = model.training
    model.eval()

    with torch.no_grad():
        right = (model(sets["validation_x"]).argmax(dim=1) == sets["validation_y"]).float().mean().item()

    model.train(was_training)

    return right


def train_on_mnist(
    model: nn.Module,
    rate: float,
    seed: int,
    images: int = 50000,
    epochs: int = 2,
    batch: int = 128,
    early_step: int = 100,
    warmup: int = 0,
) -> dict[str, float]:
    """Plain SGD on the first images of MNIST's training split, the runs behind Section 3's tables. Gives the validation
    accuracy after early_step steps and at the end, and the training loss at the end, all nan once a loss stops being
    finite. A last batch shorter than batch is left out, since BatchNorm can't take a batch of one. With warmup, the
    rate climbs in a straight line from rate / warmup at the first step to rate at step warmup."""
    sets = mnist_sets()
    x, y = sets["train_x"][:images], sets["train_y"][:images]
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    generator = torch.Generator().manual_seed(seed + 100)
    early = math.nan
    step = 0

    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)

        for start in range(0, len(x) - batch + 1, batch):
            if warmup:
                for group in optimizer.param_groups:
                    group["lr"] = rate * min(1.0, (step + 1) / warmup)

            rows = order[start : start + batch]
            loss = F.cross_entropy(model(x[rows]), y[rows])

            if not math.isfinite(loss.item()):
                return {"early": math.nan, "final": math.nan, "loss": math.nan}

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step == early_step:
                early = validation_accuracy(model)

    model.eval()

    with torch.no_grad():
        training_loss = F.cross_entropy(model(x[:10000]), y[:10000]).item()

    model.train()

    return {"early": early, "final": validation_accuracy(model), "loss": training_loss}


def kaiming_(model: nn.Module) -> nn.Module:
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    return model


def norm_run(
    kind: str,
    start: str,
    rate: float,
    seed: int,
    images: int = 50000,
    epochs: int = 2,
    batch: int = 128,
) -> dict[str, float]:
    torch.manual_seed(seed)
    model = reference_normed_mlp([784] + [64] * 8 + [10], kind)

    if start == "kaiming":
        kaiming_(model)

    return train_on_mnist(model, rate, seed, images, epochs, batch)


def measure_norms(
    kinds: tuple[str, ...] = NORM_KINDS,
    seeds: range = range(3),
) -> dict[tuple[str, str, float, int, int], dict[str, float]]:
    """norm1's table, keyed by start, kind, rate, batch and seed. An MLP of 8 hidden layers of 64 learns MNIST from
    PyTorch's default start at batch 128 over a range of rates, then at batch 4 on 10,000 images for one epoch. No norm
    and BatchNorm also train from the Kaiming start at large rates."""
    results = {}

    with pinned_threads():
        for kind in kinds:
            for rate in (0.03, 0.1, 0.3, 1.0):
                for seed in seeds:
                    results[("default", kind, rate, 128, seed)] = norm_run(kind, "default", rate, seed)

            for seed in seeds:
                results[("default", kind, 0.01, 4, seed)] = norm_run(kind, "default", 0.01, seed, 10000, 1, 4)

            if kind not in ("none", "batch"):
                continue

            for rate in (0.3, 1.0, 3.0):
                for seed in seeds:
                    results[("kaiming", kind, rate, 128, seed)] = norm_run(kind, "kaiming", rate, seed)

    return results


def measure_norm_spreads(seeds: range = range(3)) -> dict[str, list[list[float]]]:
    """norm1's spreads, the spread after each ReLU of norm1's MLP of 8 hidden layers of 64 from PyTorch's default start,
    on 1000 MNIST training images, for each kind of norm and seed."""
    x = mnist_sets()["train_x"][:1000]
    results: dict[str, list[list[float]]] = {}

    with pinned_threads():
        for kind in NORM_KINDS:
            results[kind] = []

            for seed in seeds:
                torch.manual_seed(seed)
                model = reference_normed_mlp([784] + [64] * 8 + [10], kind)
                results[kind].append(reference_spreads_after_relu(model, x))

    return results


class ReferenceLayerNorm(nn.Module):
    """layernorm1's MyLayerNorm, written out the way the learner writes it."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class ReferenceRMSNorm(nn.Module):
    """rmsnorm1's MyRMSNorm, written out the way the learner writes it."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: Tensor) -> Tensor:
        return x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-6) * self.weight


def best_milliseconds(
    run: Callable[[], object],
    calls: int = 20,
    repeats: int = 5,
) -> float:
    """The fastest of repeats timings, each the average over calls, after a few untimed calls to warm up."""
    for _ in range(5):
        run()

    best = math.inf

    for _ in range(repeats):
        start = time.perf_counter()

        for _ in range(calls):
            run()

        best = min(best, (time.perf_counter() - start) / calls)

    return 1000 * best


def measure_norm_speed(
    rows: int = 4096,
    size: int = 1024,
) -> dict[str, dict[str, float]]:
    """rmsnorm1's timing, in milliseconds on the CPU with 4 threads. layernorm1's and rmsnorm1's hand written norms,
    forward alone and forward with backward, on rows of size random numbers. The numbers move with whatever else the
    machine runs, so compare the two within one call."""
    x = torch.randn(rows, size, generator=torch.Generator().manual_seed(0), requires_grad=True)
    results: dict[str, dict[str, float]] = {}

    with pinned_threads():
        for name, norm in (("layer", ReferenceLayerNorm(size)), ("rms", ReferenceRMSNorm(size))):

            def forward() -> None:
                with torch.no_grad():
                    norm(x)

            def forward_and_backward() -> None:
                norm(x).sum().backward()

            results[name] = {
                "forward": best_milliseconds(forward),
                "forward and backward": best_milliseconds(forward_and_backward),
            }

    return results
