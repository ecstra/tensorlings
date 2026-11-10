"""90_parallel, data parallel training by hand with the workers' gradients averaged, and gradient accumulation on one device."""

from __future__ import annotations

import copy
import textwrap
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
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
from .dropout import labelled_points
from .gpt import MODEL_REFERENCE

PARALLEL = Chapter(
    folder="90_parallel",
    title="Training on several devices",
    readme=source_file(
        """
        # Training on several devices

        A big model trains on many GPUs at once, and each GPU works through
        its own share of every batch. This chapter builds data parallel
        training by hand, averaging the gradients from the workers. Then it
        gets the same result on one device with gradient accumulation.

        ## Further reading

        - [PyTorch, distributed data parallel](https://docs.pytorch.org/docs/2.14/notes/ddp.html)
        - [Li and colleagues, PyTorch distributed](https://arxiv.org/abs/2006.15704)
        - [Karpathy, nanoGPT's training loop](https://github.com/karpathy/nanoGPT/blob/master/train.py)
        """,
    ),
)

SHARD_PART = '''
def shard(
    x: Tensor,
    y: Tensor,
    workers: int,
) -> list[tuple[Tensor, Tensor]]:
    return list(zip(x.chunk(workers), y.chunk(workers)))
'''

LOCAL_PART = '''
def local_gradients(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> list[Tensor]:
    model.zero_grad()
    F.cross_entropy(model(x), y).backward()

    return [parameter.grad.clone() for parameter in model.parameters()]
'''

REDUCE_PART = '''
def all_reduce_mean(gradients: list[list[Tensor]]) -> list[Tensor]:
    return [torch.stack(parts).mean(dim=0) for parts in zip(*gradients)]
'''

STEP_PART = '''
def data_parallel_step(
    replicas: list[nn.Module],
    optimizers: list[torch.optim.Optimizer],
    x: Tensor,
    y: Tensor,
) -> None:
    shards = shard(x, y, len(replicas))
    gradients = [local_gradients(replica, shard_x, shard_y) for replica, (shard_x, shard_y) in zip(replicas, shards)]
    averaged = all_reduce_mean(gradients)

    for replica, optimizer in zip(replicas, optimizers):
        for parameter, gradient in zip(replica.parameters(), averaged):
            parameter.grad = gradient.clone()

        optimizer.step()
'''

ACCUMULATE_PART = '''
def accumulated_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    micro_batches: int,
) -> float:
    total = 0.0
    optimizer.zero_grad()

    for micro_x, micro_y in zip(x.chunk(micro_batches), y.chunk(micro_batches)):
        loss = F.cross_entropy(model(micro_x), micro_y) / micro_batches
        loss.backward()
        total += loss.item()

    optimizer.step()

    return total


def accumulation_steps(
    batch_size: int,
    micro_batch_size: int,
    workers: int,
) -> int:
    return batch_size // (micro_batch_size * workers)
'''

DDP1_SOLUTION_PARTS = SHARD_PART + LOCAL_PART + REDUCE_PART + STEP_PART


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


PARALLEL_REFERENCE = namespace_of(DDP1_SOLUTION_PARTS + ACCUMULATE_PART)


def small_classifier(
    seed: int,
    normalized: bool = False,
) -> nn.Sequential:
    torch.manual_seed(seed)

    if normalized:
        return nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 3))

    return nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 3))


def full_gradients(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> list[Tensor]:
    return PARALLEL_REFERENCE["local_gradients"](model, x, y)


def expect_tensors(
    call: str,
    actual: object,
    expected: list[Tensor],
    names: list[str],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
        raise Mismatch(f"{call} should return a list of {len(expected)} tensors, one per parameter, but it returns {value_text(actual)}.")

    for name, part, wanted in zip(names, actual, expected):
        expect_tensor(
            call=f"the {name} part of {call}",
            actual=part,
            expected=wanted,
        )


def parameter_names(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters()]


# ---------------------------------------------------------------------------
# ddp1
# ---------------------------------------------------------------------------

DDP1_LESSON = prose(
    """
    A big model trains on many GPUs at once. In **data parallel** training,
    each GPU is a worker that holds a full copy of the model, called a
    replica. Every batch is split into equal parts called shards, one per
    worker. Each worker computes the gradient of the loss on its own shard.
    Then the workers average their gradients, and every replica takes the
    same optimizer step with that average.

    This average equals the gradient on the whole batch. The mean loss over
    64 rows equals the average of the mean losses over four shards of 16
    rows. The gradient of an average is the average of the gradients, so the
    same holds for the gradients.

    ```python
    torch.stack([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 6.0])]).mean(dim=0)    # tensor([2., 4.])
    ```

    The replicas start out the same and take the same steps, so they stay the
    same. `copy.deepcopy(model)` makes a replica, a model with its own copy of
    every weight. An optimizer's step uses whatever each parameter's `.grad`
    holds, so you can set a gradient by hand.

    ```python
    twin.weight.grad = torch.ones(2, 2)    # an optimizer step on twin now uses this as weight's gradient
    ```

    Here a small classifier's gradient on 64 rows is compared with the
    average of its shards' gradients. The gap is the norm of the difference,
    divided by the norm of the whole batch's gradient.

    | Shards | Model | Gap |
    |---|---|---|
    | 4 of 16 rows | Linear and ReLU | 9.1e-8 |
    | 40 rows and 24 rows | Linear and ReLU | 0.088 |
    | 4 of 16 rows | with BatchNorm | 0.11 |

    With equal shards the two match, apart from rounding. With unequal
    shards, the small shard counts as much as the big one in the average, so
    it gets too much weight. Weighting each shard's mean by its number of
    rows fixes that. BatchNorm normalizes each shard with that shard's own
    statistics, so with BatchNorm the gradient really is different. PyTorch's
    `SyncBatchNorm` shares the statistics across workers to avoid this.

    After 200 steps with 4 workers, the replicas were identical down to the
    last bit. They differed from a single model trained on the whole batches
    by at most 2.4e-7, which is rounding again.

    Real training runs one process per GPU, with PyTorch's
    `DistributedDataParallel`. It averages the gradients with an **all
    reduce**, an operation where every worker sends in its gradients and
    receives the combined result. It sends the gradients in buckets, a few
    layers' worth at a time, while the backward pass is still running. In
    this exercise the workers are a loop in one process. So they run one
    after another, and the code works on any machine.
    """,
)

DDP1_SOURCE = source_file(
    '''
    # ddp1.py

    import copy

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    halves = torch.arange(6).chunk(2)                                       # (tensor([0, 1, 2]), tensor([3, 4, 5]))
    middle = torch.stack([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 6.0])]).mean(dim=0)    # tensor([2., 4.])
    twin = copy.deepcopy(nn.Linear(2, 2))                                   # a model with its own copy of every weight
    twin.weight.grad = torch.ones(2, 2)                                     # a gradient set by hand


    # TODO: Split x and y into workers equal shards along the batch dimension,
    #   dim 0. Return them as a list of (x, y) tuples, one per worker.
    #   Example: for x of shape (8, 3), y of shape (8,) and 4 workers, shard
    #   returns 4 tuples, each an x of shape (2, 3) and a y of shape (2,).
    def shard(
        x: Tensor,
        y: Tensor,
        workers: int,
    ) -> list[tuple[Tensor, Tensor]]:
        ...


    # TODO: Return the gradient of the mean cross entropy loss of model on x and
    #   labels y, as a list with one tensor per parameter, in the order of
    #   model.parameters(). First clear any gradients left over from before.
    def local_gradients(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> list[Tensor]:
        ...


    # TODO: gradients holds one list per worker, with one tensor per parameter.
    #   Return a list with one tensor per parameter, the average of that
    #   parameter's gradient over the workers.
    #   Example: all_reduce_mean([[torch.tensor([1.0, 2.0])],
    #   [torch.tensor([3.0, 6.0])]]) returns [tensor([2., 4.])].
    def all_reduce_mean(gradients: list[list[Tensor]]) -> list[Tensor]:
        ...


    # TODO: Finish one training step. Each replica computes the gradients on its
    #   own shard with local_gradients. Average them with all_reduce_mean. Then
    #   set every replica's parameter gradients to that average, and step its
    #   optimizer.
    def data_parallel_step(
        replicas: list[nn.Module],
        optimizers: list[torch.optim.Optimizer],
        x: Tensor,
        y: Tensor,
    ) -> None:
        shards = shard(x, y, len(replicas))
        ...

        for replica, optimizer in zip(replicas, optimizers):
            ...
    ''',
)

DDP1_SOLUTION = source_file(
    '''
    # ddp1.py, a solution for comparison

    import copy

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    halves = torch.arange(6).chunk(2)
    middle = torch.stack([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 6.0])]).mean(dim=0)
    twin = copy.deepcopy(nn.Linear(2, 2))
    twin.weight.grad = torch.ones(2, 2)


''' + indented(SHARD_PART) + '''


''' + indented(LOCAL_PART) + '''


''' + indented(REDUCE_PART) + '''


''' + indented(STEP_PART) + '''
    ''',
)


def check_shard(found: object) -> None:
    for seed, count, workers in ((905, 8, 4), (906, 6, 2)):
        x, y = labelled_points(seed, count, 3, 3)
        call = f"shard of x of shape ({count}, 3) and y of shape ({count},) into {workers} workers"
        actual = found(x, y, workers)
        expected = PARALLEL_REFERENCE["shard"](x, y, workers)
        reject_missing_return(
            call=call,
            actual=actual,
        )

        if not isinstance(actual, (list, tuple)):
            raise Mismatch(f"{call} should return a list of {workers} (x, y) pairs, but it returns {value_text(actual)}.")

        if len(actual) != workers:
            raise Mismatch(f"{call} should return a list of {workers} (x, y) pairs, but it returns {len(actual)} of them.")

        for worker, (pair, wanted) in enumerate(zip(actual, expected)):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise Mismatch(f"Shard {worker} from {call} should be an (x, y) pair, but it is {value_text(pair)}.")

            for name, part, wanted_part in zip(("x", "y"), pair, wanted):
                expect_tensor(
                    call=f"the {name} of shard {worker} from {call}",
                    actual=part,
                    expected=wanted_part,
                )


def check_local_gradients(found: object) -> None:
    model = small_classifier(907)
    names = parameter_names(model)
    x, y = labelled_points(908, 16, 6, 3)

    # Called twice on the same model, as each step does, so gradients left over from the first call show.
    for start, end in ((0, 8), (8, 16)):
        expected = full_gradients(small_classifier(907), x[start:end], y[start:end])
        expect_tensors(
            call=f"local_gradients of a small classifier on rows {start} to {end - 1}, after it already ran once"
            if start
            else f"local_gradients of a small classifier on rows {start} to {end - 1}",
            actual=found(model, x[start:end], y[start:end]),
            expected=expected,
            names=names,
        )


def check_all_reduce_mean(found: object) -> None:
    for seed, workers, shapes in ((909, 2, ((2, 3), (3,))), (910, 3, ((4,),))):
        generator = torch.Generator().manual_seed(seed)
        gradients = [[torch.randn(shape, generator=generator) for shape in shapes] for _ in range(workers)]
        expected = PARALLEL_REFERENCE["all_reduce_mean"]([[part.clone() for part in worker] for worker in gradients])
        expect_tensors(
            call=f"all_reduce_mean of {workers} workers' gradients, for parameters of shapes {', '.join(str(shape) for shape in shapes)}",
            actual=found(gradients),
            expected=expected,
            names=[f"parameter {index}" for index in range(len(shapes))],
        )


def run_with_reference_parts(
    found: Callable[..., object],
    run: Callable[[], None],
) -> None:
    """data_parallel_step runs with the reference shard, local_gradients and all_reduce_mean in place of the
    learner's, so a mistake in one of those shows in its own row."""
    globals_ = found.__globals__
    names = ("shard", "local_gradients", "all_reduce_mean")
    learners = {name: globals_.get(name) for name in names}

    for name in names:
        globals_[name] = PARALLEL_REFERENCE[name]

    try:
        run()
    finally:
        for name in names:
            globals_[name] = learners[name]


# With BatchNorm the shards' average differs from the whole batch's gradient, as ddp1's table shows, so a step
# that trains every replica on the whole batch can't match the reference. Only parameters are compared, so each
# replica's own running statistics don't matter.
def train_replicas(
    step: Callable[..., object],
    workers: int,
    steps: int,
) -> list[nn.Module]:
    model = small_classifier(911, normalized=True)
    replicas = [copy.deepcopy(model) for _ in range(workers)]
    optimizers = [torch.optim.SGD(replica.parameters(), lr=0.5) for replica in replicas]
    x, y = labelled_points(912, 16 * steps, 6, 3)

    for index in range(steps):
        step(replicas, optimizers, x[16 * index : 16 * (index + 1)], y[16 * index : 16 * (index + 1)])

    return replicas


def same_weights(
    first: nn.Module,
    second: nn.Module,
) -> bool:
    return all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))


def whole_batch_step(
    replicas: list[nn.Module],
    optimizers: list[torch.optim.Optimizer],
    x: Tensor,
    y: Tensor,
) -> None:
    """The likely mistake, every replica working out the gradient of the whole batch instead of its own shard."""
    for replica, optimizer in zip(replicas, optimizers):
        for parameter, gradient in zip(replica.parameters(), PARALLEL_REFERENCE["local_gradients"](replica, x, y)):
            parameter.grad = gradient

        optimizer.step()


def close_weights(
    first: nn.Module,
    second: nn.Module,
) -> bool:
    return all(torch.allclose(a, b, rtol=1e-5, atol=1e-6) for a, b in zip(first.parameters(), second.parameters()))


def check_data_parallel_step(found: object) -> None:
    for workers in (4, 2):
        replicas: list[nn.Module] = []
        run_with_reference_parts(
            found=found,
            run=lambda workers=workers: replicas.extend(train_replicas(found, workers, 2)),
        )
        expected = train_replicas(PARALLEL_REFERENCE["data_parallel_step"], workers, 2)
        call = f"2 steps of data_parallel_step with {workers} replicas"

        if all(same_weights(replica, small_classifier(911, normalized=True)) for replica in replicas):
            raise Mismatch(f"After {call}, every replica is as it started, so data_parallel_step is not written yet.")

        if not all(same_weights(replicas[0], replica) for replica in replicas[1:]):
            raise Mismatch(
                f"After {call}, the replicas differ from each other. Every replica should take its step with the "
                "same averaged gradient, so they all stay the same.",
            )

        whole = train_replicas(whole_batch_step, workers, 2)

        if all(close_weights(replica, wrong) for replica, wrong in zip(replicas, whole)):
            raise Mismatch(
                f"After {call}, every replica changed as if it computed the gradient on the whole batch. Each "
                "replica should compute the gradient on its own shard, the one shard(x, y, len(replicas)) gives it. "
                "Then the replicas step with the average of those gradients.",
            )

        for index, (replica, wanted) in enumerate(zip(replicas, expected)):
            for (name, parameter), wanted_parameter in zip(replica.named_parameters(), wanted.parameters()):
                expect_tensor(
                    call=f"{name} of replica {index} after 2 steps of data_parallel_step with {workers} replicas",
                    actual=parameter.detach(),
                    expected=wanted_parameter.detach(),
                )


def check_ddp1(checker: Checker) -> None:
    checker.test(
        function="shard",
        description="splits the batch into equal shards",
        run=check_shard,
    )
    checker.test(
        function="local_gradients",
        description="returns one worker's gradients, clearing old ones first",
        run=check_local_gradients,
    )
    checker.test(
        function="all_reduce_mean",
        description="averages each parameter's gradient over the workers",
        run=check_all_reduce_mean,
    )
    checker.test(
        function="data_parallel_step",
        description="steps every replica with the averaged gradient",
        run=check_data_parallel_step,
    )


DDP1 = Exercise(
    name="ddp1",
    chapter=PARALLEL,
    title="Data parallel by hand",
    lesson=DDP1_LESSON,
    source=DDP1_SOURCE,
    solution=DDP1_SOLUTION,
    hints={
        "shard": "x.chunk(workers) and y.chunk(workers) split each tensor. zip pairs the pieces up, and list() turns "
        "the pairs into a list.",
        "local_gradients": "Clear with model.zero_grad(), then call backward on F.cross_entropy of the model's scores "
        "for x against y. The list holds a .clone() of each parameter's .grad, in the order of model.parameters().",
        "all_reduce_mean": "zip(*gradients) groups each parameter's gradients from every worker. torch.stack them and "
        "take .mean(dim=0).",
        "data_parallel_step": "Call local_gradients on each replica with its shard, all_reduce_mean the lists, then set "
        "each replica's parameter.grad to a .clone() of the average before optimizer.step().",
    },
    introduces=(),
    check=check_ddp1,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# ddp2
# ---------------------------------------------------------------------------

DDP2_LESSON = prose(
    """
    Data parallel training needs several GPUs. **Gradient accumulation** takes
    the same step on one GPU. The batch is split into micro batches that run
    one after another. `backward` adds each micro batch's gradient to `.grad`
    instead of replacing it. One optimizer step at the end uses the total.

    ```python
    weight = torch.tensor(2.0, requires_grad=True)
    (3 * weight).backward()
    (4 * weight).backward()
    weight.grad    # tensor(7.)
    ```

    Each micro batch's mean loss is divided by the number of micro batches.
    That way the total equals the mean loss of the whole batch, as ddp1's
    average did. `zero_grad` runs once per step, before the first micro
    batch, and never between micro batches.

    Why bother? The backward pass needs values that the forward pass
    computed, so autograd keeps them in memory, and they grow with the number
    of rows. Here are the bytes autograd keeps for one forward pass of gpt3's
    model, with 128 tokens per row, not counting the weights.

    | Rows at once | 64 | 16 | 4 |
    |---|---|---|---|
    | Bytes kept for backward | 280 MB | 70 MB | 18 MB |

    Four micro batches of 16 rows need a quarter of the memory of 64 rows at
    once. The cost is that they run one after another. Their total gradient
    matched the whole batch's gradient, with a difference of 1.4e-6 relative
    to its size.

    The two methods can be combined. nanoGPT's default run uses micro batches
    of 12 rows of 1,024 tokens, with 40 micro batches per step. On several
    GPUs it divides those 40 micro batches among them, so a step still covers
    491,520 tokens. It also skips the all reduce on every micro batch except
    the last.
    """,
)

DDP2_SOURCE = source_file(
    '''
    # ddp2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    weight = torch.tensor(2.0, requires_grad=True)
    (3 * weight).backward()
    (4 * weight).backward()
    added_up = weight.grad    # tensor(7.), because backward adds to .grad


    # TODO: Finish one optimizer step on x and labels y, split into
    #   micro_batches equal pieces that run one after another. Clear the
    #   gradients once at the start. For each piece, divide its mean cross
    #   entropy loss by micro_batches and call backward on it. Step the
    #   optimizer once at the end. Return the sum of the divided losses, as a
    #   plain float.
    def accumulated_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        x: Tensor,
        y: Tensor,
        micro_batches: int,
    ) -> float:
        total = 0.0
        ...

        return total


    # TODO: Return how many micro batches each worker runs per step, so that
    #   all the workers together cover batch_size examples in one step.
    #   Example: accumulation_steps(480, 12, 8) returns 5.
    def accumulation_steps(
        batch_size: int,
        micro_batch_size: int,
        workers: int,
    ) -> int:
        ...
    ''',
)

DDP2_SOLUTION = source_file(
    '''
    # ddp2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    weight = torch.tensor(2.0, requires_grad=True)
    (3 * weight).backward()
    (4 * weight).backward()
    added_up = weight.grad


''' + indented(ACCUMULATE_PART) + '''
    ''',
)


def check_accumulated_step(found: object) -> None:
    for seed, count, micro_batches in ((913, 16, 4), (914, 12, 3)):
        model = small_classifier(seed)
        reference = small_classifier(seed)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.5)
        x, y = labelled_points(seed, 2 * count, 6, 3)

        # Two steps, so gradients left over from the first one show in the second.
        for step in range(2):
            rows = slice(step * count, (step + 1) * count)
            call = f"step {step + 1} of accumulated_step on {count} rows in {micro_batches} micro batches"
            before = copy.deepcopy(model)
            actual = found(model, optimizer, x[rows], y[rows], micro_batches)
            expected = PARALLEL_REFERENCE["accumulated_step"](reference, reference_optimizer, x[rows], y[rows], micro_batches)

            if same_weights(model, before):
                raise Mismatch(f"{call} left the model unchanged. It should take one optimizer step after the micro batches.")
            expect_close(
                call=f"the loss from {call}",
                actual=actual,
                expected=expected,
                tolerance=1e-5,
            )

            for (name, parameter), wanted in zip(model.named_parameters(), reference.parameters()):
                expect_tensor(
                    call=f"{name} after {call}",
                    actual=parameter.detach(),
                    expected=wanted.detach(),
                )


def expect_whole_number(
    call: str,
    actual: object,
    expected: int,
) -> None:
    if isinstance(actual, float):
        raise Mismatch(f"{call} is {actual!r}, a float, but a loop's range() needs an int. Divide with // to get one.")

    expect_value(
        call=call,
        actual=actual,
        expected=expected,
    )


def check_ddp2(checker: Checker) -> None:
    checker.test(
        function="accumulated_step",
        description="adds up the micro batches' gradients, then steps once",
        run=check_accumulated_step,
    )
    check_on_inputs(
        checker=checker,
        function="accumulation_steps",
        description="returns the number of micro batches each worker runs per step",
        inputs=(
            (480, 12, 8),
            (64, 16, 1),
            (96, 4, 3),
        ),
        reference=PARALLEL_REFERENCE["accumulation_steps"],
        expect=expect_whole_number,
    )


DDP2 = Exercise(
    name="ddp2",
    chapter=PARALLEL,
    title="Gradient accumulation",
    lesson=DDP2_LESSON,
    source=DDP2_SOURCE,
    solution=DDP2_SOLUTION,
    hints={
        "accumulated_step": "Clear the gradients once with the optimizer's zero_grad, then loop over "
        "zip(x.chunk(micro_batches), y.chunk(micro_batches)). Divide each loss by micro_batches, call backward, add "
        ".item() to total, and step after the loop.",
        "accumulation_steps": "One round of micro batches, one on every worker, covers micro_batch_size * workers "
        "examples. Divide batch_size by that with //.",
    },
    introduces=(),
    check=check_ddp2,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The measurements behind the lessons
# ---------------------------------------------------------------------------


def gradient_gap(
    shards: list[tuple[Tensor, Tensor]],
    normalized: bool,
) -> float:
    """How far the average of the shards' gradients is from the whole batch's, over the size of the whole batch's."""
    x = torch.cat([shard_x for shard_x, _ in shards])
    y = torch.cat([shard_y for _, shard_y in shards])
    full = torch.cat([part.flatten() for part in full_gradients(small_classifier(915, normalized), x, y)])
    parts = [PARALLEL_REFERENCE["local_gradients"](small_classifier(915, normalized), shard_x, shard_y) for shard_x, shard_y in shards]
    averaged = torch.cat([part.flatten() for part in PARALLEL_REFERENCE["all_reduce_mean"](parts)])

    return ((averaged - full).norm() / full.norm()).item()


def saved_bytes(
    model: nn.Module,
    ids: Tensor,
) -> int:
    """The bytes autograd keeps for the backward pass of one forward, the weights left out, each storage counted once."""
    weights = {parameter.untyped_storage().data_ptr() for parameter in model.parameters()}
    kept: dict[int, int] = {}

    def pack(tensor: Tensor) -> Tensor:
        storage = tensor.untyped_storage()

        if storage.data_ptr() not in weights:
            kept[storage.data_ptr()] = storage.nbytes()

        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        scores = model(ids)
        F.cross_entropy(scores.flatten(0, 1), ids.flatten())

    return sum(kept.values())


def measure_parallel() -> dict[str, object]:
    """Every number in the two lessons, on the CPU in under a minute. Not run by any check."""
    results: dict[str, object] = {}
    x, y = labelled_points(916, 64, 6, 3)
    shards = PARALLEL_REFERENCE["shard"]
    results["4 equal shards"] = gradient_gap(shards(x, y, 4), False)
    results["40 and 24 rows"] = gradient_gap([(x[:40], y[:40]), (x[40:], y[40:])], False)
    results["4 equal shards, BatchNorm"] = gradient_gap(shards(x, y, 4), True)

    model = small_classifier(917)
    replicas = [copy.deepcopy(model) for _ in range(4)]
    optimizers = [torch.optim.SGD(replica.parameters(), lr=0.5) for replica in replicas]
    single = copy.deepcopy(model)
    single_optimizer = torch.optim.SGD(single.parameters(), lr=0.5)
    train_x, train_y = labelled_points(918, 64 * 200, 6, 3)

    for step in range(200):
        rows = slice(64 * step, 64 * (step + 1))
        PARALLEL_REFERENCE["data_parallel_step"](replicas, optimizers, train_x[rows], train_y[rows])
        single_optimizer.zero_grad()
        F.cross_entropy(single(train_x[rows]), train_y[rows]).backward()
        single_optimizer.step()

    def largest_gap(first: nn.Module, second: nn.Module) -> float:
        return max((a - b).abs().max().item() for a, b in zip(first.parameters(), second.parameters()))

    results["replicas apart after 200 steps"] = max(largest_gap(replicas[0], replica) for replica in replicas[1:])
    results["replica against one model after 200 steps"] = largest_gap(replicas[0], single)
    results["largest weight"] = max(parameter.abs().max().item() for parameter in single.parameters())

    torch.manual_seed(0)
    gpt = MODEL_REFERENCE["GPT"](65, 128, 128, 4, 4)
    ids = torch.randint(0, 65, (64, 128), generator=torch.Generator().manual_seed(919))

    for rows in (64, 16, 4):
        results[f"gpt saved bytes, {rows} rows"] = saved_bytes(gpt, ids[:rows])

    whole = torch.cat([part.flatten() for part in full_gradients_gpt(gpt, ids, 1)])
    accumulated = torch.cat([part.flatten() for part in full_gradients_gpt(gpt, ids, 4)])
    results["gpt accumulated gap"] = ((accumulated - whole).norm() / whole.norm()).item()

    return results


def full_gradients_gpt(
    model: nn.Module,
    ids: Tensor,
    micro_batches: int,
) -> list[Tensor]:
    model.zero_grad()

    for piece in ids.chunk(micro_batches):
        scores = model(piece[:, :-1])
        (F.cross_entropy(scores.flatten(0, 1), piece[:, 1:].flatten()) / micro_batches).backward()

    return [parameter.grad.clone() for parameter in model.parameters()]
