"""84_cost, the FLOPs of a matrix multiply, a Linear and attention, the memory of the parameters and of what backward
keeps, and the rule of six FLOPs per parameter per token in training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_value,
)
from ..text import prose, source_file
from .gpt import MODEL_REFERENCE

COST = Chapter(
    folder="84_cost",
    title="Counting cost",
    readme=source_file(
        """
        # Counting cost

        Section 12 is about making models run faster and use less memory. It
        starts by counting costs before anything runs. This chapter counts the
        FLOPs of a matrix multiply, a Linear layer and attention. It also
        counts the bytes the parameters take, and the bytes that backward
        needs kept in memory. Then it checks the rule from Kaplan and
        colleagues that training costs about six FLOPs per parameter per
        token.

        ## Further reading

        - [Kaplan and colleagues, scaling laws for neural language models](https://arxiv.org/abs/2001.08361)
        - [PyTorch, hooks for saved tensors](https://docs.pytorch.org/docs/2.14/notes/autograd.html#hooks-for-saved-tensors)
        """,
    ),
)


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(code, namespace)

    return namespace


def seeded_mlp(
    sizes: tuple[int, ...],
    seed: int,
) -> nn.Sequential:
    torch.manual_seed(seed)
    layers: list[nn.Module] = []

    for into, out in zip(sizes, sizes[1:]):
        layers.extend([nn.Linear(into, out), nn.ReLU()])

    return nn.Sequential(*layers[:-1])


def tied_pair() -> nn.Sequential:
    torch.manual_seed(840)
    table = nn.Embedding(10, 4)
    head = nn.Linear(4, 10, bias=False)
    head.weight = table.weight

    return nn.Sequential(table, head)


def expect_int(
    call: str,
    actual: object,
    expected: int,
) -> None:
    """A count of FLOPs or bytes, which the TODOs ask for as a plain int."""
    if isinstance(actual, float):
        raise Mismatch(
            f"{call} is {actual!r}, a float, but it should be the plain int {expected}. Multiply whole numbers only, "
            "with no 2.0 or division, or turn the result into an int with int(...).",
        )

    expect_value(
        call=call,
        actual=actual,
        expected=expected,
    )


# ---------------------------------------------------------------------------
# cost1
# ---------------------------------------------------------------------------

COST1_LESSON = prose(
    """
    You can work out what a model costs before you run it. Compute is counted
    in FLOPs, short for floating point operations. One FLOP is one addition
    or one multiplication.

    Multiplying an (m, k) matrix by a (k, n) matrix gives an (m, n) result,
    which has `m * n` numbers. Each of those numbers is a dot product of
    length k, which takes k multiplications and k additions. So the whole
    product takes 2mkn FLOPs, which is `2 * k` for each of the `m * n`
    outputs.

    ```python
    a, b = torch.randn(32, 64), torch.randn(64, 16)
    product = a @ b    # 32 * 16 outputs, each 64 multiplies and 64 adds, so 65,536 FLOPs
    ```

    A Linear layer applied to a batch is one such product. The batch, with
    one input per row, is multiplied by the layer's weight. The bias adds one
    more addition per output. Kaplan and colleagues leave biases out of their
    counts because they are too small to matter, and this chapter does the
    same.

    Attention over T tokens, each a vector of d numbers, does two matrix
    products. The scores `Q @ K.T` multiply a (T, d) matrix by a (d, T)
    matrix. The weighted sum of the values multiplies the (T, T) weights by
    the (T, d) values. The cost of both products grows with T squared, while
    the cost of a Linear grows with T.

    The first part of a model's memory is its parameters. Each number takes
    `element_size()` bytes, which is 4 in float32 and 2 in float16. A weight
    that two layers share, like the tied output head in gpt2, is stored only
    once, and `parameters()` lists it only once.

    PyTorch can count FLOPs too. `FlopCounterMode`, from
    `torch.utils.flop_counter`, adds the FLOPs of every matrix product run
    inside it to a total. Here are its counts for the GPT that gpt3 trains,
    run on one sequence of 128 characters.

    | Part | FLOPs |
    |---|---|
    | The Linears of the 4 blocks | 201,326,592 |
    | The output head | 2,129,920 |
    | Attention's two products, 4 blocks | 33,554,432 |
    | `FlopCounterMode`'s total | 203,456,512 |

    The counter's total is exactly the Linears plus the head, so it leaves
    out attention. On the CPU, this version of PyTorch runs attention as one
    fused kernel, which is a single combined operation. The counter has no
    rule for counting it. So it is worth checking a tool's count by hand.

    Per token, the Linears in the blocks cost 1.57 million FLOPs, which is
    exactly twice their 786,432 weights. Those weights are nearly all of the
    793,344 parameters outside the embeddings. This matches Kaplan and
    colleagues' rule for a forward pass, about `2 * N` FLOPs per token. Here
    N is the number of parameters outside the embeddings. The rule leaves
    out attention's FLOPs, which is a fair approximation while the width is
    more than a twelfth of the context length. The head reuses the
    embedding's weight, so it is not counted in N either. All 818,048
    parameters of the GPT take 3.27 MB in float32.
    """,
)

COST1_SOURCE = source_file(
    '''
    # cost1.py

    import torch
    import torch.nn as nn

    product_flops = 2 * 32 * 64 * 16                     # 65536, for a (32, 64) matrix times a (64, 16) matrix
    layer = nn.Linear(64, 16)
    layer_sizes = (layer.in_features, layer.out_features)  # (64, 16)
    half_size = torch.zeros(3, dtype=torch.float16).element_size()    # 2 bytes per number


    # TODO: Return the number of FLOPs it takes to multiply an (m, k) matrix by
    #   a (k, n) matrix, as a plain int. Each of the m * n outputs takes k
    #   multiplications and k additions.
    #   Example: matmul_flops(32, 64, 16) returns 65536.
    def matmul_flops(
        m: int,
        k: int,
        n: int,
    ) -> int:
        ...


    # TODO: Return the FLOPs of applying linear to a batch of rows inputs, as a
    #   plain int. The batch has shape (rows, in_features). Use matmul_flops,
    #   and leave the bias out.
    #   Example: linear_flops(nn.Linear(64, 16), 32) returns 65536.
    def linear_flops(
        linear: nn.Linear,
        rows: int,
    ) -> int:
        ...


    # TODO: Return the FLOPs of attention's two matrix products, the scores and
    #   the weighted sum of the values, as a plain int. Use matmul_flops. The
    #   sequence has tokens positions, and each position is a vector of width
    #   numbers.
    #   Example: attention_flops(128, 128) returns 8388608.
    def attention_flops(
        tokens: int,
        width: int,
    ) -> int:
        ...


    # TODO: Return how many bytes model's parameters take, as a plain int.
    #   Count each parameter only once, and use its own element_size() for the
    #   bytes per number.
    #   Example: parameter_bytes(nn.Linear(10, 4)) returns 176.
    def parameter_bytes(
        model: nn.Module,
    ) -> int:
        ...
    ''',
)

COST1_SOLUTION = source_file(
    '''
    # cost1.py, a solution for comparison

    import torch
    import torch.nn as nn

    product_flops = 2 * 32 * 64 * 16
    layer = nn.Linear(64, 16)
    layer_sizes = (layer.in_features, layer.out_features)
    half_size = torch.zeros(3, dtype=torch.float16).element_size()


    def matmul_flops(
        m: int,
        k: int,
        n: int,
    ) -> int:
        return 2 * m * k * n


    def linear_flops(
        linear: nn.Linear,
        rows: int,
    ) -> int:
        return matmul_flops(rows, linear.in_features, linear.out_features)


    def attention_flops(
        tokens: int,
        width: int,
    ) -> int:
        return 2 * matmul_flops(tokens, width, tokens)


    def parameter_bytes(
        model: nn.Module,
    ) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    ''',
)

COST1_REFERENCE = namespace_of(COST1_SOLUTION)


def check_cost1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="matmul_flops",
        description="counts one multiply and one add for each of the k terms of every output",
        inputs=(
            (3, 5, 7),
            (1, 1, 1),
            (128, 64, 10),
        ),
        reference=COST1_REFERENCE["matmul_flops"],
        expect=expect_int,
    )
    check_on_inputs(
        checker=checker,
        function="linear_flops",
        description="counts the matrix product of the batch and the weight",
        inputs=(
            (Described(nn.Linear(10, 4), "nn.Linear(10, 4)"), 3),
            (Described(nn.Linear(7, 5, bias=False), "nn.Linear(7, 5, bias=False)"), 2),
        ),
        reference=COST1_REFERENCE["linear_flops"],
        expect=expect_int,
    )
    check_on_inputs(
        checker=checker,
        function="attention_flops",
        description="counts both products, the scores and the weighted sum",
        inputs=(
            (128, 128),
            (5, 3),
        ),
        reference=COST1_REFERENCE["attention_flops"],
        expect=expect_int,
    )
    check_on_inputs(
        checker=checker,
        function="parameter_bytes",
        description="counts every parameter once, at its own element size",
        inputs=(
            (Described(nn.Linear(10, 4), "nn.Linear(10, 4)"),),
            (Described(nn.Linear(10, 4).half(), "nn.Linear(10, 4).half()"),),
            (Described(tied_pair(), "an Embedding(10, 4) whose weight a Linear(4, 10) shares"),),
        ),
        reference=COST1_REFERENCE["parameter_bytes"],
        expect=expect_int,
    )


COST1 = Exercise(
    name="cost1",
    chapter=COST,
    title="FLOPs and bytes",
    lesson=COST1_LESSON,
    source=COST1_SOURCE,
    solution=COST1_SOLUTION,
    hints={
        "matmul_flops": "Each of the m * n outputs has k terms, and each term costs two FLOPs, one multiply and one "
        "add.",
        "linear_flops": "The batch has shape (rows, in_features), and the weight acts as an (in_features, "
        "out_features) matrix.",
        "attention_flops": "The scores multiply a (tokens, width) matrix by a (width, tokens) matrix. The weighted sum "
        "multiplies (tokens, tokens) by (tokens, width), which costs the same.",
        "parameter_bytes": "Add up parameter.numel() * parameter.element_size() over model.parameters().",
    },
    introduces=(),
    check=check_cost1,
)


# ---------------------------------------------------------------------------
# cost2
# ---------------------------------------------------------------------------

COST2_LESSON = prose(
    """
    Training needs memory for more than the parameters. The backward pass
    computes each gradient from numbers that the forward pass produced. For
    example, a Linear's weight gradient needs the Linear's input. So that
    input stays in memory until backward reaches the Linear.

    Autograd stores these saved tensors in the computation graph. Inside a
    `with torch.autograd.graph.saved_tensors_hooks(pack, unpack)` block,
    autograd calls `pack` on every tensor it saves. Later, when backward
    reads a saved tensor back, autograd calls `unpack` on it.

    ```python
    def note(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(note, lambda tensor: tensor):
        squares = torch.ones(3, 4, requires_grad=True) ** 2    # shapes gets (3, 4)
    ```

    Several saved tensors can share one block of memory, called a storage.
    For example, a ReLU saves its output, and the next Linear saves those same
    numbers as its input. `tensor.untyped_storage()` returns a tensor's
    storage. The storage's `data_ptr()` is the memory address where it
    starts, and its `nbytes()` is its size in bytes.

    Here is how much the GPT that gpt3 trains saves for backward. Its
    parameters take 3.27 MB.

    | Sequences of 128 in a batch | 1 | 8 | 64 |
    |---|---|---|---|
    | Saved for backward | 4.34 MB | 34.75 MB | 278.00 MB |

    The memory for the parameters stays the same, while the saved tensors
    grow with the batch size. At 64 sequences, the saved tensors take 85
    times as much memory as the parameters. Inside `torch.no_grad()` nothing
    is saved, which is why evaluating a model takes less memory than
    training it.

    Compute follows a rule from Kaplan and colleagues. A forward pass costs
    about 2 FLOPs per parameter per token, one multiply and one add. The
    backward pass costs about twice as much. That is because each Linear
    computes two gradients, one for its weight and one for its input. Each of
    those is a matrix product as big as the forward one. So training costs
    about 6 FLOPs per parameter per token.

    `FlopCounterMode` counts 610,369,536 FLOPs for the GPT's forward and
    backward passes on one sequence. That is 3 times its count for the
    forward pass alone. It is also within 0.2% of the rule's count,
    `6 * 793,344 * 128` for 793,344 parameters and 128 tokens. But this
    count leaves out attention, which the counter misses here, as cost1
    showed. If attention runs as plain matrix products that the counter can
    read, the total becomes 711,032,832. That is about 17% above the rule.

    The first Linear skips computing the gradient for its input when the
    input does not need one. Take an MLP with layer sizes 64, 128 and 32, run
    on 10 rows. The counter gives 573,440 FLOPs when the input does not need
    a gradient, and 737,280 when it does.
    """,
)

COST2_SOURCE = source_file(
    '''
    # cost2.py

    import torch
    import torch.nn as nn
    from torch import Tensor
    from torch.utils.flop_counter import FlopCounterMode

    grid = torch.zeros(4, 8)
    storage_bytes = grid.untyped_storage().nbytes()      # 128 bytes, for 32 float32 numbers
    shared = grid.view(32).untyped_storage().data_ptr() == grid.untyped_storage().data_ptr()    # True, a view shares the storage
    flops = 6 * 1000 * 50                                # 300000, for 1,000 parameters trained on 50 tokens

    with FlopCounterMode(display=False) as counter:      # display=False stops it printing a table
        torch.ones(2, 3) @ torch.ones(3, 4)
    counted = counter.get_total_flops()                  # 48, 2 * 2 * 3 * 4


    # TODO: Return how many bytes autograd saves for backward while model runs
    #   on x, as a plain int. Run the model inside
    #   torch.autograd.graph.saved_tensors_hooks, with a pack function that
    #   records the untyped storage of every tensor it receives. Count each
    #   storage's nbytes() only once, using its data_ptr() to tell storages
    #   apart. Leave out the storages that belong to the parameters.
    #   Example: saved_bytes(nn.Linear(6, 3), torch.randn(4, 6)) returns 96.
    def saved_bytes(
        model: nn.Module,
        x: Tensor,
    ) -> int:
        ...


    # TODO: Return the FLOPs of training, as a plain int, using the rule of 6
    #   FLOPs per parameter per token. parameters is the number of parameters
    #   in the model, and tokens is the number of tokens it trains on.
    #   Example: training_flops(1000, 50) returns 300000.
    def training_flops(
        parameters: int,
        tokens: int,
    ) -> int:
        ...


    # TODO: Return the FLOPs that FlopCounterMode(display=False) counts while
    #   model runs a forward pass on x and then a backward pass from the sum of
    #   the output. Return it as a plain int, from the counter's
    #   get_total_flops().
    #   Example: for nn.Linear(3, 4) and x = torch.ones(2, 3), it returns 96.
    def counted_training_flops(
        model: nn.Module,
        x: Tensor,
    ) -> int:
        ...
    ''',
)

COST2_SOLUTION = source_file(
    '''
    # cost2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor
    from torch.utils.flop_counter import FlopCounterMode

    grid = torch.zeros(4, 8)
    storage_bytes = grid.untyped_storage().nbytes()
    shared = grid.view(32).untyped_storage().data_ptr() == grid.untyped_storage().data_ptr()
    flops = 6 * 1000 * 50

    with FlopCounterMode(display=False) as counter:
        torch.ones(2, 3) @ torch.ones(3, 4)
    counted = counter.get_total_flops()


    def saved_bytes(
        model: nn.Module,
        x: Tensor,
    ) -> int:
        parameter_starts = {parameter.untyped_storage().data_ptr() for parameter in model.parameters()}
        sizes: dict[int, int] = {}

        def note(tensor: Tensor) -> Tensor:
            storage = tensor.untyped_storage()
            sizes[storage.data_ptr()] = storage.nbytes()

            return tensor

        with torch.autograd.graph.saved_tensors_hooks(note, lambda tensor: tensor):
            model(x)

        return sum(size for start, size in sizes.items() if start not in parameter_starts)


    def training_flops(
        parameters: int,
        tokens: int,
    ) -> int:
        return 6 * parameters * tokens


    def counted_training_flops(
        model: nn.Module,
        x: Tensor,
    ) -> int:
        with FlopCounterMode(display=False) as counter:
            model(x).sum().backward()

        return counter.get_total_flops()
    ''',
)

COST2_REFERENCE = namespace_of(COST2_SOLUTION)


def needing_gradient(x: Tensor) -> Tensor:
    return x.requires_grad_()


def expect_saved_bytes(
    call: str,
    actual: object,
    expected: int,
) -> None:
    try:
        expect_int(
            call=call,
            actual=actual,
            expected=expected,
        )
    except Mismatch as mismatch:
        if isinstance(actual, int) and actual < expected:
            raise Mismatch(
                f"{mismatch} A count that is too low can mean the pack function returns nothing. Then autograd does "
                "not keep the tensor, its memory gets reused, and a later storage starts at the same data_ptr().",
            ) from None

        raise


def check_cost2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="saved_bytes",
        description="adds up each saved storage once, leaving out the parameters",
        inputs=(
            (Described(seeded_mlp((6, 8, 3), 841), "an MLP from 6 to 8 to 3"), Described(torch.randn(4, 6), "x of shape (4, 6)")),
            (Described(seeded_mlp((6, 8, 3), 842), "an MLP from 6 to 8 to 3"), Described(torch.randn(9, 6), "x of shape (9, 6)")),
            (
                Described(seeded_mlp((5, 7, 7, 2), 843), "an MLP from 5 to 7 to 7 to 2"),
                Described(torch.randn(3, 5), "x of shape (3, 5)"),
            ),
        ),
        reference=COST2_REFERENCE["saved_bytes"],
        expect=expect_saved_bytes,
    )
    check_on_inputs(
        checker=checker,
        function="training_flops",
        description="is 6 FLOPs per parameter per token",
        inputs=(
            (818048, 128),
            (7, 3),
        ),
        reference=COST2_REFERENCE["training_flops"],
        expect=expect_int,
    )
    check_on_inputs(
        checker=checker,
        function="counted_training_flops",
        description="counts a forward and a backward pass",
        inputs=(
            (
                Described(seeded_mlp((64, 128, 32), 844), "an MLP from 64 to 128 to 32"),
                Described(torch.randn(10, 64), "x of shape (10, 64)"),
            ),
            (
                Described(seeded_mlp((6, 8, 3), 845), "an MLP from 6 to 8 to 3"),
                Described(needing_gradient(torch.randn(4, 6)), "x of shape (4, 6) that needs a gradient"),
            ),
        ),
        reference=COST2_REFERENCE["counted_training_flops"],
        expect=expect_int,
    )


COST2 = Exercise(
    name="cost2",
    chapter=COST,
    title="Memory for backward and six FLOPs a token",
    lesson=COST2_LESSON,
    source=COST2_SOURCE,
    solution=COST2_SOLUTION,
    hints={
        "saved_bytes": "First make a set of the parameters' untyped_storage().data_ptr() values. The pack function "
        "stores storage.nbytes() in a dict under the key storage.data_ptr(), then returns the tensor. At the end, "
        "sum the sizes whose key is not in the parameter set.",
        "training_flops": "Multiply 6 by parameters and by tokens.",
        "counted_training_flops": "Inside the with block, run the model on x, sum the output and call backward on "
        "that sum. Then read counter.get_total_flops().",
    },
    introduces=(),
    check=check_cost2,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The numbers in the lessons
# ---------------------------------------------------------------------------


def lesson_gpt() -> nn.Module:
    """The GPT gpt3 trains, untrained, since counting needs no trained weights."""
    torch.manual_seed(0)

    return MODEL_REFERENCE["GPT"](65, 128, 128, 4, 4)


def measure_costs() -> dict[str, object]:
    """Every number in the lessons of cost1 and cost2, on the CPU."""
    model = lesson_gpt()
    ids = torch.randint(0, 65, (1, 128), generator=torch.Generator().manual_seed(0))
    linear_flops = COST1_REFERENCE["linear_flops"]
    block_linears = sum(linear_flops(module, 128) for module in model.h.modules() if isinstance(module, nn.Linear))
    embeddings = model.wte.weight.numel() + model.wpe.weight.numel()
    parameters = sum(parameter.numel() for parameter in model.parameters())

    with FlopCounterMode(display=False) as forward_counter:
        model(ids)

    results: dict[str, object] = {
        "block linears": block_linears,
        "head": linear_flops(model.lm_head, 128),
        "attention": 4 * COST1_REFERENCE["attention_flops"](128, 128),
        "counted forward": forward_counter.get_total_flops(),
        "parameters": parameters,
        "outside the embeddings": parameters - embeddings,
        "parameter bytes": COST1_REFERENCE["parameter_bytes"](model),
        "saved MB": {
            batch: COST2_REFERENCE["saved_bytes"](model, torch.randint(0, 65, (batch, 128))) / 1e6
            for batch in (1, 8, 64)
        },
        "counted training": COST2_REFERENCE["counted_training_flops"](model, ids),
        "six N tokens": COST2_REFERENCE["training_flops"](parameters - embeddings, 128),
    }

    # The counter has no rule for the CPU's fused attention, so this runs attention as plain products it can read.
    with sdpa_kernel(SDPBackend.MATH):
        results["counted training, attention as plain products"] = COST2_REFERENCE["counted_training_flops"](model, ids)

    mlp = seeded_mlp((64, 128, 32), 0)
    results["mlp plain"] = COST2_REFERENCE["counted_training_flops"](mlp, torch.randn(10, 64))
    results["mlp needing a gradient"] = COST2_REFERENCE["counted_training_flops"](mlp, needing_gradient(torch.randn(10, 64)))

    return results
