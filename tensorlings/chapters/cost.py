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

        Section 12 is about running models fast and small. It starts by
        counting, before anything runs. This chapter counts the FLOPs of a
        matrix multiply, a Linear and attention, the bytes of the parameters,
        and the bytes backward needs kept. Then it checks Kaplan and
        colleagues' rule that training costs about six FLOPs per parameter
        per token.

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
    What a model costs can be counted before it runs. Compute is counted in
    FLOPs, floating point operations, each one add or one multiply.

    Multiplying an (m, k) matrix by a (k, n) one makes m times n outputs.
    Each is a dot product of k multiplies and k adds, so the product takes
    2mkn FLOPs.

    ```python
    a, b = torch.randn(32, 64), torch.randn(64, 16)
    product = a @ b    # 32 * 16 outputs of 64 multiplies and 64 adds, 65,536 FLOPs
    ```

    A Linear layer on a batch is one such product, the batch's rows times
    the weight. Its bias adds one add per output. Kaplan and colleagues leave
    biases out of their counts, as too small to matter, and so does this
    chapter.

    Attention over T tokens of width d runs two products. The scores, Q
    times K transposed, are (T, d) by (d, T). The weighted sum of the values
    is (T, T) by (T, d). Both grow with T squared, while a Linear grows with
    T.

    Memory starts with the parameters. Each number takes `element_size()`
    bytes, 4 in float32 and 2 in float16. A weight two layers share, like
    the tied head of gpt2, is stored once, and `parameters()` lists it once.

    PyTorch counts FLOPs too. Every product run inside `FlopCounterMode`,
    from `torch.utils.flop_counter`, adds to its total. Here is the GPT that
    gpt3 trains, on one sequence of 128 characters.

    | Part | FLOPs |
    |---|---|
    | The Linears of the 4 blocks | 201,326,592 |
    | The output head | 2,129,920 |
    | Attention's two products, 4 blocks | 33,554,432 |
    | `FlopCounterMode`'s total | 203,456,512 |

    The counter matches the Linears exactly but misses attention. On the CPU
    this Torch runs attention as one fused kernel, and the counter has no
    rule for it. A tool's count is worth checking by hand.

    Per token, the blocks' Linears cost 1.57 million FLOPs, exactly twice
    their 786,432 weights. Those are nearly all of the 793,344 parameters
    outside the embeddings. That is Kaplan and colleagues' rule of 2N FLOPs
    per token for a forward pass. They leave attention out while the width
    is more than a twelfth of the context. The head reuses the embedding's
    weight, so it sits outside N too. The GPT's 818,048 parameters take 3.27
    MB in float32.
    """,
)

COST1_SOURCE = source_file(
    '''
    # cost1.py

    import torch
    import torch.nn as nn

    product_flops = 2 * 32 * 64 * 16                     # 65536, a (32, 64) matrix times a (64, 16) one
    layer = nn.Linear(64, 16)
    layer_sizes = (layer.in_features, layer.out_features)  # (64, 16)
    half_size = torch.zeros(3, dtype=torch.float16).element_size()    # 2 bytes per number


    # TODO: Return the FLOPs of multiplying an (m, k) matrix by a (k, n) one,
    #   as a plain int. Each output takes k multiplies and k adds.
    def matmul_flops(
        m: int,
        k: int,
        n: int,
    ) -> int:
        ...


    # TODO: Return the FLOPs of linear on a batch, as a plain int, with
    #   matmul_flops. The batch holds as many rows as the number rows says.
    #   Leave the bias out.
    def linear_flops(
        linear: nn.Linear,
        rows: int,
    ) -> int:
        ...


    # TODO: Return the FLOPs of attention's two products, the scores and the
    #   weighted sum of the values, as a plain int, with matmul_flops. The
    #   sequence is tokens long, and each position holds width numbers.
    def attention_flops(
        tokens: int,
        width: int,
    ) -> int:
        ...


    # TODO: Return the bytes model's parameters take, as a plain int, each
    #   parameter counted once, with its own element_size.
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
        description="counts a multiply and an add for each of k terms of every output",
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
        description="counts the product of the rows and the weight",
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
        description="counts every parameter once at its own size",
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
        "matmul_flops": "Two FLOPs for each of the k terms of each of the m times n outputs.",
        "linear_flops": "The batch is (rows, in_features) and the weight acts as (in_features, out_features).",
        "attention_flops": "The scores are (tokens, width) by (width, tokens). The weighted sum is (tokens, tokens) by "
        "(tokens, width), which costs the same.",
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
    Training needs memory beyond the parameters. Backward works out each
    gradient from numbers the forward pass made. A Linear's weight gradient
    needs its input, so the input is kept until backward reaches it.

    Autograd keeps these saved tensors with the graph.
    `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` runs pack on
    every tensor saved inside it, and unpack when backward reads one back.

    ```python
    def note(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(note, lambda tensor: tensor):
        squares = torch.ones(3, 4, requires_grad=True) ** 2    # shapes gets (3, 4)
    ```

    Saved tensors can share one block of memory, a storage. A ReLU keeps its
    output, and the next Linear keeps the same numbers as its input.
    `untyped_storage()` gives a tensor's storage, whose `data_ptr()` says
    where it starts and `nbytes()` how big it is.

    Here is the GPT that gpt3 trains, whose parameters take 3.27 MB.

    | Sequences of 128 in a batch | 1 | 8 | 64 |
    |---|---|---|---|
    | Saved for backward | 4.34 MB | 34.75 MB | 278.00 MB |

    The parameters stay put while the saved tensors grow with the batch. At
    64 sequences they take 85 times the parameters. Inside `torch.no_grad()`
    nothing is saved, which is why evaluating takes less memory than
    training.

    Compute follows a rule from Kaplan and colleagues. A forward pass costs
    about 2 FLOPs per parameter per token, a multiply and an add. Backward
    costs about twice that. Each Linear works out two gradients, for its
    weight and for its input, each a product as big as the forward one. So
    training costs about 6 FLOPs per parameter per token.

    `FlopCounterMode` counts 610,369,536 FLOPs for the GPT's forward and
    backward on one sequence. That is 3 times its forward, and within 0.2% of
    6 times its 793,344 parameters times 128 tokens. The match leaves out
    attention, which the counter misses here, as cost1 showed. Run as plain
    products the counter can read, attention brings the total to 711,032,832,
    about 17% above the rule.

    The first Linear skips the gradient for its input when the input needs
    none. Take an MLP from 64 to 128 to 32 on 10 rows. The counter gives
    573,440 FLOPs for plain inputs and 737,280 for inputs that need a
    gradient.
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
    storage_bytes = grid.untyped_storage().nbytes()      # 128, 32 float32 numbers
    shared = grid.view(32).untyped_storage().data_ptr() == grid.untyped_storage().data_ptr()    # True, a view shares it
    flops = 6 * 1000 * 50                                # 300000, a model of 1,000 parameters on 50 tokens

    with FlopCounterMode(display=False) as counter:      # display=False keeps its table off the screen
        torch.ones(2, 3) @ torch.ones(3, 4)
    counted = counter.get_total_flops()                  # 48, 2 * 2 * 3 * 4


    # TODO: Return the bytes autograd saves for backward while model runs on x,
    #   as a plain int. Inside torch.autograd.graph.saved_tensors_hooks, note
    #   the untyped storage of every tensor packed. Count each storage's
    #   nbytes() once, by its data_ptr(), and leave out the storages of the
    #   parameters.
    def saved_bytes(
        model: nn.Module,
        x: Tensor,
    ) -> int:
        ...


    # TODO: Return the FLOPs of training, as a plain int, by the rule of 6 per
    #   parameter per token. parameters counts the model's parameters, and
    #   tokens the tokens it trains on.
    def training_flops(
        parameters: int,
        tokens: int,
    ) -> int:
        ...


    # TODO: Return the FLOPs FlopCounterMode(display=False) counts for model's
    #   forward pass on x and the backward pass of the output's sum, as a plain
    #   int, from its get_total_flops().
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
                f"{mismatch} Too few can mean the pack function gives nothing back. Autograd then lets go of each "
                "tensor, its memory is used again, and a later storage starts at the same data_ptr().",
            ) from None

        raise


def check_cost2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="saved_bytes",
        description="adds up each saved storage once, the parameters left out",
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
        "saved_bytes": "Make a set of the parameters' untyped_storage().data_ptr() first. The pack function records "
        "storage.data_ptr() against storage.nbytes() in a dict and gives the tensor back. Sum the sizes whose start "
        "is not a parameter's.",
        "training_flops": "Six times the parameters times the tokens.",
        "counted_training_flops": "Run model(x).sum().backward() inside the with block, then read "
        "counter.get_total_flops().",
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
