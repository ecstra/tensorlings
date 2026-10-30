"""91_scaling, power laws fitted in log space, Chinchilla's compute optimal split of a budget between parameters and tokens, and a sweep of your own GPTs at fixed compute."""

from __future__ import annotations

import math
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
    expect_tensor,
    expect_value,
)
from ..text import prose, source_file
from .gpt import DATA_PART, GIVEN_MODEL, GPT_FORWARD, GPT_INIT, INIT_SOLUTION

SCALING = Chapter(
    folder="91_scaling",
    title="Scaling laws",
    readme=source_file(
        """
        # Scaling laws

        A model's loss falls in a smooth curve as its parameters, its data
        and its compute grow, smooth enough to fit with a formula and to
        plan the next run with. This chapter fits power laws in log space,
        splits a compute budget between parameters and tokens the way
        Chinchilla did, and finds the best size for a budget with a sweep of
        your own small GPTs.

        ## Further reading

        - [Kaplan and colleagues, scaling laws for neural language models](https://arxiv.org/abs/2001.08361)
        - [Hoffmann and colleagues, training compute optimal large language models](https://arxiv.org/abs/2203.15556)
        - [Besiroglu and colleagues, Chinchilla scaling, a replication attempt](https://arxiv.org/abs/2404.10102)
        """,
    ),
)


def expect_relative(
    call: str,
    actual: object,
    expected: float,
    share: float = 1e-6,
) -> None:
    """A number checked to a share of its own size, since a fitted scale can run to hundreds."""
    expect_close(
        call=call,
        actual=actual,
        expected=expected,
        tolerance=share * max(abs(expected), 1.0),
    )


def expect_relative_pair(*names: str) -> Callable[..., None]:
    def expect(
        call: str,
        actual: object,
        expected: tuple[float, float],
    ) -> None:
        if actual is None:
            raise Mismatch(f"{call} is None, which usually means it is not written yet.")

        if not isinstance(actual, (tuple, list)) or len(actual) != len(names):
            raise Mismatch(f"{call} should give 2 things, {names[0]} and {names[1]}, but it gives {actual!r}.")

        for name, part, wanted in zip(names, actual, expected):
            expect_relative(
                call=f"{name} from {call}",
                actual=part,
                expected=wanted,
            )

    return expect


# ---------------------------------------------------------------------------
# scaling1
# ---------------------------------------------------------------------------

FIT_SOLUTION = '''
def power_law(
    n: Tensor,
    scale: float,
    alpha: float,
) -> Tensor:
    return scale * n ** -alpha


def fit_power_law(
    n: Tensor,
    loss: Tensor,
) -> tuple[float, float]:
    with_ones = torch.stack([torch.log(n), torch.ones_like(n)], dim=1)
    slope, intercept = torch.linalg.lstsq(with_ones, torch.log(loss)).solution.tolist()

    return math.exp(intercept), -slope


def fit_above_floor(
    n: Tensor,
    loss: Tensor,
    floor: float,
) -> tuple[float, float]:
    return fit_power_law(n, loss - floor)
'''


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "load_shakespeare": load_shakespeare,
    }
    exec(code, namespace)

    return namespace


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return "\n".join("    " + line if line else "" for line in part.strip("\n").splitlines())


FIT_REFERENCE = namespace_of(FIT_SOLUTION)

# Hoffmann and colleagues' law with unlimited data, E + A / N ** alpha, from Equation 10 of the Chinchilla paper.
FLOOR = 1.69


def chinchilla_without_data_limit(n: Tensor) -> Tensor:
    return FLOOR + 406.4 * n ** -0.34


def measure_floor_fit() -> list[tuple[float, float, float, float]]:
    """scaling1's table, a pure power law and one above the floor, both fitted between 10 million and 1 billion."""
    seen = torch.logspace(7, 9, 9, dtype=torch.float64)
    pure = FIT_REFERENCE["fit_power_law"](seen, chinchilla_without_data_limit(seen))
    above = FIT_REFERENCE["fit_above_floor"](seen, chinchilla_without_data_limit(seen), FLOOR)
    rows = []

    for size in (1e7, 1e9, 1e10, 1e11, 1e12):
        n = torch.tensor(size, dtype=torch.float64)
        rows.append(
            (
                size,
                chinchilla_without_data_limit(n).item(),
                FIT_REFERENCE["power_law"](n, *pure).item(),
                FLOOR + FIT_REFERENCE["power_law"](n, *above).item(),
            ),
        )

    return rows


SCALING1_LESSON = prose(
    """
    Kaplan and colleagues trained language models from 768 to 1.5 billion
    parameters and measured each one's test loss. The loss fell along a
    smooth curve. They counted N as the parameters outside the embeddings and
    fitted the curve with one formula.

    ```text
    L(N) = (N_c / N) ** alpha        alpha about 0.076, N_c about 8.8e13
    ```

    That is a **power law**, a number raised to a fixed power. Ten times the
    parameters multiplies the loss by `10 ** -0.076`, about 0.84, wherever on
    the curve you start.

    Take the log of both sides and it turns into a line.

    ```text
    log L = -alpha * log N + alpha * log N_c
    ```

    So log L against log N is a line with slope minus alpha. Fitting a power
    law is fitting that line, with linreg3's least squares, then undoing the
    logs with `math.exp`.

    ```python
    torch.log(torch.tensor([10.0, 100.0]))    # tensor([2.3026, 4.6052])
    torch.tensor([3.0, 4.0]).tolist()         # [3.0, 4.0], plain numbers
    ```

    A power law falls forever, but no model's loss falls below the entropy
    of the text itself. Hoffmann and colleagues, in the Chinchilla paper,
    fitted that floor in. With unlimited data their law is `1.69 + 406.4 / N
    ** 0.34`. Here are two fits to that law between 10 million and 1 billion
    parameters, then followed further out.

    | N | The law | A pure power law | A power law above 1.69 |
    |---|---|---|---|
    | 10 million | 3.384 | 3.253 | 3.384 |
    | 1 billion | 2.044 | 1.971 | 2.044 |
    | 10 billion | 1.852 | 1.534 | 1.852 |
    | 100 billion | 1.764 | 1.194 | 1.764 |
    | 1 trillion | 1.724 | 0.930 | 1.724 |

    The pure fit has alpha 0.109, where the law has 0.34. It runs below 1.69,
    a loss the law says no model reaches. Take the floor off first and what
    remains is a power law again, so the fit follows it.
    """,
)

SCALING1_SOURCE = source_file(
    '''
    # scaling1.py

    import math

    import torch
    from torch import Tensor

    sizes = torch.tensor([1e6, 1e7, 1e8])
    logs = torch.log(sizes)                                  # tensor([13.8155, 16.1181, 18.4207])
    with_ones = torch.stack([logs, torch.ones(3)], dim=1)    # shape (3, 2), as linreg3 built it
    back = math.exp(2.0)                                     # 7.389..., what torch.log(7.389) undoes


    # TODO: Return scale * n ** -alpha for every size in n.
    def power_law(
        n: Tensor,
        scale: float,
        alpha: float,
    ) -> Tensor:
        ...


    # TODO: Fit loss = scale * n ** -alpha with least squares on the logs, as
    #   linreg3 fitted a line. log loss against log n has slope -alpha and
    #   intercept log scale. Return scale and alpha as plain numbers.
    def fit_power_law(
        n: Tensor,
        loss: Tensor,
    ) -> tuple[float, float]:
        ...


    # TODO: Fit loss = floor + scale * n ** -alpha for a known floor. Fit a
    #   power law to what each loss holds above the floor, and return its
    #   scale and alpha.
    def fit_above_floor(
        n: Tensor,
        loss: Tensor,
        floor: float,
    ) -> tuple[float, float]:
        ...
    ''',
)

SCALING1_SOLUTION = source_file(
    '''
    # scaling1.py, a solution for comparison

    import math

    import torch
    from torch import Tensor

    sizes = torch.tensor([1e6, 1e7, 1e8])
    logs = torch.log(sizes)
    with_ones = torch.stack([logs, torch.ones(3)], dim=1)
    back = math.exp(2.0)


''' + indented(FIT_SOLUTION) + '''
    ''',
)


def noisy_power_law(
    seed: int,
    count: int,
    scale: float,
    alpha: float,
    floor: float,
) -> tuple[Tensor, Tensor]:
    """Sizes spread over several powers of ten, and losses off a power law by a few percent, in float64 so any
    right way of fitting agrees to many digits."""
    generator = torch.Generator().manual_seed(seed)
    n = torch.logspace(3, 3 + count * 0.7, count, dtype=torch.float64)
    wobble = 1 + 0.03 * torch.randn(count, generator=generator, dtype=torch.float64)

    return n, floor + scale * n ** -alpha * wobble


POWER_DATA = (
    noisy_power_law(911, 7, 40.0, 0.25, 0.0),
    noisy_power_law(912, 9, 6.5, 0.08, 0.0),
)

FLOOR_DATA = (
    (*noisy_power_law(913, 8, 30.0, 0.3, 1.5), 1.5),
    (*noisy_power_law(914, 6, 400.0, 0.34, 1.69), 1.69),
)


def check_scaling1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="power_law",
        description="gives scale * n ** -alpha",
        inputs=(
            (torch.tensor([1.0, 10.0, 100.0, 1000.0]), 2.0, 0.5),
            (torch.tensor([3.0, 7.0, 50.0]), 8.8, 0.076),
        ),
        reference=FIT_REFERENCE["power_law"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="fit_power_law",
        description="fits a line to the logs and gives scale and alpha",
        inputs=POWER_DATA,
        reference=FIT_REFERENCE["fit_power_law"],
        expect=expect_relative_pair("scale", "alpha"),
    )
    check_on_inputs(
        checker=checker,
        function="fit_above_floor",
        description="fits a power law to what lies above the floor",
        inputs=FLOOR_DATA,
        reference=FIT_REFERENCE["fit_above_floor"],
        expect=expect_relative_pair("scale", "alpha"),
    )


SCALING1 = Exercise(
    name="scaling1",
    chapter=SCALING,
    title="Power laws",
    lesson=SCALING1_LESSON,
    source=SCALING1_SOURCE,
    solution=SCALING1_SOLUTION,
    hints={
        "power_law": "n ** -alpha works on every entry of n at once. Multiply it by scale.",
        "fit_power_law": "Stack torch.log(n) and torch.ones_like(n) as columns with dim=1, and solve against "
        "torch.log(loss) with torch.linalg.lstsq. .solution.tolist() gives the slope and the intercept. Return "
        "math.exp of the intercept and minus the slope.",
        "fit_above_floor": "Hand fit_power_law the losses with floor taken off each one.",
    },
    introduces=("power law",),
    check=check_scaling1,
)


# ---------------------------------------------------------------------------
# scaling2
# ---------------------------------------------------------------------------

# Hoffmann and colleagues' fit as printed in the Chinchilla paper, Equation 10, and Besiroglu and colleagues' refit
# of the same data, Table 1 of their replication, each as (E, A, B, alpha, beta).
PRINTED = (1.69, 406.4, 410.7, 0.34, 0.28)
REFIT = (1.8172, 482.01, 2085.43, 0.3478, 0.3658)

BUDGET_SOLUTION = '''
def training_flops(
    parameters: float,
    tokens: float,
) -> float:
    return 6 * parameters * tokens


def chinchilla_loss(
    parameters: Tensor,
    tokens: Tensor,
    fit: tuple[float, float, float, float, float],
) -> Tensor:
    floor, a, b, alpha, beta = fit

    return floor + a / parameters**alpha + b / tokens**beta


def best_size(
    budget: float,
    sizes: Tensor,
    fit: tuple[float, float, float, float, float],
) -> float:
    tokens = budget / (6 * sizes)
    losses = chinchilla_loss(sizes, tokens, fit)

    return sizes[losses.argmin()].item()


def tokens_per_parameter(
    budget: float,
    sizes: Tensor,
    fit: tuple[float, float, float, float, float],
) -> float:
    size = best_size(budget, sizes, fit)

    return budget / (6 * size) / size
'''

BUDGET_REFERENCE = namespace_of(BUDGET_SOLUTION)

GRID = torch.logspace(7, 12, 1001, dtype=torch.float64)


def measure_budgets() -> list[tuple[float, float, float, float, float]]:
    """scaling2's table, the best size and its tokens per parameter under the printed fit and the refit."""
    rows = []

    for budget in (1e20, 1e22, 5.76e23):
        rows.append(
            (
                budget,
                BUDGET_REFERENCE["best_size"](budget, GRID, PRINTED),
                BUDGET_REFERENCE["tokens_per_parameter"](budget, GRID, PRINTED),
                BUDGET_REFERENCE["best_size"](budget, GRID, REFIT),
                BUDGET_REFERENCE["tokens_per_parameter"](budget, GRID, REFIT),
            ),
        )

    return rows


SCALING2_LESSON = prose(
    """
    Training costs about 6 FLOPs, floating point operations, for every
    parameter and every token. Kaplan and colleagues count 2 for the forward
    pass and twice that for the backward pass. So a budget of C FLOPs buys
    any split with `6 * N * D = C`, N parameters trained on D tokens.

    A bigger model on fewer tokens, or a smaller one on more? Hoffmann and
    colleagues trained over 400 models and fitted the loss to both.

    ```text
    L(N, D) = E + A / N ** alpha + B / D ** beta
    ```

    E is the floor of scaling1. The second term is what a model of N
    parameters can't learn, and the third is what D tokens of training
    haven't taught it yet. Fix the budget, give each size N its `C / (6 * N)`
    tokens, and the loss dips to a lowest point. That size is **compute
    optimal** for the budget.

    ```python
    torch.logspace(7, 9, 3)                      # tensor([1.0000e+07, 1.0000e+08, 1.0000e+09])
    torch.tensor([3.0, 1.0, 2.0]).argmin()       # tensor(1), where the smallest sits
    ```

    Gopher had 280 billion parameters and saw 300 billion tokens. Chinchilla
    spent the same compute on 70 billion parameters and 1.4 trillion tokens,
    and beat Gopher on a large range of tasks. That is 20 tokens for every
    parameter, close to the ratio of the paper's Table 3 of compute optimal
    sizes.

    The fit's printed constants disagree with that. In 2024 Besiroglu and
    colleagues refitted data read off the paper's plots. They found the
    printed fit disagreed with the paper's other two methods. Here are both
    fits, on a grid of sizes.

    | Budget in FLOPs | Printed fit, best N | Its tokens per parameter | Refit, best N | Its tokens per parameter |
    |---|---|---|---|---|
    | 1e20 | 646 million | 40 | 851 million | 23 |
    | 1e22 | 5.19 billion | 62 | 9.02 billion | 20.5 |
    | 5.76e23, Gopher's | 32.4 billion | 92 | 72.4 billion | 18.3 |

    The refit lands on Chinchilla's own size. A fitted law is worth what its
    data and its fit are worth, so check it against runs you trust.
    """,
)

SCALING2_SOURCE = source_file(
    '''
    # scaling2.py

    import torch
    from torch import Tensor

    # (E, A, B, alpha, beta) as the Chinchilla paper prints them, and as Besiroglu
    # and colleagues refitted them.
    PRINTED = (1.69, 406.4, 410.7, 0.34, 0.28)
    REFIT = (1.8172, 482.01, 2085.43, 0.3478, 0.3658)

    sizes = torch.logspace(7, 9, 3)                  # tensor([1.0000e+07, 1.0000e+08, 1.0000e+09])
    lowest = torch.tensor([3.0, 1.0, 2.0]).argmin()  # tensor(1)


    # TODO: Return the FLOPs of training parameters parameters on tokens tokens,
    #   6 for every parameter and every token.
    def training_flops(
        parameters: float,
        tokens: float,
    ) -> float:
        ...


    # TODO: Return fit's loss for each pair of a size in parameters and a count
    #   in tokens. fit holds E, A, B, alpha and beta in that order.
    def chinchilla_loss(
        parameters: Tensor,
        tokens: Tensor,
        fit: tuple[float, float, float, float, float],
    ) -> Tensor:
        ...


    # TODO: Give every size in sizes the tokens that budget buys it, and return
    #   the size with the lowest loss under fit, as a plain number.
    def best_size(
        budget: float,
        sizes: Tensor,
        fit: tuple[float, float, float, float, float],
    ) -> float:
        ...


    # TODO: Return the tokens the best size gets for budget, divided by that size.
    def tokens_per_parameter(
        budget: float,
        sizes: Tensor,
        fit: tuple[float, float, float, float, float],
    ) -> float:
        ...
    ''',
)

SCALING2_SOLUTION = source_file(
    '''
    # scaling2.py, a solution for comparison

    import torch
    from torch import Tensor

    PRINTED = (1.69, 406.4, 410.7, 0.34, 0.28)
    REFIT = (1.8172, 482.01, 2085.43, 0.3478, 0.3658)

    sizes = torch.logspace(7, 9, 3)
    lowest = torch.tensor([3.0, 1.0, 2.0]).argmin()


''' + indented(BUDGET_SOLUTION) + '''
    ''',
)

# A made up fit, so a learner who types either set of constants into the function fails.
OTHER_FIT = (2.0, 150.0, 900.0, 0.3, 0.33)


def size_grid(
    low: float,
    high: float,
    count: int,
) -> Tensor:
    return torch.logspace(low, high, count, dtype=torch.float64)


def check_scaling2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="training_flops",
        description="gives 6 FLOPs per parameter per token",
        inputs=(
            (70e9, 1.4e12),
            (818048.0, 24576000.0),
        ),
        reference=BUDGET_REFERENCE["training_flops"],
        expect=expect_relative,
    )
    check_on_inputs(
        checker=checker,
        function="chinchilla_loss",
        description="gives E + A / N ** alpha + B / D ** beta",
        inputs=(
            (torch.tensor([1e8, 1e9, 7e10], dtype=torch.float64), torch.tensor([2e9, 2e10, 1.4e12], dtype=torch.float64), PRINTED),
            (torch.tensor([5e6, 3e9], dtype=torch.float64), torch.tensor([1e11, 4e9], dtype=torch.float64), OTHER_FIT),
        ),
        reference=BUDGET_REFERENCE["chinchilla_loss"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="best_size",
        description="finds the size with the lowest loss for a budget",
        inputs=(
            (1e21, size_grid(7, 12, 201), REFIT),
            (3e19, size_grid(6, 11, 151), OTHER_FIT),
            (5.76e23, size_grid(8, 12, 81), PRINTED),
        ),
        reference=BUDGET_REFERENCE["best_size"],
        expect=expect_relative,
    )
    check_on_inputs(
        checker=checker,
        function="tokens_per_parameter",
        description="divides the best size's tokens by its size",
        inputs=(
            (1e22, size_grid(7, 12, 201), REFIT),
            (2e20, size_grid(6, 11, 151), OTHER_FIT),
        ),
        reference=BUDGET_REFERENCE["tokens_per_parameter"],
        expect=expect_relative,
    )


SCALING2 = Exercise(
    name="scaling2",
    chapter=SCALING,
    title="The compute optimal size",
    lesson=SCALING2_LESSON,
    source=SCALING2_SOURCE,
    solution=SCALING2_SOLUTION,
    hints={
        "training_flops": "Multiply 6, parameters and tokens.",
        "chinchilla_loss": "Unpack fit into five names first. Then E plus A over parameters ** alpha plus B over "
        "tokens ** beta, on whole tensors at once.",
        "best_size": "Each size gets budget / (6 * sizes) tokens. Score them all with chinchilla_loss, then index "
        "sizes with the argmin and take .item().",
        "tokens_per_parameter": "Get the size from best_size. Its tokens are budget / (6 * size), then divide by "
        "size once more.",
    },
    introduces=("compute optimal",),
    check=check_scaling2,
)


# ---------------------------------------------------------------------------
# scaling3
# ---------------------------------------------------------------------------

# gpt2's model, gpt1's data, and a short training run whose cosine ends with it, as Chinchilla matched each run's
# schedule to its tokens. Small sizes train on the CPU, where the same seeds give the same numbers every run.
SWEEP_GIVEN = '''
VOCABULARY = 65
BLOCK = 64
BATCH = 32
LAYERS = 2
PEAK = 3e-3


def learning_rate(
    step: int,
    steps: int,
) -> float:
    warmup = max(1, steps // 20)

    if step < warmup:
        return PEAK * (step + 1) / warmup

    progress = (step - warmup) / max(1, steps - warmup)

    return PEAK / 10 + 0.5 * (PEAK - PEAK / 10) * (1 + math.cos(math.pi * progress))


def train(
    model: nn.Module,
    steps: int,
    data: Tensor,
    generator: torch.Generator,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK, weight_decay=0.0)

    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)

        x, y = get_batch(data, BLOCK, BATCH, generator)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validation_score(
    model: nn.Module,
    data: Tensor,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(1)
    losses = []

    with torch.no_grad():
        for _ in range(20):
            x, y = get_batch(data, BLOCK, 64, generator)
            losses.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item())

    model.train()

    return sum(losses) / len(losses)
'''

SWEEP_SOLUTION = '''
def non_embedding_parameters(model: nn.Module) -> int:
    counts = [parameter.numel() for name, parameter in model.named_parameters() if not name.startswith(("wte", "wpe"))]

    return sum(counts)


def steps_for_budget(
    budget: float,
    parameters: int,
) -> int:
    return max(1, round(budget / (6 * parameters * BLOCK * BATCH)))


def isoflop_point(
    width: int,
    budget: float,
    train_data: Tensor,
    validation_data: Tensor,
) -> tuple[int, float]:
    torch.manual_seed(0)
    model = GPT(VOCABULARY, BLOCK, width, LAYERS, max(1, width // 16))
    parameters = non_embedding_parameters(model)
    train(model, steps_for_budget(budget, parameters), train_data, torch.Generator().manual_seed(0))

    return parameters, validation_score(model, validation_data)


def fit_parabola(points: list[tuple[int, float]]) -> float:
    sizes = torch.log(torch.tensor([size for size, _ in points], dtype=torch.float64))
    losses = torch.tensor([loss for _, loss in points], dtype=torch.float64)
    columns = torch.stack([sizes**2, sizes, torch.ones_like(sizes)], dim=1)
    a, b, _ = torch.linalg.lstsq(columns, losses).solution.tolist()

    return math.exp(-b / (2 * a))
'''

SWEEP_REFERENCE = namespace_of(GIVEN_MODEL + INIT_SOLUTION + GPT_INIT + GPT_FORWARD + DATA_PART + SWEEP_GIVEN + SWEEP_SOLUTION)

# Each budget's widths sit around its valley, since a parabola only fits near its bottom. The first row is the
# sweep the learner's file runs, kept to four widths since it runs on every check.
SWEEPS: tuple[tuple[float, tuple[int, ...]], ...] = (
    (3e10, (8, 12, 16, 24)),
    (1e11, (8, 12, 16, 24, 32)),
    (3e11, (12, 16, 24, 32, 48)),
    (1e12, (16, 24, 32, 48, 64)),
)


def measure_isoflops() -> list[dict[str, object]]:
    """scaling3's table, every sweep on the CPU, where the same seeds give the same numbers. Not run by any check."""
    _, train_data, validation_data = SWEEP_REFERENCE["shakespeare_splits"]()
    rows = []

    for budget, widths in SWEEPS:
        points = [SWEEP_REFERENCE["isoflop_point"](width, budget, train_data, validation_data) for width in widths]
        best = SWEEP_REFERENCE["fit_parabola"](points)
        lowest = min(range(len(points)), key=lambda position: points[position][1])
        rows.append(
            {
                "budget": budget,
                "points": points,
                "lowest width": widths[lowest],
                "lowest loss": points[lowest][1],
                "fitted best": best,
                "tokens per parameter": budget / (6 * best) / best,
            },
        )

    return rows


def size_exponent(rows: list[dict[str, object]]) -> float:
    """How the fitted best size grows with the budget, scaling1's power law fitted to measure_isoflops' rows."""
    budgets = torch.tensor([row["budget"] for row in rows], dtype=torch.float64)
    bests = torch.tensor([row["fitted best"] for row in rows], dtype=torch.float64)
    _, alpha = FIT_REFERENCE["fit_power_law"](budgets, bests)

    return -alpha


SCALING3_LESSON = prose(
    """
    scaling2's best sizes come from fits to hundreds of runs. Here is the
    method behind one such fit, Chinchilla's second approach, at a size
    this laptop trains.

    - Fix a budget of FLOPs.
    - Train models of several sizes, each on the tokens the budget buys it, with a cosine schedule that ends with the run.
    - Loss against log N makes a valley. Fit a parabola to it, and its bottom is the compute optimal size.

    N counts what Kaplan counted, the parameters outside the embeddings
    `wte` and `wpe`. `named_parameters()` gives each parameter with its name.
    The head shares `wte`'s table, so it is listed once, under `wte`.

    ```python
    layer = nn.Linear(2, 3)
    [(name, p.numel()) for name, p in layer.named_parameters()]    # [('weight', 6), ('bias', 3)]
    "wpe.weight".startswith(("wte", "wpe"))                         # True
    round(2.5), round(3.7)                                          # (2, 4)
    ```

    Each step trains on `BATCH * BLOCK` tokens, so a budget C buys a model
    of N parameters `C / (6 * N * BATCH * BLOCK)` steps.

    The parabola is `a * x ** 2 + b * x + c` with `x = log N`, a least
    squares fit on the columns `x ** 2`, `x` and 1. Its bottom sits at `x =
    -b / (2 * a)`. A parabola only fits near its bottom, so each budget below
    tried four or five widths around its valley, 2 blocks each.

    | Budget in FLOPs | Widths | Lowest loss, at width | Fitted best N | Its tokens per parameter |
    |---|---|---|---|---|
    | 3e10 | 8 to 24 | 2.513 at 12 | 2,797 | 639 |
    | 1e11 | 8 to 32 | 2.306 at 16 | 5,948 | 471 |
    | 3e11 | 12 to 48 | 2.054 at 24 | 10,857 | 424 |
    | 1e12 | 16 to 64 | 1.847 at 32 | 28,100 | 211 |

    The best size grows with the budget, as scaling2 said it should.
    scaling1's power law through these four sizes grows N as C to the 0.65.
    The Chinchilla paper puts Kaplan's exponent at 0.73 and its own near 0.5.

    The ratio is nowhere near Chinchilla's 20, and it falls as the budget
    grows. This toy is far below the sizes those laws were fitted on. It
    reads characters in place of words, and goes over the same play up to
    seven times. Measure your own setting before trusting a law fitted on
    someone else's.

    The last lines of your file run the first row's sweep each time the file
    is checked, four small models on the CPU. Its numbers show in the result.
    """,
)

SCALING3_SOURCE = source_file(
    '''
    # scaling3.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    layer = nn.Linear(2, 3)
    counts = [(name, p.numel()) for name, p in layer.named_parameters()]    # [('weight', 6), ('bias', 3)]
    is_position = "wpe.weight".startswith(("wte", "wpe"))                   # True


    # The model from gpt2, and get_batch and the data from gpt1.
''' + indented(GIVEN_MODEL) + '''


''' + indented(INIT_SOLUTION) + '''


''' + indented(GPT_INIT) + '''

''' + indented(GPT_FORWARD) + '''


''' + indented(DATA_PART) + '''


    # Small GPTs of 2 blocks, trained on the CPU with a cosine that ends with the run.
''' + indented(SWEEP_GIVEN) + '''


    # TODO: Return how many numbers model's parameters hold, leaving out the
    #   ones whose names start with wte or wpe.
    def non_embedding_parameters(model: nn.Module) -> int:
        ...


    # TODO: Return the steps budget FLOPs buy a model of parameters parameters,
    #   at 6 FLOPs per parameter per token and BATCH * BLOCK tokens a step.
    #   Round to the nearest whole step, and give at least 1.
    def steps_for_budget(
        budget: float,
        parameters: int,
    ) -> int:
        ...


    # TODO: Seed torch with 0 and make GPT(VOCABULARY, BLOCK, width, LAYERS,
    #   max(1, width // 16)). Train it for the steps budget buys it, drawing
    #   batches from train_data with a generator seeded with 0. Return its
    #   non embedding parameters and its validation_score on validation_data.
    def isoflop_point(
        width: int,
        budget: float,
        train_data: Tensor,
        validation_data: Tensor,
    ) -> tuple[int, float]:
        ...


    # TODO: Fit loss = a * x ** 2 + b * x + c to points, pairs of a size and a
    #   loss, with x the log of the size, by least squares. Return the size at
    #   the parabola's bottom as a plain number.
    def fit_parabola(points: list[tuple[int, float]]) -> float:
        ...


    # Your own sweep, the lesson's first row. It trains four models each time
    # the file is checked.
    characters, train_data, validation_data = shakespeare_splits()
    WIDTHS = [8, 12, 16, 24]
    points = [isoflop_point(width, 3e10, train_data, validation_data) for width in WIDTHS]
    best = fit_parabola(points)
    ''',
)

SCALING3_SOLUTION = source_file(
    '''
    # scaling3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    layer = nn.Linear(2, 3)
    counts = [(name, p.numel()) for name, p in layer.named_parameters()]
    is_position = "wpe.weight".startswith(("wte", "wpe"))


''' + indented(GIVEN_MODEL) + '''


''' + indented(INIT_SOLUTION) + '''


''' + indented(GPT_INIT) + '''

''' + indented(GPT_FORWARD) + '''


''' + indented(DATA_PART) + '''


''' + indented(SWEEP_GIVEN) + '''


''' + indented(SWEEP_SOLUTION) + '''


    characters, train_data, validation_data = shakespeare_splits()
    WIDTHS = [8, 12, 16, 24]
    points = [isoflop_point(width, 3e10, train_data, validation_data) for width in WIDTHS]
    best = fit_parabola(points)
    ''',
)

OWN_WIDTHS = (8, 12, 16, 24)


def seeded_sweep_gpt(width: int) -> nn.Module:
    torch.manual_seed(0)

    return SWEEP_REFERENCE["GPT"](65, 64, width, 2, max(1, width // 16))


def described_gpt(
    settings: tuple[int, int, int, int, int],
) -> Described:
    torch.manual_seed(0)

    return Described(
        value=SWEEP_REFERENCE["GPT"](*settings),
        text=f"GPT{settings}",
    )


def repeating_text(
    seed: int,
    length: int,
) -> Tensor:
    """Characters from a short random pattern said over and over, so a few steps of training already lower the loss."""
    pattern = torch.randint(0, 65, (37,), generator=torch.Generator().manual_seed(seed))

    return pattern.repeat(length // 37 + 1)[:length]


def check_isoflop_point(found: object) -> None:
    for width, budget, seed in ((8, 1e9, 931), (12, 2e9, 932)):
        train_data = repeating_text(seed, 3000)
        validation_data = repeating_text(seed, 1500)
        call = f"isoflop_point({width}, {budget:g}, a repeating text of 3,000 characters, one of 1,500)"
        actual = found(width, budget, train_data.clone(), validation_data.clone())
        expected = SWEEP_REFERENCE["isoflop_point"](width, budget, train_data, validation_data)

        if actual is None:
            raise Mismatch(f"{call} is None, which usually means it is not written yet.")

        if not isinstance(actual, (tuple, list)) or len(actual) != 2:
            raise Mismatch(f"{call} should give 2 things, the parameters and the loss, but it gives {actual!r}.")

        expect_value(
            call=f"the parameters from {call}",
            actual=actual[0],
            expected=expected[0],
        )
        expect_close(
            call=f"the loss from {call}",
            actual=actual[1],
            expected=expected[1],
            tolerance=1e-4,
        )


def parabola_points(
    bottom: float,
    curve: float,
    sizes: tuple[float, ...],
    seed: int,
) -> list[tuple[int, float]]:
    """Points near a parabola in log size with its bottom at bottom, a little noise added."""
    generator = torch.Generator().manual_seed(seed)
    points = []

    for size in sizes:
        noise = 0.002 * torch.randn((), generator=generator, dtype=torch.float64).item()
        points.append((int(size), 2.0 + curve * (math.log(size) - math.log(bottom)) ** 2 + noise))

    return points


def check_points(value: object) -> None:
    if not isinstance(value, list) or len(value) != len(OWN_WIDTHS):
        raise Mismatch(f"points should be a list of {len(OWN_WIDTHS)} pairs, one per width, but it is {value!r}.")

    for width, point in zip(OWN_WIDTHS, value):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise Mismatch(f"Each point should be a pair of the parameters and the loss, but one is {point!r}.")

        expect_value(
            call=f"the parameters of width {width}",
            actual=point[0],
            expected=SWEEP_REFERENCE["non_embedding_parameters"](seeded_sweep_gpt(width)),
        )

        if not isinstance(point[1], float) or not 0.5 < point[1] < 5.0:
            raise Mismatch(f"The loss of width {width} is {point[1]!r}, but a validation loss here sits between 0.5 and 5.")


def check_best(
    value: object,
    points: object,
) -> None:
    try:
        check_points(points)
    except Mismatch:
        raise Mismatch("best is fitted to points, which isn't right yet. Fix points first.") from None

    expect_relative(
        call="best",
        actual=value,
        expected=SWEEP_REFERENCE["fit_parabola"](points),
    )


def check_scaling3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="non_embedding_parameters",
        description="counts the parameters outside wte and wpe",
        inputs=(
            (described_gpt((65, 64, 24, 2, 1)),),
            (described_gpt((20, 16, 32, 3, 4)),),
        ),
        reference=SWEEP_REFERENCE["non_embedding_parameters"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="steps_for_budget",
        description="gives the steps a budget buys, rounded and at least 1",
        inputs=(
            (1e12, 25472),
            (3e10, 1760),
            (1e9, 400000),
        ),
        reference=SWEEP_REFERENCE["steps_for_budget"],
        expect=expect_value,
    )
    checker.test(
        function="isoflop_point",
        description="trains one size on its share of the budget and scores it",
        run=check_isoflop_point,
    )
    check_on_inputs(
        checker=checker,
        function="fit_parabola",
        description="gives the size at the bottom of a parabola in log size",
        inputs=(
            (parabola_points(20000.0, 0.08, (5000.0, 10000.0, 20000.0, 40000.0, 80000.0), 933),),
            (parabola_points(3000.0, 0.15, (1700.0, 3800.0, 6600.0, 14500.0), 934),),
        ),
        reference=SWEEP_REFERENCE["fit_parabola"],
        expect=expect_relative,
    )
    checker.check_name(
        name="points",
        description="holds your sweep, a pair of parameters and loss per width",
        run=check_points,
    )
    checker.check_name(
        name="best",
        description="is the bottom of the parabola through your points",
        run=lambda value: check_best(
            value=value,
            points=getattr(checker.module, "points", None),
        ),
    )


SCALING3 = Exercise(
    name="scaling3",
    chapter=SCALING,
    title="A sweep of your own",
    lesson=SCALING3_LESSON,
    source=SCALING3_SOURCE,
    solution=SCALING3_SOLUTION,
    hints={
        "non_embedding_parameters": "Sum parameter.numel() over model.named_parameters(), skipping the names "
        "that start with wte or wpe.",
        "steps_for_budget": "Divide budget by 6 * parameters * BLOCK * BATCH, round it, and take the max of that "
        "and 1.",
        "isoflop_point": "Count the parameters with non_embedding_parameters and the steps with steps_for_budget. "
        "Call train with torch.Generator().manual_seed(0), then return the count and validation_score.",
        "fit_parabola": "Make float64 tensors of torch.log of the sizes and of the losses. Stack x ** 2, x and "
        "ones as columns, solve with torch.linalg.lstsq, and turn -b / (2 * a) back with math.exp.",
        "points": "Write the four functions above first. This line only calls isoflop_point once per width.",
        "best": "This line calls fit_parabola on points.",
    },
    introduces=(),
    check=check_scaling3,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)
