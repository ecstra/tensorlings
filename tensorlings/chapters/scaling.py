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
    value_text,
)
from ..text import prose, source_file
from .gpt import DATA_PART, GIVEN_MODEL, GPT_FORWARD, GPT_INIT, INIT_SOLUTION

SCALING = Chapter(
    folder="91_scaling",
    title="Scaling laws",
    readme=source_file(
        """
        # Scaling laws

        As a model's parameters, data and compute grow, its loss falls along
        a smooth curve. The curve is smooth enough to fit with a formula, and
        that formula can plan the next training run. This chapter fits power
        laws in log space. It splits a compute budget between parameters and
        tokens the way the Chinchilla paper did. Then it finds the best model
        size for a budget with a sweep of your own small GPTs.

        ## Further reading

        - [Kaplan and colleagues, scaling laws for neural language models](https://arxiv.org/abs/2001.08361)
        - [Hoffmann and colleagues, training compute optimal large language models](https://arxiv.org/abs/2203.15556)
        - [Besiroglu and colleagues, Chinchilla scaling, a replication attempt](https://arxiv.org/abs/2404.10102)
        - [Pearce and Song, reconciling Kaplan and Chinchilla scaling laws](https://arxiv.org/abs/2406.12907)
        - [Porian and colleagues, resolving discrepancies in compute optimal scaling of language models](https://arxiv.org/abs/2406.19146)
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


# A least squares fit in float32 lands a few millionths away from one in float64. The fitted points are noisy by
# percents, so any precision is a right answer, and the fits are checked to a ten thousandth of their size.
FIT_SHARE = 1e-4


def expect_fit(
    call: str,
    actual: object,
    expected: float,
) -> None:
    expect_relative(
        call=call,
        actual=actual,
        expected=expected,
        share=FIT_SHARE,
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
            raise Mismatch(f"{call} should return 2 things, {names[0]} and {names[1]}, but it returns {actual!r}.")

        for name, part, wanted in zip(names, actual, expected):
            expect_fit(
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
    Kaplan and colleagues trained language models with 768 to 1.5 billion
    parameters, and measured each one's test loss. The loss fell along a
    smooth curve as the models grew. They let N be the number of parameters
    outside the embeddings, and fitted the curve with one formula.

    ```text
    L(N) = (N_c / N) ** alpha        alpha about 0.076, N_c about 8.8e13
    ```

    This is a **power law**, a formula where the input is raised to a fixed
    power and multiplied by a constant. Multiplying the parameters by 10
    multiplies the loss by `10 ** -0.076`, about 0.84, wherever you start on
    the curve.

    If you take the log of both sides, the formula becomes the equation of a
    line.

    ```text
    log L = -alpha * log N + alpha * log N_c
    ```

    So log L plotted against log N is a straight line with slope `-alpha`. To
    fit a power law, fit that line by least squares, as in linreg3. Then turn
    the intercept back into the constant with `math.exp`.

    ```python
    torch.log(torch.tensor([10.0, 100.0]))    # tensor([2.3026, 4.6052])
    torch.tensor([3.0, 4.0]).tolist()         # [3.0, 4.0], a list of plain floats
    ```

    A power law keeps falling forever, but no model's loss can fall below the
    entropy of the text itself. Hoffmann and colleagues, in the Chinchilla
    paper, included that floor in their fit. With unlimited data, their law
    is `1.69 + 406.4 / N ** 0.34`. Here are two fits to that law, both made
    from points between 10 million and 1 billion parameters. The table then
    follows them out to larger sizes.

    | N | The law | A pure power law | A power law above 1.69 |
    |---|---|---|---|
    | 10 million | 3.384 | 3.253 | 3.384 |
    | 1 billion | 2.044 | 1.971 | 2.044 |
    | 10 billion | 1.852 | 1.534 | 1.852 |
    | 100 billion | 1.764 | 1.194 | 1.764 |
    | 1 trillion | 1.724 | 0.930 | 1.724 |

    The pure power law fits an alpha of 0.109, while the law has 0.34.
    Further out, it goes below 1.69, a loss the law says no model can reach.
    If you subtract the floor first, what remains is a pure power law again.
    So the fit above 1.69 matches the law.
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
    with_ones = torch.stack([logs, torch.ones(3)], dim=1)    # shape (3, 2), built the same way as in linreg3
    back = math.exp(2.0)                                     # 7.389..., the number whose log is 2.0


    # TODO: Return scale * n ** -alpha for every size in n.
    #   Example: power_law(torch.tensor([1.0, 100.0]), 2.0, 0.5) returns
    #   tensor([2.0000, 0.2000]).
    def power_law(
        n: Tensor,
        scale: float,
        alpha: float,
    ) -> Tensor:
        ...


    # TODO: Fit loss = scale * n ** -alpha by least squares on the logs, the
    #   way linreg3 fitted a line. A line through log loss against log n has
    #   slope -alpha and intercept log(scale). Return scale and alpha as a
    #   tuple of plain floats.
    #   Example: for n = torch.tensor([1.0, 10.0, 100.0]) and
    #   loss = 2 * n ** -0.5, fit_power_law returns about (2.0, 0.5).
    def fit_power_law(
        n: Tensor,
        loss: Tensor,
    ) -> tuple[float, float]:
        ...


    # TODO: Fit loss = floor + scale * n ** -alpha, where floor is known.
    #   Subtract floor from each loss, fit a power law to what remains, and
    #   return that power law's scale and alpha.
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
        description="returns scale * n ** -alpha",
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
        description="fits a line to the logs and returns scale and alpha",
        inputs=POWER_DATA,
        reference=FIT_REFERENCE["fit_power_law"],
        expect=expect_relative_pair("scale", "alpha"),
    )
    check_on_inputs(
        checker=checker,
        function="fit_above_floor",
        description="fits a power law to the losses minus the floor",
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
        "power_law": "n ** -alpha applies to every entry of n at once. Multiply the result by scale.",
        "fit_power_law": "Stack torch.log(n) and torch.ones_like(n) as columns with dim=1, and solve against "
        "torch.log(loss) with torch.linalg.lstsq. .solution.tolist() gives the slope and the intercept. Return "
        "math.exp of the intercept and minus the slope.",
        "fit_above_floor": "Call fit_power_law with floor subtracted from each loss.",
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

# cost2's training_flops, given in scaling2's file so the learner doesn't write it twice.
TRAINING_FLOPS_PART = '''
def training_flops(
    parameters: float,
    tokens: float,
) -> float:
    return 6 * parameters * tokens
'''

BUDGET_SOLUTION = '''
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

BUDGET_REFERENCE = namespace_of(TRAINING_FLOPS_PART + BUDGET_SOLUTION)

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
    cost2's `training_flops` counted 6 FLOPs of training for every parameter
    and every token, 2 for the forward pass and 4 for the backward pass. Your
    file already has it. So with a budget of C FLOPs, you can train N
    parameters on D tokens for any N and D with `6 * N * D = C`.

    Is it better to train a bigger model on fewer tokens, or a smaller model
    on more tokens? Hoffmann and colleagues trained over 400 models,
    and fitted the loss as a function of both N and D.

    ```text
    L(N, D) = E + A / N ** alpha + B / D ** beta
    ```

    E is the floor from scaling1. The second term is the extra loss from
    having only N parameters. The third term is the extra loss from training
    on only D tokens. Now fix the budget C, and give each size N the
    `C / (6 * N)` tokens that the budget allows. As N grows, the loss first
    falls and then rises, so one size has the lowest loss. That size is
    **compute optimal** for the budget.

    ```python
    torch.logspace(7, 9, 3)                      # tensor([1.0000e+07, 1.0000e+08, 1.0000e+09])
    torch.tensor([3.0, 1.0, 2.0]).argmin()       # tensor(1), the index of the smallest value
    ```

    Gopher had 280 billion parameters and was trained on 300 billion tokens.
    Chinchilla spent the same compute on 70 billion parameters and 1.4
    trillion tokens, and beat Gopher on a large range of tasks. That is 20
    tokens for every parameter, close to the ratio in the paper's Table 3 of
    compute optimal sizes.

    But the constants printed for this fit disagree with that ratio. In 2024
    Besiroglu and colleagues read the data points off the paper's plots and
    fitted the formula again. They found that the printed fit also disagreed
    with the paper's other two methods. Here are the best sizes under both
    fits, searched over a grid of sizes.

    | Budget in FLOPs | Printed fit, best N | Its tokens per parameter | Refit, best N | Its tokens per parameter |
    |---|---|---|---|---|
    | 1e20 | 646 million | 40 | 851 million | 23 |
    | 1e22 | 5.19 billion | 62 | 9.02 billion | 20.5 |
    | 5.76e23, Gopher's | 32.4 billion | 92 | 72.4 billion | 18.3 |

    At Gopher's budget the refit picks 72.4 billion parameters, close to
    Chinchilla's own 70 billion. A fitted law is only as good as its data
    and its fitting, so check it against training runs you trust.
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


    # training_flops from cost2.
''' + indented(TRAINING_FLOPS_PART) + '''


    # TODO: Return the loss that fit predicts, E + A / N ** alpha + B / D ** beta,
    #   for each pair of a size N from parameters and a token count D from
    #   tokens. fit holds E, A, B, alpha and beta, in that order.
    #   Example: chinchilla_loss(torch.tensor([7e10]), torch.tensor([1.4e12]),
    #   PRINTED) returns about tensor([1.9366]).
    def chinchilla_loss(
        parameters: Tensor,
        tokens: Tensor,
        fit: tuple[float, float, float, float, float],
    ) -> Tensor:
        ...


    # TODO: For every size N in sizes, compute the tokens that budget allows,
    #   budget / (6 * N). Return the size whose loss under fit is lowest, as a
    #   plain float.
    def best_size(
        budget: float,
        sizes: Tensor,
        fit: tuple[float, float, float, float, float],
    ) -> float:
        ...


    # TODO: Find the best size for budget with best_size. Return the tokens
    #   that size gets from budget, divided by the size.
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


''' + indented(TRAINING_FLOPS_PART) + '''


''' + indented(BUDGET_SOLUTION) + '''
    ''',
)

# A made up fit, so a learner who types either set of constants into the function fails.
OTHER_FIT = (2.0, 150.0, 900.0, 0.3, 0.33)


def size_grid(
    low: float,
    high: float,
    count: int,
) -> Described:
    """A grid of sizes, shown in a failing call by the line that makes it, since its hundreds of entries would bury
    the message."""
    return Described(
        value=torch.logspace(low, high, count, dtype=torch.float64),
        text=f"torch.logspace({low}, {high}, {count}, dtype=torch.float64)",
    )


def check_scaling2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="chinchilla_loss",
        description="returns E + A / N ** alpha + B / D ** beta",
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
        "chinchilla_loss": "First unpack fit into five names. Then compute E plus A over parameters ** alpha plus B "
        "over tokens ** beta, on whole tensors at once.",
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


def flops_per_token(
    non_embedding: int,
    width: int,
) -> int:
    head = VOCABULARY * width
    attention = LAYERS * 4 * BLOCK * width

    return 3 * (2 * non_embedding + 2 * head + attention)


def steps_for_budget(
    budget: float,
    flops: int,
) -> int:
    return max(1, round(budget / (flops * BLOCK * BATCH)))


def isoflop_point(
    width: int,
    budget: float,
    train_data: Tensor,
    validation_data: Tensor,
) -> tuple[int, float]:
    torch.manual_seed(0)
    model = GPT(VOCABULARY, BLOCK, width, LAYERS, max(1, width // 16))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    flops = flops_per_token(non_embedding_parameters(model), width)
    train(model, steps_for_budget(budget, flops), train_data, torch.Generator().manual_seed(0))

    return parameters, validation_score(model, validation_data)


def fit_parabola(points: list[tuple[int, float]]) -> float:
    sizes = torch.log(torch.tensor([size for size, _ in points], dtype=torch.float64))
    losses = torch.tensor([loss for _, loss in points], dtype=torch.float64)
    columns = torch.stack([sizes**2, sizes, torch.ones_like(sizes)], dim=1)
    a, b, _ = torch.linalg.lstsq(columns, losses).solution.tolist()

    return math.exp(-b / (2 * a))
'''

SWEEP_REFERENCE = namespace_of(GIVEN_MODEL + INIT_SOLUTION + GPT_INIT + GPT_FORWARD + DATA_PART + SWEEP_GIVEN + SWEEP_SOLUTION)

# Each budget's widths sit around its valley, since a parabola only fits near its bottom. 1e10 has none, its
# loss lowest at the narrowest width, 6. The first row is the sweep the learner's file runs on every check, four widths
# that took 17 to 43 seconds on a laptop CPU shared with other work, inside the time limit.
SWEEPS: tuple[tuple[float, tuple[int, ...]], ...] = (
    (3e10, (6, 8, 12, 16)),
    (1e11, (8, 12, 16, 24, 32)),
    (3e11, (12, 16, 24, 32, 48)),
    (1e12, (16, 24, 32, 48, 64)),
    (3e12, (24, 32, 48, 64, 96)),
)


def sweep_tokens(
    width: int,
    budget: float,
) -> int:
    """The tokens isoflop_point trains a width on, worked out without training it."""
    model = seeded_sweep_gpt(width)
    flops = SWEEP_REFERENCE["flops_per_token"](SWEEP_REFERENCE["non_embedding_parameters"](model), width)

    return SWEEP_REFERENCE["steps_for_budget"](budget, flops) * SWEEP_REFERENCE["BLOCK"] * SWEEP_REFERENCE["BATCH"]


def sweep_point(
    width: int,
    budget: float,
    peak: float,
    seed: int,
    train_data: Tensor,
    validation_data: Tensor,
) -> tuple[int, float]:
    """isoflop_point trained at the peak learning rate given, with its model and batch seeds both set to seed. At a
    peak of 3e-3 and seed 0 it is isoflop_point itself."""
    usual = SWEEP_REFERENCE["PEAK"]
    SWEEP_REFERENCE["PEAK"] = peak

    try:
        torch.manual_seed(seed)
        model = SWEEP_REFERENCE["GPT"](65, 64, width, 2, max(1, width // 16))
        parameters = sum(parameter.numel() for parameter in model.parameters())
        flops = SWEEP_REFERENCE["flops_per_token"](SWEEP_REFERENCE["non_embedding_parameters"](model), width)
        steps = SWEEP_REFERENCE["steps_for_budget"](budget, flops)
        SWEEP_REFERENCE["train"](model, steps, train_data, torch.Generator().manual_seed(seed))

        return parameters, SWEEP_REFERENCE["validation_score"](model, validation_data)
    finally:
        SWEEP_REFERENCE["PEAK"] = usual


def measure_isoflops(
    peak: float = 3e-3,
    budgets: int = len(SWEEPS),
    seed: int = 0,
) -> list[dict[str, object]]:
    """scaling3's sweep with every width at one peak learning rate, on the CPU, where the same seeds give the same
    numbers. The lesson's table is seed 0, and its second seed's exponent of 0.65 is seed 1. Not run by any check."""
    _, train_data, validation_data = SWEEP_REFERENCE["shakespeare_splits"]()
    points = {
        (budget, width): sweep_point(width, budget, peak, seed, train_data, validation_data)
        for budget, widths in SWEEPS[:budgets]
        for width in widths
    }

    return isoflop_rows(points, SWEEPS[:budgets])


def isoflop_rows(
    found: dict[tuple[float, int], tuple[int, float]],
    sweeps: tuple[tuple[float, tuple[int, ...]], ...] = SWEEPS,
) -> list[dict[str, object]]:
    """Each budget's row from its points, keyed by budget and width. The best size's tokens come from a line through
    log tokens against log size, since that size sits between widths. The same losses are fitted once more against
    the non embedding count, Kaplan's N."""
    rows = []

    for budget, widths in sweeps:
        points = [found[(budget, width)] for width in widths]
        best = SWEEP_REFERENCE["fit_parabola"](points)
        lowest = min(range(len(points)), key=lambda position: points[position][1])

        sizes = torch.log(torch.tensor([size for size, _ in points], dtype=torch.float64))
        tokens = torch.log(torch.tensor([sweep_tokens(width, budget) for width in widths], dtype=torch.float64))
        columns = torch.stack([sizes, torch.ones_like(sizes)], dim=1)
        slope, intercept = torch.linalg.lstsq(columns, tokens).solution.tolist()
        best_tokens = math.exp(intercept + slope * math.log(best))

        non_embedding = [
            (SWEEP_REFERENCE["non_embedding_parameters"](seeded_sweep_gpt(width)), loss)
            for width, (_, loss) in zip(widths, points)
        ]
        rows.append(
            {
                "budget": budget,
                "points": points,
                "lowest width": widths[lowest],
                "lowest loss": points[lowest][1],
                "fitted best": best,
                "tokens per parameter": best_tokens / best,
                "fitted best non embedding": SWEEP_REFERENCE["fit_parabola"](non_embedding),
            },
        )

    return rows


# The peak learning rates the tuned sweep tries. Every width tries the first three, and the narrow ones, whose best
# rate sat at the top of those, try 2e-2 too.
TUNING_PEAKS = (3e-3, 6e-3, 1e-2)
NARROW_PEAK = 2e-2
NARROW = 24


def measure_tuned_isoflops(seed: int = 0) -> list[dict[str, object]]:
    """scaling3's sweep with the learning rate tuned for each point, as Porian and colleagues tuned each size. Every
    width trains at each of its peak rates and keeps its lowest loss. Each row also gives the rate each width kept.
    The lesson's tuned numbers, the lowest losses, the exponent of 0.62 and tokens per parameter of 192 down to 106,
    come from seed 0. Not run by any check, and it trains 86 models, nearly four of measure_isoflops' sweeps."""
    _, train_data, validation_data = SWEEP_REFERENCE["shakespeare_splits"]()
    found: dict[tuple[float, int], tuple[int, float]] = {}
    kept: dict[tuple[float, int], float] = {}

    for budget, widths in SWEEPS:
        for width in widths:
            peaks = TUNING_PEAKS + ((NARROW_PEAK,) if width <= NARROW else ())
            tried = {peak: sweep_point(width, budget, peak, seed, train_data, validation_data) for peak in peaks}
            kept[(budget, width)] = min(tried, key=lambda peak: tried[peak][1])
            found[(budget, width)] = tried[kept[(budget, width)]]

    rows = isoflop_rows(found)

    for row, (budget, widths) in zip(rows, SWEEPS):
        row["peaks"] = [kept[(budget, width)] for width in widths]

    return rows


def size_exponent(
    rows: list[dict[str, object]],
    key: str = "fitted best",
) -> float:
    """How the fitted best size grows with the budget, scaling1's power law fitted to measure_isoflops' rows."""
    budgets = torch.tensor([row["budget"] for row in rows], dtype=torch.float64)
    bests = torch.tensor([row[key] for row in rows], dtype=torch.float64)
    _, alpha = FIT_REFERENCE["fit_power_law"](budgets, bests)

    return -alpha


SCALING3_LESSON = prose(
    """
    scaling2's best sizes come from fits to hundreds of training runs. This
    exercise runs the method behind one such fit, Chinchilla's second
    approach, at sizes this laptop can train.

    - Fix a budget of FLOPs.
    - Train models of several sizes, each on the tokens the budget allows, with a cosine learning rate schedule that ends when the run ends.
    - Plot loss against log N, which makes a valley. Fit a parabola to the valley. The size at its bottom is the compute optimal size.

    N counts every parameter, as Chinchilla did. Each run is charged the
    FLOPs it actually spends, close to the count in Chinchilla's appendix F.
    In the forward pass, one token costs

    - 2 FLOPs per parameter outside the embeddings `wte` and `wpe`,
    - 2 FLOPs per number in the head, which reuses `wte`'s table of `VOCABULARY * width` numbers,
    - and `4 * BLOCK * width` FLOPs in each block's attention, which compares the token with `BLOCK` keys and values.

    Appendix F also counts the input embedding as a matrix multiply. Here,
    looking a token up in `wte` or `wpe` counts as free. The backward pass
    costs twice the forward pass, so training costs 3 times the forward pass.

    ```python
    layer = nn.Linear(2, 3)
    [(name, p.numel()) for name, p in layer.named_parameters()]    # [('weight', 6), ('bias', 3)]
    sum(p.numel() for p in layer.parameters())                      # 9
    "wpe.weight".startswith(("wte", "wpe"))                         # True
    round(2.5), round(3.7)                                          # (2, 4)
    ```

    `named_parameters()` gives each parameter together with its name. Each
    step trains on `BATCH * BLOCK` tokens. So for a model that costs F FLOPs
    per token, a budget of C FLOPs pays for `C / (F * BATCH * BLOCK)` steps.

    The parabola is `a * x ** 2 + b * x + c` with `x = log N`. Fit it by
    least squares on the columns `x ** 2`, `x` and 1. Its bottom is at
    `x = -b / (2 * a)`. A parabola only fits a valley near its bottom. So
    each budget tried four or five widths around its valley, each model with
    2 blocks.

    | Budget in FLOPs | Widths | Lowest loss, at width | Best N, to 2 figures | Tokens per parameter |
    |---|---|---|---|---|
    | 3e10 | 6 to 16 | 2.645 at 8 | 3,500 | 263 |
    | 1e11 | 8 to 32 | 2.460 at 12 | 7,500 | 212 |
    | 3e11 | 12 to 48 | 2.189 at 16 | 12,000 | 265 |
    | 1e12 | 16 to 64 | 1.927 at 32 | 29,000 | 161 |
    | 3e12 | 24 to 96 | 1.747 at 48 | 61,000 | 115 |

    The best size grows with the budget, as scaling2 predicted. Fitting
    scaling1's power law to the five best sizes gives N growing as
    `C ** 0.61`, and a second seed gives 0.65. Chinchilla's fits put the
    exponent near 0.5, which keeps tokens per parameter near 20 at every
    budget. Here tokens per parameter fall from 263 to 115.

    Porian and colleagues found three things that push a sweep's exponent
    above 0.5. The first is leaving the head's FLOPs out of the count. The
    second is a warmup that is too long for small models. The third is using
    the same optimizer settings for every size. This sweep counts the head,
    and warms up for a twentieth of each run. But every width above trains
    at the same peak learning rate, 3e-3.

    In a second sweep, the learning rate was tuned for each point. Each point
    tried peak rates from 3e-3 to 1e-2, plus 2e-2 for widths up to 24, and
    kept its best result. Every budget's lowest loss fell, for example from
    2.645 to 2.488 at 3e10. The exponent came out at 0.62, still well above
    0.5, and tokens per parameter still fell, from 192 to 106.

    Kaplan and colleagues left the embeddings out of N, and found an exponent
    of 0.73. If this sweep's N also leaves out the embeddings, its best size
    grows as `C ** 0.68`. Pearce and Song traced much of the difference
    between the two papers to this choice of N.

    The last lines of your file run the sweep from the table's first row
    every time the file is checked. Its numbers appear in the result.
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
    total = sum(p.numel() for p in layer.parameters())                      # 9
    is_position = "wpe.weight".startswith(("wte", "wpe"))                   # True


    # The model from gpt2, and get_batch and the data from gpt1.
''' + indented(GIVEN_MODEL) + '''


''' + indented(INIT_SOLUTION) + '''


''' + indented(GPT_INIT) + '''

''' + indented(GPT_FORWARD) + '''


''' + indented(DATA_PART) + '''


    # Small GPTs with 2 blocks, trained on the CPU with a cosine schedule that ends with the run.
''' + indented(SWEEP_GIVEN) + '''


    # TODO: Return the total number of entries in model's parameters, leaving
    #   out the parameters whose names start with wte or wpe.
    #   Example: for GPT(65, 64, 24, 2, 1), non_embedding_parameters returns
    #   14496.
    def non_embedding_parameters(model: nn.Module) -> int:
        ...


    # TODO: Return the FLOPs it costs to train a model on one token, which is 3
    #   times the cost of the forward pass. The forward pass costs 2 FLOPs per
    #   non embedding parameter, 2 FLOPs per number in the head's
    #   VOCABULARY * width table, and 4 * BLOCK * width FLOPs in the attention
    #   of each of the LAYERS blocks.
    #   Example: flops_per_token(1032, 6) returns 17748.
    def flops_per_token(
        non_embedding: int,
        width: int,
    ) -> int:
        ...


    # TODO: Return how many training steps budget FLOPs pay for, for a model
    #   that costs flops FLOPs per token. Each step trains on BATCH * BLOCK
    #   tokens. Round to the nearest whole step, and return at least 1.
    #   Example: steps_for_budget(3e10, 25968) returns 564.
    def steps_for_budget(
        budget: float,
        flops: int,
    ) -> int:
        ...


    # TODO: Finish isoflop_point. Set flops to the training cost of one token
    #   for this model, using flops_per_token. Then call train on the model for
    #   the number of steps that budget pays for at that cost, with batches
    #   from train_data and a generator seeded with 0.
    def isoflop_point(
        width: int,
        budget: float,
        train_data: Tensor,
        validation_data: Tensor,
    ) -> tuple[int, float]:
        torch.manual_seed(0)
        model = GPT(VOCABULARY, BLOCK, width, LAYERS, max(1, width // 16))
        parameters = sum(parameter.numel() for parameter in model.parameters())
        flops = ...
        ...

        return parameters, validation_score(model, validation_data)


    # TODO: points is a list of (size, loss) pairs. Let x be the log of each
    #   size, and fit loss = a * x ** 2 + b * x + c by least squares. Return
    #   the size at the bottom of the parabola, as a plain float.
    def fit_parabola(points: list[tuple[int, float]]) -> float:
        ...


    # Your own sweep, the first row of the lesson's table. It trains four
    # models every time the file is checked.
    characters, train_data, validation_data = shakespeare_splits()
    WIDTHS = [6, 8, 12, 16]
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
    total = sum(p.numel() for p in layer.parameters())
    is_position = "wpe.weight".startswith(("wte", "wpe"))


''' + indented(GIVEN_MODEL) + '''


''' + indented(INIT_SOLUTION) + '''


''' + indented(GPT_INIT) + '''

''' + indented(GPT_FORWARD) + '''


''' + indented(DATA_PART) + '''


''' + indented(SWEEP_GIVEN) + '''


''' + indented(SWEEP_SOLUTION) + '''


    characters, train_data, validation_data = shakespeare_splits()
    WIDTHS = [6, 8, 12, 16]
    points = [isoflop_point(width, 3e10, train_data, validation_data) for width in WIDTHS]
    best = fit_parabola(points)
    ''',
)

OWN_WIDTHS = (6, 8, 12, 16)


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


def with_reference_helpers(
    found: Callable[..., object],
    run: Callable[[], object],
) -> object:
    """isoflop_point runs with the reference non_embedding_parameters, flops_per_token and steps_for_budget in
    place of the learner's, so a mistake in one of those shows in its own row, and a step count far too large
    can't stall this one."""
    globals_ = found.__globals__
    names = ("non_embedding_parameters", "flops_per_token", "steps_for_budget")
    learners = {name: globals_.get(name) for name in names}

    for name in names:
        globals_[name] = SWEEP_REFERENCE[name]

    try:
        return run()
    finally:
        for name in names:
            globals_[name] = learners[name]


def check_isoflop_point(found: object) -> None:
    for width, budget, seed in ((8, 1e9, 931), (12, 2e9, 932)):
        train_data = repeating_text(seed, 3000)
        # Another pattern than training's, so scoring the training text gives a very different loss.
        validation_data = repeating_text(seed + 100, 1500)
        call = f"isoflop_point({width}, {budget:g}, a repeating text of 3,000 characters, another of 1,500)"
        actual = with_reference_helpers(
            found=found,
            run=lambda: found(width, budget, train_data.clone(), validation_data.clone()),
        )
        expected = SWEEP_REFERENCE["isoflop_point"](width, budget, train_data, validation_data)

        if actual is None:
            raise Mismatch(f"{call} is None, which usually means it is not written yet.")

        if not isinstance(actual, (tuple, list)) or len(actual) != 2:
            raise Mismatch(f"{call} should return 2 things, the parameters and the loss, but it returns {actual!r}.")

        expect_value(
            call=f"the parameters from {call}",
            actual=actual[0],
            expected=expected[0],
        )
        untrained = SWEEP_REFERENCE["validation_score"](seeded_sweep_gpt(width), validation_data)

        # The starting file's lines already score the model, so one left untrained gives a plausible loss.
        if isinstance(actual[1], float) and abs(actual[1] - untrained) < 1e-6:
            raise Mismatch(
                f"The loss from {call} is the model's loss before any training, so isoflop_point is not written yet. "
                "Train the model with train before it is scored.",
            )

        # Both sides train the same model with the same seeds on the CPU, so a right loss matches closely.
        expect_close(
            call=f"the loss from {call}",
            actual=actual[1],
            expected=expected[1],
            tolerance=1e-5,
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
            expected=sum(parameter.numel() for parameter in seeded_sweep_gpt(width).parameters()),
        )

        if not isinstance(point[1], float) or not 0.5 < point[1] < 5.0:
            raise Mismatch(f"The loss of width {width} is {point[1]!r}, but a validation loss here should be between 0.5 and 5.")


def check_best(
    value: object,
    points: object,
) -> None:
    try:
        check_points(points)
    except Mismatch:
        raise Mismatch("best is fitted to points, and points is not right yet. Fix points first.") from None

    expected = SWEEP_REFERENCE["fit_parabola"](points)

    try:
        expect_fit(
            call="best",
            actual=value,
            expected=expected,
        )
    except Mismatch:
        # Rounded far coarser than the check's tolerance, so a number copied from the message fails.
        rounded = float(f"{expected:.2g}")
        raise Mismatch(
            f"best is {value_text(value)}, but it should be the size at the bottom of the parabola fitted to your points, "
            f"about {rounded:,.0f}. Set it with fit_parabola(points).",
        ) from None


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
        function="flops_per_token",
        description="counts the blocks, the head and attention for one token, times 3",
        inputs=(
            (1032, 6),
            (25472, 32),
            (100, 3),
        ),
        reference=SWEEP_REFERENCE["flops_per_token"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="steps_for_budget",
        description="returns the steps a budget pays for, rounded and at least 1",
        inputs=(
            (1e12, 214464),
            (3e10, 25968),
            (1e9, 2000000),
        ),
        reference=SWEEP_REFERENCE["steps_for_budget"],
        expect=expect_value,
    )
    checker.test(
        function="isoflop_point",
        description="trains one size on the budget and scores it",
        run=check_isoflop_point,
    )
    check_on_inputs(
        checker=checker,
        function="fit_parabola",
        description="returns the size at the bottom of a parabola in log size",
        inputs=(
            (parabola_points(20000.0, 0.08, (5000.0, 10000.0, 20000.0, 40000.0, 80000.0), 933),),
            (parabola_points(3000.0, 0.15, (1700.0, 3800.0, 6600.0, 14500.0), 934),),
        ),
        reference=SWEEP_REFERENCE["fit_parabola"],
        expect=expect_fit,
    )
    checker.check_name(
        name="points",
        description="holds your sweep, a pair of parameters and loss per width",
        run=check_points,
    )
    checker.check_name(
        name="best",
        description="is the size at the bottom of the parabola fitted to your points",
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
        "flops_per_token": "In the forward pass, a token costs 2 FLOPs per non embedding parameter and 2 per number in "
        "the VOCABULARY by width head. Each of the LAYERS blocks adds 4 * BLOCK * width. Multiply the whole sum by 3.",
        "steps_for_budget": "Divide budget by flops * BLOCK * BATCH, round it, and take the max of that and 1.",
        "isoflop_point": "flops_per_token takes the model's non_embedding_parameters and its width. train takes the "
        "model, the steps that steps_for_budget returns for budget and flops, train_data, and a new generator "
        "seeded with 0.",
        "fit_parabola": "Make a float64 tensor x of torch.log of the sizes, and a float64 tensor of the losses. Stack "
        "x ** 2, x and ones as columns, and solve with torch.linalg.lstsq. Then turn -b / (2 * a) back into a size "
        "with math.exp.",
        "points": "Write the four functions above first. This line only calls isoflop_point once per width.",
        "best": "This line calls fit_parabola on points.",
    },
    introduces=(),
    check=check_scaling3,
    time_limit=240,
    imports_first=OPTIMIZER_IMPORTS,
)
