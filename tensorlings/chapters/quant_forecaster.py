"""109_quant_forecaster, Section 14's forecasting project, a made up market whose best possible score is known, the leaks that
push a forecast past it, walk forward validation, and a small model against a plain baseline."""

from __future__ import annotations

import textwrap
from collections.abc import Callable

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
    expect_parts,
    expect_tensor,
    expect_value,
    reject_missing_return,
    value_text,
)
from ..text import prose, source_file

QUANT_FORECASTER = Chapter(
    folder="109_quant_forecaster",
    title="Project: an honest forecast",
    readme=source_file(
        """
        # Project: an honest forecast

        Section 14's forecasting project. You make up a market with a weak
        signal whose size you know, so you know the best score any forecast
        can reach. Then you find three leaks that push a forecast past it,
        score forecasts the way they would have run with walk forward
        validation, and set a small model against a plain baseline.

        ## Further reading

        - [scikit-learn, TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
        - [skfolio, purging and embargo, after López de Prado's Advances in Financial Machine Learning](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html)
        """,
    ),
)

SETTINGS_PART = '''
WINDOW = 20        # the market leans on the mean of its last 20 days
STRENGTH = 0.2     # how hard it leans
HORIZON = 5        # each forecast is for the return over the next 5 days
WINDOWS = (5, 20, 60, 250)
FIRST = 249        # rows start on the first day every window above is full
DAYS = 4254        # 4,000 rows, from day 249 to the last day with a target
CUT = 3000         # the first test row
'''

MARKET_PART = '''
# The made up market. Each day's return leans on the mean of the last WINDOW days, plus noise of about 1%.
def make_market(
    days: int,
    strength: float,
    seed: int,
) -> Tensor:
    noise = (torch.randn(days, generator=torch.Generator().manual_seed(seed)) * 0.01).tolist()
    returns: list[float] = []
    recent = 0.0

    for day in range(days):
        lean = strength * recent / WINDOW if day >= WINDOW else 0.0
        returns.append(lean + noise[day])
        recent += returns[-1]

        if day >= WINDOW:
            recent -= returns[day - WINDOW]

    return torch.tensor(returns)
'''

PRICES_PART = '''
def prices(
    returns: Tensor,
    start: float,
) -> Tensor:
    return start * returns.cumsum(0).exp()
'''

FORWARD_PART = '''
def forward_returns(
    returns: Tensor,
    horizon: int,
) -> Tensor:
    return returns.unfold(0, horizon, 1)[1:].sum(dim=1)
'''

SCORE_PART = '''
def information_coefficient(
    forecasts: Tensor,
    outcomes: Tensor,
) -> float:
    return torch.corrcoef(torch.stack([forecasts, outcomes]))[0, 1].item()
'''

EXPECTED_PART = '''
def true_expected(
    returns: Tensor,
    strength: float,
    horizon: int,
) -> Tensor:
    expected = torch.zeros(len(returns) - horizon)

    for day in range(WINDOW - 1, len(returns) - horizon):
        history = returns[day - WINDOW + 1 : day + 1].tolist()
        total = 0.0

        for _ in range(horizon):
            lean = strength * sum(history[-WINDOW:]) / WINDOW
            history.append(lean)
            total += lean

        expected[day] = total

    return expected
'''

TRAILING_PART = '''
def trailing_means(
    returns: Tensor,
    windows: tuple[int, ...],
) -> Tensor:
    longest = max(windows)
    columns = [returns.unfold(0, window, 1).mean(dim=1)[longest - window :] for window in windows]

    return torch.stack(columns, dim=1)
'''

SPLIT_PART = '''
def time_split(
    x: Tensor,
    y: Tensor,
    cut: int,
    gap: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return x[: cut - gap], y[: cut - gap], x[cut:], y[cut:]
'''

PICK_PART = '''
def pick_on_training(
    indicators: Tensor,
    targets: Tensor,
    train_rows: int,
) -> int:
    scores = [information_coefficient(column[:train_rows], targets[:train_rows]) for column in indicators.T]

    return torch.tensor(scores).argmax().item()
'''

LINEAR_PART = '''
# Least squares with an intercept, fitted on the training rows, as linreg3 did.
def linear_forecast(
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
) -> Tensor:
    weights = torch.linalg.lstsq(torch.cat([train_x, torch.ones(len(train_x), 1)], dim=1), train_y.unsqueeze(1)).solution

    return (torch.cat([test_x, torch.ones(len(test_x), 1)], dim=1) @ weights).squeeze(1)
'''

NEAREST_PART = '''
# Each test row's forecast is the mean target of its count nearest training rows. The columns are scaled by the
# training rows' mean and spread first, so each one counts about the same.
def nearest_days(
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    count: int,
) -> Tensor:
    mean, spread = train_x.mean(dim=0), train_x.std(dim=0)
    distances = torch.cdist((test_x - mean) / spread, (train_x - mean) / spread)

    return train_y[distances.topk(count, largest=False).indices].mean(dim=1)
'''

INDICATORS_PART = '''
# count made up indicators, each a fixed random weighting of the last 60 days' returns, one row per day from FIRST on.
def made_up_indicators(
    returns: Tensor,
    count: int,
    seed: int,
) -> Tensor:
    weights = torch.randn(count, 60, generator=torch.Generator().manual_seed(seed))

    return (returns.unfold(0, 60, 1) @ weights.T)[FIRST - 59 :]
'''

FOLDS_PART = '''
def walk_forward(
    rows: int,
    first_train: int,
    test_size: int,
    gap: int,
) -> list[tuple[int, int, int]]:
    folds = []

    for test_start in range(first_train, rows, test_size):
        folds.append((test_start - gap, test_start, min(test_start + test_size, rows)))

    return folds
'''

FOLD_SCORES_PART = '''
def fold_scores(
    x: Tensor,
    y: Tensor,
    folds: list[tuple[int, int, int]],
) -> list[float]:
    scores = []

    for train_end, test_start, test_end in folds:
        forecast = linear_forecast(x[:train_end], y[:train_end], x[test_start:test_end])
        scores.append(information_coefficient(forecast, y[test_start:test_end]))

    return scores
'''

STANDARDIZE_PART = '''
# probe2's standardizing, with the mean and spread of the training rows alone.
def standardize(
    train: Tensor,
    other: Tensor,
) -> tuple[Tensor, Tensor]:
    mean, spread = train.mean(dim=0), train.std(dim=0)

    return (train - mean) / spread, (other - mean) / spread
'''

MLP_PART = '''
def train_mlp(
    x: Tensor,
    y: Tensor,
    seed: int,
    steps: int,
) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scale = y.std()

    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((model(x).squeeze(1) - y / scale) ** 2).mean()
        loss.backward()
        optimizer.step()

    return model


def mlp_forecast(
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    seed: int,
) -> Tensor:
    train_x, test_x = standardize(train_x, test_x)
    model = train_mlp(train_x, train_y, seed, 300)
    model.eval()

    with torch.no_grad():
        return model(test_x).squeeze(1)
'''

PAIRS_PART = '''
def paired_scores(
    x: Tensor,
    y: Tensor,
    folds: list[tuple[int, int, int]],
    seed: int,
) -> list[tuple[float, float]]:
    pairs = []

    for train_end, test_start, test_end in folds:
        train_x, train_y, test_x = x[:train_end], y[:train_end], x[test_start:test_end]
        baseline = information_coefficient(linear_forecast(train_x, train_y, test_x), y[test_start:test_end])
        model = information_coefficient(mlp_forecast(train_x, train_y, test_x, seed), y[test_start:test_end])
        pairs.append((baseline, model))

    return pairs
'''


def indented(part: str) -> str:
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "Tensor": Tensor,
    }
    exec(SETTINGS_PART + code, namespace)

    return namespace


REFERENCE = namespace_of(
    MARKET_PART
    + PRICES_PART
    + FORWARD_PART
    + SCORE_PART
    + EXPECTED_PART
    + TRAILING_PART
    + SPLIT_PART
    + PICK_PART
    + LINEAR_PART
    + NEAREST_PART
    + INDICATORS_PART
    + FOLDS_PART
    + FOLD_SCORES_PART
    + STANDARDIZE_PART
    + MLP_PART
    + PAIRS_PART,
)

WINDOW = REFERENCE["WINDOW"]
STRENGTH = REFERENCE["STRENGTH"]
HORIZON = REFERENCE["HORIZON"]
WINDOWS = REFERENCE["WINDOWS"]
FIRST = REFERENCE["FIRST"]
DAYS = REFERENCE["DAYS"]
CUT = REFERENCE["CUT"]


def seeded_returns(
    seed: int,
    days: int,
) -> Described:
    """Returns of about the market's size, so the forward sums and leans stay readable in a failing call."""
    return Described(
        value=torch.randn(days, generator=torch.Generator().manual_seed(seed)) * 0.01,
        text=f"{days} days of returns drawn with seed {seed}",
    )


def lesson_rows(seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """One market's returns, its rows of trailing means and their targets, the rows every exercise after forecast1 uses."""
    returns = REFERENCE["make_market"](DAYS, STRENGTH, seed)
    targets = REFERENCE["forward_returns"](returns, HORIZON)
    x = REFERENCE["trailing_means"](returns, WINDOWS)[: len(targets) - FIRST]

    return returns, x, targets[FIRST:]


def check_value(
    name: str,
    expected: object,
    tolerance: float = 1e-6,
) -> Callable[[object], None]:
    """A check of a name the file sets from the market, against the same lines run with the reference functions."""

    def run(value: object) -> None:
        if isinstance(expected, Tensor):
            expect_tensor(
                call=name,
                actual=value,
                expected=expected,
            )
            return

        reject_missing_return(
            call=name,
            actual=value,
        )

        if isinstance(value, Tensor):
            raise Mismatch(f"{name} is {value_text(value)}, a tensor, but it should be a plain number. Use .item() on it.")

        # Shown to 3 digits, fewer than the tolerance needs, so a number copied from this message doesn't pass.
        if not isinstance(value, (int, float)) or isinstance(value, bool) or abs(value - expected) > tolerance:
            raise Mismatch(f"{name} is {value_text(value)}, but it should be about {expected:.3g}.")

    return run


# ---------------------------------------------------------------------------
# forecast1
# ---------------------------------------------------------------------------

FORECAST1_LESSON = prose(
    """
    This project makes up a market, then forecasts it honestly. You made the
    market, so you know how good a forecast can be, and a fooled result
    shows.

    Each day's **return** here is the change in the log of the price. The
    market leans a little: a day's return is 0.2 times the mean of the last
    20 days' returns, plus noise of about 1%. Rises tend to follow rises,
    and the noise swamps them. Adding up the returns, then taking `exp`,
    gives the price back.

    ```python
    torch.tensor([0.1, -0.2, 0.3]).cumsum(0)    # tensor([ 0.1000, -0.1000,  0.2000])
    ```

    Each forecast is for the next 5 days' return, the target. stft1's
    `unfold` gives every window of days at once.

    Quants score a forecast by its correlation with what followed, the
    **information coefficient**. A correlation is 1 when two series rise
    and fall in step, and 0 when they are unrelated. It is -1 when one falls
    as the other rises. `torch.corrcoef` gives the correlation of every pair
    of rows it gets.

    You know the market's rule, so the best forecast is the rule itself with
    the noise left out. `true_expected` works it out: from each day's last
    20 returns it steps the rule 5 days ahead, each step leaning on the one
    before. No forecast scores higher for long, so its score is the ceiling.

    | Market | The rule's score |
    |---|---|
    | 100,000 days, seed 1 | 0.109 |
    | The 1,000 test days of seeds 0 to 4 | -0.040 to 0.187 |

    Squared, 0.109 is about 1.2% of the target's variance. The other 98.8%
    is noise that no forecast can touch.

    One market's test days are a rough measure, since even the rule's score
    swings from -0.040 to 0.187. forecast3 deals with that.
    """,
)

FORECAST1_SOURCE = source_file(
    '''
    # forecast1.py

    import torch
    from torch import Tensor

''' + indented(SETTINGS_PART) + '''


''' + indented(MARKET_PART) + '''


    steps = torch.tensor([0.1, -0.2, 0.3]).cumsum(0)                  # tensor([ 0.1000, -0.1000,  0.2000])
    grown = torch.tensor([0.0, 2.0]).exp()                            # tensor([1.0000, 7.3891])


    # TODO: Return the price on each day. It starts from start, and each day's
    #   return is the change in the log of the price, so day 0's price is
    #   start times exp of day 0's return.
    def prices(
        returns: Tensor,
        start: float,
    ) -> Tensor:
        ...


    pairs = torch.arange(5.0).unfold(0, 2, 1)                           # tensor([[0., 1.], [1., 2.], [2., 3.], [3., 4.]])


    # TODO: Return each day's target, the sum of the returns of the horizon days
    #   after it. Only days with horizon more days after them get one.
    def forward_returns(
        returns: Tensor,
        horizon: int,
    ) -> Tensor:
        ...


    both = torch.corrcoef(torch.tensor([[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]]))    # tensor([[1.0000, 0.5000], [0.5000, 1.0000]])


    # TODO: Return the information coefficient of forecasts against outcomes,
    #   their correlation, as a plain number.
    def information_coefficient(
        forecasts: Tensor,
        outcomes: Tensor,
    ) -> float:
        ...


    lean = 0.2 * sum([0.5, 1.0, 3.0][-2:]) / 2                         # 0.4


    # TODO: Finish the market's own forecast of each day's target, its rule
    #   with the noise left out. history starts as the day's last WINDOW
    #   returns. Then for each of the horizon days ahead, work out the lean on
    #   the last WINDOW values of history, add it to history and to a total.
    #   The total is that day's forecast. Earlier days stay at 0.
    def true_expected(
        returns: Tensor,
        strength: float,
        horizon: int,
    ) -> Tensor:
        expected = torch.zeros(len(returns) - horizon)

        for day in range(WINDOW - 1, len(returns) - horizon):
            history = returns[day - WINDOW + 1 : day + 1].tolist()
            ...

        return expected


    returns = make_market(DAYS, STRENGTH, 0)
    price = prices(returns, 100.0)
    targets = forward_returns(returns, HORIZON)
    best = true_expected(returns, STRENGTH, HORIZON)
    rule_score = information_coefficient(best[FIRST + CUT :], targets[FIRST + CUT :])
    ''',
)

FORECAST1_SOLUTION = source_file(
    '''
    # forecast1.py, a solution for comparison

    import torch
    from torch import Tensor

''' + indented(SETTINGS_PART) + '''


''' + indented(MARKET_PART) + '''


    steps = torch.tensor([0.1, -0.2, 0.3]).cumsum(0)
    grown = torch.tensor([0.0, 2.0]).exp()


''' + indented(PRICES_PART) + '''


    pairs = torch.arange(5.0).unfold(0, 2, 1)


''' + indented(FORWARD_PART) + '''


    both = torch.corrcoef(torch.tensor([[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]]))


''' + indented(SCORE_PART) + '''


    lean = 0.2 * sum([0.5, 1.0, 3.0][-2:]) / 2


''' + indented(EXPECTED_PART) + '''


    returns = make_market(DAYS, STRENGTH, 0)
    price = prices(returns, 100.0)
    targets = forward_returns(returns, HORIZON)
    best = true_expected(returns, STRENGTH, HORIZON)
    rule_score = information_coefficient(best[FIRST + CUT :], targets[FIRST + CUT :])
    ''',
)


def market_names() -> dict[str, object]:
    returns = REFERENCE["make_market"](DAYS, STRENGTH, 0)
    targets = REFERENCE["forward_returns"](returns, HORIZON)
    best = REFERENCE["true_expected"](returns, STRENGTH, HORIZON)

    return {
        "price": REFERENCE["prices"](returns, 100.0),
        "targets": targets,
        "best": best,
        "rule_score": REFERENCE["information_coefficient"](best[FIRST + CUT :], targets[FIRST + CUT :]),
    }


def series_pair(
    seed: int,
    length: int,
) -> tuple[Described, Described]:
    generator = torch.Generator().manual_seed(seed)
    first = torch.randn(length, generator=generator)
    second = 0.5 * first + torch.randn(length, generator=generator)

    return (
        Described(value=first, text=f"{length} forecasts drawn with seed {seed}"),
        Described(value=second, text=f"{length} outcomes that follow them a little"),
    )


def check_forecast1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="prices",
        description="adds up the returns and takes exp, from the start price",
        inputs=(
            (seeded_returns(1001, 6), 100.0),
            (seeded_returns(1002, 4), 2.0),
        ),
        reference=REFERENCE["prices"],
    )
    check_on_inputs(
        checker=checker,
        function="forward_returns",
        description="sums the returns of the days after each day",
        inputs=(
            (seeded_returns(1003, 8), 2),
            (seeded_returns(1004, 7), 3),
            (seeded_returns(1005, 5), 1),
        ),
        reference=REFERENCE["forward_returns"],
    )
    check_on_inputs(
        checker=checker,
        function="information_coefficient",
        description="gives the correlation of forecasts and outcomes as a plain number",
        inputs=(
            series_pair(1006, 12),
            series_pair(1007, 30),
        ),
        reference=REFERENCE["information_coefficient"],
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="true_expected",
        description="steps the market's rule ahead with the noise left out",
        inputs=(
            (seeded_returns(1008, 26), 0.2, 3),
            (seeded_returns(1009, 24), 0.5, 1),
            (seeded_returns(1010, 30), 0.3, 5),
        ),
        reference=REFERENCE["true_expected"],
    )

    expected = market_names()

    for name, description in (
        ("price", "is seed 0's prices from 100"),
        ("targets", "is seed 0's 5 day targets"),
        ("best", "is the rule's forecast of seed 0's targets"),
        ("rule_score", "is the rule's score on seed 0's test days"),
    ):
        checker.check_name(
            name=name,
            description=description,
            run=check_value(name, expected[name]),
        )


FORECAST1 = Exercise(
    name="forecast1",
    chapter=QUANT_FORECASTER,
    title="Project: a market with a known ceiling",
    lesson=FORECAST1_LESSON,
    source=FORECAST1_SOURCE,
    solution=FORECAST1_SOLUTION,
    hints={
        "prices": "returns.cumsum(0) adds them up day by day. Take .exp() of that and multiply by start.",
        "forward_returns": "returns.unfold(0, horizon, 1) holds every run of horizon days. The run that starts the day "
        "after day t is row t + 1, so drop row 0 and .sum(dim=1).",
        "information_coefficient": "torch.stack the two into rows, torch.corrcoef them, take the [0, 1] entry and "
        ".item() it.",
        "true_expected": "Set total = 0.0. Loop horizon times: lean is strength times the sum of history[-WINDOW:] "
        "over WINDOW, then append it to history and add it to total. After the loop, expected[day] = total.",
        "price": "It comes from prices. Get that right first.",
        "targets": "It comes from forward_returns. Get that right first.",
        "best": "It comes from true_expected. Get that right first.",
        "rule_score": "It comes from information_coefficient and the two above. Get those right first.",
    },
    introduces=("information coefficient",),
    check=check_forecast1,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# forecast2
# ---------------------------------------------------------------------------

FORECAST2_LESSON = prose(
    """
    A score above the ceiling can't be real, so it proves a **leak**: the
    forecast saw something from the days it is scored on. Here are three
    common ones, each in a pipeline trained on the first 3,000 of 4,000 rows
    and scored on the last 1,000.

    Each row holds the means of the last 5, 20, 60 and 250 returns, up to
    and including that day. A window that ends a day late holds the first
    day of the target. That is future data in the features.

    Rows next to each other share most of their windows and 4 of their 5
    target days. A random split puts the neighbours of every test row in
    training. `nearest_days`, which averages the targets of the 10 most
    similar training rows, then reads the answers off.

    The third is a choice made with the whole series. Of 1,000 made up
    indicators, the one with the best score over every row was partly
    picked by its test days.

    | Pipeline | Honest | Leaky |
    |---|---|---|
    | Least squares on the windows | 0.091 | 0.258, windows a day late |
    | 10 nearest days | 0.025 | 0.297, a random split |
    | Best of 1,000 indicators | 0.043 | 0.072, picked on every row |

    Each cell is the average over seeds 0 to 4. The first two leaks land far
    above the ceiling of 0.109, and on every seed above the rule's best test
    score, 0.187. The third stays under the ceiling, though it lifts the
    honest score by two thirds. So a score under the ceiling proves nothing.
    Only the rules do: features from past days, a split by time, every
    choice from training rows.

    Standardizing with the whole series' mean and spread can't change least
    squares' forecast, since its weights shift and scale to match. A column
    of past returns holds no outcome either. It is still a habit to drop,
    and the check of `trailing_means` cuts later days off to catch it.

    `time_split` leaves a gap of 5 rows too. The last 5 training rows'
    targets reach into the test days, so those rows go. López de Prado's
    2018 book calls this **purging**.
    """,
)

FORECAST2_SOURCE = source_file(
    '''
    # forecast2.py

    import torch
    from torch import Tensor

''' + indented(SETTINGS_PART) + '''


    # The market and forecast1's targets and score.
''' + indented(MARKET_PART) + '''


''' + indented(FORWARD_PART) + '''


''' + indented(SCORE_PART) + '''


''' + indented(LINEAR_PART) + '''


''' + indented(NEAREST_PART) + '''


''' + indented(INDICATORS_PART) + '''


    means = torch.arange(6.0).unfold(0, 3, 1).mean(dim=1)              # tensor([1., 2., 3., 4.])


    # TODO: Return one row per day from day max(windows) - 1 on, holding the
    #   mean of the last window returns up to and including that day, one
    #   column per window.
    def trailing_means(
        returns: Tensor,
        windows: tuple[int, ...],
    ) -> Tensor:
        ...


    early, late = torch.arange(10.0)[:6], torch.arange(10.0)[8:]        # tensor([0., 1., 2., 3., 4., 5.]), tensor([8., 9.])


    # TODO: Split by time. Return the train x and y, the rows before cut - gap,
    #   then the test x and y, the rows from cut on.
    def time_split(
        x: Tensor,
        y: Tensor,
        cut: int,
        gap: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ...


    highest = torch.tensor([0.1, 0.4, 0.2]).argmax().item()             # 1


    # TODO: Return the number of the column of indicators whose information
    #   coefficient with targets is highest over the first train_rows rows
    #   alone, as a plain int.
    def pick_on_training(
        indicators: Tensor,
        targets: Tensor,
        train_rows: int,
    ) -> int:
        ...


    returns = make_market(DAYS, STRENGTH, 0)
    targets = forward_returns(returns, HORIZON)
    x = trailing_means(returns, WINDOWS)[: len(targets) - FIRST]
    y = targets[FIRST:]
    train_x, train_y, test_x, test_y = time_split(x, y, CUT, HORIZON)
    linear_score = information_coefficient(linear_forecast(train_x, train_y, test_x), test_y)
    nearest_score = information_coefficient(nearest_days(train_x, train_y, test_x, 10), test_y)

    indicators = made_up_indicators(returns, 1000, 1)[: len(y)]
    picked = pick_on_training(indicators, y, CUT - HORIZON)
    picked_score = information_coefficient(indicators[CUT:, picked], y[CUT:])
    ''',
)

FORECAST2_SOLUTION = source_file(
    '''
    # forecast2.py, a solution for comparison

    import torch
    from torch import Tensor

''' + indented(SETTINGS_PART) + '''


''' + indented(MARKET_PART) + '''


''' + indented(FORWARD_PART) + '''


''' + indented(SCORE_PART) + '''


''' + indented(LINEAR_PART) + '''


''' + indented(NEAREST_PART) + '''


''' + indented(INDICATORS_PART) + '''


    means = torch.arange(6.0).unfold(0, 3, 1).mean(dim=1)


''' + indented(TRAILING_PART) + '''


    early, late = torch.arange(10.0)[:6], torch.arange(10.0)[8:]


''' + indented(SPLIT_PART) + '''


    highest = torch.tensor([0.1, 0.4, 0.2]).argmax().item()


''' + indented(PICK_PART) + '''


    returns = make_market(DAYS, STRENGTH, 0)
    targets = forward_returns(returns, HORIZON)
    x = trailing_means(returns, WINDOWS)[: len(targets) - FIRST]
    y = targets[FIRST:]
    train_x, train_y, test_x, test_y = time_split(x, y, CUT, HORIZON)
    linear_score = information_coefficient(linear_forecast(train_x, train_y, test_x), test_y)
    nearest_score = information_coefficient(nearest_days(train_x, train_y, test_x, 10), test_y)

    indicators = made_up_indicators(returns, 1000, 1)[: len(y)]
    picked = pick_on_training(indicators, y, CUT - HORIZON)
    picked_score = information_coefficient(indicators[CUT:, picked], y[CUT:])
    ''',
)


def expect_trailing(
    call: str,
    actual: object,
    expected: Tensor,
    returns: Tensor,
    windows: tuple[int, ...],
) -> None:
    """Names the commonest slip, windows that end a day late, before the plain comparison."""
    late = REFERENCE["trailing_means"](returns[1:], windows)

    if isinstance(actual, Tensor) and actual.shape == late.shape and torch.allclose(actual, late):
        raise Mismatch(
            f"{call} has each window end a day late, so a row holds the next day's return, the first day of its "
            "target. The first row is for day max(windows) - 1, its windows ending on that day.",
        )

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def check_trailing_means(found: object) -> None:
    for seed, days, windows in ((1012, 9, (2, 3)), (1013, 10, (1, 4, 2))):
        returns = seeded_returns(seed, days)
        expect_trailing(
            call=f"trailing_means({returns.text}, {windows})",
            actual=found(returns.value.clone(), windows),
            expected=REFERENCE["trailing_means"](returns.value.clone(), windows),
            returns=returns.value,
            windows=windows,
        )


def check_trailing_future(found: object) -> None:
    """A row may read only the days up to its own. Cutting the later days off must leave every earlier row as it was,
    which a statistic of the whole series, like its mean, would break."""
    returns = torch.randn(14, generator=torch.Generator().manual_seed(1011)) * 0.01
    windows = (2, 4)
    whole = found(returns.clone(), windows)
    reject_missing_return(
        call="trailing_means of 14 days with windows (2, 4)",
        actual=whole,
    )

    for end in (8, 11):
        shorter = found(returns[:end].clone(), windows)

        if not isinstance(whole, Tensor) or not isinstance(shorter, Tensor) or len(shorter) > len(whole):
            return

        if not torch.allclose(whole[: len(shorter)], shorter):
            raise Mismatch(
                f"trailing_means of all 14 days gives other rows than trailing_means of the first {end} days, on the "
                "days both have. A row changed when only later days changed, so it reads the future, as a statistic "
                "of the whole series would. Each row should use its own window alone.",
            )


def indicator_set(
    seed: int,
    rows: int,
    columns: int,
) -> tuple[Described, Described, int]:
    """Columns whose best on the first rows differs from their best over every row, so reading past train_rows shows."""
    generator = torch.Generator().manual_seed(seed)
    targets = torch.randn(rows, generator=generator)
    indicators = torch.randn(rows, columns, generator=generator)
    train_rows = rows // 2
    indicators[:train_rows, 0] += targets[:train_rows]
    indicators[train_rows:, 1] += 3 * targets[train_rows:]

    return (
        Described(value=indicators, text=f"{columns} indicators over {rows} rows, drawn with seed {seed}"),
        Described(value=targets, text=f"their {rows} targets"),
        train_rows,
    )


def expect_column(
    call: str,
    actual: object,
    expected: int,
) -> None:
    if isinstance(actual, Tensor) and actual.numel() == 1 and not actual.is_floating_point():
        raise Mismatch(f"{call} is {value_text(actual)}, a tensor, but it should be the plain int {expected}. Use .item() on it.")

    expect_value(
        call=call,
        actual=actual,
        expected=expected,
    )


def leak_names() -> dict[str, object]:
    returns, x, y = lesson_rows(0)
    train_x, train_y, test_x, test_y = REFERENCE["time_split"](x, y, CUT, HORIZON)
    score = REFERENCE["information_coefficient"]
    indicators = REFERENCE["made_up_indicators"](returns, 1000, 1)[: len(y)]
    picked = REFERENCE["pick_on_training"](indicators, y, CUT - HORIZON)

    return {
        "x": x,
        "train_x": train_x,
        "linear_score": score(REFERENCE["linear_forecast"](train_x, train_y, test_x), test_y),
        "nearest_score": score(REFERENCE["nearest_days"](train_x, train_y, test_x, 10), test_y),
        "picked": picked,
        "picked_score": score(indicators[CUT:, picked], y[CUT:]),
    }


def check_forecast2(checker: Checker) -> None:
    checker.test(
        function="trailing_means",
        description="gives each day's means of its last returns, one column per window",
        run=check_trailing_means,
    )
    checker.test(
        function="trailing_means",
        description="keeps its rows when later days are cut off",
        run=check_trailing_future,
    )
    check_on_inputs(
        checker=checker,
        function="time_split",
        description="trains on the rows before cut - gap and tests on the rows from cut",
        inputs=(
            (
                Described(value=torch.arange(24.0).view(12, 2), text="x of 12 rows numbered in order"),
                Described(value=torch.arange(12.0), text="y of 12 rows numbered in order"),
                8,
                2,
            ),
            (
                Described(value=torch.arange(30.0).view(10, 3), text="x of 10 rows numbered in order"),
                Described(value=torch.arange(10.0), text="y of 10 rows numbered in order"),
                6,
                1,
            ),
        ),
        reference=REFERENCE["time_split"],
        expect=expect_parts("the train x", "the train y", "the test x", "the test y"),
    )
    check_on_inputs(
        checker=checker,
        function="pick_on_training",
        description="picks the best indicator on the training rows alone",
        inputs=(
            indicator_set(1014, 40, 5),
            indicator_set(1015, 60, 8),
        ),
        reference=REFERENCE["pick_on_training"],
        expect=expect_column,
    )

    expected = leak_names()

    for name, description in (
        ("x", "holds seed 0's rows, one per day with a target"),
        ("train_x", "holds the rows before the gap"),
        ("linear_score", "is least squares' score on seed 0's test rows"),
        ("nearest_score", "is the 10 nearest days' score on seed 0's test rows"),
        ("picked", "is the indicator picked on seed 0's training rows"),
        ("picked_score", "is that indicator's score on the test rows"),
    ):
        checker.check_name(
            name=name,
            description=description,
            run=check_wins(name, expected[name]) if name == "picked" else check_value(name, expected[name]),
        )


FORECAST2 = Exercise(
    name="forecast2",
    chapter=QUANT_FORECASTER,
    title="Project: three leaks",
    lesson=FORECAST2_LESSON,
    source=FORECAST2_SOURCE,
    solution=FORECAST2_SOLUTION,
    hints={
        "trailing_means": "For each window, returns.unfold(0, window, 1).mean(dim=1) gives the means from day window - 1 "
        "on. Drop its first max(windows) - window entries so every column starts on the same day, then torch.stack "
        "them with dim=1.",
        "time_split": "Slice x and y with [: cut - gap] for training and [cut:] for testing, in the order the TODO "
        "gives.",
        "pick_on_training": "Score each column of indicators.T on its first train_rows rows against targets' first "
        "train_rows, with information_coefficient. Then torch.tensor the scores, .argmax() and .item().",
        "x": "It comes from trailing_means. Get that right first.",
        "train_x": "It comes from time_split. Get that right first.",
        "linear_score": "It comes from trailing_means and time_split. Get those right first.",
        "nearest_score": "It comes from trailing_means and time_split. Get those right first.",
        "picked": "It comes from pick_on_training. Get that right first.",
        "picked_score": "It comes from pick_on_training. Get that right first.",
    },
    introduces=(),
    check=check_forecast2,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# forecast3
# ---------------------------------------------------------------------------

FORECAST3_LESSON = prose(
    """
    forecast2 scored one test period, and forecast1 showed how far one
    period's score swings, even the rule's. **Walk forward** validation
    scores many periods in turn, the way the forecast would really have run.

    Each fold trains on every row before it, less the purged gap, then
    forecasts the next 500 rows. The next fold moves on 500 rows and trains
    on more. scikit-learn's `TimeSeriesSplit` works this way, each training
    set holding all the ones before it, with a `gap` for forecast2's purging.

    | Fold | Train rows | Test rows |
    |---|---|---|
    | 1 | 0 to 1,494 | 1,500 to 1,999 |
    | 2 | 0 to 1,994 | 2,000 to 2,499 |
    | 5 | 0 to 3,494 | 3,500 to 3,999 |

    Here are least squares and the rule, over the 5 folds of seeds 0 to 4.

    | Seed | Least squares, fold average | Lowest and highest fold | The rule, fold average |
    |---|---|---|---|
    | 0 | 0.035 | -0.124 to 0.138 | 0.098 |
    | 1 | 0.049 | -0.020 to 0.269 | 0.060 |
    | 2 | 0.105 | -0.015 to 0.225 | 0.148 |
    | 3 | 0.031 | -0.063 to 0.105 | 0.064 |
    | 4 | 0.025 | -0.083 to 0.081 | 0.056 |

    Over all 25 folds least squares averages 0.049, the rule 0.085. Single
    folds run from -0.124 to 0.269, the rule's from -0.091 to 0.280. So one
    period can make a weak forecast look strong or worthless. The spread
    across folds and seeds is the error bar on the score.

    The rule's average sits under the ceiling of 0.109 too. Least squares
    learns its weights from noisy rows, so it reaches a little over half the
    rule's score.
    """,
)

FORECAST3_SOURCE = source_file(
    '''
    # forecast3.py

    import torch
    from torch import Tensor

''' + indented(SETTINGS_PART) + '''


    # The market, forecast1's targets, score and rule, forecast2's rows, and least squares.
''' + indented(MARKET_PART) + '''


''' + indented(FORWARD_PART) + '''


''' + indented(SCORE_PART) + '''


''' + indented(EXPECTED_PART) + '''


''' + indented(TRAILING_PART) + '''


''' + indented(LINEAR_PART) + '''


    starts = list(range(1500, 4000, 500))                               # [1500, 2000, 2500, 3000, 3500]


    # TODO: Return the walk forward folds over rows, one (train_end, test_start,
    #   test_end) per fold. The first test starts at first_train, each next
    #   one test_size rows later. A test ends test_size rows on, or at rows if
    #   that comes first. Training ends gap rows before the test starts.
    def walk_forward(
        rows: int,
        first_train: int,
        test_size: int,
        gap: int,
    ) -> list[tuple[int, int, int]]:
        ...


    # TODO: Return least squares' information coefficient in each fold, as plain
    #   numbers in a list. It trains on the rows before train_end and is scored
    #   on the rows from test_start up to test_end.
    def fold_scores(
        x: Tensor,
        y: Tensor,
        folds: list[tuple[int, int, int]],
    ) -> list[float]:
        ...


    returns = make_market(DAYS, STRENGTH, 0)
    targets = forward_returns(returns, HORIZON)
    x = trailing_means(returns, WINDOWS)[: len(targets) - FIRST]
    y = targets[FIRST:]
    best = true_expected(returns, STRENGTH, HORIZON)[FIRST:]
    folds = walk_forward(len(y), 1500, 500, HORIZON)
    scores = fold_scores(x, y, folds)
    average = sum(scores) / len(scores)
    rule_average = sum(information_coefficient(best[start:end], y[start:end]) for _, start, end in folds) / len(folds)
    ''',
)

FORECAST3_SOLUTION = source_file(
    '''
    # forecast3.py, a solution for comparison

    import torch
    from torch import Tensor

''' + indented(SETTINGS_PART) + '''


''' + indented(MARKET_PART) + '''


''' + indented(FORWARD_PART) + '''


''' + indented(SCORE_PART) + '''


''' + indented(EXPECTED_PART) + '''


''' + indented(TRAILING_PART) + '''


''' + indented(LINEAR_PART) + '''


    starts = list(range(1500, 4000, 500))


''' + indented(FOLDS_PART) + '''


''' + indented(FOLD_SCORES_PART) + '''


    returns = make_market(DAYS, STRENGTH, 0)
    targets = forward_returns(returns, HORIZON)
    x = trailing_means(returns, WINDOWS)[: len(targets) - FIRST]
    y = targets[FIRST:]
    best = true_expected(returns, STRENGTH, HORIZON)[FIRST:]
    folds = walk_forward(len(y), 1500, 500, HORIZON)
    scores = fold_scores(x, y, folds)
    average = sum(scores) / len(scores)
    rule_average = sum(information_coefficient(best[start:end], y[start:end]) for _, start, end in folds) / len(folds)
    ''',
)


def expect_folds(
    call: str,
    actual: object,
    expected: list[tuple[int, int, int]],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, list):
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a list of (train_end, test_start, test_end).")

    if len(actual) != len(expected):
        raise Mismatch(f"{call} gives {len(actual)} folds, but it should give {len(expected)}: {expected}.")

    for number, (fold, wanted) in enumerate(zip(actual, expected), start=1):
        if not isinstance(fold, (tuple, list)) or len(fold) != 3:
            raise Mismatch(f"Fold {number} from {call} is {value_text(fold)}, but it should be (train_end, test_start, test_end).")

        if tuple(fold) != wanted:
            raise Mismatch(f"Fold {number} from {call} is {tuple(fold)}, but it should be {wanted}, as (train_end, test_start, test_end).")


def expect_scores(
    call: str,
    actual: object,
    expected: list[float],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, list) or len(actual) != len(expected):
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a list of {len(expected)} scores, one per fold.")

    for number, (score, wanted) in enumerate(zip(actual, expected), start=1):
        expect_close(
            call=f"fold {number}'s score from {call}",
            actual=score,
            expected=wanted,
            tolerance=1e-5,
        )


def fold_inputs(
    seed: int,
    rows: int,
    folds: list[tuple[int, int, int]],
) -> tuple[Described, Described, Described]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, 3, generator=generator)
    y = x @ torch.tensor([0.5, -0.2, 0.1]) + torch.randn(rows, generator=generator)

    return (
        Described(value=x, text=f"x of {rows} rows drawn with seed {seed}"),
        Described(value=y, text=f"y that leans on x"),
        Described(value=folds, text=f"the folds {folds}"),
    )


def walk_names() -> dict[str, object]:
    returns, x, y = lesson_rows(0)
    best = REFERENCE["true_expected"](returns, STRENGTH, HORIZON)[FIRST:]
    folds = REFERENCE["walk_forward"](len(y), 1500, 500, HORIZON)
    scores = REFERENCE["fold_scores"](x, y, folds)
    score = REFERENCE["information_coefficient"]

    return {
        "average": sum(scores) / len(scores),
        "rule_average": sum(score(best[start:end], y[start:end]) for _, start, end in folds) / len(folds),
    }


def check_forecast3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="walk_forward",
        description="gives each fold's training end and test rows",
        inputs=(
            (4000, 1500, 500, 5),
            (23, 10, 5, 2),
            (12, 4, 3, 0),
        ),
        reference=REFERENCE["walk_forward"],
        expect=expect_folds,
    )
    check_on_inputs(
        checker=checker,
        function="fold_scores",
        description="scores least squares in each fold, trained on the rows before it",
        inputs=(
            fold_inputs(1016, 40, [(18, 20, 30), (28, 30, 40)]),
            fold_inputs(1017, 50, [(10, 12, 25), (23, 25, 38), (36, 38, 50)]),
        ),
        reference=REFERENCE["fold_scores"],
        expect=expect_scores,
    )

    expected = walk_names()

    for name, description in (
        ("average", "is least squares' average over seed 0's folds"),
        ("rule_average", "is the rule's average over the same folds"),
    ):
        checker.check_name(
            name=name,
            description=description,
            run=check_value(name, expected[name]),
        )


FORECAST3 = Exercise(
    name="forecast3",
    chapter=QUANT_FORECASTER,
    title="Project: walk forward validation",
    lesson=FORECAST3_LESSON,
    source=FORECAST3_SOURCE,
    solution=FORECAST3_SOLUTION,
    hints={
        "walk_forward": "Loop test_start over range(first_train, rows, test_size). Each fold is (test_start - gap, "
        "test_start, min(test_start + test_size, rows)).",
        "fold_scores": "For each (train_end, test_start, test_end), call linear_forecast on x[:train_end], y[:train_end] "
        "and x[test_start:test_end], then information_coefficient against y[test_start:test_end].",
        "average": "It comes from walk_forward and fold_scores. Get those right first.",
        "rule_average": "It comes from walk_forward. Get that right first.",
    },
    introduces=("walk forward",),
    check=check_forecast3,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# forecast4
# ---------------------------------------------------------------------------

FORECAST4_LESSON = prose(
    """
    design2 sets a design against the simplest baseline first. Least squares
    on the four windows is the baseline here. The model to beat it is a small
    MLP, 4 inputs to 32 hidden to 1. It trains for 300 steps of Adam on
    windows standardized with the training rows.

    The MLP learns the target divided by its spread, so the numbers stay
    near 1. A correlation ignores that scale, so the score is the same.

    Here are both over the 5 walk forward folds of seeds 0 to 4, beside the
    rule.

    | | Average over 25 folds | Lowest and highest fold | Folds won by the MLP |
    |---|---|---|---|
    | The rule | 0.085 | -0.091 to 0.280 | |
    | Least squares | 0.049 | -0.124 to 0.269 | |
    | The MLP | 0.039 | -0.163 to 0.188 | 11 of 25 |

    On seed 0, the market your file runs, the MLP leads, 0.122 against
    0.035. Seed 4 favours it a little too. Seeds 1 to 3 favour least
    squares, and so does the average. One market would have fooled you.

    design2's diagnosis asks whether a model failed to fit or failed to
    generalize. Scored on its own training rows, least squares reaches 0.138
    on average and the MLP 0.412. The MLP fits its training rows far better
    and tests no better, so it failed to generalize. With 32 hidden units it
    learns the noise of its training days.

    That is the honest result: the small model ties least squares at best,
    winning 11 folds of 25. The market's rule is a straight line in its last
    20 returns, so a straight line is the right shape. More freedom only buys
    more noise.

    The habit carries to any module you design. Score it against the
    simplest baseline, across folds and seeds, under a ceiling where you
    have one. Believe a win only when it holds across the spread.
    """,
)

MLP_SOURCE_PART = '''
    # TODO: Train an MLP on x and y and return it. Seed torch with seed, then
    #   build Linear(x.shape[1], 32), ReLU and Linear(32, 1) in an
    #   nn.Sequential. Take steps steps of Adam at a rate of 1e-2, each on the
    #   mean squared error between the model's outputs, squeezed to one per
    #   row, and y divided by y.std().
    def train_mlp(
        x: Tensor,
        y: Tensor,
        seed: int,
        steps: int,
    ) -> nn.Module:
        ...


    # TODO: Standardize train_x and test_x with standardize, train an MLP on
    #   the training rows for 300 steps, and return its forecasts for test_x,
    #   one per row, in eval mode with no autograd recording.
    def mlp_forecast(
        train_x: Tensor,
        train_y: Tensor,
        test_x: Tensor,
        seed: int,
    ) -> Tensor:
        ...


    # TODO: For each fold, train both forecasts on the rows before train_end
    #   and score them on the rows from test_start to test_end. Return one
    #   (least squares' score, the MLP's score) pair per fold.
    def paired_scores(
        x: Tensor,
        y: Tensor,
        folds: list[tuple[int, int, int]],
        seed: int,
    ) -> list[tuple[float, float]]:
        ...
'''

FORECAST4_SOURCE = source_file(
    '''
    # forecast4.py

    import torch
    import torch.nn as nn
    from torch import Tensor

''' + indented(SETTINGS_PART) + '''


    # The market, forecast1's targets and score, forecast2's rows, least squares and forecast3's folds.
''' + indented(MARKET_PART) + '''


''' + indented(FORWARD_PART) + '''


''' + indented(SCORE_PART) + '''


''' + indented(TRAILING_PART) + '''


''' + indented(LINEAR_PART) + '''


''' + indented(FOLDS_PART) + '''


''' + indented(STANDARDIZE_PART) + '''


    small = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
    outputs = small(torch.zeros(3, 4)).squeeze(1).shape                  # torch.Size([3])


''' + MLP_SOURCE_PART + '''


    returns = make_market(DAYS, STRENGTH, 0)
    targets = forward_returns(returns, HORIZON)
    x = trailing_means(returns, WINDOWS)[: len(targets) - FIRST]
    y = targets[FIRST:]
    folds = walk_forward(len(y), 1500, 500, HORIZON)
    pairs = paired_scores(x, y, folds, 0)
    linear_average = sum(baseline for baseline, _ in pairs) / len(pairs)
    mlp_average = sum(model for _, model in pairs) / len(pairs)
    mlp_wins = sum(model > baseline for baseline, model in pairs)
    mlp_training_score = information_coefficient(mlp_forecast(x[:3495], y[:3495], x[:3495], 0), y[:3495])
    linear_training_score = information_coefficient(linear_forecast(x[:3495], y[:3495], x[:3495]), y[:3495])
    ''',
)

FORECAST4_SOLUTION = source_file(
    '''
    # forecast4.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

''' + indented(SETTINGS_PART) + '''


''' + indented(MARKET_PART) + '''


''' + indented(FORWARD_PART) + '''


''' + indented(SCORE_PART) + '''


''' + indented(TRAILING_PART) + '''


''' + indented(LINEAR_PART) + '''


''' + indented(FOLDS_PART) + '''


''' + indented(STANDARDIZE_PART) + '''


    small = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
    outputs = small(torch.zeros(3, 4)).squeeze(1).shape


''' + indented(MLP_PART) + '''


''' + indented(PAIRS_PART) + '''


    returns = make_market(DAYS, STRENGTH, 0)
    targets = forward_returns(returns, HORIZON)
    x = trailing_means(returns, WINDOWS)[: len(targets) - FIRST]
    y = targets[FIRST:]
    folds = walk_forward(len(y), 1500, 500, HORIZON)
    pairs = paired_scores(x, y, folds, 0)
    linear_average = sum(baseline for baseline, _ in pairs) / len(pairs)
    mlp_average = sum(model for _, model in pairs) / len(pairs)
    mlp_wins = sum(model > baseline for baseline, model in pairs)
    mlp_training_score = information_coefficient(mlp_forecast(x[:3495], y[:3495], x[:3495], 0), y[:3495])
    linear_training_score = information_coefficient(linear_forecast(x[:3495], y[:3495], x[:3495]), y[:3495])
    ''',
)


def mlp_data(
    seed: int,
    rows: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, 4, generator=generator)
    y = 0.01 * (x @ torch.tensor([0.5, -0.3, 0.2, 0.1]) + torch.randn(rows, generator=generator))

    return x, y


def check_train_mlp(found: object) -> None:
    for seed, rows, steps in ((1018, 24, 2), (1019, 16, 5)):
        x, y = mlp_data(seed, rows)
        call = f"train_mlp of {rows} rows of 4 numbers, with seed {seed}, for {steps} steps"
        model = found(x.clone(), y.clone(), seed, steps)
        reference = REFERENCE["train_mlp"](x.clone(), y.clone(), seed, steps)
        reject_missing_return(
            call=call,
            actual=model,
        )

        if not isinstance(model, nn.Module):
            raise Mismatch(f"{call} gives {value_text(model)}, but it should give the trained nn.Sequential.")

        shapes = [tuple(parameter.shape) for parameter in model.parameters()]
        wanted_shapes = [tuple(parameter.shape) for parameter in reference.parameters()]

        if shapes != wanted_shapes:
            raise Mismatch(
                f"The model from {call} has parameters of shapes {shapes}, but they should be {wanted_shapes}: Linear(4, 32), "
                "ReLU and Linear(32, 1).",
            )

        for (name, parameter), wanted in zip(model.named_parameters(), reference.parameters()):
            expect_tensor(
                call=f"{name} of the model from {call}",
                actual=parameter.detach(),
                expected=wanted.detach(),
            )


def run_with_reference(
    found: Callable[..., object],
    names: tuple[str, ...],
    run: Callable[[], object],
) -> object:
    """Calls run with the reference versions of names in the learner's module, so a mistake in one of those shows in its
    own row, and puts the learner's back after."""
    globals_ = found.__globals__
    learners = {name: globals_.get(name) for name in names}

    for name in names:
        globals_[name] = REFERENCE[name]

    try:
        return run()
    finally:
        for name in names:
            globals_[name] = learners[name]


def check_mlp_forecast(found: object) -> None:
    for seed, rows in ((1020, 30), (1021, 20)):
        x, y = mlp_data(seed, rows)
        test_x = x[-6:] * 2
        call = f"mlp_forecast trained on {rows} rows with seed {seed}, for 6 test rows"
        states: list[tuple[bool, bool]] = []

        def record(
            module: nn.Module,
            inputs: object,
        ) -> None:
            states.append((module.training, torch.is_grad_enabled()))

        # The reference train_mlp, with a hook on the model it hands back, so the mode and autograd at forecast time show.
        def recording_train(*arguments: object) -> nn.Module:
            model = REFERENCE["train_mlp"](*arguments)
            model.register_forward_pre_hook(record)

            return model

        globals_ = found.__globals__
        learners = {name: globals_.get(name) for name in ("train_mlp", "standardize")}
        globals_["train_mlp"] = recording_train
        globals_["standardize"] = REFERENCE["standardize"]

        try:
            actual = found(x.clone(), y.clone(), test_x.clone(), seed)
        finally:
            for name, value in learners.items():
                globals_[name] = value

        expected = REFERENCE["mlp_forecast"](x.clone(), y.clone(), test_x.clone(), seed)
        expect_tensor(
            call=call,
            actual=actual,
            expected=expected,
        )

        if not states:
            raise Mismatch(f"{call} never ran the model from train_mlp. Train it with train_mlp, then run it on test_x.")

        training, recording = states[-1]

        if training:
            raise Mismatch(f"{call} ran the model in training mode. Call model.eval() before forecasting.")

        if recording:
            raise Mismatch(f"{call} recorded for autograd. Forecast inside torch.no_grad().")


def check_paired_scores(found: object) -> None:
    for seed, rows, folds in (
        (1022, 60, [(28, 30, 45), (43, 45, 60)]),
        (1023, 50, [(18, 20, 35), (33, 35, 50)]),
    ):
        x, y = mlp_data(seed, rows)
        call = f"paired_scores of {rows} rows with seed {seed} over the folds {folds}"
        actual = run_with_reference(
            found=found,
            names=("mlp_forecast", "train_mlp", "standardize"),
            run=lambda x=x, y=y, folds=folds, seed=seed: found(x.clone(), y.clone(), folds, seed),
        )
        expected = REFERENCE["paired_scores"](x.clone(), y.clone(), folds, seed)
        reject_missing_return(
            call=call,
            actual=actual,
        )

        if not isinstance(actual, list) or len(actual) != len(expected):
            raise Mismatch(f"{call} is {value_text(actual)}, but it should be a list of {len(expected)} pairs, one per fold.")

        for number, (pair, wanted) in enumerate(zip(actual, expected), start=1):
            expect_parts("least squares' score", "the MLP's score")(
                call=f"fold {number} of {call}",
                actual=pair,
                expected=wanted,
            )


def model_names() -> dict[str, object]:
    _, x, y = lesson_rows(0)
    folds = REFERENCE["walk_forward"](len(y), 1500, 500, HORIZON)
    pairs = REFERENCE["paired_scores"](x, y, folds, 0)
    score = REFERENCE["information_coefficient"]

    return {
        "linear_average": sum(baseline for baseline, _ in pairs) / len(pairs),
        "mlp_average": sum(model for _, model in pairs) / len(pairs),
        "mlp_wins": sum(model > baseline for baseline, model in pairs),
        "mlp_training_score": score(REFERENCE["mlp_forecast"](x[:3495], y[:3495], x[:3495], 0), y[:3495]),
        "linear_training_score": score(REFERENCE["linear_forecast"](x[:3495], y[:3495], x[:3495]), y[:3495]),
    }


def check_wins(
    name: str,
    expected: int,
) -> Callable[[object], None]:
    def run(value: object) -> None:
        expect_value(
            call=name,
            actual=value,
            expected=expected,
        )

    return run


def check_forecast4(checker: Checker) -> None:
    checker.test(
        function="train_mlp",
        description="builds the MLP and trains it, gradients cleared every step",
        run=check_train_mlp,
    )
    checker.test(
        function="mlp_forecast",
        description="standardizes, trains, then forecasts in eval mode without autograd",
        run=check_mlp_forecast,
    )
    checker.test(
        function="paired_scores",
        description="scores both forecasts in every fold",
        run=check_paired_scores,
    )

    expected = model_names()

    for name, description in (
        ("linear_average", "is least squares' average over seed 0's folds"),
        ("mlp_average", "is the MLP's average over the same folds"),
        ("mlp_training_score", "is the MLP's score on its own training rows"),
        ("linear_training_score", "is least squares' score on its own training rows"),
    ):
        checker.check_name(
            name=name,
            description=description,
            run=check_value(name, expected[name], 1e-4),
        )

    checker.check_name(
        name="mlp_wins",
        description="counts the folds the MLP won",
        run=check_wins("mlp_wins", expected["mlp_wins"]),
    )


FORECAST4 = Exercise(
    name="forecast4",
    chapter=QUANT_FORECASTER,
    title="Project: a model against the baseline",
    lesson=FORECAST4_LESSON,
    source=FORECAST4_SOURCE,
    solution=FORECAST4_SOLUTION,
    hints={
        "train_mlp": "torch.manual_seed(seed) before building the nn.Sequential. In each step: zero_grad, the mean of "
        "(model(x).squeeze(1) - y / scale) ** 2 with scale = y.std(), backward, step.",
        "mlp_forecast": "train_x, test_x = standardize(train_x, test_x), then train_mlp(train_x, train_y, seed, 300). "
        "model.eval(), and inside torch.no_grad() return model(test_x).squeeze(1).",
        "paired_scores": "For each fold, slice the training rows with [:train_end] and the test rows with "
        "[test_start:test_end]. Score linear_forecast and mlp_forecast with information_coefficient and append the "
        "pair.",
        "linear_average": "It comes from paired_scores. Get that right first.",
        "mlp_average": "It comes from paired_scores. Get that right first.",
        "mlp_wins": "It comes from paired_scores. Get that right first.",
        "mlp_training_score": "It comes from mlp_forecast. Get that right first.",
        "linear_training_score": "It comes from the given least squares, so it passes once the file runs.",
    },
    introduces=(),
    check=check_forecast4,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The measurements behind the lessons
# ---------------------------------------------------------------------------


def column_scores(
    columns: Tensor,
    targets: Tensor,
) -> Tensor:
    """The information coefficient of every column against targets at once."""
    centered = columns - columns.mean(dim=0)
    target = targets - targets.mean()

    return (centered * target.unsqueeze(1)).sum(dim=0) / (centered.norm(dim=0) * target.norm())


def measure_forecaster(seeds: range = range(5)) -> dict[str, object]:
    """Every number in the four lessons, on the CPU with 4 threads, in about a minute. Not run by any check."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    score = REFERENCE["information_coefficient"]
    results: dict[str, object] = {}

    try:
        long = REFERENCE["make_market"](100_000, STRENGTH, 1)
        long_targets = REFERENCE["forward_returns"](long, HORIZON)
        long_best = REFERENCE["true_expected"](long, STRENGTH, HORIZON)
        results["ceiling"] = score(long_best[FIRST:], long_targets[FIRST:])

        for seed in seeds:
            returns, x, y = lesson_rows(seed)
            best = REFERENCE["true_expected"](returns, STRENGTH, HORIZON)[FIRST:]
            train_x, train_y, test_x, test_y = REFERENCE["time_split"](x, y, CUT, HORIZON)
            late = REFERENCE["trailing_means"](returns[1:], WINDOWS)[: len(y)]
            late_train, _, late_test, _ = REFERENCE["time_split"](late, y, CUT, HORIZON)
            shuffled = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
            random_train, random_test = shuffled[:CUT], shuffled[CUT:]
            indicators = REFERENCE["made_up_indicators"](returns, 1000, 1)[: len(y)]
            picked_train = column_scores(indicators[: CUT - HORIZON], y[: CUT - HORIZON]).argmax()
            picked_whole = column_scores(indicators, y).argmax()

            results[f"seed {seed}"] = {
                "rule on test rows": score(best[CUT:], test_y),
                "least squares": score(REFERENCE["linear_forecast"](train_x, train_y, test_x), test_y),
                "least squares, late windows": score(REFERENCE["linear_forecast"](late_train, train_y, late_test), test_y),
                "nearest": score(REFERENCE["nearest_days"](train_x, train_y, test_x, 10), test_y),
                "nearest, random split": score(
                    REFERENCE["nearest_days"](x[random_train], y[random_train], x[random_test], 10),
                    y[random_test],
                ),
                "picked on training": score(indicators[CUT:, picked_train], test_y),
                "picked on every row": score(indicators[CUT:, picked_whole], test_y),
            }

            folds = REFERENCE["walk_forward"](len(y), 1500, 500, HORIZON)
            pairs = REFERENCE["paired_scores"](x, y, folds, seed)
            results[f"seed {seed} folds"] = [
                {
                    "rule": score(best[start:end], y[start:end]),
                    "least squares": baseline,
                    "mlp": model,
                    "least squares on training": score(REFERENCE["linear_forecast"](x[:train_end], y[:train_end], x[:train_end]), y[:train_end]),
                    "mlp on training": score(REFERENCE["mlp_forecast"](x[:train_end], y[:train_end], x[:train_end], seed), y[:train_end]),
                }
                for (train_end, start, end), (baseline, model) in zip(folds, pairs)
            ]
    finally:
        torch.set_num_threads(threads)

    return results
