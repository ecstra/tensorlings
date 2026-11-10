"""95_time_series, a series read in patches: patches as tokens as in PatchTST, each window normalized on its own and
restored after as in RevIN, and PatchTST's two choices, patches and columns apart, each tested against its other way,
on the CDC's weekly flu table."""

from __future__ import annotations

import contextlib
import math
import textwrap
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import load_illness
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
    reject_missing_return,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

TIME_SERIES = Chapter(
    folder="95_time_series",
    title="Time series",
    readme=source_file(
        """
        # Time series

        A time series is a sequence of readings taken at a steady pace. This
        chapter cuts a series into patches and turns each patch into a token.
        It normalizes every window with its own mean and standard deviation,
        and undoes that normalization on the forecast. Then it tests
        PatchTST's two design choices, patches and keeping the columns apart,
        each against the alternative. The data is the CDC's weekly count of
        doctor visits for flu-like illness in the US, from 2002 to 2020.

        ## Further reading

        - [Nie and colleagues, PatchTST](https://arxiv.org/abs/2211.14730)
        - [Kim and colleagues, RevIN](https://github.com/ts-kim/RevIN)
        - [Zeng and colleagues, are transformers effective for time series forecasting?](https://arxiv.org/abs/2205.13504)
        - [The CDC's FluView, where the flu counts come from](https://www.cdc.gov/fluview/index.html)
        """,
    ),
)


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """The CPU's thread count moves the last digits of a run, so the measurements pin it."""
    before = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        yield
    finally:
        torch.set_num_threads(before)


def indented(part: str) -> str:
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(code, namespace)

    return namespace


def seeded_model(
    make: Callable[..., nn.Module],
    *arguments: object,
) -> nn.Module:
    torch.manual_seed(0)

    return make(*arguments)


def recording_hook(
    model: nn.Module,
    seen: list[tuple[bool, bool]],
) -> torch.utils.hooks.RemovableHandle:
    """Notes on every call whether the model was in training mode and autograd was recording."""
    return model.register_forward_pre_hook(lambda module, inputs: seen.append((module.training, torch.is_grad_enabled())))


# The split, the standardizing and the windows, given in every file of the chapter.
GIVEN_PART = '''
LOOKBACK = 104          # two years of weeks, how far back PatchTST reads in the flu table
TRAIN_END = 676         # the first 70% of the 966 weeks are for training
VALIDATION_END = 773    # the next 10% pick the best epoch, and the last 20% are for testing


def standardized_series(table: Tensor) -> Tensor:
    training = table[:TRAIN_END]

    return (table - training.mean(dim=0)) / training.std(dim=0)


# Cuts series into windows of LOOKBACK weeks, each paired with the horizon weeks that follow it. It keeps the
# pairs whose following weeks start at or after week start and end by week end.
def windows(
    series: Tensor,
    start: int,
    end: int,
    horizon: int,
) -> tuple[Tensor, Tensor]:
    starts = range(max(start, LOOKBACK), end - horizon + 1)
    past = torch.stack([series[week - LOOKBACK : week] for week in starts])
    future = torch.stack([series[week : week + horizon] for week in starts])

    return past, future
'''

COUNT_SOLUTION = '''
def patch_count(
    lookback: int,
    length: int,
    stride: int,
) -> int:
    return (lookback - length) // stride + 2
'''

PATCHES_SOLUTION = '''
def patches(
    series: Tensor,
    length: int,
    stride: int,
) -> Tensor:
    padded = torch.cat([series, series[:, -1:].expand(-1, stride)], dim=1)

    return padded.unfold(1, length, stride)
'''

EMBEDDING_INIT = '''
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        lookback: int,
        length: int,
        stride: int,
        size: int,
    ) -> None:
        super().__init__()
        self.length = length
        self.stride = stride
        self.project = nn.Linear(length, size)
        self.position = nn.Parameter(torch.randn(patch_count(lookback, length, stride), size) * 0.02)
'''

EMBEDDING_FORWARD = '''
    def forward(
        self,
        series: Tensor,
    ) -> Tensor:
        return self.project(patches(series, self.length, self.stride)) + self.position
'''

PATCH_SOLUTION = COUNT_SOLUTION + PATCHES_SOLUTION + EMBEDDING_INIT + EMBEDDING_FORWARD

APART_SOLUTION = '''
def channels_apart(windows: Tensor) -> Tensor:
    return windows.transpose(1, 2).reshape(-1, windows.shape[1])


def channels_together(
    forecasts: Tensor,
    channels: int,
) -> Tensor:
    return forecasts.reshape(-1, channels, forecasts.shape[1]).transpose(1, 2)
'''

NORMALIZE_SOLUTION = '''
def normalized(series: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    mean = series.mean(dim=1, keepdim=True)
    spread = series.std(dim=1, keepdim=True) + 1e-5

    return (series - mean) / spread, mean, spread


def restored(
    forecasts: Tensor,
    mean: Tensor,
    spread: Tensor,
) -> Tensor:
    return forecasts * spread + mean
'''

LINEAR_PART = '''
class LinearForecaster(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(lookback, horizon)

    def forward(
        self,
        series: Tensor,
    ) -> Tensor:
        return self.linear(series)
'''

NORMALIZED_INIT = '''
class NormalizedLinear(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(lookback, horizon)
'''

NORMALIZED_FORWARD = '''
    def forward(
        self,
        series: Tensor,
    ) -> Tensor:
        scaled, mean, spread = normalized(series)

        return restored(self.linear(scaled), mean, spread)
'''

REVIN_SOLUTION = APART_SOLUTION + NORMALIZE_SOLUTION + LINEAR_PART + NORMALIZED_INIT + NORMALIZED_FORWARD

FORECASTER_INIT = '''
class PatchForecaster(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        length: int = 24,
        stride: int = 2,
        size: int = 16,
        blocks: int = 3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.embed = PatchEmbedding(lookback, length, stride, size)
        self.dropout = nn.Dropout(0.3)
        self.blocks = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    size,
                    heads,
                    128,
                    dropout=0.3,
                    batch_first=True,
                )
                for _ in range(blocks)
            ],
        )
        self.head = nn.Linear(patch_count(lookback, length, stride) * size, horizon)
'''

FORECASTER_FORWARD = '''
    def forward(
        self,
        series: Tensor,
    ) -> Tensor:
        scaled, mean, spread = normalized(series)
        tokens = self.blocks(self.dropout(self.embed(scaled)))

        return restored(self.head(tokens.flatten(1)), mean, spread)
'''

ERROR_SOLUTION = '''
def forecast_error(
    model: nn.Module,
    past: Tensor,
    future: Tensor,
) -> float:
    model.eval()

    with torch.no_grad():
        error = F.mse_loss(model(channels_apart(past)), channels_apart(future)).item()

    model.train()

    return error
'''

FORECASTER_SOLUTION = FORECASTER_INIT + FORECASTER_FORWARD + ERROR_SOLUTION

REFERENCE = namespace_of(GIVEN_PART + PATCH_SOLUTION + REVIN_SOLUTION + FORECASTER_SOLUTION)


def flu_series() -> Tensor:
    return REFERENCE["standardized_series"](load_illness())


# ---------------------------------------------------------------------------
# series1
# ---------------------------------------------------------------------------

SERIES1_LESSON = prose(
    """
    A time series is a sequence of readings taken at a steady pace, such as
    once a week. Here it is the CDC's flu table, 966 weeks from 2002 to 2020.
    Each week holds 7 numbers, such as the percentage of doctor visits that
    were for flu-like illness and the number of patients seen.

    The task is to forecast the coming weeks of each column from the last
    104 weeks, which is two years of history. Section 6 read sequences one
    step at a time. A transformer could instead take each week as a token, as
    gpt3 took each character as a token.

    A single reading says little on its own. A patch, a run of consecutive
    readings, shows a shape such as a rise or a peak. Nie and colleagues cut
    the series into patches and make each patch a token. They call the model
    **PatchTST**.

    ```python
    torch.arange(6.0).unsqueeze(0).unfold(1, 3, 2)    # tensor([[[0., 1., 2.], [2., 3., 4.]]])
    ```

    `unfold(1, 3, 2)` slides a window of 3 readings along dimension 1 in
    steps of 2, and returns each window as a patch. The step between patches
    is called the stride. Here the last reading, 5, is in no patch at all. So
    before cutting, PatchTST adds `stride` copies of the last reading to the
    end of the series. That gives the final readings a patch of their own,
    and adds one patch to the count.

    PatchTST's own scripts for the flu data use patches of 24 weeks, with a
    new patch every 2 weeks. So 104 weeks become 42 patches. A Linear layer
    turns each patch into a token, a vector of size 16. Then a learned
    position embedding is added, as in gpt2.

    Attention computes a score for every pair of tokens. With 104 tokens that
    is 104 * 104 = 10,816 scores in each head of each block. With 42 patches
    it is 1,764, about 6 times fewer. The paper uses a stride of 8 on bigger
    tables. The number of tokens shrinks by about the stride, so the number
    of scores shrinks by about the square of the stride.

    PatchTST also treats every column as a separate series, and uses one
    shared model for all 7 columns. series2 splits the columns apart this
    way.
    """,
)

SERIES1_SOURCE = source_file(
    '''
    # series1.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import ILLNESS_COLUMNS, load_illness

'''
    + indented(GIVEN_PART)
    + '''


    count = (10 - 4) // 2 + 2    # a series of 10 readings makes 5 patches of 4 readings with a stride of 2


    # TODO: Return how many patches a series of `lookback` readings makes.
    #   First `stride` copies of the last reading are added at the end, as
    #   PatchTST does. Then each patch holds `length` readings, and a new patch
    #   starts every `stride` readings. Return a plain int.
    #   Example: patch_count(104, 24, 2) returns 42.
    def patch_count(
        lookback: int,
        length: int,
        stride: int,
    ) -> int:
        ...


    sliding = torch.arange(6.0).unsqueeze(0).unfold(1, 3, 2)    # tensor([[[0., 1., 2.], [2., 3., 4.]]])


    # TODO: Return the patches of each series. `series` has shape
    #   (rows, readings). Add `stride` copies of each row's last reading to
    #   the end of that row. Then cut patches of `length` readings, with a new
    #   patch every `stride` readings. The result has shape
    #   (rows, patch_count(readings, length, stride), length).
    #   Example: patches(torch.arange(6.0).unsqueeze(0), 3, 2) returns
    #   tensor([[[0., 1., 2.], [2., 3., 4.], [4., 5., 5.]]]).
    def patches(
        series: Tensor,
        length: int,
        stride: int,
    ) -> Tensor:
        ...


    # TODO: Finish forward. Cut series into patches with your patches function,
    #   self.length and self.stride. Turn each patch into a token with
    #   self.project, then add self.position. The result has shape
    #   (rows, patches, size).
    class PatchEmbedding(nn.Module):
        def __init__(
            self,
            lookback: int,
            length: int,
            stride: int,
            size: int,
        ) -> None:
            super().__init__()
            self.length = length
            self.stride = stride
            self.project = nn.Linear(length, size)
            self.position = nn.Parameter(torch.randn(patch_count(lookback, length, stride), size) * 0.02)

        def forward(
            self,
            series: Tensor,
        ) -> Tensor:
            ...


    series = standardized_series(load_illness())
    past, future = windows(series, 0, TRAIN_END, 24)
    flu_patches = patches(past[:, :, 0], 24, 2)
    ''',
)

SERIES1_SOLUTION = source_file(
    '''
    # series1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import ILLNESS_COLUMNS, load_illness

'''
    + indented(GIVEN_PART)
    + '''


    count = (10 - 4) // 2 + 2

'''
    + indented(COUNT_SOLUTION)
    + '''


    sliding = torch.arange(6.0).unsqueeze(0).unfold(1, 3, 2)

'''
    + indented(PATCHES_SOLUTION)
    + "\n\n\n"
    + indented(EMBEDDING_INIT)
    + "\n"
    + textwrap.indent(EMBEDDING_FORWARD.strip("\n"), "    ")
    + '''


    series = standardized_series(load_illness())
    past, future = windows(series, 0, TRAIN_END, 24)
    flu_patches = patches(past[:, :, 0], 24, 2)
    ''',
)


def expect_whole_number(
    call: str,
    actual: object,
    expected: int,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if isinstance(actual, float):
        raise Mismatch(f"{call} is {actual}, a float. A count of patches is a whole number, so use // and return an int.")

    expect_value(
        call=call,
        actual=actual,
        expected=expected,
    )


class ReferencePatchEmbedding(nn.Module):
    def __init__(
        self,
        lookback: int,
        length: int,
        stride: int,
        size: int,
    ) -> None:
        super().__init__()
        self.length = length
        self.stride = stride
        self.project = nn.Linear(length, size)
        self.position = nn.Parameter(torch.randn(REFERENCE["patch_count"](lookback, length, stride), size) * 0.02)

    def forward(
        self,
        series: Tensor,
    ) -> Tensor:
        return self.project(REFERENCE["patches"](series, self.length, self.stride)) + self.position


def check_patch_embedding(found: type) -> None:
    for lookback, length, stride, size, rows, seed in ((10, 4, 2, 3, 2, 9501), (12, 3, 3, 2, 3, 9502)):
        module = expect_module(
            f"PatchEmbedding({lookback}, {length}, {stride}, {size})",
            seeded_model(found, lookback, length, stride, size),
        )
        reference = seeded_model(ReferencePatchEmbedding, lookback, length, stride, size)

        # A learned position of near zeros would hide a missing one, so the check sets it.
        for made in (module, reference):
            with torch.no_grad():
                made.position.copy_(torch.linspace(-1, 1, made.position.numel()).view_as(made.position))

        series = small_tensor(seed, rows, lookback)

        with torch.no_grad():
            actual = module(series)
            expected = reference(series)

        expect_tensor(
            call=f"PatchEmbedding({lookback}, {length}, {stride}, {size}) on {rows} series",
            actual=actual,
            expected=expected,
        )


def check_flu_patches(value: object) -> None:
    past, _ = REFERENCE["windows"](flu_series(), 0, REFERENCE["TRAIN_END"], 24)

    expect_tensor(
        call="flu_patches",
        actual=value,
        expected=REFERENCE["patches"](past[:, :, 0], 24, 2),
    )


def check_series1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="patch_count",
        description="counts the patches, including the extra one from the copies at the end",
        inputs=(
            (104, 24, 2),
            (336, 16, 8),
            (10, 3, 3),
        ),
        reference=REFERENCE["patch_count"],
        expect=expect_whole_number,
    )
    check_on_inputs(
        checker=checker,
        function="patches",
        description="adds stride copies of the last reading, then cuts a patch every stride readings",
        inputs=(
            (small_tensor(9503, 2, 10), 4, 2),
            (small_tensor(9504, 3, 9), 3, 3),
            (small_tensor(9505, 1, 7), 2, 1),
        ),
        reference=REFERENCE["patches"],
    )
    checker.test_method(
        class_name="PatchEmbedding",
        method="forward",
        description="turns each patch into a token and adds its position",
        run=check_patch_embedding,
    )
    checker.check_name(
        name="flu_patches",
        description="is the first column's training windows, cut into patches of 24 weeks",
        run=check_flu_patches,
    )


SERIES1 = Exercise(
    name="series1",
    chapter=TIME_SERIES,
    title="Patches as tokens",
    lesson=SERIES1_LESSON,
    source=SERIES1_SOURCE,
    solution=SERIES1_SOLUTION,
    hints={
        "patch_count": "(lookback - length) // stride counts the patches that start after the first one. Add 1 for "
        "the first patch, and 1 more for the extra patch that the copies at the end make.",
        "patches": "series[:, -1:] takes each row's last reading as a column of shape (rows, 1). .expand(-1, stride) "
        "repeats it stride times. torch.cat that onto the end of series along dim=1, then call "
        ".unfold(1, length, stride).",
        "PatchEmbedding.forward": "patches(series, self.length, self.stride) has shape (rows, patches, length). "
        "self.project turns the last dimension into size, then add self.position.",
        "flu_patches": "It comes from patches. Get that right first.",
    },
    introduces=("PatchTST",),
    check=check_series1,
)


# ---------------------------------------------------------------------------
# series2
# ---------------------------------------------------------------------------

SERIES2_LESSON = prose(
    """
    PatchTST treats each of the 7 columns as a separate series, and all 7
    columns share one model. So a batch of windows of shape (rows, 104, 7)
    becomes a batch of 7 times as many series, each of 104 readings.

    ```python
    torch.zeros(2, 5, 3).transpose(1, 2).reshape(-1, 5).shape    # torch.Size([6, 5])
    ```

    A series can drift, which means its typical level changes over time.
    Over 18 years, more doctors started reporting to the CDC, and they saw
    more patients. Each column was standardized with the mean and standard
    deviation of the training weeks, so its mean over the training weeks is
    0. Here is each column's mean over the test weeks.

    | Share of visits, weighted | Visits, ages 0 to 4 | Flu visits | Doctors | Patients |
    |---|---|---|---|---|
    | 0.40 | 1.50 | 1.97 | 2.44 | 2.90 |

    On average, the number of patients seen in the test weeks is 2.9 training
    standard deviations above the training mean. A model fitted to the
    training levels misses by about that much.

    **RevIN**, from Kim and colleagues, normalizes each window with the
    window's own mean and standard deviation before the model sees it. Then
    it reverses the normalization on the forecast. So the model sees only the
    shape of the window, and the level of the forecast comes from the window
    itself. RevIN also learns a scale and a shift, which this chapter leaves
    out.

    ```text
    forecast = model((window - mean) / spread) * spread + mean
    ```

    Here `spread` is the window's standard deviation, the name the code uses
    for it. This file computes it with `.std()` and adds 0.00001, so a flat
    window never divides by 0. RevIN's own code differs in two small ways. It
    adds the 0.00001 to the variance inside the square root, and its variance
    divides by n instead of n - 1. It also stops gradients from flowing
    through the mean and the spread.

    Here is a plain Linear layer that maps 104 weeks to the following weeks,
    with and without RevIN. Each trained with Adam for 60 epochs on batches
    of 128, with 3 seeds. The error is the mean squared error over the test
    weeks, measured on the standardized values.

    | Weeks ahead | 24 | 36 | 48 | 60 |
    |---|---|---|---|---|
    | Linear | 2.21 to 2.32 | 2.16 to 2.31 | 2.14 to 2.30 | 2.29 to 2.37 |
    | Linear with RevIN | 2.05 to 2.10 | 1.89 to 1.94 | 1.89 to 1.92 | 1.95 to 2.01 |

    The horizon is how many weeks ahead the model forecasts. RevIN lowers
    the error at every horizon on every seed. Averaged over the 3 seeds, it
    cuts the error by 8% at 24 weeks and by 14% to 15% at the longer
    horizons. For single seeds the cut ranges from 5% to 17%. The further
    ahead the forecast, the more the level matters, and the more RevIN helps.
    """,
)

SERIES2_SOURCE = source_file(
    '''
    # series2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import ILLNESS_COLUMNS, load_illness

'''
    + indented(GIVEN_PART)
    + '''


    rows = torch.zeros(2, 5, 3).transpose(1, 2).reshape(-1, 5).shape    # torch.Size([6, 5])


    # TODO: Return the windows with every column as a separate series. windows
    #   has shape (count, readings, columns). Return shape
    #   (count * columns, readings), with rows in window order, so the first
    #   window's columns come first, in column order.
    #   Example: for windows of shape (2, 5, 3), row 4 of the result is
    #   windows[1, :, 1], column 1 of window 1.
    def channels_apart(windows: Tensor) -> Tensor:
        ...


    # TODO: Undo channels_apart for forecasts. forecasts has shape
    #   (count * channels, horizon), with rows in the order channels_apart
    #   makes. Return shape (count, horizon, channels).
    def channels_together(
        forecasts: Tensor,
        channels: int,
    ) -> Tensor:
        ...


    spread = torch.tensor([[1.0, 3.0]]).std(dim=1, keepdim=True)    # tensor([[1.4142]])


    # TODO: Return each series scaled to a mean of 0 and a standard deviation
    #   of 1, together with the mean and spread used. series has shape
    #   (rows, readings). Compute the mean with .mean() and the spread with
    #   .std(), both along dim 1 and keeping that dimension, then add 1e-5 to
    #   the spread. That is the lesson's form, not RevIN's own square root of
    #   the variance. Return the tuple (scaled series, mean, spread).
    def normalized(series: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ...


    # TODO: Undo normalized on the forecasts. forecasts has shape
    #   (rows, horizon), and mean and spread have shape (rows, 1). Multiply
    #   each row by its spread, then add its mean.
    def restored(
        forecasts: Tensor,
        mean: Tensor,
        spread: Tensor,
    ) -> Tensor:
        ...


    # The plain Linear from the first row of the lesson's table.
'''
    + indented(LINEAR_PART)
    + '''


    # TODO: Finish forward. Normalize the series with normalized, run
    #   self.linear on the scaled series, and undo the normalization on the
    #   forecasts with restored, using the mean and spread from normalized.
    class NormalizedLinear(nn.Module):
        def __init__(
            self,
            lookback: int,
            horizon: int,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(lookback, horizon)

        def forward(
            self,
            series: Tensor,
        ) -> Tensor:
            ...
    ''',
)

SERIES2_SOLUTION = source_file(
    '''
    # series2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import ILLNESS_COLUMNS, load_illness

'''
    + indented(GIVEN_PART)
    + '''


    rows = torch.zeros(2, 5, 3).transpose(1, 2).reshape(-1, 5).shape

'''
    + indented(APART_SOLUTION)
    + '''


    spread = torch.tensor([[1.0, 3.0]]).std(dim=1, keepdim=True)

'''
    + indented(NORMALIZE_SOLUTION)
    + '''


    # The plain Linear from the first row of the lesson's table.
'''
    + indented(LINEAR_PART)
    + "\n\n\n"
    + indented(NORMALIZED_INIT)
    + "\n"
    + textwrap.indent(NORMALIZED_FORWARD.strip("\n"), "    ")
    + "\n",
)


def check_normalized_linear(found: type) -> None:
    for lookback, horizon, rows, seed in ((6, 3, 2, 9511), (5, 2, 3, 9512)):
        module = expect_module(f"NormalizedLinear({lookback}, {horizon})", seeded_model(found, lookback, horizon))
        reference = seeded_model(REFERENCE["NormalizedLinear"], lookback, horizon)
        series = small_tensor(seed, rows, lookback) + 5

        with torch.no_grad():
            actual = module(series)
            expected = reference(series)

        expect_tensor(
            call=f"NormalizedLinear({lookback}, {horizon}) on {rows} series around 5",
            actual=actual,
            expected=expected,
        )


def check_series2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="channels_apart",
        description="makes each column of each window a series of its own",
        inputs=(
            (small_tensor(9513, 2, 4, 3),),
            (small_tensor(9514, 3, 2, 2),),
        ),
        reference=REFERENCE["channels_apart"],
    )
    check_on_inputs(
        checker=checker,
        function="channels_together",
        description="puts the columns of each window back together",
        inputs=(
            (small_tensor(9515, 6, 4), 3),
            (small_tensor(9516, 4, 2), 2),
        ),
        reference=REFERENCE["channels_together"],
    )
    check_on_inputs(
        checker=checker,
        function="normalized",
        description="normalizes each series with its own mean and spread, and returns both",
        inputs=(
            (small_tensor(9517, 2, 5) + 3,),
            (small_tensor(9518, 3, 4),),
        ),
        reference=REFERENCE["normalized"],
        expect=expect_parts("the scaled series", "the mean", "the spread"),
    )
    check_on_inputs(
        checker=checker,
        function="restored",
        description="multiplies by each row's spread and adds its mean",
        inputs=(
            (small_tensor(9519, 2, 3), small_tensor(9520, 2, 1), small_tensor(9521, 2, 1).abs() + 1),
            (small_tensor(9522, 3, 2), small_tensor(9523, 3, 1), small_tensor(9524, 3, 1).abs() + 1),
        ),
        reference=REFERENCE["restored"],
    )
    checker.test_method(
        class_name="NormalizedLinear",
        method="forward",
        description="normalizes, runs the Linear, and restores the level",
        run=check_normalized_linear,
    )


SERIES2 = Exercise(
    name="series2",
    chapter=TIME_SERIES,
    title="Normalizing each window",
    lesson=SERIES2_LESSON,
    source=SERIES2_SOURCE,
    solution=SERIES2_SOLUTION,
    hints={
        "channels_apart": ".transpose(1, 2) puts the columns before the readings. Then .reshape(-1, windows.shape[1]) "
        "makes each column of each window a row.",
        "channels_together": ".reshape(-1, channels, forecasts.shape[1]) groups each window's rows, then "
        ".transpose(1, 2) puts the columns last.",
        "normalized": "Take series.mean(dim=1, keepdim=True), and the same with .std, plus 1e-5. That is this file's "
        "form, not RevIN's square root of the variance. Return (series - mean) / spread, mean and spread.",
        "restored": "Multiply by spread, then add mean. Both have one column, so they broadcast over the forecasts.",
        "NormalizedLinear.forward": "normalized(series) returns three things, the scaled series, its mean and its "
        "spread. Run self.linear on the scaled series, then pass the result, the mean and the spread to restored.",
    },
    introduces=("RevIN",),
    check=check_series2,
)


# ---------------------------------------------------------------------------
# series3
# ---------------------------------------------------------------------------

SERIES3_LESSON = prose(
    """
    Together, series1's patches and series2's RevIN make PatchTST. A window
    is normalized, cut into patches, turned into tokens, and passed through 3
    transformer blocks. The outputs of the last block are flattened into one
    long vector, and a Linear layer maps that vector to the forecast.
    Finally the window's mean and spread are put back on the forecast.

    The sizes are the ones PatchTST uses for this table. Tokens have size 16,
    attention has 4 heads, and Adam trains with a learning rate of 0.0025.
    Dropout is 0.3, and `nn.TransformerEncoderLayer` also applies it to the
    attention weights. PatchTST itself applies no dropout to the attention
    weights. It also normalizes with BatchNorm instead of LayerNorm.

    Zeng and colleagues found that a one layer linear model beat the time
    series transformers built before PatchTST. Nie and colleagues answered
    with two design choices. Here each choice is tested against the
    alternative.

    - **Patches**: each token covers 24 weeks, and a new token starts every 2 weeks. The alternative makes every week its own token, with `PatchForecaster(lookback, horizon, 1, 1)`.
    - **Columns apart**: each column is a separate series, as in series2. The alternative puts the same patch of all 7 columns into one token.

    The paper argues that patches let a model look further back with fewer
    tokens. So PatchTST here reads 148 weeks, nearly three years, as 64
    patches. With one token a week, this file's 104 weeks take 105 tokens,
    and the same 148 weeks take 149. The extra token comes from the copy of
    the last week added at the end.

    Each model trained for 60 epochs with 3 seeds, on batches of 128 series.
    The mixed model reads whole windows, so its batches held 18 windows,
    which is 126 series. Each number is the mean squared error over the test
    weeks, averaged over the seeds.

    | Weeks ahead | 24 | 36 | 48 | 60 |
    |---|---|---|---|---|
    | PatchTST, 148 weeks in 64 patches | 1.89 | 1.87 | 1.83 | 1.86 |
    | A token a week, 104 weeks | 1.99 | 1.93 | 1.92 | 2.01 |
    | A token a week, 148 weeks | 1.85 | 1.93 | 1.97 | 2.13 |
    | Columns mixed, 148 weeks | 2.52 | 2.04 | 2.04 | 2.55 |

    PatchTST uses 64 tokens instead of 105. At every horizon, its error is
    3% to 7% lower than one token a week over 104 weeks. It has the lower
    error in 10 of the 12 pairs of runs with the same seed and horizon. One
    token a week over the same 148 weeks needs 149 tokens, so each head
    computes 22,201 scores instead of 4,096. That model has a lower error
    than the patches at 24 weeks. At 36, 48 and 60 weeks its error is higher,
    by up to 14%.

    Mixing the columns raises the error at every horizon, by 33% at 24
    weeks, 9% at 36, 11% at 48 and 37% at 60. The mixed model is worse on
    every seed at 24 and 60 weeks. At 36 and 48 weeks, one of its seeds has
    a lower error than PatchTST. The paper also finds mixing worse at every
    horizon on this table, and shows why on its Weather table. There, a
    mixed model's test error starts rising after a few epochs, while a model
    with its columns apart keeps improving.

    `forecast_error` splits each window's columns apart, the way the model
    reads them. Then it scores the model in eval mode without recording for
    autograd.
    """,
)

SERIES3_SOURCE = source_file(
    '''
    # series3.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import ILLNESS_COLUMNS, load_illness

'''
    + indented(GIVEN_PART)
    + '''


    # The patch code from series1 and the column and RevIN code from series2, already finished.
'''
    + indented(PATCH_SOLUTION)
    + "\n\n"
    + indented(APART_SOLUTION)
    + "\n\n"
    + indented(NORMALIZE_SOLUTION)
    + '''


    flat = torch.zeros(2, 42, 16).flatten(1).shape    # torch.Size([2, 672]), each row's tokens in one vector


    # TODO: Finish forward. series has shape (rows, lookback). Normalize it with
    #   normalized, and turn the scaled series into patch tokens with
    #   self.embed. Apply self.dropout to the tokens and run them through
    #   self.blocks. Flatten each row's tokens into one vector, run self.head
    #   on it, and undo the normalization with restored, using the mean and
    #   spread from normalized. The result has shape (rows, horizon).
    class PatchForecaster(nn.Module):
        def __init__(
            self,
            lookback: int,
            horizon: int,
            length: int = 24,
            stride: int = 2,
            size: int = 16,
            blocks: int = 3,
            heads: int = 4,
        ) -> None:
            super().__init__()
            self.embed = PatchEmbedding(lookback, length, stride, size)
            self.dropout = nn.Dropout(0.3)
            self.blocks = nn.Sequential(
                *[
                    nn.TransformerEncoderLayer(
                        size,
                        heads,
                        128,
                        dropout=0.3,
                        batch_first=True,
                    )
                    for _ in range(blocks)
                ],
            )
            self.head = nn.Linear(patch_count(lookback, length, stride) * size, horizon)

        def forward(
            self,
            series: Tensor,
        ) -> Tensor:
            ...


    # TODO: Return the mean squared error between model's forecasts from past
    #   and the true values in future, as a plain Python float. past has shape
    #   (count, lookback, columns) and future has shape
    #   (count, horizon, columns). Split both with channels_apart, since the
    #   model reads one column at a time. Score in eval mode inside
    #   torch.no_grad(), then put the model back in training mode.
    def forecast_error(
        model: nn.Module,
        past: Tensor,
        future: Tensor,
    ) -> float:
        ...
    ''',
)

SERIES3_SOLUTION = source_file(
    '''
    # series3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import ILLNESS_COLUMNS, load_illness

'''
    + indented(GIVEN_PART)
    + '''


    # The patch code from series1 and the column and RevIN code from series2, already finished.
'''
    + indented(PATCH_SOLUTION)
    + "\n\n"
    + indented(APART_SOLUTION)
    + "\n\n"
    + indented(NORMALIZE_SOLUTION)
    + '''


    flat = torch.zeros(2, 42, 16).flatten(1).shape

'''
    + indented(FORECASTER_INIT)
    + "\n"
    + textwrap.indent(FORECASTER_FORWARD.strip("\n"), "    ")
    + "\n\n\n"
    + indented(ERROR_SOLUTION)
    + "\n",
)


def check_patch_forecaster(found: type) -> None:
    for lookback, horizon, length, stride, rows, seed in ((12, 3, 4, 2, 2, 9531), (10, 2, 3, 3, 3, 9532)):
        call = f"PatchForecaster({lookback}, {horizon}, {length}, {stride}, 8, 1, 2)"
        module = expect_module(call, seeded_model(found, lookback, horizon, length, stride, 8, 1, 2))
        reference = seeded_model(REFERENCE["PatchForecaster"], lookback, horizon, length, stride, 8, 1, 2)

        for made in (module, reference):
            with torch.no_grad():
                made.embed.position.copy_(torch.linspace(-1, 1, made.embed.position.numel()).view_as(made.embed.position))

        module.eval()
        reference.eval()
        series = small_tensor(seed, rows, lookback) + 4

        with torch.no_grad():
            actual = module(series)
            expected = reference(series)

        expect_tensor(
            call=f"{call} on {rows} series around 4, in eval mode",
            actual=actual,
            expected=expected,
        )

        # Eval mode turns dropout off, so it can't see a dropout left out or moved. In training mode the same seed
        # before each call gives the same dropout masks only when self.dropout sits where the reference has it.
        module.train()
        reference.train()

        with torch.no_grad():
            torch.manual_seed(seed)
            actual = module(series)
            torch.manual_seed(seed)
            expected = reference(series)

        if not torch.allclose(actual, expected, atol=1e-5):
            raise Mismatch(
                f"{call} matches in eval mode but not in training mode, with the same seed before each call. Apply "
                "self.dropout to the patch tokens from self.embed, before self.blocks.",
            )


class ShiftModel(nn.Module):
    """A stand in model whose forecast is each series' last reading plus a learned step, repeated over the horizon."""

    def __init__(self, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.step = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        series: Tensor,
    ) -> Tensor:
        return (series[:, -1:] + self.step).expand(-1, self.horizon)


def check_forecast_error(found: object) -> None:
    for count, lookback, horizon, columns, seed in ((3, 5, 2, 2, 9533), (2, 4, 3, 3, 9534)):
        past = small_tensor(seed, count, lookback, columns)
        future = small_tensor(seed + 1, count, horizon, columns)
        model = ShiftModel(horizon)
        seen: list[tuple[bool, bool]] = []
        handle = recording_hook(model, seen)

        try:
            actual = found(model, past, future)
        finally:
            handle.remove()

        call = f"forecast_error(a model, {count} windows of {columns} columns)"
        reject_missing_return(
            call=call,
            actual=actual,
        )

        if any(training for training, _ in seen):
            raise Mismatch(f"{call} ran the model in training mode, where dropout is on. Call model.eval() first.")

        if any(recording for _, recording in seen):
            raise Mismatch(f"{call} ran the model while recording for autograd. Score it inside torch.no_grad().")

        if not model.training:
            raise Mismatch(f"{call} left the model in eval mode. Put it back with model.train() at the end.")

        expected = REFERENCE["forecast_error"](ShiftModel(horizon), past, future)

        expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        )


def check_series3(checker: Checker) -> None:
    checker.test_method(
        class_name="PatchForecaster",
        method="forward",
        description="normalizes, embeds the patches, runs the blocks, computes the forecast and restores the level",
        run=check_patch_forecaster,
    )
    checker.test(
        function="forecast_error",
        description="splits the columns apart and scores in eval mode without autograd",
        run=check_forecast_error,
    )


SERIES3 = Exercise(
    name="series3",
    chapter=TIME_SERIES,
    title="PatchTST's two choices",
    lesson=SERIES3_LESSON,
    source=SERIES3_SOURCE,
    solution=SERIES3_SOLUTION,
    hints={
        "PatchForecaster.forward": "normalized(series) returns the scaled series, its mean and its spread. Run "
        "self.embed on the scaled series, then self.dropout, then self.blocks. Call .flatten(1) on the result, run "
        "self.head, and pass its output, the mean and the spread to restored.",
        "forecast_error": "Call model.eval(). Inside torch.no_grad(), take F.mse_loss of model(channels_apart(past)) "
        "and channels_apart(future), then .item(). Call model.train() before returning.",
    },
    introduces=(),
    check=check_series3,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

HORIZONS = (24, 36, 48, 60)


class MixedPatchForecaster(nn.Module):
    """series3's PatchTST with its columns mixed: each token is one patch of every column at once, and the head writes
    every column's forecast. Each column is still normalized on its own, so RevIN is the same."""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        columns: int,
        length: int = 24,
        stride: int = 2,
        size: int = 16,
        blocks: int = 3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.length = length
        self.stride = stride
        self.columns = columns
        self.horizon = horizon
        count = REFERENCE["patch_count"](lookback, length, stride)
        self.project = nn.Linear(length * columns, size)
        self.position = nn.Parameter(torch.randn(count, size) * 0.02)
        self.dropout = nn.Dropout(0.3)
        self.blocks = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    size,
                    heads,
                    128,
                    dropout=0.3,
                    batch_first=True,
                )
                for _ in range(blocks)
            ],
        )
        self.head = nn.Linear(count * size, horizon * columns)

    def forward(
        self,
        windows: Tensor,
    ) -> Tensor:
        rows = windows.shape[0]
        scaled, mean, spread = REFERENCE["normalized"](REFERENCE["channels_apart"](windows))
        cut = REFERENCE["patches"](scaled, self.length, self.stride)
        tokens = cut.reshape(rows, self.columns, cut.shape[1], self.length).permute(0, 2, 1, 3).flatten(2)
        forecasts = self.head(self.blocks(self.dropout(self.project(tokens) + self.position)).flatten(1))
        forecasts = REFERENCE["restored"](forecasts.reshape(rows * self.columns, self.horizon), mean, spread)

        return REFERENCE["channels_together"](forecasts, self.columns)


def mixed_error(
    model: nn.Module,
    past: Tensor,
    future: Tensor,
) -> float:
    """forecast_error for a model that reads whole windows, scored over the same numbers."""
    model.eval()

    with torch.no_grad():
        error = F.mse_loss(model(past), future).item()

    model.train()

    return error


def train_forecaster(
    model: nn.Module,
    series: Tensor,
    horizon: int,
    epochs: int,
    seed: int,
    device: torch.device,
    batch: int = 16,
    columns_apart: bool = True,
) -> float:
    """Adam at PatchTST's rate of 0.0025, on batches of 16 as its flu script uses unless told otherwise, keeping the
    epoch with the lowest validation error, whose test error it gives. With columns_apart False the model reads whole
    windows, as MixedPatchForecaster does."""
    series = series.to(device)
    model = model.to(device)
    past, future = REFERENCE["windows"](series, 0, REFERENCE["TRAIN_END"], horizon)
    validation = REFERENCE["windows"](series, REFERENCE["TRAIN_END"], REFERENCE["VALIDATION_END"], horizon)
    test = REFERENCE["windows"](series, REFERENCE["VALIDATION_END"], len(series), horizon)
    score: Callable[[nn.Module, Tensor, Tensor], float] = REFERENCE["forecast_error"]
    x, y = REFERENCE["channels_apart"](past), REFERENCE["channels_apart"](future)

    if not columns_apart:
        score = mixed_error
        x, y = past, future

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
    generator = torch.Generator().manual_seed(seed)
    best = (math.inf, math.nan)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator).to(device)

        for start in range(0, len(order), batch):
            rows = order[start : start + batch]
            loss = F.mse_loss(model(x[rows]), y[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validation_error = score(model, *validation)

        if validation_error < best[0]:
            best = (validation_error, score(model, *test))

    return best[1]


def measure_series(
    seeds: range = range(3),
    epochs: int = 60,
    device: str = "mps",
    batch: int = 128,
    names: tuple[str, ...] = ("linear", "linear with RevIN", "PatchTST"),
) -> dict[str, list[float]]:
    """series2's table: each model's test error at each horizon, on 3 seeds, on the Mac's GPU for 60 epochs on
    batches of 128."""
    series = flu_series()
    makers: dict[str, Callable[[int], nn.Module]] = {
        "linear": lambda horizon: REFERENCE["LinearForecaster"](REFERENCE["LOOKBACK"], horizon),
        "linear with RevIN": lambda horizon: REFERENCE["NormalizedLinear"](REFERENCE["LOOKBACK"], horizon),
        "PatchTST": lambda horizon: REFERENCE["PatchForecaster"](REFERENCE["LOOKBACK"], horizon),
    }
    results: dict[str, list[float]] = {}

    with pinned_threads():
        for name in names:
            for horizon in HORIZONS:
                errors = []

                for seed in seeds:
                    torch.manual_seed(seed)
                    errors.append(train_forecaster(makers[name](horizon), series, horizon, epochs, seed, torch.device(device), batch))

                results[f"{name}, {horizon}"] = errors

    return results


@contextlib.contextmanager
def looking_back(weeks: int) -> Iterator[None]:
    """windows reads LOOKBACK from the files' given part, so a run at a longer lookback sets it there for the run."""
    before = REFERENCE["LOOKBACK"]
    REFERENCE["LOOKBACK"] = weeks

    try:
        yield
    finally:
        REFERENCE["LOOKBACK"] = before


def measure_choices(
    seeds: range = range(3),
    epochs: int = 60,
    device: str = "mps",
    batch: int = 128,
    lookback: int = 148,
    horizons: tuple[int, ...] = HORIZONS,
    names: tuple[str, ...] = ("PatchTST", "every week a token, 104 weeks", "every week a token", "columns mixed"),
) -> dict[str, list[float]]:
    """series3's table: PatchTST as series3 builds it reading `lookback` weeks, 148 making 64 patches, the same model
    reading every week as its own token (patches of 1 every 1) over series3's own 104 weeks and over the same
    `lookback` weeks, and PatchTST with its columns mixed over `lookback` weeks. Each one's test error at each horizon
    on 3 seeds, on the Mac's GPU for 60 epochs on batches of 128 series, as series2's table was. Every lookback scores
    the same test windows. The mixed model reads whole windows, so its batches hold 128 // 7 = 18 windows: the same 126
    series and the same steps per epoch as the others."""
    series = flu_series()
    columns = series.shape[1]
    given = REFERENCE["LOOKBACK"]
    rows: dict[str, tuple[int, Callable[[int], nn.Module]]] = {
        "PatchTST": (lookback, lambda horizon: REFERENCE["PatchForecaster"](lookback, horizon)),
        "every week a token, 104 weeks": (given, lambda horizon: REFERENCE["PatchForecaster"](given, horizon, 1, 1)),
        "every week a token": (lookback, lambda horizon: REFERENCE["PatchForecaster"](lookback, horizon, 1, 1)),
        "columns mixed": (lookback, lambda horizon: MixedPatchForecaster(lookback, horizon, columns)),
    }
    results: dict[str, list[float]] = {}

    with pinned_threads():
        for horizon in horizons:
            for name in names:
                weeks, make = rows[name]
                errors = []

                for seed in seeds:
                    torch.manual_seed(seed)

                    with looking_back(weeks):
                        errors.append(
                            train_forecaster(
                                make(horizon),
                                series,
                                horizon,
                                epochs,
                                seed,
                                torch.device(device),
                                batch if name != "columns mixed" else batch // columns,
                                columns_apart=name != "columns mixed",
                            ),
                        )

                results[f"{name}, {horizon}"] = errors

    return results


def measure_shift() -> dict[str, tuple[float, float]]:
    """series2's first table: each column's mean over the training weeks and over the test weeks, standardized."""
    series = flu_series()
    names = ("% WEIGHTED ILI", "%UNWEIGHTED ILI", "AGE 0-4", "AGE 5-24", "ILITOTAL", "NUM. OF PROVIDERS", "OT")

    return {
        name: (series[: REFERENCE["TRAIN_END"], column].mean().item(), series[REFERENCE["VALIDATION_END"] :, column].mean().item())
        for column, name in enumerate(names)
    }
