"""94_tables, a table read one column at a time: each column its own token, periodic embeddings that let a network see
fine steps in a number, and attention over the column tokens, on California's housing blocks of 1990."""

from __future__ import annotations

import contextlib
import math
import textwrap
from collections.abc import Callable, Iterator
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import load_housing
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    reject_missing_return,
    small_tensor,
    value_text,
)
from ..text import prose, source_file
from .linear_layers import expect_module

TABLES = Chapter(
    folder="94_tables",
    title="Tables",
    readme=source_file(
        """
        # Tables

        A table holds rows of examples and columns of different kinds, each
        on its own scale. This chapter turns every column into a token of
        its own, gives a number column periodic features so a network can
        see fine steps in it, and runs attention over the column tokens. The
        data is California's 20,640 housing blocks of 1990.

        ## Further reading

        - [Gorishniy and colleagues, revisiting deep learning for tabular data](https://arxiv.org/abs/2106.11959)
        - [Gorishniy and colleagues, embeddings for numerical features](https://arxiv.org/abs/2203.05556)
        - [Tancik and colleagues, Fourier features](https://arxiv.org/abs/2006.10739)
        - [Pace and Barry, the California housing data](https://doi.org/10.1016/S0167-7152(96)00140-X)
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


# The split, the targets and the standardizing, given in every file of the chapter.
GIVEN_PART = '''
LOCATION = 2      # the first two number columns, longitude and latitude
PLACES = 5        # the categories of the distance to the ocean


def split_rows(
    count: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    train_end = int(0.64 * count)
    validation_end = int(0.8 * count)

    return order[:train_end], order[train_end:validation_end], order[validation_end:]


def standardized(
    numbers: Tensor,
    rows: Tensor,
) -> Tensor:
    mean = numbers[rows].mean(dim=0)
    spread = numbers[rows].std(dim=0)

    return (numbers - mean) / spread


# The log of each block's value, standardized with the training rows, so an error of 1 is one spread of it.
def log_targets(
    values: Tensor,
    rows: Tensor,
) -> Tensor:
    return standardized(values.log().unsqueeze(1), rows).squeeze(1)
'''

FILL_SOLUTION = '''
def fill_missing(
    numbers: Tensor,
    rows: Tensor,
) -> Tensor:
    medians = numbers[rows].nanmedian(dim=0).values

    return torch.where(numbers.isnan(), medians, numbers)
'''

NUMBER_TOKENS_SOLUTION = '''
def number_tokens(
    numbers: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Tensor:
    return numbers.unsqueeze(-1) * weight + bias
'''

COLUMN_TOKENS_INIT = '''
class ColumnTokens(nn.Module):
    def __init__(
        self,
        columns: int,
        places: int,
        size: int,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(columns, size) / math.sqrt(size))
        self.bias = nn.Parameter(torch.zeros(columns, size))
        self.place = nn.Embedding(places, size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, size))
'''

COLUMN_TOKENS_FORWARD = '''
    def forward(
        self,
        numbers: Tensor,
        places: Tensor,
    ) -> Tensor:
        first = self.class_token.expand(len(numbers), -1, -1)
        place = self.place(places).unsqueeze(1)

        return torch.cat([first, number_tokens(numbers, self.weight, self.bias), place], dim=1)
'''

TOKENS_SOLUTION = NUMBER_TOKENS_SOLUTION + COLUMN_TOKENS_INIT + COLUMN_TOKENS_FORWARD

PERIODIC_SOLUTION = '''
def periodic(
    values: Tensor,
    frequencies: Tensor,
) -> Tensor:
    angles = 2 * math.pi * frequencies * values.unsqueeze(-1)

    return torch.cat([angles.sin(), angles.cos()], dim=-1)
'''

EMBEDDING_INIT = '''
class PeriodicEmbedding(nn.Module):
    def __init__(
        self,
        columns: int,
        frequencies: int,
        spread: float,
        size: int,
    ) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(columns, frequencies) * spread)
        self.weight = nn.Parameter(torch.randn(columns, 2 * frequencies, size) / math.sqrt(2 * frequencies))
        self.bias = nn.Parameter(torch.zeros(columns, size))
'''

EMBEDDING_FORWARD = '''
    def forward(
        self,
        values: Tensor,
    ) -> Tensor:
        waves = periodic(values, self.frequencies)
        mixed = (waves.unsqueeze(-1) * self.weight).sum(dim=-2)

        return torch.relu(mixed + self.bias)
'''

TABLE_MLP_PART = '''
class TableMLP(nn.Module):
    def __init__(
        self,
        numbers: int,
        width: int = 256,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(numbers + PLACES, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        numbers: Tensor,
        places: Tensor,
    ) -> Tensor:
        joined = torch.cat([numbers, F.one_hot(places, PLACES).float()], dim=1)

        return self.layers(joined).squeeze(1)
'''

LOCATION_INIT = '''
class LocationMLP(nn.Module):
    def __init__(
        self,
        numbers: int = 8,
        width: int = 256,
        spread: float = 1.0,
    ) -> None:
        super().__init__()
        self.location = PeriodicEmbedding(LOCATION, 16, spread, 16)
        self.mlp = TableMLP(LOCATION * 16 + numbers - LOCATION, width)
'''

LOCATION_FORWARD = '''
    def forward(
        self,
        numbers: Tensor,
        places: Tensor,
    ) -> Tensor:
        where = self.location(numbers[:, :LOCATION]).flatten(1)

        return self.mlp(torch.cat([where, numbers[:, LOCATION:]], dim=1), places)
'''

EMBEDDINGS_SOLUTION = PERIODIC_SOLUTION + EMBEDDING_INIT + EMBEDDING_FORWARD + TABLE_MLP_PART + LOCATION_INIT + LOCATION_FORWARD

REGLU_PART = '''
class ReGLU(nn.Module):
    def forward(
        self,
        hidden: Tensor,
    ) -> Tensor:
        value, gate = hidden.chunk(2, dim=-1)

        return value * torch.relu(gate)
'''

TRANSFORMER_INIT = '''
class ColumnTransformer(nn.Module):
    def __init__(
        self,
        size: int = 64,
        blocks: int = 3,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.tokens = ColumnTokens(8, PLACES, size)
        self.blocks = nn.ModuleList()
        width = 4 * size // 3

        for _ in range(blocks):
            block = nn.TransformerEncoderLayer(
                size,
                heads,
                2 * width,
                dropout=0.1,
                activation=ReGLU(),
                batch_first=True,
                norm_first=True,
            )
            # ReGLU halves the 2 * width numbers it gets, so the second Linear takes width of them.
            block.linear2 = nn.Linear(width, size)
            # The paper's dropout, 0.2 on attention's weights, 0.1 in the feed forward and none on the residual adds.
            block.self_attn.dropout = 0.2
            block.dropout1 = nn.Identity()
            block.dropout2 = nn.Identity()
            self.blocks.append(block)

        # The first block's tokens come straight from ColumnTokens, so it skips the norm before attention.
        self.blocks[0].norm1 = nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(size),
            nn.ReLU(),
            nn.Linear(size, 1),
        )
'''

TRANSFORMER_FORWARD = '''
    def forward(
        self,
        numbers: Tensor,
        places: Tensor,
    ) -> Tensor:
        missing = numbers.isnan()
        known = numbers.nan_to_num()
        tokens = self.tokens(known, places)
        always = torch.zeros(len(numbers), 1, dtype=torch.bool)
        hidden = torch.cat([always, missing, always], dim=1)

        for block in self.blocks:
            tokens = block(tokens, src_key_padding_mask=hidden)

        return self.head(tokens[:, 0]).squeeze(1)
'''

RMSE_SOLUTION = '''
def rmse(
    model: nn.Module,
    numbers: Tensor,
    places: Tensor,
    targets: Tensor,
) -> float:
    model.eval()

    with torch.no_grad():
        error = F.mse_loss(model(numbers, places), targets).sqrt().item()

    model.train()

    return error
'''

TRANSFORMER_SOLUTION = REGLU_PART + TRANSFORMER_INIT + TRANSFORMER_FORWARD + RMSE_SOLUTION

REFERENCE = namespace_of(GIVEN_PART + FILL_SOLUTION + TOKENS_SOLUTION + EMBEDDINGS_SOLUTION + TRANSFORMER_SOLUTION)


def prepared_housing(seed: int) -> dict[str, Tensor]:
    """The housing table as every file of the chapter prepares it, the split, the filled and standardized numbers,
    the places and the log targets."""
    numbers, places, values = load_housing()
    train_rows, validation_rows, test_rows = REFERENCE["split_rows"](len(values), seed)
    filled = REFERENCE["fill_missing"](numbers, train_rows)

    return {
        "numbers": REFERENCE["standardized"](filled, train_rows),
        "places": places,
        "targets": REFERENCE["log_targets"](values, train_rows),
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }


# ---------------------------------------------------------------------------
# table1
# ---------------------------------------------------------------------------

TABLE1_LESSON = prose(
    """
    Part A of this section takes each kind of input on its own. The course
    met numbers in Section 2, grids in Section 5, sequences in Section 6 and
    text in Section 7. Tables, time series, audio and video come next.

    A table holds rows of examples and columns of different kinds. Here each
    row is one of California's 20,640 housing blocks of 1990. Eight columns
    are numbers, from longitude to median income. One is a category, how far
    the block sits from the ocean. The target is the block's median value.

    A real table has holes. 207 blocks have no bedroom count, so it reads as
    nan. `fill_missing` puts each column's median over the training rows in
    every hole, so the test rows never shape the fill. `nanmedian` skips the
    holes. With an even count it takes the lower of the two middle values,
    where `nanquantile(0.5)` takes their average.

    ```python
    nan = float("nan")
    torch.tensor([[1.0, nan], [nan, 4.0], [3.0, 6.0], [8.0, 5.0]]).nanmedian(dim=0).values    # tensor([3., 5.])
    ```

    An MLP reads a row as one long vector. Gorishniy and colleagues give each
    column a token of its own instead, their **FT-Transformer**. A number
    column scales a learned vector by its value and adds a learned bias. A
    category column picks a row of an embedding table, as embedding1 did.

    ```text
    number token = value * weight[column] + bias[column]
    ```

    Each column gets its own weight, so each can mean something different.
    A class token goes in front, as in vit1, and the model reads its answer
    there. So a row of 8 numbers and 1 category becomes 10 tokens.

    Tokens are what attention reads, so table3 runs attention over them. A
    token per column also lets each column choose its own embedding, which
    table2 uses for the two location columns.
    """,
)

TABLE1_SOURCE = source_file(
    '''
    # table1.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import HOUSING_COLUMNS, HOUSING_PLACES, load_housing

'''
    + indented(GIVEN_PART)
    + '''


    holes = torch.tensor([float("nan"), 2.0]).isnan()    # tensor([ True, False])


    # TODO: Return numbers with every nan replaced by its column's median over
    #   the given rows. numbers has shape (count, columns) and rows holds the
    #   row numbers to take the medians from. Leave every other entry as it is.
    def fill_missing(
        numbers: Tensor,
        rows: Tensor,
    ) -> Tensor:
        ...


    scaled = torch.tensor([[2.0], [3.0]]).unsqueeze(-1) * torch.ones(1, 4)    # shape (2, 1, 4)


    # TODO: Return one token per number. numbers has shape (rows, columns), and
    #   weight and bias have shape (columns, size). Each value scales its
    #   column's weight, then adds its column's bias, giving (rows, columns, size).
    def number_tokens(
        numbers: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. Return the class token, one token per number
    #   column from number_tokens, then the place's token from self.place, in
    #   that order along dimension 1, for a shape of (rows, 1 + columns + 1, size).
    class ColumnTokens(nn.Module):
        def __init__(
            self,
            columns: int,
            places: int,
            size: int,
        ) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(columns, size) / math.sqrt(size))
            self.bias = nn.Parameter(torch.zeros(columns, size))
            self.place = nn.Embedding(places, size)
            self.class_token = nn.Parameter(torch.zeros(1, 1, size))

        def forward(
            self,
            numbers: Tensor,
            places: Tensor,
        ) -> Tensor:
            ...


    numbers, places, values = load_housing()
    train_rows, validation_rows, test_rows = split_rows(len(values), 0)
    prepared = standardized(fill_missing(numbers, train_rows), train_rows)

    torch.manual_seed(0)
    tokens = ColumnTokens(8, PLACES, 32)(prepared[:4], places[:4])
    ''',
)

TABLE1_SOLUTION = source_file(
    '''
    # table1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import HOUSING_COLUMNS, HOUSING_PLACES, load_housing

'''
    + indented(GIVEN_PART)
    + '''


    holes = torch.tensor([float("nan"), 2.0]).isnan()

'''
    + indented(FILL_SOLUTION)
    + '''


    scaled = torch.tensor([[2.0], [3.0]]).unsqueeze(-1) * torch.ones(1, 4)

'''
    + indented(NUMBER_TOKENS_SOLUTION)
    + "\n\n"
    + indented(COLUMN_TOKENS_INIT)
    + "\n\n"
    + textwrap.indent(COLUMN_TOKENS_FORWARD.strip("\n"), "    ")
    + '''


    numbers, places, values = load_housing()
    train_rows, validation_rows, test_rows = split_rows(len(values), 0)
    prepared = standardized(fill_missing(numbers, train_rows), train_rows)

    torch.manual_seed(0)
    tokens = ColumnTokens(8, PLACES, 32)(prepared[:4], places[:4])
    ''',
)


NAN = math.nan


def expect_filled(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    if isinstance(actual, Tensor) and actual.isnan().any():
        raise Mismatch(f"{call} still has nan in it, at {actual.isnan().nonzero().tolist()[:3]}. Every hole should be filled.")

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


class ReferenceColumnTokens(nn.Module):
    def __init__(
        self,
        columns: int,
        places: int,
        size: int,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(columns, size) / math.sqrt(size))
        self.bias = nn.Parameter(torch.zeros(columns, size))
        self.place = nn.Embedding(places, size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, size))

    def forward(
        self,
        numbers: Tensor,
        places: Tensor,
    ) -> Tensor:
        first = self.class_token.expand(len(numbers), -1, -1)
        place = self.place(places).unsqueeze(1)
        numbered = numbers.unsqueeze(-1) * self.weight + self.bias

        return torch.cat([first, numbered, place], dim=1)


def settled_tokens(module: nn.Module) -> nn.Module:
    """A class token of zeros and a bias of zeros would hide a mistake in either, so the check sets both."""
    with torch.no_grad():
        module.class_token.copy_(torch.linspace(-1, 1, module.class_token.numel()).view_as(module.class_token))
        module.bias.copy_(torch.linspace(0.5, -0.5, module.bias.numel()).view_as(module.bias))

    return module


def check_column_tokens(found: type) -> None:
    for columns, places, size, rows, seed in ((3, 4, 5, 2, 9401), (2, 3, 4, 3, 9402)):
        made = expect_module(f"ColumnTokens({columns}, {places}, {size})", seeded_model(found, columns, places, size))
        module = settled_tokens(made)
        reference = settled_tokens(seeded_model(ReferenceColumnTokens, columns, places, size))
        numbers = small_tensor(seed, rows, columns) / 2
        ids = torch.randint(0, places, (rows,), generator=torch.Generator().manual_seed(seed + 1))

        with torch.no_grad():
            actual = module(numbers, ids)
            expected = reference(numbers, ids)

        expect_tensor(
            call=f"ColumnTokens({columns}, {places}, {size}) on {rows} rows",
            actual=actual,
            expected=expected,
        )


def check_tokens(value: object) -> None:
    data = prepared_housing(0)

    torch.manual_seed(0)
    expected = ReferenceColumnTokens(8, 5, 32)(data["numbers"][:4], data["places"][:4])

    expect_tensor(
        call="tokens",
        actual=value,
        expected=expected.detach(),
    )


def check_table1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="fill_missing",
        description="puts each column's median over the given rows in every hole",
        inputs=(
            # Every column holds an odd count of numbers in the given rows, so its median is one middle value and
            # torch's nanmedian, which takes the lower of two, agrees with nanquantile's average of them.
            (
                torch.tensor([[1.0, NAN, 3.0], [6.0, 2.0, 0.0], [4.0, 5.0, NAN], [2.0, -1.0, 1.0], [-3.0, NAN, NAN], [NAN, 9.0, 8.0], [10.0, NAN, 7.0]]),
                torch.tensor([0, 1, 2, 3, 4]),
            ),
            (
                torch.tensor([[NAN, 1.0], [2.0, 6.0], [-1.0, 0.0], [3.0, 2.0], [NAN, -2.0], [5.0, NAN], [7.0, 9.0]]),
                torch.tensor([4, 2, 0, 3, 1]),
            ),
        ),
        reference=REFERENCE["fill_missing"],
        expect=expect_filled,
    )
    check_on_inputs(
        checker=checker,
        function="number_tokens",
        description="scales each column's weight by the value and adds the column's bias",
        inputs=(
            (small_tensor(9405, 2, 3), small_tensor(9406, 3, 4), small_tensor(9407, 3, 4)),
            (small_tensor(9408, 4, 2), small_tensor(9409, 2, 3), small_tensor(9410, 2, 3)),
        ),
        reference=REFERENCE["number_tokens"],
    )
    checker.test_method(
        class_name="ColumnTokens",
        method="forward",
        description="puts the class token, the number tokens and the place token in a row",
        run=check_column_tokens,
    )
    checker.check_name(
        name="tokens",
        description="the first 4 blocks as 10 tokens each",
        run=check_tokens,
    )


TABLE1 = Exercise(
    name="table1",
    chapter=TABLES,
    title="A token per column",
    lesson=TABLE1_LESSON,
    source=TABLE1_SOURCE,
    solution=TABLE1_SOLUTION,
    hints={
        "fill_missing": "numbers[rows].nanmedian(dim=0).values gives each column's median, skipping the holes. "
        "torch.where with numbers.isnan() picks it in the holes and the number everywhere else.",
        "number_tokens": "numbers.unsqueeze(-1) has shape (rows, columns, 1), so times weight it gives one vector per "
        "value. Add bias after.",
        "ColumnTokens.forward": "The class token has shape (1, 1, size). Copy it for every row with .expand(len(numbers), "
        "-1, -1). The place token from self.place(places) needs .unsqueeze(1). torch.cat the three along dim=1.",
        "tokens": "It comes from fill_missing and ColumnTokens. Get those right first.",
    },
    introduces=("FT-Transformer",),
    check=check_table1,
)


# ---------------------------------------------------------------------------
# table2
# ---------------------------------------------------------------------------

TABLE2_LESSON = prose(
    """
    A house's value changes fast across a map. A street by the coast and
    one a mile inland can differ a lot, so the value takes fine steps along
    latitude and longitude.

    An MLP learns smooth functions of its inputs first. Tancik and colleagues
    show a plain MLP fails to learn fine detail in a low dimensional input.
    Feeding it waves of the input fixes that.

    Gorishniy and colleagues give each number column its own waves. A value
    x becomes the sine and cosine of 2π c x for k frequencies c. The
    frequencies are learned, starting from a normal of a chosen spread.

    ```python
    angles = 2 * math.pi * torch.tensor([[0.5, 1.0]]) * 0.25    # frequencies 0.5 and 1 at x = 0.25
    torch.cat([angles.sin(), angles.cos()], dim=-1)          # tensor([[0.7071, 1.0000, 0.7071, 0.0000]])
    ```

    A high frequency flips sign over a small change in x, so the network can
    tell close values apart. Then a Linear per column and a ReLU make each
    column's embedding, the paper's PLR, for periodic, linear, ReLU.

    Here the two location columns get the waves and the other columns stay
    as they are. Each model trained for 40 epochs with AdamW, keeping the
    epoch with the lowest validation error, on 3 splits and seeds.

    | Columns | Plain MLP | Periodic location |
    |---|---|---|
    | longitude and latitude alone | 0.60 to 0.62 | 0.50 to 0.51 |
    | all of them | 0.43 to 0.44 | 0.38 to 0.40 |

    The error is the root mean squared error of the log value, measured in
    spreads of it. An error of 0.4 is a typical miss of about 26% of a
    block's value.

    With location alone the waves cut the error by 17%, on every seed. On the
    whole table they cut it by 9%. The paper reports ensembles of MLP-PLR at
    0.453 on this data, against 0.486 for plain MLPs, in its own units.
    """,
)

TABLE2_SOURCE = source_file(
    '''
    # table2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

'''
    + indented(GIVEN_PART)
    + '''


    angles = 2 * math.pi * torch.tensor([[0.5, 1.0]]) * 0.25
    waves = torch.cat([angles.sin(), angles.cos()], dim=-1)    # tensor([[0.7071, 1.0000, 0.7071, 0.0000]])


    # TODO: Return the periodic features of values. values has shape (rows,
    #   columns) and frequencies (columns, k). Take angles of 2π times each
    #   frequency times its column's value, then the sines and the cosines
    #   side by side, for a shape of (rows, columns, 2k).
    def periodic(
        values: Tensor,
        frequencies: Tensor,
    ) -> Tensor:
        ...


    per_column = (torch.ones(3, 2, 4).unsqueeze(-1) * torch.ones(2, 4, 5)).sum(dim=-2)    # shape (3, 2, 5)


    # TODO: Finish forward. Take periodic(values, self.frequencies), mix each
    #   column's waves with that column's weight, of shape (2k, size), add its
    #   bias and take a ReLU. The result has shape (rows, columns, size).
    class PeriodicEmbedding(nn.Module):
        def __init__(
            self,
            columns: int,
            frequencies: int,
            spread: float,
            size: int,
        ) -> None:
            super().__init__()
            self.frequencies = nn.Parameter(torch.randn(columns, frequencies) * spread)
            self.weight = nn.Parameter(torch.randn(columns, 2 * frequencies, size) / math.sqrt(2 * frequencies))
            self.bias = nn.Parameter(torch.zeros(columns, size))

        def forward(
            self,
            values: Tensor,
        ) -> Tensor:
            ...


    # The plain MLP of the lesson's table, with the place joined as a one hot.
'''
    + indented(TABLE_MLP_PART)
    + '''


    # TODO: Finish forward. Embed the first LOCATION columns of numbers with
    #   self.location and flatten each row's embeddings. Join them with the
    #   other columns and run self.mlp on that and the places.
    class LocationMLP(nn.Module):
        def __init__(
            self,
            numbers: int = 8,
            width: int = 256,
            spread: float = 1.0,
        ) -> None:
            super().__init__()
            self.location = PeriodicEmbedding(LOCATION, 16, spread, 16)
            self.mlp = TableMLP(LOCATION * 16 + numbers - LOCATION, width)

        def forward(
            self,
            numbers: Tensor,
            places: Tensor,
        ) -> Tensor:
            ...
    ''',
)

TABLE2_SOLUTION = source_file(
    '''
    # table2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

'''
    + indented(GIVEN_PART)
    + '''


    angles = 2 * math.pi * torch.tensor([[0.5, 1.0]]) * 0.25
    waves = torch.cat([angles.sin(), angles.cos()], dim=-1)

'''
    + indented(PERIODIC_SOLUTION)
    + '''


    per_column = (torch.ones(3, 2, 4).unsqueeze(-1) * torch.ones(2, 4, 5)).sum(dim=-2)

'''
    + indented(EMBEDDING_INIT)
    + "\n\n"
    + textwrap.indent(EMBEDDING_FORWARD.strip("\n"), "    ")
    + '''


    # The plain MLP of the lesson's table, with the place joined as a one hot.
'''
    + indented(TABLE_MLP_PART)
    + "\n\n\n"
    + indented(LOCATION_INIT)
    + "\n\n"
    + textwrap.indent(LOCATION_FORWARD.strip("\n"), "    ")
    + "\n",
)


class ReferencePeriodicEmbedding(nn.Module):
    def __init__(
        self,
        columns: int,
        frequencies: int,
        spread: float,
        size: int,
    ) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(columns, frequencies) * spread)
        self.weight = nn.Parameter(torch.randn(columns, 2 * frequencies, size) / math.sqrt(2 * frequencies))
        self.bias = nn.Parameter(torch.zeros(columns, size))

    def forward(
        self,
        values: Tensor,
    ) -> Tensor:
        waves = REFERENCE["periodic"](values, self.frequencies)

        return torch.relu(torch.einsum("rcw,cws->rcs", waves, self.weight) + self.bias)


def settled_bias(module: nn.Module) -> nn.Module:
    """A bias of zeros would hide a missing bias, so the check sets one."""
    with torch.no_grad():
        module.bias.copy_(torch.linspace(-0.3, 0.3, module.bias.numel()).view_as(module.bias))

    return module


def check_periodic_embedding(found: type) -> None:
    for columns, frequencies, size, rows, seed in ((2, 3, 4, 3, 9411), (3, 2, 5, 2, 9412)):
        made = expect_module(
            f"PeriodicEmbedding({columns}, {frequencies}, 1.0, {size})",
            seeded_model(found, columns, frequencies, 1.0, size),
        )
        module = settled_bias(made)
        reference = settled_bias(seeded_model(ReferencePeriodicEmbedding, columns, frequencies, 1.0, size))
        values = small_tensor(seed, rows, columns) / 4

        with torch.no_grad():
            actual = module(values)
            expected = reference(values)

        expect_tensor(
            call=f"PeriodicEmbedding({columns}, {frequencies}, 1.0, {size}) on {rows} rows",
            actual=actual,
            expected=expected,
        )


class ReferenceLocationMLP(nn.Module):
    def __init__(
        self,
        numbers: int = 8,
        width: int = 256,
        spread: float = 1.0,
    ) -> None:
        super().__init__()
        self.location = ReferencePeriodicEmbedding(2, 16, spread, 16)
        self.mlp = REFERENCE["TableMLP"](2 * 16 + numbers - 2, width)

    def forward(
        self,
        numbers: Tensor,
        places: Tensor,
    ) -> Tensor:
        where = self.location(numbers[:, :2]).flatten(1)

        return self.mlp(torch.cat([where, numbers[:, 2:]], dim=1), places)


def check_location_mlp(found: type) -> None:
    for numbers_count, width, rows, seed in ((8, 16, 3, 9413), (5, 8, 4, 9414)):
        module = expect_module(f"LocationMLP({numbers_count}, {width})", seeded_model(found, numbers_count, width))
        reference = seeded_model(ReferenceLocationMLP, numbers_count, width)
        numbers = small_tensor(seed, rows, numbers_count) / 3
        places = torch.randint(0, 5, (rows,), generator=torch.Generator().manual_seed(seed + 1))

        with torch.no_grad():
            actual = module(numbers, places)
            expected = reference(numbers, places)

        expect_tensor(
            call=f"LocationMLP({numbers_count}, {width}) on {rows} rows of {numbers_count} numbers",
            actual=actual,
            expected=expected,
        )


def check_table2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="periodic",
        description="gives the sines and cosines of 2π times each frequency times the value",
        inputs=(
            (small_tensor(9415, 2, 3) / 4, small_tensor(9416, 3, 2) / 3),
            (small_tensor(9417, 3, 1) / 4, small_tensor(9418, 1, 4) / 3),
        ),
        reference=REFERENCE["periodic"],
    )
    checker.test_method(
        class_name="PeriodicEmbedding",
        method="forward",
        description="mixes each column's waves with its own weight, adds its bias and takes a ReLU",
        run=check_periodic_embedding,
    )
    checker.test_method(
        class_name="LocationMLP",
        method="forward",
        description="embeds the location columns, joins the rest and runs the MLP",
        run=check_location_mlp,
    )


TABLE2 = Exercise(
    name="table2",
    chapter=TABLES,
    title="Number embeddings",
    lesson=TABLE2_LESSON,
    source=TABLE2_SOURCE,
    solution=TABLE2_SOLUTION,
    hints={
        "periodic": "values.unsqueeze(-1) has shape (rows, columns, 1), so times frequencies it gives every column's "
        "k frequencies. Multiply by 2 * math.pi, then torch.cat the .sin() and the .cos() along the last dim.",
        "PeriodicEmbedding.forward": "The waves have shape (rows, columns, 2k). waves.unsqueeze(-1) times self.weight "
        "has shape (rows, columns, 2k, size). Sum over dim=-2, add self.bias, then torch.relu.",
        "LocationMLP.forward": "self.location(numbers[:, :LOCATION]) has shape (rows, 2, 16). .flatten(1) makes it "
        "(rows, 32). torch.cat it with numbers[:, LOCATION:] along dim=1 and pass that and places to self.mlp.",
    },
    introduces=(),
    check=check_table2,
)


# ---------------------------------------------------------------------------
# table3
# ---------------------------------------------------------------------------

TABLE3_LESSON = prose(
    """
    table1's tokens meet attention here. ColumnTokens turns a row's 8
    numbers and its place into 10 tokens, the class token first. Then
    attention reads them with PyTorch's `nn.TransformerEncoderLayer`, as vit2
    read its patches. No position table is needed, since each column's own
    weight marks where a token came from. The class token's output goes
    through a LayerNorm, a ReLU and a Linear, the head Gorishniy and
    colleagues use.

    Their paper puts this FT-Transformer at 0.459 on California's housing
    against 0.499 for an MLP, in its own units. It wins with the paper's
    setup, not with table2's plain one.

    - quantile transform: each number becomes its share of the training
      values below it, then the normal's value with that share below it.
      Long tails like the rooms' spread out.
    - size: tokens of 64 numbers through 3 blocks of 8 heads. The first
      block skips its norm, since its tokens come straight from ColumnTokens.
    - ReGLU: llama2's SwiGLU with a ReLU in place of the SiLU, as the feed
      forward.
    - dropout: 0.2 on attention's weights, 0.1 in the feed forward and none
      on the residual adds.
    - training: until 16 epochs pass with no lower validation error, with
      attention at a rate of 3e-4 and the MLP at its better 1e-3.

    ```python
    torch.special.ndtri(torch.tensor([0.1, 0.5, 0.9]))    # tensor([-1.2816,  0.0000,  1.2816])
    ```

    Both read the same inputs, on 3 splits and seeds, scored as in table2.
    Attention wins on every seed, by 0.009 to 0.011.

    | Model | Test error |
    |---|---|
    | MLP | 0.418 to 0.425 |
    | attention over the column tokens | 0.407 to 0.415 |

    A token per column also lets a row have holes. An MLP has an input for
    every column, so a hole needs a number, and table1 put the median there.
    Attention can leave the hole's token out. `src_key_padding_mask` hides
    tokens from attention, True where a token is hidden, as mask1's padding
    masks were. The hole still needs a 0 for its nan, which would spoil
    attention's sums.

    ```python
    torch.tensor([[1.0, float("nan")]]).nan_to_num()    # tensor([[1., 0.]])
    ```

    Both trained again with a tenth of the numbers dropped, as fuse3 will do
    with whole inputs. So the MLP learns that a median may mark a hole. Then
    a column went missing from every test row.

    | Missing from the test rows | MLP, median in the hole | Attention, token left out |
    |---|---|---|
    | nothing | 0.407 to 0.426 | 0.413 to 0.427 |
    | longitude and latitude | 0.544 to 0.567 | 0.538 to 0.555 |
    | total rooms | 0.453 to 0.467 | 0.436 to 0.455 |
    | population | 0.483 to 0.505 | 0.467 to 0.485 |
    | households | 0.429 to 0.444 | 0.419 to 0.435 |

    Trained with holes, the MLP does as well on complete rows, so column
    tokens pay off when columns go missing. Rooms, bedrooms, households and
    population each tell of the others. With any of them missing, or the
    location, attention wins on every seed. No column tells of income, and
    there the two tie.

    `rmse` scores in eval mode without autograd, then switches back, as
    gpt3's validation_loss did.
    """,
)

TABLE3_SOURCE = source_file(
    '''
    # table3.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

'''
    + indented(GIVEN_PART)
    + '''


    # table1's tokens, done.
'''
    + indented(TOKENS_SOLUTION)
    + '''


    first = torch.tensor([[[1.0], [2.0]]])[:, 0]                  # tensor([[1.]]), the first token of each row
    holes = torch.tensor([[1.0, float("nan")]]).isnan()           # tensor([[False,  True]])
    zeroed = torch.tensor([[1.0, float("nan")]]).nan_to_num()     # tensor([[1., 0.]])
    mask = torch.cat([torch.zeros(1, 1, dtype=torch.bool), holes], dim=1)      # tensor([[False, False,  True]])
    layer = nn.TransformerEncoderLayer(4, 2, 8, batch_first=True)
    shape = layer(torch.ones(1, 3, 4), src_key_padding_mask=mask).shape      # torch.Size([1, 3, 4])


    # The paper's feed forward, llama2's SwiGLU with a ReLU in place of the SiLU.
'''
    + indented(REGLU_PART)
    + '''


    # TODO: Finish forward. A nan in numbers is a missing number. Note where
    #   they are, then turn them into 0 so no nan reaches a token. Make the
    #   tokens with self.tokens, which puts the class token first, then one
    #   per number, then the place. Run each block in self.blocks with
    #   src_key_padding_mask, True at the token of every missing number and
    #   False at the class and place tokens. Return self.head of the class
    #   token's output, one answer per row, of shape (rows,).
'''
    + indented(TRANSFORMER_INIT)
    + '''

        def forward(
            self,
            numbers: Tensor,
            places: Tensor,
        ) -> Tensor:
            ...


    # TODO: Return the root mean squared error of model's answers against
    #   targets, as a plain number. Put the model in eval mode, score it
    #   without recording for autograd, then put it back in training mode.
    def rmse(
        model: nn.Module,
        numbers: Tensor,
        places: Tensor,
        targets: Tensor,
    ) -> float:
        ...
    ''',
)

TABLE3_SOLUTION = source_file(
    '''
    # table3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

'''
    + indented(GIVEN_PART)
    + '''


    # table1's tokens, done.
'''
    + indented(TOKENS_SOLUTION)
    + '''


    first = torch.tensor([[[1.0], [2.0]]])[:, 0]
    holes = torch.tensor([[1.0, float("nan")]]).isnan()
    zeroed = torch.tensor([[1.0, float("nan")]]).nan_to_num()
    mask = torch.cat([torch.zeros(1, 1, dtype=torch.bool), holes], dim=1)
    layer = nn.TransformerEncoderLayer(4, 2, 8, batch_first=True)
    shape = layer(torch.ones(1, 3, 4), src_key_padding_mask=mask).shape


    # The paper's feed forward, llama2's SwiGLU with a ReLU in place of the SiLU.
'''
    + indented(REGLU_PART)
    + '''


'''
    + indented(TRANSFORMER_INIT)
    + "\n\n"
    + textwrap.indent(TRANSFORMER_FORWARD.strip("\n"), "    ")
    + "\n\n\n"
    + indented(RMSE_SOLUTION)
    + "\n",
)


def check_column_transformer(found: type) -> None:
    for size, blocks, rows, seed, holes in (
        (8, 1, 3, 9421, ()),
        (12, 2, 2, 9422, ()),
        (8, 2, 4, 9423, ((0, 7), (1, 0), (2, 3), (2, 5), (3, 1))),
    ):
        module = expect_module(f"ColumnTransformer({size}, {blocks}, 2)", seeded_model(found, size, blocks, 2))
        reference = seeded_model(REFERENCE["ColumnTransformer"], size, blocks, 2)

        # The class token starts at zeros, so the check moves it to show which token the head reads.
        for made in (module, reference):
            with torch.no_grad():
                made.tokens.class_token.copy_(torch.linspace(-1, 1, size).view(1, 1, size))

        module.eval()
        reference.eval()
        numbers = small_tensor(seed, rows, 8) / 3
        places = torch.randint(0, 5, (rows,), generator=torch.Generator().manual_seed(seed + 1))

        for row, column in holes:
            numbers[row, column] = NAN

        masks: list[object] = []
        handle = module.blocks[0].register_forward_pre_hook(
            lambda block, arguments, keywords: masks.append(keywords.get("src_key_padding_mask")),
            with_kwargs=True,
        )

        try:
            with torch.no_grad():
                actual = module(numbers, places)
                expected = reference(numbers, places)
        finally:
            handle.remove()

        call = f"ColumnTransformer({size}, {blocks}, 2) on {rows} rows, in eval mode"

        if holes:
            call += f", with nan at (row, column) {', '.join(map(str, holes))}"

        first_mask = masks[0] if masks else None

        if isinstance(first_mask, Tensor) and first_mask.dim() == 2 and first_mask[:, 0].any():
            raise Mismatch(
                f"{call} hides the class token, whose output the head reads. The mask should be True only at the "
                "tokens of missing numbers, and False at the class and place tokens.",
            )

        if isinstance(actual, Tensor) and actual.isnan().any():
            raise Mismatch(
                f"{call} gives nan. A hidden token still meets the others in attention's weighted sum, and 0 "
                "times nan is nan, so turn the missing numbers into 0 before making any token.",
            )

        expect_tensor(
            call=call,
            actual=actual,
            expected=expected,
        )


class ScoredModel(nn.Module):
    """A stand in model whose answer is the sum of each row's numbers plus a number per place."""

    def __init__(self) -> None:
        super().__init__()
        self.place = nn.Parameter(torch.tensor([0.5, -0.5, 1.0, 0.0, -1.0]))

    def forward(
        self,
        numbers: Tensor,
        places: Tensor,
    ) -> Tensor:
        return numbers.sum(dim=1) + self.place[places]


def check_rmse(found: object) -> None:
    for rows, seed in ((5, 9423), (7, 9424)):
        numbers = small_tensor(seed, rows, 3) / 2
        places = torch.randint(0, 5, (rows,), generator=torch.Generator().manual_seed(seed + 1))
        targets = small_tensor(seed + 2, rows) / 2
        model = ScoredModel()
        seen: list[tuple[bool, bool]] = []
        handle = recording_hook(model, seen)

        try:
            actual = found(model, numbers, places, targets)
        finally:
            handle.remove()

        call = f"rmse(a model, {rows} rows)"
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

        with torch.no_grad():
            expected = F.mse_loss(ScoredModel()(numbers, places), targets).sqrt().item()

        expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        )


def check_table3(checker: Checker) -> None:
    checker.test_method(
        class_name="ColumnTransformer",
        method="forward",
        description="runs attention over the class, column and place tokens and reads the class token",
        run=check_column_transformer,
    )
    checker.test(
        function="rmse",
        description="scores in eval mode without autograd, then switches back",
        run=check_rmse,
    )


TABLE3 = Exercise(
    name="table3",
    chapter=TABLES,
    title="Attention over the columns",
    lesson=TABLE3_LESSON,
    source=TABLE3_SOURCE,
    solution=TABLE3_SOLUTION,
    hints={
        "ColumnTransformer.forward": "Keep numbers.isnan() and give self.tokens the numbers with nan_to_num() and the "
        "places. The tokens run class, the 8 numbers in order, then place. So the mask is a column of False, the isnan "
        "columns, and a column of False, joined along dim=1. Loop over self.blocks passing src_key_padding_mask=, then "
        "keep [:, 0], run self.head and .squeeze(1).",
        "rmse": "Call model.eval(). Inside torch.no_grad(), take F.mse_loss of the model's answers and targets, then "
        ".sqrt().item(). Call model.train() before returning.",
    },
    introduces=(),
    check=check_table3,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


# table3's second table: the columns left out of every test row, by HOUSING_COLUMNS' order.
MISSING_AT_TEST: dict[str, tuple[int, ...]] = {
    "nothing": (),
    "longitude and latitude": (0, 1),
    "longitude": (0,),
    "latitude": (1,),
    "housing median age": (2,),
    "total rooms": (3,),
    "total bedrooms": (4,),
    "population": (5,),
    "households": (6,),
    "median income": (7,),
}


class MedianFilled(nn.Module):
    """An MLP that reads a missing number, a nan, as its column's median over the training rows, table1's fill."""

    def __init__(
        self,
        inner: nn.Module,
        data: dict[str, Tensor],
    ) -> None:
        super().__init__()
        self.inner = inner
        self.register_buffer("medians", data["numbers"][data["train"]].nanmedian(dim=0).values)

    def forward(
        self,
        numbers: Tensor,
        places: Tensor,
    ) -> Tensor:
        return self.inner(torch.where(numbers.isnan(), self.medians, numbers), places)


def quantile_normal(
    numbers: Tensor,
    rows: Tensor,
) -> Tensor:
    """The quantile transform Gorishniy and colleagues give every model's inputs. Each number becomes its share of the
    training rows' values below it, ties counting half, sent through the inverse of the normal's cumulative
    distribution. A heavy tail like the rooms or the population then spreads as evenly as any other column."""
    ordered = numbers[rows].sort(dim=0).values.T.contiguous()
    columns = numbers.T.contiguous()
    below = torch.searchsorted(ordered, columns, right=False)
    up_to = torch.searchsorted(ordered, columns, right=True)
    share = ((below + up_to).double() / (2 * len(rows))).clamp(0.5 / len(rows), 1 - 0.5 / len(rows))

    return torch.special.ndtri(share).float().T


def quantile_housing(seed: int) -> dict[str, Tensor]:
    """prepared_housing with the quantile transform in place of the standardizing, table3's inputs for every model."""
    data = prepared_housing(seed)
    filled = REFERENCE["fill_missing"](load_housing()[0], data["train"])

    return {**data, "numbers": quantile_normal(filled, data["train"])}


def train_table_model(
    model: nn.Module,
    data: dict[str, Tensor],
    seed: int,
    drop: float = 0.0,
    missing_at_test: bool = False,
    rate: float = 1e-3,
    patience: int | None = None,
) -> dict[str, float]:
    """table2's budget: AdamW at rate with decay 1e-4 and batches of 256 for 40 epochs, or with patience until that
    many epochs pass with no lower validation error, keeping the epoch with the lowest. Gives that epoch's test error
    with nothing missing, and with missing_at_test with each of MISSING_AT_TEST's columns turned to nan in every test
    row. drop turns each training number to nan at that rate."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    holes = torch.Generator().manual_seed(1000 + seed)
    numbers, places, targets = data["numbers"], data["places"], data["targets"]
    test = data["test"]
    best = math.inf
    epochs_since_best = 0
    scores: dict[str, float] = {}

    for _ in range(40 if patience is None else 1000):
        model.train()
        order = data["train"][torch.randperm(len(data["train"]), generator=generator)]

        for start in range(0, len(order), 256):
            rows = order[start : start + 256]
            seen = numbers[rows]

            if drop > 0:
                seen = torch.where(torch.rand(seen.shape, generator=holes) < drop, math.nan, seen)

            loss = F.mse_loss(model(seen, places[rows]), targets[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validation = REFERENCE["rmse"](model, numbers[data["validation"]], places[data["validation"]], targets[data["validation"]])

        if validation >= best:
            epochs_since_best += 1

            if epochs_since_best == patience:
                break

            continue

        best = validation
        epochs_since_best = 0
        tests = MISSING_AT_TEST if missing_at_test else {"nothing": ()}

        for name, columns in tests.items():
            left_out = numbers[test].clone()
            left_out[:, list(columns)] = math.nan
            scores[name] = REFERENCE["rmse"](model, left_out, places[test], targets[test])

    return scores


def location_only(data: dict[str, Tensor]) -> dict[str, Tensor]:
    """The table with longitude and latitude alone and every place set to the same one."""
    return {**data, "numbers": data["numbers"][:, :2], "places": torch.zeros_like(data["places"])}


class TableRun(NamedTuple):
    make: Callable[[dict[str, Tensor]], nn.Module]
    inputs: Callable[[int], dict[str, Tensor]]
    rate: float
    patience: int | None
    drops: tuple[float, ...]
    missing_at_test: bool


# table2's runs keep its standardized inputs and 40 epochs. table3's read the quantile transform and stop at the
# paper's patience of 16, each model at the rate that suits it, since the MLP does worse at attention's.
TABLE_RUNS: dict[str, TableRun] = {
    "location, plain MLP": TableRun(
        make=lambda data: REFERENCE["TableMLP"](2),
        inputs=lambda seed: location_only(prepared_housing(seed)),
        rate=1e-3,
        patience=None,
        drops=(0.0,),
        missing_at_test=False,
    ),
    "location, periodic": TableRun(
        make=lambda data: REFERENCE["LocationMLP"](2),
        inputs=lambda seed: location_only(prepared_housing(seed)),
        rate=1e-3,
        patience=None,
        drops=(0.0,),
        missing_at_test=False,
    ),
    "all, plain MLP": TableRun(
        make=lambda data: MedianFilled(REFERENCE["TableMLP"](8), data),
        inputs=prepared_housing,
        rate=1e-3,
        patience=None,
        drops=(0.0,),
        missing_at_test=False,
    ),
    "all, periodic location": TableRun(
        make=lambda data: MedianFilled(REFERENCE["LocationMLP"](8), data),
        inputs=prepared_housing,
        rate=1e-3,
        patience=None,
        drops=(0.0,),
        missing_at_test=False,
    ),
    "quantile, plain MLP": TableRun(
        make=lambda data: MedianFilled(REFERENCE["TableMLP"](8), data),
        inputs=quantile_housing,
        rate=1e-3,
        patience=16,
        drops=(0.0, 0.1),
        missing_at_test=True,
    ),
    "quantile, column transformer": TableRun(
        make=lambda data: REFERENCE["ColumnTransformer"](),
        inputs=quantile_housing,
        rate=3e-4,
        patience=16,
        drops=(0.0, 0.1),
        missing_at_test=True,
    ),
}


def measured_run(
    name: str,
    seed: int,
    drop: float = 0.0,
) -> tuple[dict[str, float], int]:
    """One of TABLE_RUNS on one split and seed: its test errors and its weight count."""
    run = TABLE_RUNS[name]
    data = run.inputs(seed)
    torch.manual_seed(seed)
    model = run.make(data)

    with pinned_threads():
        scores = train_table_model(
            model=model,
            data=data,
            seed=seed,
            drop=drop,
            missing_at_test=run.missing_at_test,
            rate=run.rate,
            patience=run.patience,
        )

    return scores, sum(parameter.numel() for parameter in model.parameters())


def measure_tables(seeds: range = range(3)) -> dict[str, object]:
    """table2's and table3's tables: the test error of each of TABLE_RUNS on 3 splits and seeds, and their weight
    counts. table3's runs are also scored with columns missing from every test row, the MLPs reading table1's median in
    the hole and attention leaving the token out, and trained again with a tenth of the numbers dropped."""
    results: dict[str, object] = {}

    for name, run in TABLE_RUNS.items():
        for drop in run.drops:
            runs = [measured_run(name, seed, drop) for seed in seeds]
            label = f"{name}, trained with {drop:g} dropped" if drop else name
            results[label] = {test: [scores[test] for scores, _ in runs] for test in runs[0][0]}
            results[f"{name} weights"] = runs[0][1]

    results["log value spread"] = load_housing()[2].log().std().item()

    return results
