"""01_tensors, making tensors, picking parts of them, math, broadcasting and reductions."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator

import torch
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_shape,
    expect_tensor,
    expect_value,
    reject_missing_return,
    value_text,
)
from ..text import prose, source_file

TENSORS = Chapter(
    folder="01_tensors",
    title="Tensors",
    readme=source_file(
        """
        # Tensors

        Every number a model reads, stores or produces is kept in a tensor.
        This chapter covers making tensors, reading parts of them, math on
        whole tensors, broadcasting and reductions. Every later chapter builds
        its models from these operations.

        ## Further reading

        - [PyTorch, introduction to tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
        - [PyTorch, broadcasting semantics](https://docs.pytorch.org/docs/2.14/notes/broadcasting.html)
        - [Dive into Deep Learning, data manipulation](https://d2l.ai/chapter_preliminaries/ndarray.html)
        """,
    ),
)

TENSORS1_LESSON = prose(
    """
    A **tensor** is a grid of numbers. PyTorch is the Python library that
    `import torch` loads. It keeps every number a model reads, learns and
    outputs in tensors. A **model** is a program that turns input numbers into
    output numbers. **Training** adjusts the numbers inside a model until its
    outputs are close to the right answers.

    `torch.tensor` builds a tensor from a list, or from a list of lists with
    one inner list per row.

    ```python
    import torch

    v = torch.tensor([7, 8, 9])
    m = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    v.shape       # torch.Size([3])
    m.shape       # torch.Size([2, 3])
    m.shape[0]    # 2
    ```

    The **shape** of a tensor gives the size of each **dimension**. `m` has 2
    rows and 3 columns, so its first dimension has size 2 and its second has
    size 3. `v` is a flat list of numbers, so it has a single dimension. This
    course writes shapes in a shorter form, as `(2, 3)` and `(3,)`.

    A tensor can have more than two dimensions. Ten images of 28 by 28 pixels
    fit in one tensor of shape `(10, 28, 28)`.
    """,
)

TENSORS1_SOURCE = source_file(
    '''
    # tensors1.py

    import torch

    # torch.tensor turns a list into a tensor.
    a = torch.tensor([3, 5, 2])            # a has shape (3,)

    # TODO: Turn [10, 20, 30, 40] into a tensor called b, the same way.
    b = ...

    # A list of lists gives a tensor with rows and columns.
    c = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])          # c has shape (2, 3), 2 rows of 3 numbers

    # TODO: Make d, a tensor with 3 rows of 2 numbers each, so its shape is
    #   (3, 2). Any numbers work.
    d = ...

    # TODO: This line crashes. Save the file, read the error in the Result tab,
    #   then fix the line so e still has two rows.
    e = torch.tensor([[1, 2], [3, 4, 5]])
    ''',
)

TENSORS1_SOLUTION = source_file(
    '''
    # tensors1.py, a solution for comparison

    import torch

    a = torch.tensor([3, 5, 2])

    b = torch.tensor([10, 20, 30, 40])

    c = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])

    d = torch.tensor([[1, 2],
                      [3, 4],
                      [5, 6]])

    # Every row of a tensor must have the same length.
    e = torch.tensor([[1, 2], [3, 4]])
    ''',
)


def check_tensors1(checker: Checker) -> None:
    checker.check_name(
        name="b",
        description="is a tensor of 10, 20, 30 and 40",
        run=lambda b: expect_tensor(
            call="b",
            actual=b,
            expected=torch.tensor([10, 20, 30, 40]),
        ),
    )
    checker.check_name(
        name="d",
        description="is a tensor with 3 rows of 2 numbers",
        run=lambda d: expect_shape(
            call="d",
            actual=d,
            shape=(3, 2),
        ),
    )
    checker.check_name(
        name="e",
        description="is a tensor with 2 rows of the same length",
        run=lambda e: expect_shape(
            call="e",
            actual=e,
            shape=(2, None),
        ),
    )


TENSORS1 = Exercise(
    name="tensors1",
    chapter=TENSORS,
    title="Tensors and shapes",
    lesson=TENSORS1_LESSON,
    source=TENSORS1_SOURCE,
    solution=TENSORS1_SOLUTION,
    hints={
        "b": "Pass the list to torch.tensor, as the line that makes a does.",
        "d": "Write a list of 3 inner lists with 2 numbers in each, and pass it to torch.tensor, as the line that "
        "makes c does.",
        "e": "The error says it expected a row of length 2 and got one of length 3. Make both rows the same length.",
    },
    introduces=("model", "training", "tensor", "shape", "dimension"),
    check=check_tensors1,
)

INDEXING1_LESSON = prose(
    """
    Try the first TODO before you read on, with what you know from Python
    lists.

    If you looped over the rows, that works. Square brackets on a tensor can
    do the same thing in one step, and the TODOs after the first one use them.

    ```python
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    ```

    Inside the brackets, give one position per dimension, separated by commas.
    Positions start at 0, and negative positions count back from the end. If
    you give fewer positions than the tensor has dimensions, you get all of
    each remaining dimension. So `x[0]` is the whole of row 0.

    | Code | Result | Picks |
    |---|---|---|
    | `x[0]` | `tensor([1, 2, 3])` | row 0 |
    | `x[-1]` | `tensor([7, 8, 9])` | the last row |
    | `x[1, 2]` | `tensor(6)` | row 1, column 2 |
    | `x[:, 0]` | `tensor([1, 4, 7])` | column 0 |
    | `x[0:2]` | `tensor([[1, 2, 3], [4, 5, 6]])` | rows 0 and 1 |
    | `x[:, 0:2]` | `tensor([[1, 2], [4, 5], [7, 8]])` | columns 0 and 1 |

    A single position is called an **index**. A `:` on its own takes the
    whole dimension, so `x[:, 0]` is column 0 of every row. `start:stop` is a
    **slice**, which takes the positions from `start` up to, but not
    including, `stop`. Leave out `start` to begin at 0, or leave out `stop` to
    go to the end. So `x[1:]` is every row after the first.

    `x[1, 2]` picks a single number, so it returns `tensor(6)`, a tensor with
    no dimensions. Its shape is `()`, an empty list of sizes.

    Models pick parts of tensors this way all the time. When `images` holds
    ten images in a tensor of shape `(10, 28, 28)`, `images[0]` is the first
    image and `images[0:4]` is the first four.
    """,
)

INDEXING1_SOURCE = source_file(
    '''
    # indexing1.py

    import torch
    from torch import Tensor


    # TODO: Try this before reading the lesson, with what you know from Python
    #   lists. grid is a tensor with rows and columns. Return its first column
    #   as a tensor.
    #   Example: for the grid [[1, 2, 3], [4, 5, 6], [7, 8, 9]], first_column
    #   returns tensor([1, 4, 7]).
    def first_column(grid: Tensor) -> Tensor:
        ...


    # The TODOs below use the square brackets the lesson shows. grid can have
    # any number of rows and columns.


    # TODO: Return row 1 of grid.
    #   Example: for the grid above, row_one returns tensor([4, 5, 6]).
    def row_one(grid: Tensor) -> Tensor:
        ...


    # TODO: Return the entry in the last row and the last column of grid.
    #   Example: for the grid above, bottom_right returns tensor(9).
    def bottom_right(grid: Tensor) -> Tensor:
        ...


    # TODO: Return the last two columns of grid, with every row.
    #   Example: for the grid above, last_two_columns returns
    #   tensor([[2, 3], [5, 6], [8, 9]]).
    def last_two_columns(grid: Tensor) -> Tensor:
        ...
    ''',
)

INDEXING1_SOLUTION = source_file(
    '''
    # indexing1.py, a solution for comparison

    import torch
    from torch import Tensor


    # A loop like torch.tensor([row[0] for row in grid]) works too.
    def first_column(grid: Tensor) -> Tensor:
        return grid[:, 0]


    def row_one(grid: Tensor) -> Tensor:
        return grid[1]


    def bottom_right(grid: Tensor) -> Tensor:
        return grid[-1, -1]


    def last_two_columns(grid: Tensor) -> Tensor:
        return grid[:, -2:]
    ''',
)

# Grids of four shapes, so an answer typed in for the grid in the file fails. In the last one the
# bottom right entry is the smallest, so taking the largest entry fails too.
INDEXING_GRIDS: tuple[tuple[Tensor], ...] = (
    (torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),),
    (torch.arange(20).reshape(4, 5),),
    (torch.tensor([[7, -1, 4], [0, 2, 9]]),),
    (torch.tensor([[5, 9], [3, 1]]),),
)


def check_indexing1(checker: Checker) -> None:
    for function, description, reference in (
        ("first_column", "gives column 0", lambda grid: grid[:, 0]),
        ("row_one", "gives row 1", lambda grid: grid[1]),
    ):
        check_on_inputs(
            checker=checker,
            function=function,
            description=description,
            inputs=INDEXING_GRIDS,
            reference=reference,
        )

    # The shape is checked too, so a 1 by 1 slice of the corner fails.
    check_on_inputs(
        checker=checker,
        function="bottom_right",
        description="gives the entry in the last row and column",
        inputs=INDEXING_GRIDS,
        reference=lambda grid: grid[-1, -1],
    )
    check_on_inputs(
        checker=checker,
        function="last_two_columns",
        description="gives the last two columns",
        inputs=INDEXING_GRIDS,
        reference=lambda grid: grid[:, -2:],
    )


INDEXING1 = Exercise(
    name="indexing1",
    chapter=TENSORS,
    title="Picking parts of a tensor",
    lesson=INDEXING1_LESSON,
    source=INDEXING1_SOURCE,
    solution=INDEXING1_SOLUTION,
    hints={
        "first_column": "Each row of grid works like an inner list. Collect the first entry of every row in a list, "
        "then pass the list to torch.tensor.",
        "row_one": "A single index picks a whole row. Counting starts at 0.",
        "bottom_right": "-1 is the last position. Give it for the row and for the column, separated by a comma.",
        "last_two_columns": "Take every row with a colon on its own. For the columns, take a slice that starts 2 "
        "positions from the end.",
    },
    introduces=("index", "slice"),
    check=check_indexing1,
)

MATH1_LESSON = prose(
    """
    Arithmetic between two tensors of the same shape works entry by entry.
    Each entry of the result comes from the two entries at the same position
    in the inputs. This is called **elementwise** math. `+`, `-`, `*`, `/` and
    `**`, which raises to a power, all work this way. Arithmetic between a
    tensor and a plain number applies the number to every entry.

    Python computes `**` first, then `*` and `/`, then `+` and `-`. So
    `2 * a ** 2 + 1` means `(2 * (a ** 2)) + 1`. Add brackets when you want a
    different order.

    ```python
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])
    a + b     # tensor([11., 22., 33.])
    a * 2     # tensor([2., 4., 6.])
    ```

    These numbers are written with a decimal point, as in `1.0`. That makes a
    float tensor, one that holds decimal numbers instead of whole numbers. The
    math inside models runs on floats.

    Comparisons work entry by entry too. Each one gives a tensor of `True` and
    `False`, one per entry.

    ```python
    c = torch.tensor([-2.0, 0.5, -1.0, 4.0])
    c > 1    # tensor([False, False, False,  True])
    ```

    `torch.where(condition, x, y)` builds a new tensor. Each entry comes from
    `x` where the condition is `True`, and from `y` where it is `False`. You
    can pass a plain number in place of `x` or `y`.

    ```python
    torch.where(c > 1, 1.0, c)    # tensor([-2.0000,  0.5000, -1.0000,  1.0000])
    ```

    Models apply operations like these to every number they hold. Replacing
    the negative entries with 0, which the last TODO asks for, is one of the
    most common operations inside a model.
    """,
)

MATH1_SOURCE = source_file(
    '''
    # math1.py

    import torch
    from torch import Tensor

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])

    # + adds two tensors entry by entry.
    added = a + b                               # tensor([11., 22., 33.])


    # TODO: Return a and b multiplied entry by entry. a and b have the same
    #   shape.
    #   Example: multiply(a, b) with the a and b above returns
    #   tensor([10., 40., 90.]).
    def multiply(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        ...


    # Arithmetic with a plain number applies the number to every entry.
    doubled = a * 2                             # tensor([2., 4., 6.])


    # TODO: Return a tensor with every entry of a squared. ** raises to a
    #   power, as in plain Python.
    #   Example: square(a) with the a above returns tensor([1., 4., 9.]).
    def square(a: Tensor) -> Tensor:
        ...


    c = torch.tensor([-2.0, 0.5, 4.0])

    # A comparison gives True or False for every entry.
    above_one = c > 1                           # tensor([False, False,  True])


    # TODO: Return a tensor that holds True for each entry of c below 0, and
    #   False for the other entries.
    #   Example: below_zero(c) with the c above returns
    #   tensor([ True, False, False]).
    def below_zero(c: Tensor) -> Tensor:
        ...


    # torch.where(condition, x, y) takes each entry from x where the condition is
    # True, and from y where it is False.
    capped = torch.where(c > 1, 1.0, c)         # tensor([-2.0000,  0.5000,  1.0000])


    # TODO: Return a new tensor like c, with its negative entries replaced by
    #   0.0 and the other entries unchanged. Leave c itself unchanged.
    #   Example: zero_negatives(c) with the c above returns
    #   tensor([0.0000, 0.5000, 4.0000]).
    def zero_negatives(c: Tensor) -> Tensor:
        ...
    ''',
)

MATH1_SOLUTION = source_file(
    '''
    # math1.py, a solution for comparison

    import torch
    from torch import Tensor

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])

    added = a + b


    def multiply(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return a * b


    doubled = a * 2


    def square(a: Tensor) -> Tensor:
        return a ** 2


    c = torch.tensor([-2.0, 0.5, 4.0])

    above_one = c > 1


    def below_zero(c: Tensor) -> Tensor:
        return c < 0


    capped = torch.where(c > 1, 1.0, c)


    def zero_negatives(c: Tensor) -> Tensor:
        return torch.where(c < 0, 0.0, c)
    ''',
)

# A flat tensor and a grid, with a 0 in each, so a comparison like c <= 0 fails.
MATH_TENSORS: tuple[tuple[Tensor], ...] = (
    (torch.tensor([-2.0, 0.5, -1.0, 4.0, 0.0]),),
    (torch.tensor([[0.0, -0.5, 3.0], [-7.0, 2.0, 1.5]]),),
)


def expect_true_false(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )

    if actual.dtype != torch.bool:
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but a comparison gives True and False. Return the comparison itself.")


def expect_positions(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )

    if actual.dtype.is_floating_point:
        raise Mismatch(
            f"{call} holds {actual.dtype} numbers, but positions are whole numbers, as argmax gives them. "
            "Return argmax's result unchanged.",
        )


def check_leaves_input(found: object) -> None:
    """Each call of a check gets a fresh copy, so a function that writes into its input passes the others."""
    for (c,) in MATH_TENSORS:
        given = c.clone()
        result = found(given)
        reject_missing_return(
            call=f"zero_negatives({value_text(c)})",
            actual=result,
        )

        if not torch.equal(given, c):
            raise Mismatch(
                f"zero_negatives({value_text(c)}) changed the tensor it was given to {value_text(given)}. "
                "Build a new tensor with torch.where, and leave c unchanged.",
            )


def check_math1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="multiply",
        description="multiplies entry by entry",
        inputs=(
            (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([10.0, 20.0, 30.0])),
            (torch.tensor([[2.0, -1.0], [0.5, 3.0]]), torch.tensor([[4.0, 4.0], [-2.0, 0.0]])),
        ),
        reference=lambda a, b: a * b,
    )
    check_on_inputs(
        checker=checker,
        function="square",
        description="squares every entry",
        inputs=MATH_TENSORS,
        reference=lambda c: c ** 2,
    )
    check_on_inputs(
        checker=checker,
        function="below_zero",
        description="gives True for the entries below 0",
        inputs=MATH_TENSORS,
        reference=lambda c: c < 0,
        expect=expect_true_false,
    )
    check_on_inputs(
        checker=checker,
        function="zero_negatives",
        description="replaces the negative entries with 0",
        inputs=MATH_TENSORS,
        reference=lambda c: torch.where(c < 0, 0.0, c),
    )
    checker.test(
        function="zero_negatives",
        description="leaves the tensor it was given unchanged",
        run=check_leaves_input,
    )


MATH1 = Exercise(
    name="math1",
    chapter=TENSORS,
    title="Math on whole tensors",
    lesson=MATH1_LESSON,
    source=MATH1_SOURCE,
    solution=MATH1_SOLUTION,
    hints={
        "multiply": "* works entry by entry, the way + does in the line that sets added.",
        "square": "Raise the whole tensor to the power 2. That squares every entry at once.",
        "below_zero": "Compare c with 0, the way the line that sets above_one compares it with 1.",
        "zero_negatives": "Use torch.where the way the line that sets capped does. The condition is that c is below "
        "0, and where the condition is True the entry should be 0.0.",
    },
    introduces=("elementwise",),
    check=check_math1,
)

BROADCASTING1_LESSON = prose(
    """
    Elementwise math also works between some tensors of different shapes.
    PyTorch compares the two shapes one size at a time, starting from the
    right. Two sizes are compatible when they are equal or when one of them
    is 1. If one shape has fewer dimensions, each missing size counts as 1, so
    `(3,)` acts like `(1, 3)`. Along a dimension of size 1, the same numbers
    are then repeated to match the other size. This is called
    **broadcasting**.

    ```python
    m = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    m + torch.tensor([10.0, 20.0, 30.0])
    # tensor([[11., 22., 33.], [14., 25., 36.]])
    ```

    ```text
    (2, 3) with (3,)      works, the (3,) is repeated for every row
    (2, 3) with (2, 1)    works, the (2, 1) is repeated for every column
    (2, 3) with (2,)      an error, since 3 and 2 differ and neither is 1
    ```

    To combine each row with its own number, the numbers need shape
    `(rows, 1)`. `.unsqueeze(1)` inserts a new dimension of size 1 at
    position 1, which is the second dimension. So a tensor of shape `(2,)`
    becomes shape `(2, 1)`.

    ```python
    r = torch.tensor([1.0, 10.0])
    r.unsqueeze(1)        # tensor([[ 1.], [10.]])
    m * r.unsqueeze(1)    # tensor([[ 1.,  2.,  3.], [40., 50., 60.]])
    ```

    Square tensors have a trap. Say `s` has shape `(3, 3)`, `t` has shape
    `(3,)`, and you want to divide each row of `s` by its own number in `t`.
    `s / t` runs without an error. But `t` lines up with the last dimension,
    the columns, so each column gets divided by its own number instead. The
    result is wrong, and nothing warns you.

    Models use broadcasting all the time. When each row of a tensor holds one
    example, adding a single row of numbers adds the same numbers to every
    example.
    """,
)

BROADCASTING1_SOURCE = source_file(
    '''
    # broadcasting1.py

    import torch
    from torch import Tensor

    grid = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])

    # Shapes (2, 3) and (3,) are lined up from the right, so the row of 3
    # numbers is added to every row of grid.
    plus_row = grid + torch.tensor([10.0, 20.0, 30.0])   # tensor([[11., 22., 33.], [14., 25., 36.]])


    # TODO: Return every row of grid multiplied by row, entry by entry. grid has
    #   shape (rows, columns) and row has shape (columns,).
    #   Example: multiply_rows(grid, torch.tensor([1.0, 0.0, -1.0])) with the
    #   grid above returns tensor([[ 1.,  0., -3.], [ 4.,  0., -6.]]).
    def multiply_rows(
        grid: Tensor,
        row: Tensor,
    ) -> Tensor:
        ...


    # .unsqueeze(1) turns shape (2,) into (2, 1), one number for each row.
    scaled = grid * torch.tensor([1.0, 2.0]).unsqueeze(1)   # tensor([[1., 2., 3.], [8., 10., 12.]])


    # TODO: numbers holds one number per row of grid, so it has shape (rows,).
    #   This function should add each row's own number to every entry of that
    #   row. It crashes when the checker calls it. Read the error in the Result
    #   tab, then fix it.
    #   Example: add_per_row(grid, torch.tensor([100.0, 200.0])) with the grid
    #   above should return tensor([[101., 102., 103.], [204., 205., 206.]]).
    def add_per_row(
        grid: Tensor,
        numbers: Tensor,
    ) -> Tensor:
        return grid + numbers
    ''',
)

BROADCASTING1_SOLUTION = source_file(
    '''
    # broadcasting1.py, a solution for comparison

    import torch
    from torch import Tensor

    grid = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])

    plus_row = grid + torch.tensor([10.0, 20.0, 30.0])


    def multiply_rows(
        grid: Tensor,
        row: Tensor,
    ) -> Tensor:
        return grid * row


    scaled = grid * torch.tensor([1.0, 2.0]).unsqueeze(1)


    # Shape (rows, 1) lines up with the rows, so each row gets its own number.
    def add_per_row(
        grid: Tensor,
        numbers: Tensor,
    ) -> Tensor:
        return grid + numbers.unsqueeze(1)
    ''',
)


def check_broadcasting1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="multiply_rows",
        description="multiplies every row by row",
        inputs=(
            (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([1.0, 0.0, -1.0])),
            (torch.tensor([[2.0, 3.0], [-1.0, 0.5], [4.0, 4.0]]), torch.tensor([10.0, 2.0])),
        ),
        reference=lambda grid, row: grid * row,
    )
    # The square grid runs without an error when the numbers line up with the columns instead.
    check_on_inputs(
        checker=checker,
        function="add_per_row",
        description="adds each row's own number to that row",
        inputs=(
            (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([100.0, 200.0])),
            (torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]), torch.tensor([10.0, 20.0, 30.0])),
        ),
        reference=lambda grid, numbers: grid + numbers.unsqueeze(1),
    )


BROADCASTING1 = Exercise(
    name="broadcasting1",
    chapter=TENSORS,
    title="Broadcasting",
    lesson=BROADCASTING1_LESSON,
    source=BROADCASTING1_SOURCE,
    solution=BROADCASTING1_SOLUTION,
    hints={
        "multiply_rows": "Multiply grid by row, the way the line that sets plus_row adds a row of numbers.",
        "add_per_row": "numbers needs shape (rows, 1) to line up with the rows. Give it that shape the way the "
        "line that sets scaled does.",
    },
    introduces=("broadcasting",),
    check=check_broadcasting1,
)

REDUCTIONS1_LESSON = prose(
    """
    Try the first TODO before you read on, with what you know from Python.

    If you added up the rows in a loop, that works. `.sum()` does the same in
    one step. With no arguments, it adds up every entry. With `dim`, it adds
    along one dimension only, and that dimension disappears from the shape of
    the result.

    ```python
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    x.sum()         # tensor(21.)
    x.sum(dim=0)    # tensor([5., 7., 9.]), one total per column
    x.sum(dim=1)    # tensor([ 6., 15.]), one total per row
    ```

    `.sum()` is also much faster than a loop. On a 1000 by 1000 tensor, a loop
    with Python's `sum` took 0.7 to 1.0 seconds in three runs. `.sum()` took
    about 0.04 milliseconds, which is 18,000 to 25,000 times faster.

    `.mean()` takes `dim` the same way, so `x.mean(dim=0)` is
    `tensor([2.5000, 3.5000, 4.5000])`. `.mean()` only works on float tensors.
    `.argmax()` returns the position of the largest entry. Without `dim`, it
    treats the tensor as one long row of entries. With `dim`, it returns one
    position per row or per column.

    ```python
    x.argmax()         # tensor(5), counted along one long row of 6
    x.argmax(dim=1)    # tensor([2, 2]), the column of the largest entry in each row
    ```

    Operations like these, which combine many entries into fewer, are called
    **reductions**. `keepdim=True` keeps the reduced dimension in the shape,
    with size 1. So `x.sum(dim=1, keepdim=True)` has shape `(2, 1)` instead of
    `(2,)`. As broadcasting1 showed, a tensor of shape `(2, 1)` lines up with
    the rows of `x`, so each row gets combined with its own total.

    Models use reductions all the time. A model often outputs one score per
    possible answer, and argmax picks the answer with the highest score.
    """,
)

REDUCTIONS1_SOURCE = source_file(
    '''
    # reductions1.py

    import torch
    from torch import Tensor


    # TODO: Try this before reading the lesson, with what you know from Python.
    #   Finish the line in the loop, so total returns the sum of every number
    #   in grid.
    #   Example: for the grid [[1., 2., 3.], [4., 5., 6.]], total returns
    #   tensor(21.).
    def total(grid: Tensor) -> Tensor:
        result = 0.0

        for row in grid:
            result = ...

        return result


    # The TODOs below use what the lesson shows.

    grid = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])

    # grid.sum(dim=0) adds down each column, one total per column.
    column_totals = grid.sum(dim=0)              # tensor([5., 7., 9.])


    # TODO: Return the sum of each row of grid instead, one total per row.
    #   Example: for the grid above, row_totals returns tensor([ 6., 15.]).
    def row_totals(grid: Tensor) -> Tensor:
        ...


    # TODO: Return the mean of each column of grid, one mean per column.
    #   Example: for the grid above, column_means returns
    #   tensor([2.5000, 3.5000, 4.5000]), of shape (3,).
    def column_means(grid: Tensor) -> Tensor:
        ...


    # argmax gives the position of the largest entry. With dim=1, it gives one
    # position per row.
    best_in_each_row = grid.argmax(dim=1)        # tensor([2, 2])


    # TODO: Return the position of the largest entry in each column of grid.
    #   Example: for the grid above, best_in_each_column returns
    #   tensor([1, 1, 1]).
    def best_in_each_column(grid: Tensor) -> Tensor:
        ...


    # keepdim=True keeps the summed dimension with size 1, so the totals have
    # shape (2, 1) and line up with the rows of grid.
    kept_totals = grid.sum(dim=1, keepdim=True)  # tensor([[ 6.], [15.]])


    # TODO: Finish the first line, so this divides each row of grid by that
    #   row's total. Then every row of the result adds up to 1.
    #   Example: for the grid above, the first row of the result is
    #   tensor([0.1667, 0.3333, 0.5000]).
    def rows_to_fractions(grid: Tensor) -> Tensor:
        totals = ...

        return grid / totals
    ''',
)

REDUCTIONS1_SOLUTION = source_file(
    '''
    # reductions1.py, a solution for comparison

    import torch
    from torch import Tensor


    # grid.sum() adds up every number in one step, as the lesson shows.
    def total(grid: Tensor) -> Tensor:
        result = 0.0

        for row in grid:
            result = result + row.sum()

        return result


    grid = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])

    column_totals = grid.sum(dim=0)


    def row_totals(grid: Tensor) -> Tensor:
        return grid.sum(dim=1)


    def column_means(grid: Tensor) -> Tensor:
        return grid.mean(dim=0)


    best_in_each_row = grid.argmax(dim=1)


    def best_in_each_column(grid: Tensor) -> Tensor:
        return grid.argmax(dim=0)


    kept_totals = grid.sum(dim=1, keepdim=True)


    # Without keepdim the totals have shape (rows,), which lines up with the columns
    # instead of the rows.
    def rows_to_fractions(grid: Tensor) -> Tensor:
        totals = grid.sum(dim=1, keepdim=True)

        return grid / totals
    ''',
)

# A square grid among them, where adding along the wrong dimension still runs.
# No column has a tie for its largest entry, and every row adds up to more than 0.
REDUCTION_GRIDS: tuple[tuple[Tensor], ...] = (
    (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),),
    (torch.tensor([[2.0, 0.0, 1.0], [1.0, 5.0, 1.0], [0.0, 3.0, 9.0]]),),
    (torch.tensor([[1.0, 7.0], [3.0, 3.0], [8.0, 2.0], [0.0, 4.0]]),),
)


def check_reductions1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="total",
        description="adds up every number",
        inputs=REDUCTION_GRIDS,
        reference=lambda grid: grid.sum(),
        expect=expect_value,
    )

    for function, description, reference, expect in (
        ("row_totals", "gives one total per row", lambda grid: grid.sum(dim=1), expect_tensor),
        ("column_means", "gives one mean per column", lambda grid: grid.mean(dim=0), expect_tensor),
        ("best_in_each_column", "gives the position of each column's largest entry", lambda grid: grid.argmax(dim=0), expect_positions),
        ("rows_to_fractions", "divides each row by its own total", lambda grid: grid / grid.sum(dim=1, keepdim=True), expect_tensor),
    ):
        check_on_inputs(
            checker=checker,
            function=function,
            description=description,
            inputs=REDUCTION_GRIDS,
            reference=reference,
            expect=expect,
        )


REDUCTIONS1 = Exercise(
    name="reductions1",
    chapter=TENSORS,
    title="Reductions",
    lesson=REDUCTIONS1_LESSON,
    source=REDUCTIONS1_SOURCE,
    solution=REDUCTIONS1_SOLUTION,
    hints={
        "total": "Each row works like a list of numbers. Add up the row's numbers with Python's sum, and add that "
        "total to result.",
        "row_totals": "dim=1 is the dimension that runs across each row. Adding along it leaves one total per row.",
        "column_means": ".mean takes dim the same way .sum does in the line that sets column_totals.",
        "best_in_each_column": "Use argmax the way the line that sets best_in_each_row does, but along the other "
        "dimension.",
        "rows_to_fractions": "The totals need shape (rows, 1) to line up with the rows. The lesson shows a .sum "
        "option that keeps the summed dimension with size 1.",
    },
    introduces=("reduction",),
    check=check_reductions1,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """Timings move with the thread count, so every measurement runs on 4."""
    before = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        yield
    finally:
        torch.set_num_threads(before)


def fastest_seconds(
    run: Callable[[], object],
    repeats: int,
) -> float:
    """The quickest of several runs, since other work on the machine only ever slows a run down."""
    times: list[float] = []

    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)

    return min(times)


def measure_loop_against_sum(seeds: tuple[int, ...] = (0, 1, 2)) -> list[tuple[float, float]]:
    """reductions1's timing: a 1000 by 1000 grid added up by the first TODO's loop with Python's sum, against
    .sum(). Per seed, the loop's seconds, then .sum()'s."""
    results: list[tuple[float, float]] = []

    with pinned_threads():
        for seed in seeds:
            grid = torch.randn(1000, 1000, generator=torch.Generator().manual_seed(seed))

            def loop() -> object:
                result = 0.0

                for row in grid:
                    result = result + sum(row)

                return result

            results.append(
                (
                    fastest_seconds(
                        run=loop,
                        repeats=1,
                    ),
                    fastest_seconds(
                        run=grid.sum,
                        repeats=20,
                    ),
                ),
            )

    return results
