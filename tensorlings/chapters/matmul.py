"""02_matmul, weighted sums and matrix multiplication, and quiz1, which closes the first two chapters."""

from __future__ import annotations

import torch
from torch import Tensor

from ..course import QUIZZES, Chapter, Exercise
from ..engine import Checker, check_on_inputs, expect_value
from ..text import prose, source_file
from .tensors import expect_positions

MATMUL = Chapter(
    folder="02_matmul",
    title="Matrix multiplication",
    readme=source_file(
        """
        # Matrix multiplication

        A weighted sum multiplies each input by how much it counts and adds the
        results. Matrix multiplication does many weighted sums in one call, and
        most of the work in deep learning is matrix multiplication.

        ## Further reading

        - [Dive into Deep Learning, linear algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html)
        - [3Blue1Brown, the essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra)
        """,
    ),
)

MATMUL1_LESSON = prose(
    """
    Try the first TODO before you read on, with what you already know.

    A **weighted sum** multiplies each value by its own **weight** and adds the
    results. With tensors that is an elementwise multiply and a sum, as you may
    have written. The **dot product**, written `@` between two flat tensors,
    does both in one step.

    ```python
    v = torch.tensor([2.0, 3.0])
    w = torch.tensor([10.0, 100.0])
    (v * w).sum()    # tensor(320.)
    v @ w            # tensor(320.)
    ```

    A tensor with two dimensions is a **matrix**. With a matrix on the left and
    a flat tensor on the right, `@` gives one dot product per row.

    With a matrix on each side, `@` takes the dot product of every row on the
    left with every column on the right. This is **matrix multiplication**.
    `(rows, k) @ (k, columns)` gives `(rows, columns)`, and the two middle
    sizes, both `k`, must match.

    ```python
    a = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    b = torch.tensor([[1.0, 0.0, 2.0],
                      [0.0, 10.0, 1.0]])
    a @ b                           # tensor([[ 1., 20.,  4.], [ 3., 40., 10.]])
    a @ torch.tensor([1.0, 1.0])    # tensor([3., 7.])
    ```

    `.T` swaps the two dimensions of a matrix, so its rows become columns and
    `b.T` has shape `(3, 2)`. It is for matrices only, and PyTorch warns when
    it meets a flat tensor.

    Most of the work inside a model is matrix multiplication. A model's inputs
    often form a matrix with one example per row. One `@` with a matrix of
    weights then gives every example many weighted sums at once.
    """,
)

MATMUL1_SOURCE = source_file(
    '''
    # matmul1.py

    import torch
    from torch import Tensor


    # TODO: Try this before reading the lesson, with what you already know.
    #   Multiply each entry of v by the matching entry of w, then add up the
    #   results. For v = [2., 3.] and w = [10., 100.] that is tensor(320.).
    def weighted_sum(
        v: Tensor,
        w: Tensor,
    ) -> Tensor:
        ...


    # The TODOs below use what the lesson shows.

    v = torch.tensor([2.0, 3.0])
    w = torch.tensor([10.0, 100.0])

    # v @ w takes the dot product of two flat tensors in one step.
    dot = v @ w                                   # tensor(320.)


    # TODO: Return the dot product of every row of matrix with w, one number per
    #   row.
    def row_dots(
        matrix: Tensor,
        w: Tensor,
    ) -> Tensor:
        ...


    a = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    b = torch.tensor([[1.0, 0.0, 2.0],
                      [0.0, 10.0, 1.0]])

    # A matrix @ a matrix pairs every row on the left with every column on the
    # right, so (2, 2) @ (2, 3) gives (2, 3).
    product = a @ b                               # tensor([[1., 20., 4.], [3., 40., 10.]])

    # .T swaps rows and columns, so b.T has shape (3, 2).
    b_turned = b.T                                # tensor([[ 1.,  0.], [ 0., 10.], [ 2.,  1.]])


    # TODO: This crashes when the checker calls it. It should return the dot
    #   product of every row of a with every row of b, one row per row of a,
    #   holding one number per row of b. Read the error in the Result tab, then
    #   fix it.
    def row_pairs(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return a @ b


    # TODO: Finish the first line, so scores holds the dot product of every row
    #   of a with every row of b, the way row_pairs gives it. The second line
    #   then picks, for each row of a, the position of the row of b it scores
    #   highest with.
    def best_match(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        scores = ...

        return scores.argmax(dim=1)
    ''',
)

MATMUL1_SOLUTION = source_file(
    '''
    # matmul1.py, a solution for comparison

    import torch
    from torch import Tensor


    # (v * w).sum() works too.
    def weighted_sum(
        v: Tensor,
        w: Tensor,
    ) -> Tensor:
        return v @ w


    v = torch.tensor([2.0, 3.0])
    w = torch.tensor([10.0, 100.0])

    dot = v @ w


    def row_dots(
        matrix: Tensor,
        w: Tensor,
    ) -> Tensor:
        return matrix @ w


    a = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    b = torch.tensor([[1.0, 0.0, 2.0],
                      [0.0, 10.0, 1.0]])

    product = a @ b

    b_turned = b.T


    # b.T turns the rows of b into columns, so (n, size) @ (size, m) gives (n, m).
    def row_pairs(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return a @ b.T


    def best_match(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        scores = a @ b.T

        return scores.argmax(dim=1)
    ''',
)


def check_matmul1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="weighted_sum",
        description="multiplies matching entries and adds them up",
        inputs=(
            (torch.tensor([2.0, 3.0]), torch.tensor([10.0, 100.0])),
            (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])),
            (torch.tensor([0.5, -1.0]), torch.tensor([4.0, 2.0])),
        ),
        reference=lambda v, w: (v * w).sum(),
        expect=expect_value,
    )
    # A square matrix among them, where w @ matrix runs but gives other numbers.
    check_on_inputs(
        checker=checker,
        function="row_dots",
        description="gives one dot product per row",
        inputs=(
            (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([10.0, 100.0])),
            (torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]]), torch.tensor([1.0, 2.0, 3.0])),
        ),
        reference=lambda matrix, w: matrix @ w,
    )
    # Two square matrices among them, where a @ b runs but pairs rows with columns.
    check_on_inputs(
        checker=checker,
        function="row_pairs",
        description="dots every row of a with every row of b",
        inputs=(
            (torch.tensor([[1.0, 0.0, 2.0], [0.0, 10.0, 1.0]]), torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 2.0, 3.0]])),
            (torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [3.0, 0.0, 1.0]]), torch.tensor([[2.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 4.0, 2.0]])),
        ),
        reference=lambda a, b: a @ b.T,
    )
    # In the square pair, a @ b runs without .T but picks other rows.
    check_on_inputs(
        checker=checker,
        function="best_match",
        description="picks the row of b each row of a scores highest with",
        inputs=(
            (torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]), torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 2.0, 3.0]])),
            (torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]])),
        ),
        reference=lambda a, b: (a @ b.T).argmax(dim=1),
    )


MATMUL1 = Exercise(
    name="matmul1",
    chapter=MATMUL,
    title="Matrix multiplication",
    lesson=MATMUL1_LESSON,
    source=MATMUL1_SOURCE,
    solution=MATMUL1_SOLUTION,
    hints={
        "weighted_sum": "Multiply v and w entry by entry, then add up the result, as math1 and reductions1 showed.",
        "row_dots": "Put matrix on the left of @ and w on the right.",
        "row_pairs": "The error says the middle sizes don't match. Turn the rows of b into columns with .T, as the "
        "line that sets b_turned does.",
        "best_match": "scores is every row of a dotted with every row of b, the same thing row_pairs returns once "
        "it is fixed.",
    },
    introduces=("weight", "weighted sum", "dot product", "matrix", "matrix multiplication"),
    check=check_matmul1,
)

QUIZ1_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines moves
    from the earlier exercises. If a step is unclear, the exercise in the
    table shows it.

    | Function | Moves | Shown in |
    |---|---|---|
    | `biggest_row` | add up each row, find the position of the largest | reductions1 |
    | `clean_fractions` | replace negatives with 0, divide each row by its total | math1, reductions1 |
    | `center_columns` | take each column's mean, take it away from every row | reductions1, broadcasting1 |
    | `weighted_first_columns` | keep the first two columns, take one weighted sum per row | indexing1, matmul1 |
    """,
)

QUIZ1_SOURCE = source_file(
    '''
    # quiz1.py
    #
    # No examples and no new tools. Each function combines moves from the
    # earlier exercises.

    import torch
    from torch import Tensor


    # TODO: Return the position of the row of grid with the largest total. For
    #   [[1., 2., 3.], [4., 5., 6.]] that is tensor(1).
    def biggest_row(grid: Tensor) -> Tensor:
        ...


    # TODO: Fill in both lines. The first replaces the negative entries of grid
    #   with 0.0. The second divides each row of cleaned by its own total, so
    #   every row adds up to 1.
    def clean_fractions(grid: Tensor) -> Tensor:
        cleaned = ...

        return ...


    # TODO: Take each column's mean away from that column, so every column ends
    #   up with a mean of 0.
    def center_columns(grid: Tensor) -> Tensor:
        ...


    # TODO: Fill in both lines. The first keeps the first two columns of grid,
    #   and the second gives each row of them its weighted sum with weights,
    #   one number per row. weights holds 2 numbers.
    def weighted_first_columns(
        grid: Tensor,
        weights: Tensor,
    ) -> Tensor:
        first_two = ...

        return ...
    ''',
)

QUIZ1_SOLUTION = source_file(
    '''
    # quiz1.py, a solution for comparison

    import torch
    from torch import Tensor


    def biggest_row(grid: Tensor) -> Tensor:
        return grid.sum(dim=1).argmax()


    def clean_fractions(grid: Tensor) -> Tensor:
        cleaned = torch.where(grid < 0, 0.0, grid)

        return cleaned / cleaned.sum(dim=1, keepdim=True)


    # The means have shape (columns,), which lines up with every row.
    def center_columns(grid: Tensor) -> Tensor:
        return grid - grid.mean(dim=0)


    def weighted_first_columns(
        grid: Tensor,
        weights: Tensor,
    ) -> Tensor:
        first_two = grid[:, 0:2]

        return first_two @ weights
    ''',
)


def check_quiz1(checker: Checker) -> None:
    # In the square grid the row totals and the column totals have their largest in different places.
    # In the last grid row 0 holds the largest single entry, but row 1 has the largest total.
    check_on_inputs(
        checker=checker,
        function="biggest_row",
        description="gives the position of the row with the largest total",
        inputs=(
            (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),),
            (torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 9.0], [5.0, 0.0, 0.0]]),),
            (torch.tensor([[1.0, 7.0], [3.0, 3.0], [8.0, 2.0], [0.0, 4.0]]),),
            (torch.tensor([[9.0, 0.0, 0.0], [4.0, 4.0, 4.0]]),),
        ),
        reference=lambda grid: grid.sum(dim=1).argmax(),
        expect=expect_positions,
    )
    check_on_inputs(
        checker=checker,
        function="clean_fractions",
        description="zeroes the negatives and makes every row add up to 1",
        inputs=(
            (torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]),),
            (torch.tensor([[2.0, -1.0, 2.0], [-3.0, 1.0, 1.0], [0.0, 4.0, 4.0]]),),
        ),
        reference=lambda grid: torch.where(grid < 0, 0.0, grid) / torch.where(grid < 0, 0.0, grid).sum(dim=1, keepdim=True),
    )
    check_on_inputs(
        checker=checker,
        function="center_columns",
        description="takes each column's mean away from it",
        inputs=(
            (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),),
            (torch.tensor([[2.0, 0.0, 1.0], [1.0, 5.0, 1.0], [0.0, 3.0, 9.0]]),),
            (torch.tensor([[1.0, 7.0], [3.0, 3.0], [8.0, 2.0], [0.0, 4.0]]),),
        ),
        reference=lambda grid: grid - grid.mean(dim=0),
    )
    # In the grid with 2 rows, weights @ first_two runs but gives other numbers.
    check_on_inputs(
        checker=checker,
        function="weighted_first_columns",
        description="gives each row's weighted sum of its first two columns",
        inputs=(
            (torch.tensor([[1.0, 2.0, 9.0, 9.0], [0.0, 1.0, 9.0, 9.0], [3.0, -1.0, 9.0, 9.0]]), torch.tensor([10.0, 1.0])),
            (torch.tensor([[1.0, 2.0, 5.0], [3.0, 0.0, 5.0]]), torch.tensor([2.0, -1.0])),
        ),
        reference=lambda grid, weights: grid[:, 0:2] @ weights,
    )


QUIZ1 = Exercise(
    name="quiz1",
    chapter=QUIZZES,
    title="Putting the moves together",
    lesson=QUIZ1_LESSON,
    source=QUIZ1_SOURCE,
    solution=QUIZ1_SOLUTION,
    hints={
        "biggest_row": "Add up each row, as row_totals does in reductions1, then find the position of the largest "
        "total with argmax.",
        "clean_fractions": "Replace the negatives with torch.where, as zero_negatives does in math1. Then divide "
        "cleaned by its row totals, kept with size 1 so they line up with the rows, as rows_to_fractions does in "
        "reductions1.",
        "center_columns": "Take the mean of each column, as column_means does in reductions1. Its shape lines up "
        "with every row, so take it away from grid.",
        "weighted_first_columns": "Slice the first two columns, as in indexing1. Then put first_two on the left of @ "
        "and weights on the right, as row_dots does in matmul1.",
    },
    introduces=(),
    check=check_quiz1,
)
