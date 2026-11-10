"""02_matmul, weighted sums and matrix multiplication, and quiz1, which closes the first two chapters."""

from __future__ import annotations

import torch
from torch import Tensor

from ..course import QUIZZES, Chapter, Exercise
from ..engine import Checker, check_on_inputs, expect_value
from ..text import prose, source_file
from .tensors import expect_positions, fastest_seconds, pinned_threads

MATMUL = Chapter(
    folder="02_matmul",
    title="Matrix multiplication",
    readme=source_file(
        """
        # Matrix multiplication

        A weighted sum multiplies each input by its own weight, a number that
        says how much that input counts, and adds up the results. Matrix
        multiplication computes many weighted sums in one call. Most of the
        computation in deep learning is matrix multiplication.

        ## Further reading

        - [Dive into Deep Learning, linear algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html)
        - [3Blue1Brown, the essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra)
        """,
    ),
)

MATMUL1_LESSON = prose(
    """
    Try the first TODO before you read on, with what you already know.

    A **weighted sum** multiplies each value by its own **weight** and adds up
    the results. With tensors, that is an elementwise multiplication followed
    by a sum, which you may have written for the first TODO. The **dot
    product** of two flat tensors computes the same thing in one step. PyTorch
    writes it as `@` between the two tensors.

    ```python
    v = torch.tensor([2.0, 3.0])
    w = torch.tensor([10.0, 100.0])
    (v * w).sum()    # tensor(320.)
    v @ w            # tensor(320.)
    ```

    A tensor with two dimensions is a **matrix**. Put a matrix on the left of
    `@` and a flat tensor on the right. Then `@` takes the dot product of each
    row of the matrix with the flat tensor, which gives one number per row.

    With a matrix on each side, `@` takes the dot product of every row of the
    left matrix with every column of the right matrix. This is called
    **matrix multiplication**. Entry `[i, j]` of the result is row `i` of the
    left matrix dotted with column `j` of the right one. So
    `(rows, k) @ (k, columns)` gives shape `(rows, columns)`. The two middle
    sizes must both be `k`, because a dot product needs two lists of the same
    length.

    ```python
    a = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    b = torch.tensor([[1.0, 0.0, 2.0],
                      [0.0, 10.0, 1.0]])
    a @ b                           # tensor([[ 1., 20.,  4.], [ 3., 40., 10.]])
    a @ torch.tensor([1.0, 1.0])    # tensor([3., 7.])
    ```

    `.T` swaps the two dimensions of a matrix, so its rows become its columns.
    `b` has shape `(2, 3)`, so `b.T` has shape `(3, 2)`. Use `.T` only on
    matrices, because PyTorch warns when you use it on a flat tensor.

    Most of the computation inside a model is matrix multiplication. A model's
    inputs often form a matrix with one example per row. One `@` with a matrix
    of weights then computes many weighted sums for every example at once.

    One `@` is also much faster than a loop. On a 1000 by 1000 matrix, a
    Python loop that takes one dot product per row took about 2 milliseconds
    in three runs. One `@` took about 0.015 milliseconds, which is over 140
    times faster.
    """,
)

MATMUL1_SOURCE = source_file(
    '''
    # matmul1.py

    import torch
    from torch import Tensor


    # TODO: Try this before reading the lesson, with what you already know.
    #   v and w are flat tensors of the same length. Multiply each entry of v
    #   by the matching entry of w, then add up the results.
    #   Example: for v = tensor([2., 3.]) and w = tensor([10., 100.]),
    #   weighted_sum returns tensor(320.).
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
    #   row. matrix has shape (rows, k) and w has shape (k,), so the result has
    #   shape (rows,).
    #   Example: for matrix = tensor([[1., 2.], [3., 4.]]) and the w above,
    #   row_dots returns tensor([210., 430.]).
    def row_dots(
        matrix: Tensor,
        w: Tensor,
    ) -> Tensor:
        ...


    a = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    b = torch.tensor([[1.0, 0.0, 2.0],
                      [0.0, 10.0, 1.0]])

    # A matrix @ a matrix takes the dot product of every row on the left with
    # every column on the right, so (2, 2) @ (2, 3) gives (2, 3).
    product = a @ b                               # tensor([[1., 20., 4.], [3., 40., 10.]])

    # .T swaps rows and columns, so b.T has shape (3, 2).
    b_turned = b.T                                # tensor([[ 1.,  0.], [ 0., 10.], [ 2.,  1.]])


    # TODO: This crashes when the checker calls it. It should return the dot
    #   product of every row of a with every row of b. Entry [i, j] of the
    #   result is row i of a dotted with row j of b. a has shape (n, k) and b
    #   has shape (m, k), so the result has shape (n, m). Read the error in the
    #   Result tab, then fix it.
    #   Example: for a = [[1., 2.], [3., 4.]] and
    #   b = [[1., 0.], [0., 1.], [1., 1.]], row_pairs should return
    #   tensor([[1., 2., 3.], [3., 4., 7.]]).
    def row_pairs(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return a @ b


    # TODO: Finish the first line, so scores holds the dot product of every row
    #   of a with every row of b, the same tensor row_pairs returns. The second
    #   line then finds, for each row of a, the position of the row of b that
    #   gives the largest dot product with it.
    #   Example: for the a and b in the row_pairs example, best_match returns
    #   tensor([2, 2]).
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


    # b.T turns the rows of b into columns, so (n, k) @ (k, m) gives (n, m).
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
        description="picks, for each row of a, the row of b with the largest dot product",
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
        "weighted_sum": "Multiply v and w entry by entry, then add up the products, as math1 and reductions1 "
        "showed.",
        "row_dots": "Put matrix on the left of @ and w on the right.",
        "row_pairs": "The error says the shapes cannot be multiplied, because the two middle sizes differ. Turn the "
        "rows of b into columns with .T, as the line that sets b_turned does.",
        "best_match": "scores is the dot product of every row of a with every row of b. That is the same tensor "
        "row_pairs returns once you fix it.",
    },
    introduces=("weight", "weighted sum", "dot product", "matrix", "matrix multiplication"),
    check=check_matmul1,
)

QUIZ1_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines
    operations from the earlier exercises. If a step is unclear, the exercise
    listed in the table shows how to do it.

    | Function | Steps | Shown in |
    |---|---|---|
    | `biggest_row` | sum each row, then find the position of the largest total | reductions1 |
    | `clean_fractions` | replace negatives with 0, divide each row by its total | math1, reductions1 |
    | `center_columns` | compute each column's mean, subtract it from every row | reductions1, broadcasting1 |
    | `weighted_first_columns` | keep the first two columns, compute one weighted sum per row | indexing1, matmul1 |
    """,
)

QUIZ1_SOURCE = source_file(
    '''
    # quiz1.py
    #
    # This quiz has no examples and no new tools. Each function combines
    # operations from the earlier exercises.

    import torch
    from torch import Tensor


    # TODO: Return the position of the row of grid whose entries add up to the
    #   largest total.
    #   Example: for the grid [[1., 2., 3.], [4., 5., 6.]], biggest_row returns
    #   tensor(1).
    def biggest_row(grid: Tensor) -> Tensor:
        ...


    # TODO: Fill in both lines. The first replaces the negative entries of grid
    #   with 0.0. The second divides each row of cleaned by that row's total,
    #   so every row adds up to 1.
    def clean_fractions(grid: Tensor) -> Tensor:
        cleaned = ...

        return ...


    # TODO: Subtract each column's mean from every entry in that column, so
    #   every column of the result has a mean of 0.
    def center_columns(grid: Tensor) -> Tensor:
        ...


    # TODO: Fill in both lines. The first keeps only the first two columns of
    #   grid. The second computes the weighted sum of each row of first_two
    #   with weights, one number per row. weights holds 2 numbers.
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
        description="replaces the negatives with 0 and makes every row add up to 1",
        inputs=(
            (torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]),),
            (torch.tensor([[2.0, -1.0, 2.0], [-3.0, 1.0, 1.0], [0.0, 4.0, 4.0]]),),
        ),
        reference=lambda grid: torch.where(grid < 0, 0.0, grid) / torch.where(grid < 0, 0.0, grid).sum(dim=1, keepdim=True),
    )
    check_on_inputs(
        checker=checker,
        function="center_columns",
        description="subtracts each column's mean from that column",
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
        "cleaned by its row totals, as rows_to_fractions does in reductions1. Keep the summed dimension with size "
        "1, so the totals line up with the rows.",
        "center_columns": "Compute the mean of each column, as column_means does in reductions1. The means have "
        "shape (columns,), which lines up with every row, so subtract them from grid.",
        "weighted_first_columns": "Slice the first two columns, as in indexing1. Then put first_two on the left of @ "
        "and weights on the right, as row_dots does in matmul1.",
    },
    introduces=(),
    check=check_quiz1,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_loop_against_matmul(seeds: tuple[int, ...] = (0, 1, 2)) -> list[tuple[float, float]]:
    """matmul1's timing: one weighted sum per row of a 1000 by 1000 matrix, from a Python loop of (row * w).sum(),
    against one matrix @ w. Per seed, the loop's seconds, then @'s."""
    results: list[tuple[float, float]] = []

    with pinned_threads():
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)
            matrix = torch.randn(1000, 1000, generator=generator)
            w = torch.randn(1000, generator=generator)
            results.append(
                (
                    fastest_seconds(
                        run=lambda: torch.stack([(row * w).sum() for row in matrix]),
                        repeats=5,
                    ),
                    fastest_seconds(
                        run=lambda: matrix @ w,
                        repeats=20,
                    ),
                ),
            )

    return results
