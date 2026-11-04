"""06_tensor_backprop, backpropagation by hand through whole tensors, and quiz2, which closes Section 1's calculus."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from ..course import QUIZZES, Chapter, Exercise
from ..engine import Checker, Described, check_on_inputs, expect_parts, expect_tensor, expect_value, small_tensor
from ..text import prose, source_file

TENSOR_BACKPROP = Chapter(
    folder="06_tensor_backprop",
    title="Backprop through tensors",
    readme=source_file(
        """
        # Backprop through tensors

        Models work on whole tensors, so their gradients are whole tensors too.
        This chapter works them out by hand, step by step, for elementwise math,
        sums, broadcasting and matrix multiplication, and checks every one
        against PyTorch.

        ## Further reading

        - [Karpathy, becoming a backprop ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)
        - [Karpathy, yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)
        - [CS231n, derivatives, backpropagation and vectorization](https://cs231n.stanford.edu/handouts/derivatives.pdf)
        """,
    ),
)


def autograd_backward(
    forward: Callable[..., Tensor],
    inputs: tuple[Tensor, ...],
    grad_out: Tensor,
) -> tuple[Tensor, ...]:
    """The gradient of every input, given the gradient handed down to forward's result."""
    leaves = [tensor.clone().requires_grad_() for tensor in inputs]
    # The sum of the result times grad_out hands the result exactly grad_out. Passing grad_out to
    # .backward() would do the same, but its first call imports sympy, which takes seconds and can
    # run past the check's time limit.
    (forward(*leaves) * grad_out).sum().backward()

    return tuple(leaf.grad for leaf in leaves)


# ---------------------------------------------------------------------------
# backprop1
# ---------------------------------------------------------------------------

BACKPROP1_LESSON = prose(
    """
    autograd4 showed that a tensor's gradient has the tensor's shape, one
    partial per entry. Real training leaves this to autograd. Working a few out
    by hand shows what autograd does, and that knowledge is what finds the
    problem when training goes wrong.

    Take one step that makes `y` from `x`, inside a bigger calculation that
    ends in a single number `L`. Backpropagation hands the step `grad_y`, the
    gradient of `L` with respect to `y`, shaped like `y`. The step's job is to
    hand back `grad_x`, shaped like `x`.

    In an elementwise step, each entry of `y` depends only on the same entry
    of `x`. So the chain rule works entry by entry, and `grad_x` is `grad_y`
    times the local derivative.

    ```python
    x = torch.tensor([1.0, -2.0, 3.0])
    y = x ** 2
    grad_y = torch.tensor([1.0, 0.5, 2.0])    # handed down by later steps
    grad_x = grad_y * 2 * x                    # tensor([ 2., -2., 12.])
    ```

    A multiplication `y = a * b` of two tensors of the same shape gives each
    input `grad_y` times the other input, entry by entry, as a `Value` did.
    `torch.exp(x)` takes exp of every entry, as `math.exp` does for one
    number, so `torch.exp(torch.tensor([0.0, 1.0]))` is
    `tensor([1.0000, 2.7183])`.

    A sum `s = x.sum()` makes one number, and each entry of `x` moves it one
    for one. So every entry gets the same gradient, `grad_s`.
    `torch.full_like(x, 5.0)` makes a tensor shaped like `x` with 5 in every
    entry. A mean divides the sum by the number of entries, which `x.numel()`
    gives.
    """,
)

BACKPROP1_SOURCE = source_file(
    '''
    # backprop1.py
    #
    # Each function is one step of a bigger calculation. It gets the gradient
    # handed down to its result, and returns the gradient of its input.

    import torch
    from torch import Tensor


    # y = x ** 2 is elementwise, so grad_x is grad_y times the local derivative
    # 2 * x, entry by entry.
    def square_backward(
        x: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        return grad_y * 2 * x


    # TODO: y = a * b, entry by entry. Return grad_a and grad_b, given grad_y.
    def multiply_backward(
        a: Tensor,
        b: Tensor,
        grad_y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: y = torch.exp(x), entry by entry. Return grad_x, given grad_y. exp is
    #   its own derivative.
    def exp_backward(
        x: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        ...


    # s = x.sum() moves one for one with every entry, so every entry gets grad_s.
    def sum_backward(
        x: Tensor,
        grad_s: Tensor,
    ) -> Tensor:
        return torch.full_like(x, grad_s.item())


    # TODO: m = x.mean(). Return grad_x, given grad_m, which holds one number.
    def mean_backward(
        x: Tensor,
        grad_m: Tensor,
    ) -> Tensor:
        ...
    ''',
)

BACKPROP1_SOLUTION = source_file(
    '''
    # backprop1.py, a solution for comparison

    import torch
    from torch import Tensor


    def square_backward(
        x: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        return grad_y * 2 * x


    def multiply_backward(
        a: Tensor,
        b: Tensor,
        grad_y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return grad_y * b, grad_y * a


    def exp_backward(
        x: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        return grad_y * torch.exp(x)


    def sum_backward(
        x: Tensor,
        grad_s: Tensor,
    ) -> Tensor:
        return torch.full_like(x, grad_s.item())


    # The mean is the sum divided by the count, so every entry gets grad_m / count.
    def mean_backward(
        x: Tensor,
        grad_m: Tensor,
    ) -> Tensor:
        return torch.full_like(x, grad_m.item() / x.numel())
    ''',
)


def check_backprop1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="multiply_backward",
        description="gives each input grad_y times the other input",
        inputs=(
            (small_tensor(1, 3), small_tensor(2, 3), small_tensor(3, 3)),
            (small_tensor(4, 2, 3), small_tensor(5, 2, 3), small_tensor(6, 2, 3)),
        ),
        reference=lambda a, b, grad_y: autograd_backward(lambda a, b: a * b, (a, b), grad_y),
        expect=expect_parts("grad_a", "grad_b"),
    )
    check_on_inputs(
        checker=checker,
        function="exp_backward",
        description="gives grad_y times the local derivative of exp",
        inputs=(
            (small_tensor(7, 4), small_tensor(8, 4)),
            (small_tensor(9, 2, 2), small_tensor(10, 2, 2)),
        ),
        reference=lambda x, grad_y: autograd_backward(torch.exp, (x,), grad_y)[0],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="mean_backward",
        description="gives every entry grad_m divided by the number of entries",
        inputs=(
            (small_tensor(11, 4), torch.tensor(1.0)),
            (small_tensor(12, 2, 3), torch.tensor(3.0)),
        ),
        reference=lambda x, grad_m: autograd_backward(lambda x: x.mean(), (x,), grad_m)[0],
        expect=expect_tensor,
    )


BACKPROP1 = Exercise(
    name="backprop1",
    chapter=TENSOR_BACKPROP,
    title="Gradients of whole tensors",
    lesson=BACKPROP1_LESSON,
    source=BACKPROP1_SOURCE,
    solution=BACKPROP1_SOLUTION,
    hints={
        "multiply_backward": "a gets grad_y times b, and b gets grad_y times a, entry by entry.",
        "exp_backward": "Multiply grad_y by the local derivative, which for exp is torch.exp(x).",
        "mean_backward": "Follow sum_backward, with grad_m divided by x.numel() as the number in every entry.",
    },
    introduces=(),
    check=check_backprop1,
)


# ---------------------------------------------------------------------------
# backprop2
# ---------------------------------------------------------------------------

BACKPROP2_LESSON = prose(
    """
    Broadcasting reuses a tensor. In `y = grid + row`, with `grid` of shape
    `(2, 3)` and `row` of shape `(3,)`, each entry of `row` is added into both
    rows of `grid`. So each entry of `row` takes two paths to `y`, and as in
    chain2 the paths add up. The gradient of `row` is `grad_y` added up over
    the rows `row` was copied into.

    ```python
    grad_y = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]])
    grad_grid = grad_y                # an addition passes it on unchanged
    grad_row = grad_y.sum(dim=0)      # tensor([5., 7., 9.]), shaped like row
    ```

    The rule holds in general. Backward through broadcasting adds the
    gradient up along every dimension the tensor was copied along, back to
    the tensor's own shape. When the tensor had a size 1 dimension, like a
    column of shape `(rows, 1)`, add up along it with `keepdim=True` so it
    keeps that shape.

    A sum along a dimension goes the other way. `s = x.sum(dim=1)` gives one
    total per row, and every entry of a row moves its total one for one. So
    each entry gets its row's gradient, copied across the row, and
    broadcasting does the copying. `torch.ones(2, 3)` makes ones of that
    shape, and `torch.ones_like(x)` makes ones shaped like `x`.

    ```python
    x = torch.ones(2, 3)
    grad_s = torch.tensor([10.0, 20.0])               # one per row, like s
    grad_x = grad_s.unsqueeze(1) * torch.ones_like(x)
    # tensor([[10., 10., 10.], [20., 20., 20.]])
    ```

    Along dimension 0 there is one number per column instead. A shape
    `(columns,)` already lines up with every row, so it needs no
    `.unsqueeze`.
    """,
)

BACKPROP2_SOURCE = source_file(
    '''
    # backprop2.py

    import torch
    from torch import Tensor


    # y = grid + row, where row has shape (columns,) and is copied into every
    # row of grid. Its gradient adds grad_y up over those rows.
    def add_row_backward(grad_y: Tensor) -> tuple[Tensor, Tensor]:
        grad_grid = grad_y
        grad_row = grad_y.sum(dim=0)

        return grad_grid, grad_row


    # TODO: y = grid + column, where column has shape (rows, 1) and is copied
    #   across every column of grid. Return grad_grid and grad_column, with
    #   grad_column shaped like column.
    def add_column_backward(grad_y: Tensor) -> tuple[Tensor, Tensor]:
        ...


    # TODO: s = x.mean(dim=0), one mean per column. Return grad_x, shaped like x,
    #   given grad_s, which holds one number per column.
    def column_mean_backward(
        x: Tensor,
        grad_s: Tensor,
    ) -> Tensor:
        ...


    # TODO: y = grid * scales.unsqueeze(1), so each row of grid is multiplied by
    #   its own number from scales. This should return the gradient of scales,
    #   but its shape comes out wrong. Read the result, then fix it.
    def scale_rows_backward(
        grid: Tensor,
        scales: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        return grad_y * grid
    ''',
)

BACKPROP2_SOLUTION = source_file(
    '''
    # backprop2.py, a solution for comparison

    import torch
    from torch import Tensor


    def add_row_backward(grad_y: Tensor) -> tuple[Tensor, Tensor]:
        grad_grid = grad_y
        grad_row = grad_y.sum(dim=0)

        return grad_grid, grad_row


    # column was copied along dimension 1, so its gradient adds up along it.
    def add_column_backward(grad_y: Tensor) -> tuple[Tensor, Tensor]:
        return grad_y, grad_y.sum(dim=1, keepdim=True)


    # Every entry of a column moves that column's mean by one over the row count.
    def column_mean_backward(
        x: Tensor,
        grad_s: Tensor,
    ) -> Tensor:
        return grad_s / x.shape[0] * torch.ones_like(x)


    # Each scale was copied across its row, so its gradient adds up along the row.
    def scale_rows_backward(
        grid: Tensor,
        scales: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        return (grad_y * grid).sum(dim=1)
    ''',
)


def check_backprop2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="add_column_backward",
        description="passes grad_y to grid and adds it up along the columns for column",
        inputs=(
            (small_tensor(13, 2, 3),),
            (small_tensor(14, 3, 3),),
        ),
        reference=lambda grad_y: autograd_backward(
            lambda grid, column: grid + column,
            (torch.zeros_like(grad_y), torch.zeros(grad_y.shape[0], 1)),
            grad_y,
        ),
        expect=expect_parts("grad_grid", "grad_column"),
    )
    # A square x among them, where the row count and the column count can't be told apart.
    check_on_inputs(
        checker=checker,
        function="column_mean_backward",
        description="copies each column's gradient down the column, divided by the row count",
        inputs=(
            (small_tensor(15, 4, 2), small_tensor(16, 2)),
            (small_tensor(17, 3, 3), small_tensor(18, 3)),
        ),
        reference=lambda x, grad_s: autograd_backward(lambda x: x.mean(dim=0), (x,), grad_s)[0],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="scale_rows_backward",
        description="gives the gradient of scales, one number per row",
        inputs=(
            (small_tensor(19, 2, 3), small_tensor(20, 2), small_tensor(21, 2, 3)),
            (small_tensor(22, 3, 3), small_tensor(23, 3), small_tensor(24, 3, 3)),
        ),
        reference=lambda grid, scales, grad_y: autograd_backward(
            lambda scales: grid * scales.unsqueeze(1),
            (scales,),
            grad_y,
        )[0],
        expect=expect_tensor,
    )


BACKPROP2 = Exercise(
    name="backprop2",
    chapter=TENSOR_BACKPROP,
    title="Through broadcasting and sums",
    lesson=BACKPROP2_LESSON,
    source=BACKPROP2_SOURCE,
    solution=BACKPROP2_SOLUTION,
    hints={
        "add_column_backward": "column was copied along dimension 1, across the columns. Add grad_y up along that "
        "dimension, and keep it with keepdim=True so the shape stays (rows, 1).",
        "column_mean_backward": "A mean is a sum divided by the count, here the number of rows, x.shape[0]. "
        "grad_s has one number per column, which broadcasting copies down every row.",
        "scale_rows_backward": "grad_y * grid is right entry by entry, but each scale was copied across its whole "
        "row. Add it up along the row, so there is one number per row, as scales has.",
    },
    introduces=(),
    check=check_backprop2,
)


# ---------------------------------------------------------------------------
# backprop3
# ---------------------------------------------------------------------------

BACKPROP3_LESSON = prose(
    """
    Take `y = a @ b`, with `a` of shape `(n, k)`, `b` of shape `(k, m)` and
    `y` of shape `(n, m)`. Each `y[i, j]` is the dot product of row `i` of `a`
    with column `j` of `b`, `a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + ...`.

    So `a[i, t]` moves `y[i, j]` by `b[t, j]`, for every column `j`. By the
    chain rule, with the paths added up,

    ```text
    grad_a[i, t] = grad_y[i, 0] * b[t, 0] + grad_y[i, 1] * b[t, 1] + ...
    ```

    That is the dot product of row `i` of `grad_y` with row `t` of `b`. For
    every `i` and `t` at once it is `grad_y @ b.T`.

    A shortcut finds the same answer from the shapes alone. `grad_a` must have
    the shape of `a`, `(n, k)`. It can only be made from `grad_y`, `(n, m)`,
    and `b`, `(k, m)`, and only one way of putting them together gives
    `(n, k)`.

    ```text
    grad_y @ b.T      (n, m) @ (m, k) gives (n, k), the shape of a
    ```

    The shortcut needs care when sizes are equal, since then a wrong
    arrangement has the right shape too. Check it against autograd, as
    autograd4 did.

    When `v` is flat, `y = a @ v` gives one number per row. `a[i, t]` then
    moves `y[i]` by `v[t]`, so `grad_a[i, t]` is `grad_y[i] * v[t]`, one
    product for every pair of `i` and `t`. Broadcasting makes every pair, from
    a column of shape `(n, 1)` and a flat tensor of shape `(k,)`.

    ```python
    torch.tensor([1.0, 2.0]).unsqueeze(1) * torch.tensor([10.0, 20.0, 30.0])
    # tensor([[10., 20., 30.], [20., 40., 60.]])
    ```
    """,
)

BACKPROP3_SOURCE = source_file(
    '''
    # backprop3.py

    import torch
    from torch import Tensor


    # For y = a @ b, the gradient of a is grad_y @ b.T, as the lesson works out.
    def matmul_grad_a(
        b: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        return grad_y @ b.T


    # TODO: Return the gradient of b for y = a @ b, shaped like b. Work it out
    #   from the shapes, the way the lesson does for a.
    def matmul_grad_b(
        a: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        ...


    # TODO: y = matrix @ v, where v is flat with shape (k,) and y has shape (n,).
    #   Return the gradient of v, shaped like v.
    def matvec_grad_v(
        matrix: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        ...


    # TODO: The same y = matrix @ v. Return the gradient of matrix, shaped like
    #   matrix.
    def matvec_grad_matrix(
        v: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        ...
    ''',
)

BACKPROP3_SOLUTION = source_file(
    '''
    # backprop3.py, a solution for comparison

    import torch
    from torch import Tensor


    def matmul_grad_a(
        b: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        return grad_y @ b.T


    # (k, n) @ (n, m) gives (k, m), the shape of b.
    def matmul_grad_b(
        a: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        return a.T @ grad_y


    # (k, n) @ (n,) gives (k,), the shape of v.
    def matvec_grad_v(
        matrix: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        return matrix.T @ grad_y


    # Shape (n, 1) times shape (k,) broadcasts to (n, k), every product of a pair.
    def matvec_grad_matrix(
        v: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        return grad_y.unsqueeze(1) * v
    ''',
)


def check_backprop3(checker: Checker) -> None:
    # The square cases are where a wrong arrangement still has the right shape.
    check_on_inputs(
        checker=checker,
        function="matmul_grad_b",
        description="gives the gradient of b, shaped like b",
        inputs=(
            (small_tensor(25, 2, 3), small_tensor(26, 2, 4)),
            (small_tensor(27, 3, 3), small_tensor(28, 3, 3)),
        ),
        reference=lambda a, grad_y: autograd_backward(
            lambda b: a @ b,
            (torch.zeros(a.shape[1], grad_y.shape[1]),),
            grad_y,
        )[0],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="matvec_grad_v",
        description="gives the gradient of v, shaped like v",
        inputs=(
            (small_tensor(29, 2, 3), small_tensor(30, 2)),
            (small_tensor(31, 3, 3), small_tensor(32, 3)),
        ),
        reference=lambda matrix, grad_y: autograd_backward(
            lambda v: matrix @ v,
            (torch.zeros(matrix.shape[1]),),
            grad_y,
        )[0],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="matvec_grad_matrix",
        description="gives the gradient of matrix, shaped like matrix",
        inputs=(
            (small_tensor(33, 3), small_tensor(34, 2)),
            (small_tensor(35, 3), small_tensor(36, 3)),
        ),
        reference=lambda v, grad_y: autograd_backward(
            lambda matrix: matrix @ v,
            (torch.zeros(grad_y.shape[0], v.shape[0]),),
            grad_y,
        )[0],
        expect=expect_tensor,
    )


BACKPROP3 = Exercise(
    name="backprop3",
    chapter=TENSOR_BACKPROP,
    title="The gradient of matrix multiplication",
    lesson=BACKPROP3_LESSON,
    source=BACKPROP3_SOURCE,
    solution=BACKPROP3_SOLUTION,
    hints={
        "matmul_grad_b": "grad_b needs the shape of b, (k, m). You have a, (n, k), and grad_y, (n, m). Turn a so "
        "its n meets the n of grad_y in the middle of @.",
        "matvec_grad_v": "It is the same as for b in matmul_grad_b, with v as a b that has one column. grad_v "
        "needs shape (k,).",
        "matvec_grad_matrix": "Give grad_y shape (n, 1) with unsqueeze, and multiply it by v. Broadcasting then "
        "pairs every grad_y[i] with every v[t].",
    },
    introduces=(),
    check=check_backprop3,
)


# ---------------------------------------------------------------------------
# backprop4
# ---------------------------------------------------------------------------

BACKPROP4_LESSON = prose(
    """
    A real calculation chains many steps. Running them from the inputs to the
    result is the forward pass. Backward runs them in reverse, and each step
    hands the gradient of its input to the step before it. Keep every value
    from the forward pass, since the backward steps need them.

    ```python
    # forward
    h = x @ w               # (n, k) @ (k, m)
    sq = h ** 2
    L = sq.mean()           # a single number

    # backward, starting from the gradient of L, which is 1
    grad_sq = torch.full_like(sq, 1.0 / sq.numel())    # through the mean
    grad_h = grad_sq * 2 * h                           # through the square
    grad_w = x.T @ grad_h                              # through the matrix multiplication
    ```

    Each backward line uses one rule from the last three exercises.

    Checking against PyTorch is cheap, so do it every time.
    `torch.allclose(a, b)` says whether two tensors match up to rounding.

    ```python
    w.requires_grad_()
    ((x @ w) ** 2).mean().backward()
    torch.allclose(w.grad, grad_w)    # True
    ```

    A wrong backward often runs without an error. With square tensors, `@ w`
    and `@ w.T` give the same shape, so only the numbers show the mistake.
    """,
)

BACKPROP4_SOURCE = source_file(
    '''
    # backprop4.py

    import torch
    from torch import Tensor


    # TODO: Finish the backward lines, so this returns the gradients of L with
    #   respect to w and c. c holds one number per column of z and is copied
    #   into every row. Each line passes the gradient back through one forward
    #   step, from the last forward line up.
    def backward_through_steps(
        x: Tensor,
        w: Tensor,
        c: Tensor,
    ) -> tuple[Tensor, Tensor]:
        z = x @ w
        u = z + c
        e = torch.exp(u)
        s = e.sum(dim=1)
        L = s.mean()

        grad_s = torch.full_like(s, 1.0 / s.numel())
        grad_e = grad_s.unsqueeze(1) * torch.ones_like(e)
        grad_u = ...
        grad_c = ...
        grad_z = grad_u
        grad_w = ...

        return grad_w, grad_c


    # TODO: z = x @ w with a square w. This should return the gradient of x, but
    #   its numbers are wrong even though its shape is right. Fix it.
    def grad_of_x(
        w: Tensor,
        grad_z: Tensor,
    ) -> Tensor:
        return grad_z @ w
    ''',
)

BACKPROP4_SOLUTION = source_file(
    '''
    # backprop4.py, a solution for comparison

    import torch
    from torch import Tensor


    def backward_through_steps(
        x: Tensor,
        w: Tensor,
        c: Tensor,
    ) -> tuple[Tensor, Tensor]:
        z = x @ w
        u = z + c
        e = torch.exp(u)
        s = e.sum(dim=1)
        L = s.mean()

        grad_s = torch.full_like(s, 1.0 / s.numel())
        grad_e = grad_s.unsqueeze(1) * torch.ones_like(e)
        grad_u = grad_e * e
        grad_c = grad_u.sum(dim=0)
        grad_z = grad_u
        grad_w = x.T @ grad_z

        return grad_w, grad_c


    # Like grad_a in backprop3, the gradient of the left side is grad_z @ w.T.
    def grad_of_x(
        w: Tensor,
        grad_z: Tensor,
    ) -> Tensor:
        return grad_z @ w.T
    ''',
)


def steps_forward(
    x: Tensor,
    w: Tensor,
    c: Tensor,
) -> Tensor:
    return torch.exp(x @ w + c).sum(dim=1).mean()


def check_backprop4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="backward_through_steps",
        description="gives the gradients of w and c",
        inputs=(
            (small_tensor(37, 4, 3), small_tensor(38, 3, 2) / 2, small_tensor(39, 2) / 2),
            (small_tensor(40, 3, 3), small_tensor(41, 3, 3) / 2, small_tensor(42, 3) / 2),
        ),
        reference=lambda x, w, c: autograd_backward(steps_forward, (x, w, c), torch.tensor(1.0))[1:],
        expect=expect_parts("grad_w", "grad_c"),
    )
    check_on_inputs(
        checker=checker,
        function="grad_of_x",
        description="gives the gradient of x for z = x @ w",
        inputs=(
            (small_tensor(43, 3, 3), small_tensor(44, 2, 3)),
            (small_tensor(45, 2, 2), small_tensor(46, 4, 2)),
        ),
        reference=lambda w, grad_z: autograd_backward(
            lambda x: x @ w,
            (torch.zeros(grad_z.shape[0], w.shape[0]),),
            grad_z,
        )[0],
        expect=expect_tensor,
    )


BACKPROP4 = Exercise(
    name="backprop4",
    chapter=TENSOR_BACKPROP,
    title="Backward through several steps",
    lesson=BACKPROP4_LESSON,
    source=BACKPROP4_SOURCE,
    solution=BACKPROP4_SOLUTION,
    hints={
        "backward_through_steps": "Work from the last forward line up. e = exp(u), so grad_u is grad_e times "
        "exp's local derivative. c was copied into every row, so its gradient adds grad_u up over the rows. w is "
        "on the right of x @ w, like b in backprop3.",
        "grad_of_x": "x is on the left of @, like a in backprop3, whose gradient is grad_y @ b.T.",
    },
    introduces=(),
    check=check_backprop4,
)


# ---------------------------------------------------------------------------
# quiz2
# ---------------------------------------------------------------------------

QUIZ2_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines moves
    from the derivatives, chain rule, autograd and backprop chapters. If a
    step is unclear, the exercise in the table shows it.

    | Function | Moves | Shown in |
    |---|---|---|
    | `partials_of_x_times_exp` | the product rule, the chain rule through exp, a variable reaching f along two paths | derivatives2, chain1, chain2 |
    | `largest_partial` | a gradient from autograd, the position of its largest entry | autograd4, reductions1 |
    | `squared_matmul_grad_w` | backward through a square, then through a matrix multiplication | backprop1, backprop3 |
    | `shifted_row_sums_grad` | backward through a row sum, then through broadcasting | backprop2 |
    """,
)

QUIZ2_SOURCE = source_file(
    '''
    # quiz2.py
    #
    # No examples and no new tools. Each function combines moves from the
    # derivatives, chain rule, autograd and backprop chapters.

    import math
    from collections.abc import Callable

    import torch
    from torch import Tensor


    # TODO: Return the partial derivatives of f(x, y) = x * exp(x * y), first
    #   with respect to x, then with respect to y. x and y are plain numbers,
    #   so use math.exp.
    def partials_of_x_times_exp(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        ...


    # TODO: Return the position of the largest entry of the gradient of f at v,
    #   as a plain int, using PyTorch's autograd. The largest value counts, so
    #   a partial of -8 is smaller than one of 2. f gives a single number.
    def largest_partial(
        f: Callable[[Tensor], Tensor],
        v: Tensor,
    ) -> int:
        ...


    # TODO: Finish both lines. y = (x @ w) ** 2, and grad_y is handed down.
    #   Return the gradient of w, shaped like w.
    def squared_matmul_grad_w(
        x: Tensor,
        w: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        z = x @ w
        grad_z = ...

        return ...


    # TODO: y = (grid + row).sum(dim=1), one total per row, with row of shape
    #   (columns,) added into every row of grid. Return the gradient of row,
    #   shaped like row, given grad_y, which has one number per row of grid.
    def shifted_row_sums_grad(
        grid: Tensor,
        row: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        ...
    ''',
)

QUIZ2_SOLUTION = source_file(
    '''
    # quiz2.py, a solution for comparison

    import math
    from collections.abc import Callable

    import torch
    from torch import Tensor


    # x reaches f twice, as the left factor and inside exp, so the product rule
    # adds a term for each. The one inside exp chains through exp to x * y,
    # whose partial is y for x and x for y.
    def partials_of_x_times_exp(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        exp_of_product = math.exp(x * y)

        return exp_of_product + x * y * exp_of_product, x * x * exp_of_product


    def largest_partial(
        f: Callable[[Tensor], Tensor],
        v: Tensor,
    ) -> int:
        v.requires_grad_()
        f(v).backward()

        return v.grad.argmax().item()


    def squared_matmul_grad_w(
        x: Tensor,
        w: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        z = x @ w
        grad_z = grad_y * 2 * z

        return x.T @ grad_z


    # Each row's total copies its gradient across the row, and row was copied into
    # every row, so its gradient adds those copies up over the rows.
    def shifted_row_sums_grad(
        grid: Tensor,
        row: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        grad_sum = grad_y.unsqueeze(1) * torch.ones_like(grid)

        return grad_sum.sum(dim=0)
    ''',
)


def check_quiz2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="partials_of_x_times_exp",
        description="gives both partials of x * exp(x * y)",
        inputs=(
            (0.5, 2.0),
            (-1.5, 0.5),
            (2.0, -0.25),
        ),
        reference=lambda x, y: ((1 + x * y) * math.exp(x * y), x * x * math.exp(x * y)),
        expect=expect_parts("the x partial", "the y partial"),
    )
    check_on_inputs(
        checker=checker,
        function="largest_partial",
        description="gives the position of the largest partial derivative",
        inputs=(
            (Described(lambda v: (v ** 2).sum(), "(v ** 2).sum()"), torch.tensor([1.0, 3.0, -4.0, 2.0])),
            (Described(lambda v: v[0] * v[1] + v[2] ** 3, "v[0] * v[1] + v[2] ** 3"), torch.tensor([2.0, 5.0, 1.0])),
        ),
        reference=lambda f, v: autograd_backward(f, (v,), torch.tensor(1.0))[0].argmax().item(),
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="squared_matmul_grad_w",
        description="gives the gradient of w for (x @ w) ** 2",
        inputs=(
            (small_tensor(47, 3, 2), small_tensor(48, 2, 4), small_tensor(49, 3, 4)),
            (small_tensor(50, 3, 3), small_tensor(51, 3, 3), small_tensor(52, 3, 3)),
        ),
        reference=lambda x, w, grad_y: autograd_backward(lambda w: (x @ w) ** 2, (w,), grad_y)[0],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="shifted_row_sums_grad",
        description="gives the gradient of row, shaped like row",
        inputs=(
            (small_tensor(53, 2, 3), small_tensor(54, 3), small_tensor(55, 2)),
            (small_tensor(56, 4, 4), small_tensor(57, 4), small_tensor(58, 4)),
        ),
        reference=lambda grid, row, grad_y: autograd_backward(
            lambda row: (grid + row).sum(dim=1),
            (row,),
            grad_y,
        )[0],
        expect=expect_tensor,
    )


QUIZ2 = Exercise(
    name="quiz2",
    chapter=QUIZZES,
    title="Gradients, start to finish",
    lesson=QUIZ2_LESSON,
    source=QUIZ2_SOURCE,
    solution=QUIZ2_SOLUTION,
    hints={
        "partials_of_x_times_exp": "f is x times exp(x * y). For x, the product rule from derivatives2 gives one "
        "term per factor. The derivative of exp(x * y) comes from the chain rule, as in chain1, with the inner step "
        "x * y. y only reaches f inside exp, so its partial keeps the x in front.",
        "largest_partial": "Get the gradient of f at v as autograd_gradient does in autograd4, then find the "
        "position of its largest entry with argmax.",
        "squared_matmul_grad_w": "The square is elementwise, so grad_z is grad_y times its local derivative at z. "
        "Then w is on the right of @, as b is in backprop3.",
        "shifted_row_sums_grad": "Backward through the row sum copies each grad_y across its row of grid. Then row "
        "was copied into every row, so add those copies up over the rows.",
    },
    introduces=(),
    check=check_quiz2,
)
