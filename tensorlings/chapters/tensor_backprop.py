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

        Models compute with whole tensors, so their gradients are whole tensors
        too. This chapter works out those gradients by hand, one step at a time.
        It covers elementwise math, sums, broadcasting and matrix
        multiplication, and checks every result against PyTorch.

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
    autograd4 showed that a tensor's gradient has the same shape as the
    tensor, with one partial derivative per entry. In real training, autograd
    computes these gradients. Working a few out by hand shows you what
    autograd does. That understanding is what helps you find the problem when
    training goes wrong.

    Take one step that computes `y` from `x`, inside a bigger calculation that
    ends in a single number `L`. Backpropagation gives the step `grad_y`, the
    gradient of `L` with respect to `y`, which has the shape of `y`. The step
    must return `grad_x`, the gradient of `L` with respect to `x`, which has
    the shape of `x`.

    In an elementwise step, each entry of `y` depends only on the entry of `x`
    at the same position. So the chain rule works entry by entry, and `grad_x`
    is `grad_y` times the local derivative.

    ```python
    x = torch.tensor([1.0, -2.0, 3.0])
    y = x ** 2
    grad_y = torch.tensor([1.0, 0.5, 2.0])    # handed down by later steps
    grad_x = grad_y * 2 * x                    # tensor([ 2., -2., 12.])
    ```

    For a multiplication `y = a * b` of two tensors with the same shape, each
    input gets `grad_y` times the other input, entry by entry. This is the
    same rule your `Value` used. `torch.exp(x)` applies exp to every entry,
    the way `math.exp` does for one number. So
    `torch.exp(torch.tensor([0.0, 1.0]))` is `tensor([1.0000, 2.7183])`.

    A sum `s = x.sum()` gives one number. Increasing any one entry of `x` by
    some amount increases `s` by the same amount. So the derivative of `s`
    with respect to each entry is 1, and every entry of `x` gets the same
    gradient, `grad_s`. `torch.full_like(x, 5.0)` makes a tensor with the
    shape of `x` and 5 in every entry. A mean divides the sum by the number
    of entries, and `x.numel()` gives that number.
    """,
)

BACKPROP1_SOURCE = source_file(
    '''
    # backprop1.py
    #
    # Each function below is the backward step of one operation inside a
    # bigger calculation. It receives the gradient of the operation's result
    # and returns the gradient of the operation's input.

    import torch
    from torch import Tensor


    # y = x ** 2 is elementwise, so grad_x is grad_y times the local derivative
    # 2 * x, entry by entry.
    def square_backward(
        x: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        return grad_y * 2 * x


    # TODO: y = a * b, entry by entry, where a and b have the same shape.
    #   Return grad_a and grad_b, the gradients of a and b, given grad_y.
    def multiply_backward(
        a: Tensor,
        b: Tensor,
        grad_y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: y = torch.exp(x), applied to every entry. Return grad_x, given
    #   grad_y. exp is its own derivative.
    def exp_backward(
        x: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        ...


    # Increasing any entry of x by some amount increases s = x.sum() by the
    # same amount. So every entry of x gets the gradient grad_s.
    def sum_backward(
        x: Tensor,
        grad_s: Tensor,
    ) -> Tensor:
        return torch.full_like(x, grad_s.item())


    # TODO: m = x.mean(), the mean of every entry of x. Return grad_x, shaped
    #   like x, given grad_m, a tensor that holds one number.
    #   Example: for x of shape (2, 2) and grad_m = torch.tensor(1.0), it
    #   returns a (2, 2) tensor with 0.25 in every entry.
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


    # The mean is the sum divided by the number of entries, so every entry gets
    # grad_m divided by that number.
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
        "mean_backward": "Copy the pattern of sum_backward, but put grad_m divided by x.numel() in every entry.",
    },
    introduces=(),
    check=check_backprop1,
)


# ---------------------------------------------------------------------------
# backprop2
# ---------------------------------------------------------------------------

BACKPROP2_LESSON = prose(
    """
    Broadcasting uses one tensor in several places. In `y = grid + row`,
    `grid` has shape `(2, 3)` and `row` has shape `(3,)`. Each entry of `row`
    is added to both rows of `grid`. So each entry of `row` reaches `y` along
    two paths, and as in chain2 the paths add up. The gradient of `row` is
    `grad_y` summed over the rows that `row` was copied into.

    ```python
    grad_y = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]])
    grad_grid = grad_y                # an addition passes it on unchanged
    grad_row = grad_y.sum(dim=0)      # tensor([5., 7., 9.]), shaped like row
    ```

    This rule holds in general. To go backward through broadcasting, sum the
    gradient along every dimension the tensor was copied along. The result
    has the tensor's own shape. Sometimes the tensor had a dimension of size
    1, like a column of shape `(rows, 1)`. Then sum along that dimension with
    `keepdim=True`, so the result keeps the shape `(rows, 1)`.

    A sum along a dimension works the other way around. `s = x.sum(dim=1)`
    gives one total per row. The derivative of a row's total with respect to
    each entry in that row is 1. So each entry gets the gradient of its row's
    total, copied across the row, and broadcasting can do the copying.
    `torch.ones(2, 3)` makes a tensor of ones with shape `(2, 3)`, and
    `torch.ones_like(x)` makes ones with the shape of `x`.

    ```python
    x = torch.ones(2, 3)
    grad_s = torch.tensor([10.0, 20.0])               # one per row, like s
    grad_x = grad_s.unsqueeze(1) * torch.ones_like(x)
    # tensor([[10., 10., 10.], [20., 20., 20.]])
    ```

    A sum along dimension 0 gives one number per column instead. Its gradient
    has shape `(columns,)`, which broadcasting already lines up with every
    row. So it needs no `.unsqueeze`.
    """,
)

BACKPROP2_SOURCE = source_file(
    '''
    # backprop2.py

    import torch
    from torch import Tensor


    # y = grid + row, where row has shape (columns,) and is copied into every
    # row of grid. So the gradient of row is grad_y summed over those rows.
    def add_row_backward(grad_y: Tensor) -> tuple[Tensor, Tensor]:
        grad_grid = grad_y
        grad_row = grad_y.sum(dim=0)

        return grad_grid, grad_row


    # TODO: y = grid + column, where column has shape (rows, 1) and is copied
    #   across every column of grid. Return grad_grid and grad_column, with
    #   grad_column shaped like column.
    #   Example: for grad_y = torch.tensor([[1.0, 2.0], [3.0, 4.0]]), it
    #   returns grad_y and tensor([[3.], [7.]]).
    def add_column_backward(grad_y: Tensor) -> tuple[Tensor, Tensor]:
        ...


    # TODO: s = x.mean(dim=0), one mean per column. Return grad_x, shaped like x,
    #   given grad_s, which holds one number per column.
    #   Example: for x of shape (2, 3) and grad_s = torch.tensor([2.0, 4.0, 6.0]),
    #   it returns tensor([[1., 2., 3.], [1., 2., 3.]]).
    def column_mean_backward(
        x: Tensor,
        grad_s: Tensor,
    ) -> Tensor:
        ...


    # TODO: y = grid * scales.unsqueeze(1), so each row of grid is multiplied by
    #   its own number from scales. This should return the gradient of scales,
    #   shaped like scales, but it returns the wrong shape. Save the file, read
    #   the error in the Result tab, then fix it.
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


    # column was copied along dimension 1, so its gradient is grad_y summed
    # along dimension 1.
    def add_column_backward(grad_y: Tensor) -> tuple[Tensor, Tensor]:
        return grad_y, grad_y.sum(dim=1, keepdim=True)


    # The derivative of a column's mean with respect to each entry of that
    # column is 1 divided by the number of rows.
    def column_mean_backward(
        x: Tensor,
        grad_s: Tensor,
    ) -> Tensor:
        return grad_s / x.shape[0] * torch.ones_like(x)


    # Each scale was copied across its row, so its gradient is summed along the row.
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
        "add_column_backward": "column was copied along dimension 1, across the columns. Sum grad_y along that "
        "dimension, with keepdim=True so the result has shape (rows, 1).",
        "column_mean_backward": "A mean is a sum divided by the number of entries, which here is the number of rows, "
        "x.shape[0]. grad_s has one number per column, and broadcasting copies it down every row.",
        "scale_rows_backward": "grad_y * grid is correct entry by entry, but each scale was copied across its whole "
        "row. So sum grad_y * grid along each row, which gives one number per row, the same shape as scales.",
    },
    introduces=(),
    check=check_backprop2,
)


# ---------------------------------------------------------------------------
# backprop3
# ---------------------------------------------------------------------------

BACKPROP3_LESSON = prose(
    """
    Take `y = a @ b`, where `a` has shape `(n, k)`, `b` has shape `(k, m)` and
    `y` has shape `(n, m)`. Each `y[i, j]` is the dot product of row `i` of `a`
    with column `j` of `b`. That is,
    `y[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + ...`.

    So the derivative of `y[i, j]` with respect to `a[i, t]` is `b[t, j]`. The
    entry `a[i, t]` appears in every entry of row `i` of `y`, so it reaches
    `y` along one path per column `j`. By the chain rule, adding up those
    paths gives

    ```text
    grad_a[i, t] = grad_y[i, 0] * b[t, 0] + grad_y[i, 1] * b[t, 1] + ...
    ```

    That sum is the dot product of row `i` of `grad_y` with row `t` of `b`.
    Computed for every `i` and `t` at once, it is `grad_y @ b.T`.

    A shortcut finds the same answer from the shapes alone. `grad_a` must have
    the shape of `a`, which is `(n, k)`. It can only be made from `grad_y`,
    with shape `(n, m)`, and `b`, with shape `(k, m)`. Only one way of
    multiplying them with `@`, transposing where needed, gives `(n, k)`.

    ```text
    grad_y @ b.T      (n, m) @ (m, k) gives (n, k), the shape of a
    ```

    Be careful with the shortcut when some sizes are equal, because then a
    wrong arrangement can have the right shape too. Check the result against
    autograd, as autograd4 did.

    When `v` is a flat tensor of shape `(k,)`, `y = a @ v` gives one number
    per row of `a`. The derivative of `y[i]` with respect to `a[i, t]` is then
    `v[t]`. So `grad_a[i, t]` is `grad_y[i] * v[t]`, one product for every
    pair of `i` and `t`. Broadcasting can compute every pair at once. Multiply
    a column of shape `(n, 1)` by a flat tensor of shape `(k,)`, and the
    result has shape `(n, k)`.

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


    # TODO: Return the gradient of b for y = a @ b, shaped like b. a has shape
    #   (n, k) and grad_y has shape (n, m). Work it out from the shapes, the way
    #   the lesson does for a.
    def matmul_grad_b(
        a: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        ...


    # TODO: y = matrix @ v, where matrix has shape (n, k), v is flat with shape
    #   (k,) and y has shape (n,). Return the gradient of v, shaped like v.
    def matvec_grad_v(
        matrix: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        ...


    # TODO: The same y = matrix @ v as above. Return the gradient of matrix,
    #   shaped like matrix, which is (n, k).
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


    # Shape (n, 1) times shape (k,) broadcasts to (n, k), one product per pair.
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
        "matmul_grad_b": "grad_b needs the shape of b, (k, m). You have a, with shape (n, k), and grad_y, with shape "
        "(n, m). Transpose a so that its n sits next to the n of grad_y in the middle of @.",
        "matvec_grad_v": "This works like the gradient of b in matmul_grad_b, if you think of v as a b with one "
        "column. grad_v needs shape (k,).",
        "matvec_grad_matrix": "Give grad_y shape (n, 1) with unsqueeze, and multiply it by v. Broadcasting then "
        "multiplies every grad_y[i] by every v[t].",
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
    result is called the forward pass. The backward pass goes through the same
    steps in reverse order. Each step computes the gradient of its input and
    passes it back to the step that produced that input. Keep every value
    from the forward pass, because the backward steps need them.

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

    Checking your gradients against PyTorch takes little effort, so do it
    every time. `torch.allclose(a, b)` returns `True` when two tensors are
    equal apart from small rounding errors.

    ```python
    w.requires_grad_()
    ((x @ w) ** 2).mean().backward()
    torch.allclose(w.grad, grad_w)    # True
    ```

    A wrong backward pass often runs without an error. With square tensors,
    `@ w` and `@ w.T` give the same shape, so only the numbers reveal the
    mistake.
    """,
)

BACKPROP4_SOURCE = source_file(
    '''
    # backprop4.py

    import torch
    from torch import Tensor


    # TODO: Finish the backward lines, so this returns the gradients of L with
    #   respect to w and c. c holds one number per column of z, and
    #   broadcasting adds it to every row of z. Each backward line passes the
    #   gradient back through one forward step, starting from the last forward
    #   line and working up.
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
        "backward_through_steps": "Work from the last forward line up. Since e = exp(u), grad_u is grad_e times "
        "the local derivative of exp. c was copied into every row, so its gradient is grad_u summed over the rows. "
        "w is on the right of x @ w, like b in backprop3.",
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
    This quiz has no examples and no new tools. Each function combines
    techniques from the derivatives, chain rule, autograd and backprop
    chapters. If a step is unclear, the exercise named in the table shows how
    it works.

    | Function | Uses | Shown in |
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
    # This quiz has no examples and no new tools. Each function combines
    # techniques from the derivatives, chain rule, autograd and backprop
    # chapters.

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
    #   as a plain int, using PyTorch's autograd. Compare the signed values, so
    #   a partial of -8 counts as smaller than a partial of 2. f returns a
    #   tensor that holds a single number.
    def largest_partial(
        f: Callable[[Tensor], Tensor],
        v: Tensor,
    ) -> int:
        ...


    # TODO: Finish both lines. y = (x @ w) ** 2, and grad_y is the gradient passed
    #   back to y by later steps. Return the gradient of w, shaped like w.
    def squared_matmul_grad_w(
        x: Tensor,
        w: Tensor,
        grad_y: Tensor,
    ) -> Tensor:
        z = x @ w
        grad_z = ...

        return ...


    # TODO: y = (grid + row).sum(dim=1), which gives one total per row. row has
    #   shape (columns,) and is added to every row of grid. Return the gradient of
    #   row, shaped like row, given grad_y, which holds one number per row of grid.
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


    # x appears twice in f, as the left factor and inside exp, so the product
    # rule adds one term for each. For the x inside exp, the chain rule goes
    # through exp to x * y, whose partial is y for x and x for y.
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


    # Backward through the row sum copies each total's gradient across its row.
    # row was copied into every row, so its gradient sums those copies over the
    # rows.
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
            (Described(lambda v: v[0] * v[1] + v[2] ** 3, "v[0] * v[1] + (v[2] ** 3)"), torch.tensor([2.0, 5.0, 1.0])),
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
        "x * y. y appears in f only inside exp, so its partial keeps the factor x in front.",
        "largest_partial": "Get the gradient of f at v as autograd_gradient does in autograd4, then find the "
        "position of its largest entry with argmax.",
        "squared_matmul_grad_w": "The square is elementwise, so grad_z is grad_y times its local derivative at z. "
        "Then w is on the right of @, as b is in backprop3.",
        "shifted_row_sums_grad": "Going backward through the row sum copies each entry of grad_y across its row of "
        "grid. row was copied into every row, so sum those copies over the rows.",
    },
    introduces=(),
    check=check_quiz2,
)
