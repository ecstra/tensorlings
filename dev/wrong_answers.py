"""Run common wrong answers through each exercise's checks and print the messages.

Each case takes an exercise's solution and swaps one piece of it for a likely
mistake, so the other functions stay right and only the mistake is tested.
Every case must fail, and the script exits with 1 if one passes, since a check
that accepts a wrong answer is testing nothing. Read the printed messages too,
because a beginner sees exactly these.

    .venv/bin/python dev/wrong_answers.py
    .venv/bin/python dev/wrong_answers.py init1 dropout2    # only these exercises
"""

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Loading the course would otherwise leave __pycache__ folders in the package.
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))

from tensorlings.course import attempt
from tensorlings.progress import exercise_named

# (exercise, what is wrong, a piece of the solution, what a learner might write instead)
CASES: list[tuple[str, str, str, str]] = [
    ("tensors1", "b left as a list", "b = torch.tensor([10, 20, 30, 40])", "b = [10, 20, 30, 40]"),
    ("tensors1", "b with the wrong numbers", "b = torch.tensor([10, 20, 30, 40])", "b = torch.tensor([10, 20, 30])"),
    ("tensors1", "d with 2 rows of 3", "d = torch.tensor([[1, 2],\n                  [3, 4],\n                  [5, 6]])", "d = torch.tensor([[1, 2, 3], [4, 5, 6]])"),
    ("tensors1", "e still crashes", "e = torch.tensor([[1, 2], [3, 4]])", "e = torch.tensor([[1, 2], [3, 4, 5]])"),
    ("tensors1", "e fixed by flattening it", "e = torch.tensor([[1, 2], [3, 4]])", "e = torch.tensor([1, 2, 3, 4, 5])"),
    ("tensors1", "syntax error", "b = torch.tensor([10, 20, 30, 40])", "b = torch.tensor([10, 20, 30, 40]"),
    ("indexing1", "first_column as a Python list", "return grid[:, 0]", "return [row[0] for row in grid]"),
    ("indexing1", "first_column typed in for the grid in the file", "return grid[:, 0]", "return torch.tensor([1, 4, 7])"),
    ("indexing1", "first_column takes the first row", "return grid[:, 0]", "return grid[0]"),
    ("indexing1", "row_one takes a column", "return grid[1]", "return grid[:, 1]"),
    ("indexing1", "bottom_right only works for a 3 by 3 grid", "return grid[-1, -1]", "return grid[2, 2]"),
    ("indexing1", "last_two_columns only works for 3 columns", "return grid[:, -2:]", "return grid[:, 1:]"),
    ("indexing1", "last_two_columns takes the last two rows", "return grid[:, -2:]", "return grid[-2:]"),
    ("math1", "multiply adds instead", "return a * b", "return a + b"),
    ("math1", "multiply typed in for the tensors in the file", "return a * b", "return torch.tensor([10.0, 40.0, 90.0])"),
    ("math1", "square doubles instead", "return a ** 2", "return a * 2"),
    ("math1", "below_zero counts 0 as below", "return c < 0", "return c <= 0"),
    ("math1", "zero_negatives keeps only the negatives", "return torch.where(c < 0, 0.0, c)", "return torch.where(c < 0, c, 0.0)"),
    ("broadcasting1", "multiply_rows adds instead", "return grid * row", "return grid + row"),
    ("broadcasting1", "add_per_row still crashes", "return grid + numbers.unsqueeze(1)", "return grid + numbers"),
    ("reductions1", "total keeps only the last row", "result = result + row.sum()", "result = row.sum()"),
    ("reductions1", "total typed in for the grid in the file", "return result", "return 21.0"),
    ("reductions1", "row_totals adds down the columns", "return grid.sum(dim=1)", "return grid.sum(dim=0)"),
    ("reductions1", "column_means adds instead", "return grid.mean(dim=0)", "return grid.sum(dim=0)"),
    ("reductions1", "best_in_each_column looks along the rows", "return grid.argmax(dim=0)", "return grid.argmax(dim=1)"),
    ("reductions1", "rows_to_fractions without keepdim", "    totals = grid.sum(dim=1, keepdim=True)", "    totals = grid.sum(dim=1)"),
    ("matmul1", "weighted_sum only multiplies", "return (v @ w).item()", "return v * w"),
    ("matmul1", "weighted_sum typed in for the tensors in the comment", "return (v @ w).item()", "return 320.0"),
    ("matmul1", "row_dots in the wrong order", "return matrix @ w", "return w @ matrix"),
    ("matmul1", "row_pairs still crashes", "return a @ b.T", "return a @ b"),
    ("matmul1", "row_pairs turns the wrong side", "return a @ b.T", "return a.T @ b"),
    ("matmul1", "best_match without .T", "scores = a @ b.T", "scores = a @ b"),
    ("matmul1", "best_match scores b against a", "scores = a @ b.T", "scores = b @ a.T"),
    ("quiz1", "biggest_row adds down the columns", "return grid.sum(dim=1).argmax().item()", "return grid.sum(dim=0).argmax().item()"),
    ("quiz1", "biggest_row typed in for the grid in the comment", "return grid.sum(dim=1).argmax().item()", "return 1"),
    ("quiz1", "clean_fractions without keepdim", "return cleaned / cleaned.sum(dim=1, keepdim=True)", "return cleaned / cleaned.sum(dim=1)"),
    ("quiz1", "center_columns takes each row's mean", "return grid - grid.mean(dim=0)", "return grid - grid.mean(dim=1)"),
    ("quiz1", "weighted_first_columns keeps the first two rows", "first_two = grid[:, 0:2]", "first_two = grid[0:2]"),
    ("quiz1", "weighted_first_columns in the wrong order", "return first_two @ weights", "return weights @ first_two"),
    ("quiz1", "a loop that never ends", "    cleaned = torch.where(grid < 0, 0.0, grid)", "    while True:\n        pass\n    cleaned = torch.where(grid < 0, 0.0, grid)"),
    ("derivatives1", "nudge_slope divides by the wrong nudge", "return (f(x + 0.001) - f(x)) / 0.001", "return (f(x + 0.001) - f(x)) / 0.01"),
    ("derivatives1", "nudge_slope typed in for x ** 2 at 3", "return (f(x + 0.001) - f(x)) / 0.001", "return 6.001"),
    ("derivatives1", "central_slope nudges one way", "    h: float,\n) -> float:\n    return (f(x + h) - f(x - h)) / (2 * h)", "    h: float,\n) -> float:\n    return (f(x + h) - f(x)) / h"),
    ("derivatives1", "central_slope divides by h alone", "    h: float,\n) -> float:\n    return (f(x + h) - f(x - h)) / (2 * h)", "    h: float,\n) -> float:\n    return (f(x + h) - f(x - h)) / h"),
    ("derivatives1", "careful_slope keeps the tiny nudge", "h = 1e-6", "h = 1e-15"),
    ("derivatives2", "d_quartic keeps the constant", "return 20 * x ** 3 - 2", "return 20 * x ** 3 - 2 + 7"),
    ("derivatives2", "d_quartic forgets the 5", "return 20 * x ** 3 - 2", "return 4 * x ** 3 - 2"),
    ("derivatives2", "d_cube_times_log multiplies the two derivatives", "return 3 * x ** 2 * math.log(x) + x ** 2", "return 3 * x ** 2 * (1 / x)"),
    ("derivatives2", "d_three_terms gets the square root wrong", "return math.exp(x) + 1 / x + 0.5 * x ** -0.5", "return math.exp(x) + 1 / x + 0.5 * x ** 0.5"),
    ("derivatives3", "partial nudges one way", "return (f(v + nudge) - f(v - nudge)) / (2 * h)", "return (f(v + nudge) - f(v)) / h"),
    ("derivatives3", "partial keeps the 0.02 of the example", "return (f(v + nudge) - f(v - nudge)) / (2 * h)", "return (f(v + nudge) - f(v - nudge)) / 0.02"),
    ("derivatives3", "numeric_gradient with a big nudge", "gradient[i] = partial(f, v, i, 0.001)", "gradient[i] = partial(f, v, i, 0.5)"),
    ("derivatives3", "gradient_of_cubes forgets the 3", "return 3 * v ** 2", "return v ** 2"),
    ("derivatives3", "gradient_of_cubes typed in", "return 3 * v ** 2", "return torch.tensor([3.0, 12.0, 0.75])"),
    ("chain1", "d_cube_of_line forgets the inner derivative", "    inner = 2\n", "    inner = 1\n"),
    ("chain1", "d_cube_of_line forgets the power rule's 3", "outer = 3 * u ** 2", "outer = u ** 2"),
    ("chain1", "d_log_of_square leaves out the inner step", "inner = 2 * x\n", "inner = 1\n"),
    ("chain1", "d_three_steps forgets the 3 of 3 * x", "return 2 * b * math.exp(a) * 3", "return 2 * b * math.exp(a)"),
    ("chain2", "partials_of_reuse counts one path for x", "return 2 * x * y, x * x", "return x * y, x * x"),
    ("chain2", "partials_through_a counts one path through a * a", "d_a = 2 * a + 3", "d_a = a + 3"),
    ("chain2", "partials_of_two_paths misses the path through x + y", "return (x + y) + x, x + 1", "return x + y, x + 1"),
    ("chain3", "backward_small swaps s and z", "d_s = d_L * z", "d_s = d_L * s"),
    ("chain3", "backward_reuse counts q once", "d_q = d_L * q + d_L * q", "d_q = d_L * q"),
    ("chain3", "backward_reuse misses the second path of x", "d_x = d_p * y + d_q", "d_x = d_p * y"),
    ("autograd1", "__mul__ adds", 'return Value(self.data * other.data, (self, other), "*")', 'return Value(self.data + other.data, (self, other), "*")'),
    ("autograd1", "__mul__ forgets its children", 'return Value(self.data * other.data, (self, other), "*")', "return Value(self.data * other.data)"),
    ("autograd1", "build in the wrong order", "return a * b + c", "return c + a * b"),
    ("autograd2", "__mul__ sets with =", "self.grad += other.data * out.grad\n            other.grad += self.data * out.grad", "self.grad = other.data * out.grad\n            other.grad = self.data * out.grad"),
    ("autograd2", "__mul__ forgets the gradient handed down", "self.grad += other.data * out.grad", "self.grad += other.data"),
    ("autograd2", "walk_backward in the wrong order", "q.backward_step()\n    p.backward_step()", "p.backward_step()\n    q.backward_step()"),
    ("autograd3", "backward still in forward order", "for value in reversed(order):", "for value in order:"),
    ("autograd3", "topological_order adds a value before its children", "        for child in value.children:\n            visit(child)\n\n        order.append(value)", "        order.append(value)\n\n        for child in value.children:\n            visit(child)"),
    ("autograd3", "topological_order without the seen set", "        if value in seen:\n            return\n", ""),
    ("autograd4", "autograd_slope forgets backward", "    y = f(t)\n    y.backward()\n", "    y = f(t)\n"),
    ("autograd4", "autograd_gradient returns v itself", "    return v.grad\n", "    return v\n"),
    ("autograd4", "second_gradient still adds up", "    t.grad = None\n", ""),
    ("backprop1", "multiply_backward swaps the inputs", "return grad_y * b, grad_y * a", "return grad_y * a, grad_y * b"),
    ("backprop1", "exp_backward forgets grad_y", "return grad_y * torch.exp(x)", "return torch.exp(x)"),
    ("backprop1", "mean_backward forgets to divide", "return torch.full_like(x, grad_m.item() / x.numel())", "return torch.full_like(x, grad_m.item())"),
    ("backprop2", "add_column_backward without keepdim", "return grad_y, grad_y.sum(dim=1, keepdim=True)", "return grad_y, grad_y.sum(dim=1)"),
    ("backprop2", "add_column_backward adds along the wrong dimension", "return grad_y, grad_y.sum(dim=1, keepdim=True)", "return grad_y, grad_y.sum(dim=0, keepdim=True)"),
    ("backprop2", "column_mean_backward divides by the column count", "return grad_s / x.shape[0] * torch.ones_like(x)", "return grad_s / x.shape[1] * torch.ones_like(x)"),
    ("backprop2", "scale_rows_backward still the wrong shape", "return (grad_y * grid).sum(dim=1)", "return grad_y * grid"),
    ("backprop3", "matmul_grad_b arranged the wrong way", "return a.T @ grad_y", "return grad_y.T @ a"),
    ("backprop3", "matvec_grad_v without .T", "return matrix.T @ grad_y", "return matrix @ grad_y"),
    ("backprop3", "matvec_grad_matrix pairs the wrong way", "return grad_y.unsqueeze(1) * v", "return v.unsqueeze(1) * grad_y"),
    ("backprop4", "backward_through_steps forgets exp's local derivative", "grad_u = grad_e * e", "grad_u = grad_e"),
    ("backprop4", "backward_through_steps adds grad_c up along the rows", "grad_c = grad_u.sum(dim=0)", "grad_c = grad_u.sum(dim=1)"),
    ("backprop4", "grad_of_x still without .T", "return grad_z @ w.T", "return grad_z @ w"),
    ("quiz2", "partials_of_exp_product swaps the partials", "return outer * y, outer * x", "return outer * x, outer * y"),
    ("quiz2", "largest_partial takes the largest input", "return v.grad.argmax().item()", "return v.argmax().item()"),
    ("quiz2", "squared_matmul_grad_w forgets the square", "grad_z = grad_y * 2 * z", "grad_z = grad_y"),
    ("quiz2", "shifted_row_sums_grad adds along the rows", "return grad_sum.sum(dim=0)", "return grad_sum.sum(dim=1)"),
    ("micrograd1", "__mul__ without wrapping a number", '        other = other if isinstance(other, Value) else Value(other)\n        out = Value(self.data * other.data, (self, other), "*")', '        out = Value(self.data * other.data, (self, other), "*")'),
    ("micrograd1", "__pow__ forgets the gradient handed down", "self.grad += n * self.data ** (n - 1) * out.grad", "self.grad += n * self.data ** (n - 1)"),
    ("micrograd1", "__pow__ keeps the power as it was", "self.grad += n * self.data ** (n - 1) * out.grad", "self.grad += n * self.data ** n * out.grad"),
    ("micrograd1", "__sub__ adds", "return self + (-other)", "return self + other"),
    ("micrograd1", "__truediv__ divides the wrong way round", "return self * other ** -1", "return other * self ** -1"),
    ("micrograd1", "__neg__ gives the Value back", "return self * -1", "return self"),
    ("micrograd2", "log forgets the gradient handed down", "self.grad += (1 / self.data) * out.grad", "self.grad += 1 / self.data"),
    ("micrograd2", "log divides by its result", "self.grad += (1 / self.data) * out.grad", "self.grad += (1 / out.data) * out.grad"),
    ("micrograd2", "log_sum_exp adds the plain values", "total = total + value.exp()", "total = total + value"),
    ("micrograd3", "the bug in log left in place", "self.grad += (1 / self.data) * out.grad", "self.grad += (1 / out.data) * out.grad"),
    ("micrograd3", "engine_gradients forgets backward", "build(*values).backward()", "build(*values)"),
    ("micrograd3", "numeric_gradients divides by one nudge", "gradients.append((output(up) - output(down)) / 2e-6)", "gradients.append((output(up) - output(down)) / 1e-6)"),
    ("micrograd3", "torch_gradients in single precision", "torch.tensor(number, dtype=torch.float64, requires_grad=True)", "torch.tensor(number, requires_grad=True)"),
    ("linreg1", "predict forgets the bias", "return x @ w + b\n\n\n# (examples", "return x @ w\n\n\n# (examples"),
    ("linreg1", "predict_outputs leaves out the biases", "return x @ W + biases", "return x @ W"),
    ("linreg1", "count_parameters forgets the biases", "return features * outputs + outputs", "return features * outputs"),
    ("linreg1", "predict_rows still crashes", "of @.\ndef predict_rows(\n    x: Tensor,\n    w: Tensor,\n    b: float,\n) -> Tensor:\n    return x @ w + b", "of @.\ndef predict_rows(\n    x: Tensor,\n    w: Tensor,\n    b: float,\n) -> Tensor:\n    return w @ x + b"),
    ("linreg2", "mean_squared_error forgets to square", "return ((predictions - targets) ** 2).mean()\n\n\ndef mean_absolute_error", "return (predictions - targets).mean()\n\n\ndef mean_absolute_error"),
    ("linreg2", "mean_absolute_error keeps the sign", "return (predictions - targets).abs().mean()", "return (predictions - targets).mean()"),
    ("linreg2", "column_model_loss still broadcasts", "predictions = (x @ w + b).squeeze(1)", "predictions = x @ w + b"),
    ("linreg3", "add_ones_column only for 4 rows", "return torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)", "return torch.cat([x, torch.ones(4, 1)], dim=1)"),
    ("linreg3", "add_ones_column adds a row", "return torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)", "return torch.cat([x, torch.ones(1, x.shape[1])], dim=0)"),
    ("linreg3", "best_fit without the column of ones", "solution = torch.linalg.lstsq(with_ones, y).solution", "solution = torch.linalg.lstsq(x, y).solution"),
    ("linreg3", "lowest_loss forgets the bias", "predictions = x @ w + b", "predictions = x @ w"),
    ("descent1", "step goes uphill", "return w - learning_rate * grad_w", "return w + learning_rate * grad_w"),
    ("descent1", "mse_gradients forgets the mean", "grad_errors = 2 * errors / errors.numel()", "grad_errors = 2 * errors"),
    ("descent1", "mse_gradients averages b's gradient", "grad_b = grad_errors.sum()", "grad_b = grad_errors.mean()"),
    ("descent1", "descend never moves b", "b = step(b, grad_b, learning_rate)", "b = b"),
    ("descent2", "train still never clears", "        w.grad = None\n        b.grad = None\n\n    return w.detach(), b.detach()\n\n\ndef train_absolute(", "\n    return w.detach(), b.detach()\n\n\ndef train_absolute("),
    ("descent2", "train_absolute trains on squared error", "loss = (x @ w + b - y).abs().mean()", "loss = ((x @ w + b - y) ** 2).mean()"),
    ("descent2", "loss_history keeps tensors", "history.append(loss.item())", "history.append(loss)"),
    ("descent3", "best_learning_rate takes the largest rate", "    return best_rate", "    return max(rates)"),
    ("descent3", "halve_until_stable only looks for inf and nan", "return not math.isfinite(loss) or loss > start_loss", "return not math.isfinite(loss)"),
    ("descent3", "halve_until_stable lets a nan through", "return not math.isfinite(loss) or loss > start_loss", "return loss > start_loss"),
    ("descent3", "best_learning_rate lets a nan rate win", "    best_rate = None\n    best_loss = math.inf\n\n    for rate in rates:\n        loss = final_loss(x, y, rate, steps)\n\n        if math.isfinite(loss) and loss < best_loss:\n            best_rate = rate\n            best_loss = loss\n\n    return best_rate", "    return min(rates, key=lambda rate: final_loss(x, y, rate, steps))"),
    ("descent4", "column_stats along the rows", "return x.mean(dim=0), x.std(dim=0)", "return x.mean(dim=1), x.std(dim=1)"),
    ("descent4", "standardize_with divides first", "return (x - mean) / spread", "return x - mean / spread"),
    ("descent4", "fit_loss still trains on the raw features", "return final_loss(standardize_with(x, mean, spread), y, 0.1, 100)", "return final_loss(x, y, 0.1, 100)"),
    ("minibatch1", "random_minibatch takes the first rows", "positions = torch.randperm(len(x), generator=generator)[:size]\n\n    return x[positions], y[positions]", "return x[:size], y[:size]"),
    ("minibatch1", "random_minibatch ignores the generator", "positions = torch.randperm(len(x), generator=generator)[:size]\n\n    return x[positions], y[positions]", "positions = torch.randperm(len(x))[:size]\n\n    return x[positions], y[positions]"),
    ("minibatch1", "minibatch_gradients uses every row", "loss = ((x[positions] @ w + b - y[positions]) ** 2).mean()", "loss = ((x @ w + b - y) ** 2).mean()"),
    ("minibatch1", "mean_minibatch_gradient keeps only the last draw", "total = total + grad_w", "total = grad_w * draws"),
    ("minibatch2", "epoch_batches without a shuffle", "    order = torch.randperm(n, generator=generator)\n\n    return [order[start:start + size] for start in range(0, n, size)]", "    order = torch.arange(n)\n\n    return [order[start:start + size] for start in range(0, n, size)]"),
    ("minibatch2", "full_batches keeps the short piece", "range(0, n - size + 1, size)", "range(0, n, size)"),
    ("minibatch2", "train_epochs uses every row every step", "xb = x[positions]\n            yb = y[positions]", "xb = x\n            yb = y"),
    ("minibatch3", "make_loader keeps the short minibatch", "        drop_last=True,\n", ""),
    ("minibatch3", "make_loader without a shuffle", "        shuffle=True,\n        drop_last=True,", "        shuffle=False,\n        drop_last=True,"),
    ("minibatch3", "count_steps always rounds up", "per_epoch = n // size if drop_last else math.ceil(n / size)", "per_epoch = math.ceil(n / size)"),
    ("softmax1", "softmax_rows without keepdim", "return exps / exps.sum(dim=1, keepdim=True)", "return exps / exps.sum(dim=1)"),
    ("softmax1", "softmax_rows over the whole matrix", "return exps / exps.sum(dim=1, keepdim=True)", "return exps / exps.sum()"),
    ("softmax1", "most_likely down the columns", "return scores.argmax(dim=1)", "return scores.argmax(dim=0)"),
    ("softmax2", "softmax_rows still overflows", "exps = torch.exp(scores - scores.amax(dim=1, keepdim=True))", "exps = torch.exp(scores)"),
    ("softmax2", "softmax_rows takes one largest for the whole matrix", "exps = torch.exp(scores - scores.amax(dim=1, keepdim=True))", "exps = torch.exp(scores - scores.max())"),
    ("softmax2", "confidence gives the class", "return torch.softmax(scores, dim=1).amax(dim=1)", "return torch.softmax(scores, dim=1).argmax(dim=1)"),
    ("softmax2", "confidence with max and a dim", "return torch.softmax(scores, dim=1).amax(dim=1)", "return torch.softmax(scores, dim=1).max(dim=1)"),
    ("softmax3", "class_probabilities without softmax", "return torch.softmax(x @ W + b, dim=1)", "return x @ W + b"),
    ("softmax3", "predict_classes down the columns", "return (x @ W + b).argmax(dim=1)", "return (x @ W + b).argmax(dim=0)"),
    ("softmax3", "accuracy still crashes", "return (predicted == labels).float().mean().item()", "return (predicted == labels).mean().item()"),
    ("likelihood1", "label_probabilities for 3 rows only", "return probabilities[torch.arange(len(labels)), labels]", "return probabilities[torch.arange(3), labels]"),
    ("likelihood1", "label_probabilities takes whole columns", "return probabilities[torch.arange(len(labels)), labels]", "return probabilities[:, labels]"),
    ("likelihood1", "log_likelihood still underflows", "return torch.log(picked).sum()", "return torch.log(picked.prod())"),
    ("likelihood1", "negative_log_likelihood keeps the sign", "return -torch.log(label_probabilities(probabilities, labels)).mean()", "return torch.log(label_probabilities(probabilities, labels)).mean()"),
    ("likelihood2", "log_softmax_rows by softmax then log", "return scores - torch.logsumexp(scores, dim=1, keepdim=True)", "return torch.log(torch.softmax(scores, dim=1))"),
    ("likelihood2", "log_softmax_rows without keepdim", "return scores - torch.logsumexp(scores, dim=1, keepdim=True)", "return scores - torch.logsumexp(scores, dim=1)"),
    ("likelihood2", "cross_entropy keeps the sign", "return -picked.mean()", "return picked.mean()"),
    ("likelihood2", "starting_loss typed in for 10 classes", "return math.log(classes)", "return 2.302585092994046"),
    ("likelihood3", "gaussian_nll drops the constant", "return (squared + math.log(sigma * math.sqrt(2 * math.pi))).mean()", "return squared.mean()"),
    ("likelihood3", "nll_from_mse forgets the 2", "return mse / (2 * sigma ** 2) + math.log(sigma * math.sqrt(2 * math.pi))", "return mse / sigma ** 2 + math.log(sigma * math.sqrt(2 * math.pi))"),
    ("likelihood3", "laplace_nll squares the error", "return ((targets - predictions).abs() / b + math.log(2 * b)).mean()", "return ((targets - predictions) ** 2 / b + math.log(2 * b)).mean()"),
    ("likelihood4", "one_hot puts every 1 in column 0", "encoded[torch.arange(len(labels)), labels] = 1.0", "encoded[:, 0] = 1.0"),
    ("likelihood4", "cross_entropy_grad forgets the mean", "return (probabilities - one_hot(labels, logits.shape[1])) / len(labels)", "return probabilities - one_hot(labels, logits.shape[1])"),
    ("likelihood4", "cross_entropy_grad adds the one hot", "return (probabilities - one_hot(labels, logits.shape[1])) / len(labels)", "return (probabilities + one_hot(labels, logits.shape[1])) / len(labels)"),
    ("likelihood4", "softmax_regression_grads arranged the wrong way", "grad_W = x.T @ grad_logits", "grad_W = grad_logits.T @ x"),
    ("splits1", "split_positions without a shuffle", "    order = torch.randperm(n, generator=generator)\n    first_cut", "    order = torch.arange(n)\n    first_cut"),
    ("splits1", "split_rows still cuts without a shuffle", "    train, validation, test = split_positions(len(x), generator)\n\n    return x[train], y[train], x[validation], y[validation], x[test], y[test]", "    first_cut = int(0.8 * len(x))\n    second_cut = int(0.9 * len(x))\n\n    return x[:first_cut], y[:first_cut], x[first_cut:second_cut], y[first_cut:second_cut], x[second_cut:], y[second_cut:]"),
    ("splits1", "class_counts misses classes that never appear", "return torch.bincount(labels, minlength=classes)", "return torch.bincount(labels)"),
    ("overfitting1", "polynomial_features starts at x ** 0", "return torch.stack([x ** power for power in range(1, degree + 1)], dim=1)", "return torch.stack([x ** power for power in range(degree)], dim=1)"),
    ("overfitting1", "polynomial_features stacks rows", "return torch.stack([x ** power for power in range(1, degree + 1)], dim=1)", "return torch.stack([x ** power for power in range(1, degree + 1)], dim=0)"),
    ("overfitting1", "fit_errors measures validation on the training points", "validation_predictions = polynomial_features(validation_x, degree) @ w + b", "validation_predictions = polynomial_features(train_x, degree) @ w + b"),
    ("overfitting1", "best_degree picks on the training error", "errors = [fit_errors(train_x, train_y, validation_x, validation_y, degree)[1] for degree in degrees]", "errors = [fit_errors(train_x, train_y, validation_x, validation_y, degree)[0] for degree in degrees]"),
    ("overfitting2", "train_with_early_stopping keeps w itself", "best_w = w.detach().clone()", "best_w = w"),
    ("overfitting2", "train_with_early_stopping keeps the last step", "best_step = step", "best_step = steps - 1"),
    ("overfitting2", "should_stop counts the best step too", "return len(history) - 1 - best_step > patience", "return len(history) - best_step > patience"),
    ("decay1", "loss_with_decay decays the bias too", "return ((x @ w + b - y) ** 2).mean() + strength * (w ** 2).sum()", "return ((x @ w + b - y) ** 2).mean() + strength * ((w ** 2).sum() + b ** 2)"),
    ("decay1", "decayed_step forgets the 2", "return w - learning_rate * (grad_w + 2 * strength * w)", "return w - learning_rate * (grad_w + strength * w)"),
    ("decay1", "fit_with_decay trains without the decay", "loss = loss_with_decay(x, y, w, b, strength)", "loss = ((x @ w + b - y) ** 2).mean()"),
    ("decay2", "validation_error measures on the training set", "return ((validation_x @ w + b - validation_y) ** 2).mean().item()", "return ((train_x @ w + b - train_y) ** 2).mean().item()"),
    ("decay2", "pick_strength picks the largest error", "return strengths[errors.index(min(errors))]", "return strengths[errors.index(max(errors))]"),
    ("decay2", "test_error measures on the validation set", "return ((test_x @ w + b - test_y) ** 2).mean().item()", "return ((validation_x @ w + b - validation_y) ** 2).mean().item()"),
    ("quiz3", "classifier_loss on probabilities", "return F.cross_entropy(x @ W + b, labels)\n\n\n# The test set", "return F.cross_entropy(torch.softmax(x @ W + b, dim=1), labels)\n\n\n# The test set"),
    ("quiz3", "standardize_both uses the test set's own numbers", "return (train_x - mean) / spread, (test_x - mean) / spread", "return (train_x - mean) / spread, (test_x - test_x.mean(dim=0)) / test_x.std(dim=0)"),
    ("quiz3", "minibatch_step on every row", "loss = ((x[positions] @ w + b - y[positions]) ** 2).mean()", "loss = ((x @ w + b - y) ** 2).mean()"),
    ("quiz3", "decayed_classifier_loss forgets the strength", "return F.cross_entropy(x @ W + b, labels) + strength * (W ** 2).sum()", "return F.cross_entropy(x @ W + b, labels) + (W ** 2).sum()"),
    ("mnist1", "flatten_and_scale forgets to scale", "return images.reshape(len(images), -1).float() / 255", "return images.reshape(len(images), -1).float()"),
    ("mnist1", "train_validation_split without a shuffle", "train, validation = order[:-validation_size], order[-validation_size:]", "train, validation = torch.arange(len(x))[:-validation_size], torch.arange(len(x))[-validation_size:]"),
    ("mnist1", "average_image adds up", "return x[labels == digit].mean(dim=0)", "return x[labels == digit].sum(dim=0)"),
    ("mnist2", "train on the first rows every step", "loss = F.cross_entropy(x[positions] @ W + b, y[positions])", "loss = F.cross_entropy(x[:batch_size] @ W + b, y[:batch_size])"),
    ("mnist2", "accuracy down the columns", "return ((x @ W + b).argmax(dim=1) == y).float().mean().item()", "return ((x @ W + b).argmax(dim=0) == y).float().mean().item()"),
    ("mnist3", "pick_strength picks the lowest accuracy", "return strengths[scores.index(max(scores))]", "return strengths[scores.index(min(scores))]"),
    ("mnist3", "test_accuracy measures on the validation set", "return accuracy(test_x, test_y, W, b)", "return accuracy(validation_x, validation_y, W, b)"),
    ("mnist3", "confusion_matrix swaps rows and columns", "counts[true_class, guess] += 1", "counts[guess, true_class] += 1"),
    ("mnist3", "most_confused counts the diagonal", "    mistakes.fill_diagonal_(0)\n", ""),
    ("init1", "predicted_spread forgets the depth", "return (math.sqrt(n) * weight_spread) ** depth", "return math.sqrt(n) * weight_spread"),
    ("init1", "predicted_spread grows by n", "return (math.sqrt(n) * weight_spread) ** depth", "return (n * weight_spread) ** depth"),
    ("init1", "spreads_through with the weight turned", "x = x @ weight\n", "x = x @ weight.T\n"),
    ("init1", "scaled_weights divides by the width", "/ math.sqrt(width) for _ in range(depth)]", "/ width for _ in range(depth)]"),
    ("init1", "scaled_weights without the generator", "torch.randn(width, width, generator=generator) / math.sqrt(width)", "torch.randn(width, width) / math.sqrt(width)"),
    ("init2", "default_std forgets the sqrt(3)", "return 1 / math.sqrt(fan_in) / math.sqrt(3)", "return 1 / math.sqrt(fan_in)"),
    ("init2", "kaiming_std without the 2", "return math.sqrt(2 / fan_in)", "return math.sqrt(1 / fan_in)"),
    ("init2", "xavier_std with fan_in only", "return math.sqrt(2 / (fan_in + fan_out))", "return math.sqrt(2 / fan_in)"),
    ("init2", "kaiming_init_ leaves the bias", "        layer.weight.normal_(0, kaiming_std(layer.in_features))\n        layer.bias.zero_()", "        layer.weight.normal_(0, kaiming_std(layer.in_features))"),
    ("init2", "kaiming_init_ with the fan out", "layer.weight.normal_(0, kaiming_std(layer.in_features))", "layer.weight.normal_(0, kaiming_std(layer.out_features))"),
    ("init2", "kaiming_init_ outside no_grad", "    with torch.no_grad():\n        layer.weight.normal_(0, kaiming_std(layer.in_features))\n        layer.bias.zero_()", "    layer.weight.normal_(0, kaiming_std(layer.in_features))\n    layer.bias.zero_()"),
    ("init3", "start_loss on probabilities", "return F.cross_entropy(model(x), y).item()", "return F.cross_entropy(torch.softmax(model(x), dim=1), y).item()"),
    ("init3", "start_gap with the number of examples", "return start_loss(model, x, y) - math.log(classes)", "return start_loss(model, x, y) - math.log(len(x))"),
    ("init3", "kaiming_mlp_ without the isinstance check", "        if isinstance(module, nn.Linear):\n            kaiming_init_(module)", "        kaiming_init_(module)"),
    ("init3", "kaiming_mlp_ fills the first layer only", "for module in model.modules():", "for module in list(model.modules())[:2]:"),
    ("init4", "init_std gives tanh Kaiming", "if activation in (\"relu\", \"gelu\"):", "if activation in (\"relu\", \"gelu\", \"tanh\"):"),
    ("init4", "init_std gives none Kaiming", "return 1 / math.sqrt(fan_in)", "return math.sqrt(2 / fan_in)"),
    ("init4", "final_spread forgets the activation", "x = ACTIVATIONS[activation](x @ weight)", "x = x @ weight"),
    ("init4", "final_spread without the generator", "weight = torch.randn(width, width, generator=generator) *", "weight = torch.randn(width, width) *"),
    ("init4", "last_bias without the log", "return (counts / len(y)).log()", "return counts / len(y)"),
    ("dropout1", "keep_mask keeps the small draws", "return torch.rand(shape, generator=generator) >= p", "return torch.rand(shape, generator=generator) < p"),
    ("dropout1", "dropout forgets to scale", "return x * keep / (1 - p)", "return x * keep"),
    ("dropout1", "dropout scales by p", "return x * keep / (1 - p)", "return x * keep / p"),
    ("dropout1", "MyDropout drops in eval mode too", "        if not self.training:\n            return x\n\n", ""),
    ("dropout1", "MyDropout drops the same numbers every call", "keep = torch.rand(x.shape) >= self.p", "keep = torch.rand(x.shape, generator=torch.Generator().manual_seed(0)) >= self.p"),
    ("dropout1", "MyDropout forgets to scale", "return x * keep / (1 - self.p)", "return x * keep"),
    ("dropout2", "train_epochs left unfixed", "        model.train()\n", ""),
    ("dropout2", "evaluate without eval mode", "    model.eval()\n\n    with torch.no_grad():", "    with torch.no_grad():"),
    ("dropout2", "evaluate takes the argmax down the columns", ".argmax(dim=1) == y", ".argmax(dim=0) == y"),
    ("dropout2", "best_epoch counts from 0", "return validation_losses.index(min(validation_losses)) + 1", "return validation_losses.index(min(validation_losses))"),
    ("dropout3", "paper_dropout scales in training", "return x * keep_mask(x.shape, p, generator)\n", "return x * keep_mask(x.shape, p, generator) / (1 - p)\n"),
    ("dropout3", "paper_dropout passes x through at test time", "return x * (1 - p)", "return x"),
    ("dropout3", "average_of_draws adds up", ".mean(dim=0)", ".sum(dim=0)"),
    ("dropout3", "dropout_mlp without the hidden dropout", "layers.extend([nn.Linear(inputs, outputs), nn.ReLU(), nn.Dropout(p_hidden)])", "layers.extend([nn.Linear(inputs, outputs), nn.ReLU()])"),
    ("dropout3", "dropout_mlp drops before the ReLU", "layers.extend([nn.Linear(inputs, outputs), nn.ReLU(), nn.Dropout(p_hidden)])", "layers.extend([nn.Linear(inputs, outputs), nn.Dropout(p_hidden), nn.ReLU()])"),
]


def main() -> int:
    accepted: list[str] = []
    only = set(sys.argv[1:])
    cases = [case for case in CASES if not only or case[0] in only]

    with tempfile.TemporaryDirectory() as folder:
        for number, (name, label, piece, mistake) in enumerate(cases):
            exercise = exercise_named(name)

            # A case whose piece is gone from the solution would test the solution itself.
            if exercise.solution.count(piece) != 1:
                raise SystemExit(f"{name}, {label}: the piece {piece!r} is not in the solution exactly once.")

            path = Path(folder) / str(number) / f"{name}.py"
            path.parent.mkdir()
            path.write_text(exercise.solution.replace(piece, mistake))
            outcome = attempt(
                exercise=exercise,
                path=path,
            )

            print(f"=== {name}, {label}: {'PASSED, which is wrong' if outcome.passed else 'fails as it should'}")

            if outcome.passed:
                accepted.append(f"{name}, {label}")

            if outcome.problem:
                print(f"    {outcome.problem}")

            for result in outcome.results:
                if result.passed:
                    continue

                print(f"  ✗ {result.function}, {result.description}")
                print("    " + result.detail.replace("\n", "\n    "))

    print()

    if accepted:
        print("These wrong answers passed: " + ", ".join(accepted))
        return 1

    print(f"All {len(cases)} wrong answers fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
