"""106_papers, Power and colleagues' grokking paper read into code, then its result reproduced small and judged by its spread across seeds."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    DeviceRecorder,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_on_device,
    expect_tensor,
    expect_value,
    reject_missing_return,
    value_text,
)
from ..text import prose, source_file
from ..training import MissingCheckpoint, best_device, load_checkpoint
from .gpt import GIVEN_MODEL, GPT_FORWARD, GPT_INIT, INIT_SOLUTION

PAPERS = Chapter(
    folder="106_papers",
    title="Papers",
    readme=source_file(
        """
        # Papers

        Most new modules first appear in a research paper. This chapter turns
        one paper into code, Power and colleagues' paper on grokking. Then it
        reproduces the paper's main result at a small scale. It trains with
        several seeds and reports how much the results vary between them.

        ## Further reading

        - [Power and colleagues, grokking](https://arxiv.org/abs/2201.02177)
        - [Nanda and colleagues, progress measures for grokking](https://arxiv.org/abs/2301.05217)
        - [Liu and colleagues, Omnigrok](https://arxiv.org/abs/2210.01117)
        - [Karpathy, a recipe for training neural networks](https://karpathy.github.io/2019/04/25/recipe/)
        """,
    ),
)

# The modulus, the paper's 97, and the two tokens after the numbers.
SETTINGS_PART = '''
P = 97
OP = P
EQUALS = P + 1
'''

START_PART = '''
# gpt2's GPT, set up as the paper's model. The head gets its own weight matrix, as in Nanda
# and colleagues' grokking model. Every layer keeps PyTorch's default initialization.
class PaperGPT(GPT):
    def __init__(
        self,
        vocabulary: int,
        block_size: int,
        size: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__(vocabulary, block_size, size, layers, heads)
        self.lm_head = nn.Linear(size, vocabulary, bias=False)

        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
'''

EQUATIONS_SOLUTION = '''
def equations(p: int) -> tuple[Tensor, Tensor]:
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    inputs = torch.stack([a, torch.full_like(a, p), b, torch.full_like(a, p + 1)], dim=1)

    return inputs, (a + b) % p
'''

SPLIT_SOLUTION = '''
def split_equations(
    count: int,
    fraction: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    order = torch.randperm(count, generator=generator)
    cut = int(fraction * count)

    return order[:cut], order[cut:]
'''

LOSS_SOLUTION = '''
def answer_loss(
    model: nn.Module,
    inputs: Tensor,
    answers: Tensor,
) -> Tensor:
    return F.cross_entropy(model(inputs)[:, -1], answers)


def answer_accuracy(
    model: nn.Module,
    inputs: Tensor,
    answers: Tensor,
) -> float:
    model.eval()

    with torch.no_grad():
        right = (model(inputs)[:, -1].argmax(dim=-1) == answers).float().mean().item()

    model.train()

    return right
'''

OPTIMIZER_SOLUTION = '''
def paper_optimizer(
    model: nn.Module,
    weight_decay: float,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 10))

    return optimizer, scheduler
'''

COUNT_SOLUTION = '''
def non_embedding_parameters(model: nn.Module) -> int:
    inside = [*model.h.parameters(), *model.ln_f.parameters()]

    return sum(parameter.numel() for parameter in inside)
'''

PAPER1_PARTS = EQUATIONS_SOLUTION + SPLIT_SOLUTION + LOSS_SOLUTION + OPTIMIZER_SOLUTION + COUNT_SOLUTION

GROK_SOLUTION = '''
def grok(
    model: nn.Module,
    inputs: Tensor,
    answers: Tensor,
    splits: tuple[Tensor, Tensor],
    weight_decay: float,
    steps: int,
    every: int,
    stop_at: float | None,
    generator: torch.Generator,
    device: torch.device,
) -> list[tuple[int, float, float]]:
    train, validation = splits
    inputs, answers = inputs.to(device), answers.to(device)
    optimizer, scheduler = paper_optimizer(model, weight_decay)
    batch = min(512, len(train) // 2)
    curve = []

    for step in range(1, steps + 1):
        picked = train[torch.randint(0, len(train), (batch,), generator=generator)]
        loss = answer_loss(model, inputs[picked], answers[picked])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % every != 0:
            continue

        trained = answer_accuracy(model, inputs[train], answers[train])
        held_out = answer_accuracy(model, inputs[validation], answers[validation])
        curve.append((step, trained, held_out))

        if stop_at is not None and held_out >= stop_at:
            break

    return curve
'''

FIRST_STEP_SOLUTION = '''
def first_step(
    curve: list[tuple[int, float, float]],
    column: int,
    threshold: float,
) -> int | None:
    for point in curve:
        if point[column] >= threshold:
            return point[0]

    return None
'''

SPREAD_SOLUTION = '''
def mean_and_spread(values: list[float]) -> tuple[float, float]:
    numbers = torch.tensor(values, dtype=torch.float64)

    return numbers.mean().item(), numbers.std().item()
'''

PAPER2_PARTS = GROK_SOLUTION + FIRST_STEP_SOLUTION + SPREAD_SOLUTION


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
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


MODEL_PARTS = GIVEN_MODEL + INIT_SOLUTION + GPT_INIT + GPT_FORWARD
REFERENCE = namespace_of(MODEL_PARTS + SETTINGS_PART + START_PART + PAPER1_PARTS + PAPER2_PARTS)


def paper_model(seed: int) -> nn.Module:
    """The paper's model, PaperGPT at 2 layers, width 128 and 4 heads, on mod 97's 99 tokens."""
    torch.manual_seed(seed)

    return REFERENCE["PaperGPT"](99, 4, 128, 2, 4)


def small_model(seed: int) -> nn.Module:
    """A PaperGPT small enough for a check, on mod 5's 7 tokens."""
    torch.manual_seed(seed)

    return REFERENCE["PaperGPT"](7, 4, 16, 1, 2)


# ---------------------------------------------------------------------------
# paper1
# ---------------------------------------------------------------------------

PAPER1_LESSON = prose(
    """
    Most new modules first appear in a paper, so building your own module
    often starts with turning a paper into code. Three habits do most of the
    work.

    - Shapes: list every symbol the method uses, with its shape.
    - Lines: turn each sentence of the setup into one line of code.
    - Counts: count the model's parameters, and compare the count with the paper's number.

    Power and colleagues' grokking paper is a good example. Grokking is their
    name for a model that gets every training example right early, but gets
    the held out examples right only much later. The data is the full table
    of an operation such as a + b mod 97, which is `(a + b) % 97` in Python.
    Every number is a separate token, so the model never sees digits. It
    learns what each number means only from how the numbers combine.

    - Equations: each equation is four tokens, a, then the operation token, then b, then =. With p numbers, the vocabulary has p + 2 tokens.
    - Data: every pair of numbers makes one equation, p squared in all. A random fraction is used for training, and the rest is held out.
    - Model: a decoder only transformer with 2 layers, width 128 and 4 heads. It has about 400,000 parameters outside the embeddings.
    - Loss: cross entropy on the answer only, computed from the scores at the = token. The question tokens are never scored.
    - Optimizer: AdamW with learning rate 0.001, betas 0.9 and 0.98, and weight decay 1. The learning rate warms up over the first 10 updates.
    - Batches: 512 equations, or half the training set when that is smaller.

    gpt2's GPT with 2 layers, width 128 and 4 heads is nearly that model. A
    weight decay of 1 is ten times gpt3's 0.1. paper2 shows what such strong
    weight decay does.

    The paper never says how its weights are initialized, or whether its
    output head shares the embedding table `wte`. Its released code gives the
    head its own weight matrix and keeps PyTorch's default initialization for
    every layer. The given `PaperGPT` does the same, and so does Nanda and
    colleagues' grokking model. PyTorch's default initial weights are larger
    than gpt2's, which have a standard deviation of 0.02. paper2 shows what
    gpt2's two choices would change.

    The warmup needs a learning rate scheduler that schedule1 did not show.
    `LambdaLR(optimizer, f)` sets the learning rate to the starting rate times
    `f(n)`. Here n is the number of calls to `scheduler.step()` so far,
    starting from 0.

    The parameter count leaves out the token and position embedding tables
    and the head. It counts the transformer blocks and the final LayerNorm,
    the part of the model that grows with its depth.
    """,
)

PAPER1_SOURCE = source_file(
    '''
    # paper1.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    firsts = torch.arange(3).repeat_interleave(3)       # tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    seconds = torch.arange(3).repeat(3)                 # tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    sevens = torch.full_like(torch.arange(3), 7)        # tensor([7, 7, 7])
    shuffled = torch.randperm(5, generator=torch.Generator().manual_seed(0))    # tensor([4, 0, 1, 3, 2])
    last = torch.zeros(2, 4, 99)[:, -1].shape           # torch.Size([2, 99]), the scores at the last token
    warmup = [min(1.0, (step + 1) / 10) for step in (0, 1, 9, 20)]    # [0.1, 0.2, 1.0, 1.0]
    sgd = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    halved = torch.optim.lr_scheduler.LambdaLR(sgd, lambda step: 0.5)    # sgd's learning rate is now 0.1 * 0.5 = 0.05


    # SelfAttention, GPT2MLP and GPT2Block from block2, and init_weights,
    # scale_residuals_ and GPT from gpt2.
''' + indented(MODEL_PARTS) + '''

''' + indented(SETTINGS_PART) + '''

''' + indented(START_PART) + '''


    # TODO: Return every equation of a + b mod p, as two integer tensors.
    #   inputs has one row per pair (a, b), holding a, OP, b, EQUALS, where OP
    #   is p and EQUALS is p + 1. The rows go through every pair with a
    #   changing slowest, as firsts and seconds show. answers holds
    #   (a + b) % p for each row.
    #   Example: equations(2) returns tensor([[0, 2, 0, 3], [0, 2, 1, 3],
    #   [1, 2, 0, 3], [1, 2, 1, 3]]) and tensor([0, 1, 1, 0]).
    def equations(p: int) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Shuffle the numbers 0 to count - 1 with torch.randperm and
    #   generator. Return the first int(fraction * count) of the shuffled
    #   numbers as the training row indices, and the rest as the validation
    #   row indices.
    #   Example: split_equations(5, 0.4, torch.Generator().manual_seed(0))
    #   returns (tensor([4, 0]), tensor([1, 3, 2])).
    def split_equations(
        count: int,
        fraction: float,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return the cross entropy between the model's scores at the last
    #   position, the = token, and answers, as a tensor. model(inputs) has
    #   shape (batch, 4, vocabulary) and answers has shape (batch,).
    def answer_loss(
        model: nn.Module,
        inputs: Tensor,
        answers: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the fraction of answers the model gets right, as a Python
    #   float. The model's answer is the token with the highest score at the
    #   last position. Call model.eval() and score inside torch.no_grad().
    #   Then call model.train() to put the model back in training mode.
    def answer_accuracy(
        model: nn.Module,
        inputs: Tensor,
        answers: Tensor,
    ) -> float:
        ...


    # TODO: Return the paper's optimizer and scheduler. The optimizer is AdamW
    #   over every parameter of model, with learning rate 0.001, betas 0.9 and
    #   0.98, and the given weight_decay. The scheduler is a LambdaLR that
    #   multiplies the learning rate by min(1.0, (step + 1) / 10), which warms
    #   it up over the first 10 updates.
    def paper_optimizer(
        model: nn.Module,
        weight_decay: float,
    ) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
        ...


    # TODO: Return the total number of entries in the parameters of model.h
    #   and model.ln_f. These are all the parameters outside the embedding
    #   tables and the head.
    def non_embedding_parameters(model: nn.Module) -> int:
        ...


    counted = non_embedding_parameters(PaperGPT(P + 2, 4, 128, 2, 4))
    ''',
)

PAPER1_SOLUTION = source_file(
    '''
    # paper1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    firsts = torch.arange(3).repeat_interleave(3)
    seconds = torch.arange(3).repeat(3)
    sevens = torch.full_like(torch.arange(3), 7)
    shuffled = torch.randperm(5, generator=torch.Generator().manual_seed(0))
    last = torch.zeros(2, 4, 99)[:, -1].shape
    warmup = [min(1.0, (step + 1) / 10) for step in (0, 1, 9, 20)]
    sgd = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    halved = torch.optim.lr_scheduler.LambdaLR(sgd, lambda step: 0.5)


''' + indented(MODEL_PARTS) + '''

''' + indented(SETTINGS_PART) + '''

''' + indented(START_PART) + '''


''' + indented(EQUATIONS_SOLUTION) + '''


''' + indented(SPLIT_SOLUTION) + '''


''' + indented(LOSS_SOLUTION) + '''


''' + indented(OPTIMIZER_SOLUTION) + '''


''' + indented(COUNT_SOLUTION) + '''


    counted = non_embedding_parameters(PaperGPT(P + 2, 4, 128, 2, 4))
    ''',
)


def expect_whole_numbers(
    call: str,
    actual: Tensor,
) -> None:
    if actual.dtype not in (torch.int64, torch.int32):
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but tokens and answers are whole numbers, torch.long, for an embedding and cross entropy.")


def expect_tensor_pair(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be two tensors.")

    for name, got, wanted in zip(("the first tensor", "the second tensor"), actual, expected):
        expect_tensor(
            call=f"{name} of {call}",
            actual=got,
            expected=wanted,
        )
        expect_whole_numbers(
            call=f"{name} of {call}",
            actual=got,
        )


class LastTokenScores(nn.Module):
    """A stand in model whose scores differ at every place, so a loss read at the wrong place shows."""

    def __init__(
        self,
        vocabulary: int,
        seed: int,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.table = nn.Parameter(torch.randn(4, vocabulary, generator=generator))
        self.mixing = nn.Parameter(torch.randn(vocabulary, vocabulary, generator=generator) / 3)

    def forward(self, ids: Tensor) -> Tensor:
        return F.one_hot(ids, self.mixing.shape[0]).float() @ self.mixing + self.table


def scored_inputs(seed: int) -> tuple[Tensor, Tensor]:
    inputs, answers = REFERENCE["equations"](5)
    rows = torch.randperm(25, generator=torch.Generator().manual_seed(seed))[:9]

    return inputs[rows], answers[rows]


def check_answer_loss(found: object) -> None:
    for seed in (720, 721):
        inputs, answers = scored_inputs(seed)
        actual = found(LastTokenScores(7, seed), inputs, answers)
        expected = REFERENCE["answer_loss"](LastTokenScores(7, seed), inputs, answers)
        reject_missing_return(
            call="answer_loss",
            actual=actual,
        )

        if not isinstance(actual, Tensor):
            raise Mismatch(f"answer_loss gives {value_text(actual)}, but it should give the loss as a tensor, so it can be backpropagated.")

        expect_close(
            call=f"answer_loss on 9 equations of mod 5, seed {seed}",
            actual=actual.item(),
            expected=expected.item(),
            tolerance=1e-5,
        )


def check_answer_accuracy(found: object) -> None:
    for seed in (722, 723):
        inputs, answers = scored_inputs(seed)
        model = small_model(seed)
        model.train()
        modes: list[bool] = []
        recording: list[bool] = []

        def note(module: nn.Module, arguments: tuple[object, ...]) -> None:
            modes.append(module.training)
            recording.append(torch.is_grad_enabled())

        hook = model.register_forward_pre_hook(note)
        actual = found(model, inputs, answers)
        hook.remove()

        if any(modes):
            raise Mismatch("answer_accuracy scored the model in training mode. Call model.eval() first.")

        if any(recording):
            raise Mismatch("answer_accuracy recorded for autograd. Score inside torch.no_grad().")

        if not model.training:
            raise Mismatch("answer_accuracy left the model in eval mode. Call model.train() before returning, so training goes on as it was.")

        expect_close(
            call=f"answer_accuracy of an untrained GPT on 9 equations of mod 5, seed {seed}",
            actual=actual,
            expected=REFERENCE["answer_accuracy"](small_model(seed), inputs, answers),
        )


def check_paper_optimizer(found: object) -> None:
    for weight_decay in (1.0, 0.25):
        model = small_model(724)
        actual = found(model, weight_decay)
        reject_missing_return(
            call="paper_optimizer",
            actual=actual,
        )

        if not isinstance(actual, (tuple, list)) or len(actual) != 2:
            raise Mismatch(f"paper_optimizer gives {value_text(actual)}, but it should give the optimizer and the scheduler.")

        optimizer, scheduler = actual

        if type(optimizer) is not torch.optim.AdamW:
            raise Mismatch(f"paper_optimizer's optimizer is {type(optimizer).__name__}, but the paper uses AdamW, whose weight decay stays out of Adam's averages.")

        settings = optimizer.param_groups[0]

        if tuple(settings["betas"]) != (0.9, 0.98):
            raise Mismatch(f"paper_optimizer's betas are {tuple(settings['betas'])}, but the paper uses (0.9, 0.98).")

        if settings["weight_decay"] != weight_decay:
            raise Mismatch(f"paper_optimizer({weight_decay}) has a weight decay of {settings['weight_decay']}. Pass weight_decay through.")

        if sum(len(group["params"]) for group in optimizer.param_groups) != len(list(model.parameters())):
            raise Mismatch("paper_optimizer's AdamW doesn't train every parameter of the model.")

        rates = []

        for _ in range(12):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        expected = [1e-3 * min(1.0, (step + 1) / 10) for step in range(12)]

        for step, (got, wanted) in enumerate(zip(rates, expected)):
            if not math.isclose(got, wanted, rel_tol=1e-9):
                raise Mismatch(f"Before update {step + 1} the rate is {got:.6g}, but it should be {wanted:.6g}, rising linearly to 0.001 over the first 10 updates.")


def check_counted(value: object) -> None:
    expect_value(
        call="counted",
        actual=value,
        expected=REFERENCE["non_embedding_parameters"](REFERENCE["PaperGPT"](99, 4, 128, 2, 4)),
    )


def check_paper1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="equations",
        description="writes every equation of a + b mod p and its answer",
        inputs=((3,), (5,)),
        reference=REFERENCE["equations"],
        expect=expect_tensor_pair,
    )
    check_on_inputs(
        checker=checker,
        function="split_equations",
        description="shuffles the rows and cuts off the training fraction",
        inputs=(
            (10, 0.5, Described(torch.Generator().manual_seed(7), "a generator seeded with 7")),
            (9, 0.3, Described(torch.Generator().manual_seed(8), "a generator seeded with 8")),
        ),
        reference=REFERENCE["split_equations"],
        expect=expect_tensor_pair,
    )
    checker.test(
        function="answer_loss",
        description="scores only the answer, at the = token",
        run=check_answer_loss,
    )
    checker.test(
        function="answer_accuracy",
        description="counts right answers in eval mode, without autograd",
        run=check_answer_accuracy,
    )
    checker.test(
        function="paper_optimizer",
        description="builds the paper's AdamW and its warmup",
        run=check_paper_optimizer,
    )
    check_on_inputs(
        checker=checker,
        function="non_embedding_parameters",
        description="counts the parameters outside the embedding tables and the head",
        inputs=(
            (Described(REFERENCE["PaperGPT"](99, 4, 128, 2, 4), "PaperGPT(99, 4, 128, 2, 4)"),),
            (Described(REFERENCE["GPT"](20, 8, 32, 3, 4), "GPT(20, 8, 32, 3, 4)"),),
        ),
        reference=REFERENCE["non_embedding_parameters"],
        expect=expect_value,
    )
    checker.check_name(
        name="counted",
        description="the paper's model's parameter count outside its embeddings",
        run=check_counted,
    )


PAPER1 = Exercise(
    name="paper1",
    chapter=PAPERS,
    title="A paper read into code",
    lesson=PAPER1_LESSON,
    source=PAPER1_SOURCE,
    solution=PAPER1_SOLUTION,
    hints={
        "equations": "The firsts and seconds lines in the file show how to list every pair of a and b. Stack a, a "
        "tensor full of p, b and a tensor full of p + 1 along dim=1. The answers are the sums mod p.",
        "split_equations": "The shuffled line in the file shows the shuffle. Slice the shuffled numbers at "
        "int(fraction * count) to get the two parts.",
        "answer_loss": "model(inputs) has shape (batch, 4, vocabulary). The line near the top of the file that sets "
        "last shows how to pick the scores at the last position.",
        "answer_accuracy": "Take the argmax over the last dimension of the scores at the last position to get each "
        "answer. Compare with answers, convert to float, take the mean and call .item().",
        "paper_optimizer": "torch.optim.AdamW takes lr, betas and weight_decay as keyword arguments. The warmup line "
        "in the file holds the function that LambdaLR needs.",
        "non_embedding_parameters": "Add up .numel() over model.h.parameters() and model.ln_f.parameters(). "
        "PaperGPT's head has its own weight, so model.parameters() would count it too.",
        "counted": "It comes from non_embedding_parameters. If it fails while the function passes, rerun the file.",
    },
    introduces=(),
    check=check_paper1,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# paper2
# ---------------------------------------------------------------------------

# The fraction and the steps behind paper2's table and main.
RUN_PART = '''
FRACTION = 0.3
STEPS = 6000
EVERY = 100
'''

# What measure_grokking gave, the step each of 5 seeds first passed 99%, on training and on validation.
MEASURED_RUNS = '''
TRAIN_STEPS = [300, 300, 300, 300, 300]
VALIDATION_STEPS = [4600, 3600, 3200, 3700, 4900]
'''

PAPER2_MAIN = '''
    # Run by .venv/bin/python run.py train paper2, never by the app. It trains the paper's
    # model on seed 0 until it generalizes, prints its curve, then saves it for the check.
    def main() -> None:
        inputs, answers = equations(P)
        splits = split_equations(len(inputs), FRACTION, torch.Generator().manual_seed(0))
        device = best_device()
        torch.manual_seed(0)
        model = PaperGPT(P + 2, 4, 128, 2, 4).to(device)
        curve = grok(model, inputs, answers, splits, 1.0, STEPS, EVERY, 0.99, torch.Generator().manual_seed(1), device)

        for step, trained, held_out in curve:
            print(f"step {step}, training {trained:.3f}, validation {held_out:.3f}")

        print(f"Training passed 99% at step {first_step(curve, 1, 0.99)}, validation at {first_step(curve, 2, 0.99)}.")
        save_checkpoint(model, "paper2")
'''

PAPER2_LESSON = prose(
    """
    A paper's result is a claim about training runs, so you check it with
    training runs of your own. Power and colleagues claim three things.

    - Grokking: training accuracy reaches 100% early, and validation accuracy catches up only much later.
    - Weight decay: of every trick they tried, it helped generalization the most.
    - Data: the smaller the fraction of the table used for training, the longer validation accuracy takes to jump.

    `grok` runs the training. main calls it on seed 0 and trains on 30% of
    the table, the fraction Nanda and colleagues used. Here are the training
    and validation accuracy at several updates.

    | Update | 100 | 300 | 1,000 | 2,000 | 3,000 | 4,000 | 4,200 | 4,400 | 4,600 |
    |---|---|---|---|---|---|---|---|---|---|
    | Training | 19% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
    | Validation | 0.1% | 5% | 13% | 15% | 21% | 36% | 59% | 95% | 99% |

    By update 300 the model gets every training equation right and almost
    none of the others. For thousands of updates little changes. Then
    validation accuracy jumps from 36% to 99% in 600 updates. Training
    accuracy sometimes dips and recovers, so main stops at the first point
    where validation accuracy passes 99%.

    One run is only an anecdote. The next table shows 5 seeds, with the
    update where each one first passed 99%. Seed 0 is the run above.

    | Seed | 0 | 1 | 2 | 3 | 4 |
    |---|---|---|---|---|---|
    | Training | 300 | 300 | 300 | 300 | 300 |
    | Validation | 4,600 | 3,600 | 3,200 | 3,700 | 4,900 |

    `mean_and_spread` summarizes them with their mean and their standard
    deviation. The standard deviation measures how far the values typically
    are from their mean. Validation passes 99% at update 4,000 on average,
    with a standard deviation of 718. That is 13 times as late as training.
    A run on a GPU does not repeat exactly. Seed 0 has passed 99% at 4,400,
    4,600 and 4,800 in three runs, so yours may finish a few hundred updates
    earlier or later.

    Next, what makes the jump come earlier or later? This table shows the
    update where validation accuracy first passed 99% in each kind of run.

    | Run | Seeds | Validation passes 99% at |
    |---|---|---|
    | 30% of the table, as above | 5 | 3,200 to 4,900 |
    | 50% | 5 | 1,300 to 1,500 |
    | 70% | 3 | 700 to 1,200 |
    | 30%, gpt3's weight decay of 0.1 | 3 | not in 6,000 updates, at 3% to 14% |
    | 30%, no weight decay | 3 | not in 6,000 updates, at 3% to 9% |
    | 30%, gpt2's 0.02 initialization, own head | 5 | 2,500 to 3,400 |
    | 30%, gpt2's 0.02 initialization, tied head | 5 | 1,200 to 2,700 |
    | 50%, gpt2's 0.02 initialization, tied head | 3 | 300, together with training |

    All three claims hold. With gpt3's weight decay of 0.1, the jump does not
    come within 6,000 updates. In the paper's first figure, with no weight
    decay at all, it came after close to a million updates.

    The last three rows add a fourth finding. Liu and colleagues' Omnigrok
    traced grokking to weights that are too large overall, measured by their
    norm, for the solution that generalizes. Weight decay shrinks the
    weights, and a smaller initialization starts them closer to the right
    size. gpt2's 0.02 initialization alone brings every seed's jump earlier,
    by 500 to 1,600 updates. Tying the head to `wte` as well brings every
    seed's jump earlier still. At 50% of the table, it removes the wait
    entirely.

    `run.py train paper2` runs main. The check loads your saved model and
    requires it to get nearly every held out equation right.
    """,
)

PAPER2_SOURCE = source_file(
    '''
    # paper2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.training import best_device, save_checkpoint

    pairs = [(100, 0.2, 0.0), (200, 0.6, 0.1)]
    found = next(point[0] for point in pairs if point[1] >= 0.5)    # 200, the step of the first point whose entry 1 reaches 0.5
    values = torch.tensor([3.0, 5.0], dtype=torch.float64)
    center, wobble = values.mean().item(), values.std().item()     # 4.0 and 1.414..., the mean and the standard deviation that divides by n - 1


    # The model from gpt2 and every function from paper1.
''' + indented(MODEL_PARTS) + '''

''' + indented(SETTINGS_PART) + '''

''' + indented(START_PART) + '''

''' + indented(PAPER1_PARTS) + '''

''' + indented(RUN_PART) + '''


    # TODO: Train model as the paper does, for up to `steps` updates, and
    #   return its curve. splits holds the training and the validation row
    #   indices. Move inputs and answers to device first. Make the optimizer
    #   and scheduler with paper_optimizer and weight_decay. Each update draws
    #   a batch of min(512, len(train) // 2) training rows at random, using
    #   torch.randint with generator. It computes answer_loss on them, zeroes
    #   the old gradients, backpropagates, steps the optimizer and steps the
    #   scheduler. Count updates from 1. After every `every` updates, append
    #   (step, training accuracy, validation accuracy) to the curve, each
    #   accuracy from answer_accuracy. If stop_at is not None, stop right
    #   after the first point whose validation accuracy reaches stop_at.
    def grok(
        model: nn.Module,
        inputs: Tensor,
        answers: Tensor,
        splits: tuple[Tensor, Tensor],
        weight_decay: float,
        steps: int,
        every: int,
        stop_at: float | None,
        generator: torch.Generator,
        device: torch.device,
    ) -> list[tuple[int, float, float]]:
        ...


    # TODO: Return the step of the first point in curve whose entry at index
    #   column is at least threshold, or None if no point reaches it. Column 1
    #   is the training accuracy and column 2 is the validation accuracy.
    #   Example: first_step([(100, 0.2, 0.0), (200, 0.995, 0.1)], 1, 0.99)
    #   returns 200.
    def first_step(
        curve: list[tuple[int, float, float]],
        column: int,
        threshold: float,
    ) -> int | None:
        ...


    # TODO: Return the mean of values and their standard deviation, as two
    #   Python floats. Use the standard deviation that divides by n - 1, which
    #   is what torch's .std() computes.
    #   Example: mean_and_spread([3.0, 5.0]) returns (4.0, 1.4142135623730951).
    def mean_and_spread(values: list[float]) -> tuple[float, float]:
        ...


''' + indented(MEASURED_RUNS) + '''
    memorized = mean_and_spread(TRAIN_STEPS)
    generalized = mean_and_spread(VALIDATION_STEPS)

''' + PAPER2_MAIN,
)

PAPER2_SOLUTION = source_file(
    '''
    # paper2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.training import best_device, save_checkpoint

    pairs = [(100, 0.2, 0.0), (200, 0.6, 0.1)]
    found = next(point[0] for point in pairs if point[1] >= 0.5)
    values = torch.tensor([3.0, 5.0], dtype=torch.float64)
    center, wobble = values.mean().item(), values.std().item()


''' + indented(MODEL_PARTS) + '''

''' + indented(SETTINGS_PART) + '''

''' + indented(START_PART) + '''

''' + indented(PAPER1_PARTS) + '''

''' + indented(RUN_PART) + '''


''' + indented(GROK_SOLUTION) + '''


''' + indented(FIRST_STEP_SOLUTION) + '''


''' + indented(SPREAD_SOLUTION) + '''


''' + indented(MEASURED_RUNS) + '''
    memorized = mean_and_spread(TRAIN_STEPS)
    generalized = mean_and_spread(VALIDATION_STEPS)

''' + PAPER2_MAIN,
)

RUNS = namespace_of(MEASURED_RUNS + RUN_PART)


def expect_pair(
    call: str,
    actual: object,
    expected: tuple[float, float],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be two plain numbers, the mean and the standard deviation.")

    for name, got, wanted in zip(("the mean", "the standard deviation"), actual, expected):
        expect_close(
            call=f"{name} from {call}",
            actual=got,
            expected=wanted,
            tolerance=1e-6,
        )


def expect_step(
    call: str,
    actual: object,
    expected: int | None,
) -> None:
    """first_step's answer, where None is a right answer when no point reaches the threshold."""
    if expected is None:
        if actual is not None:
            raise Mismatch(f"{call} is {value_text(actual)}, but no point reaches the threshold, so it should be None.")

        return

    expect_value(
        call=call,
        actual=actual,
        expected=expected,
    )


def expect_curve(
    call: str,
    actual: object,
    expected: list[tuple[int, float, float]],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, list) or len(actual) != len(expected):
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a list of {len(expected)} points, one for each time it measures the accuracies.")

    for got, wanted in zip(actual, expected):
        if not isinstance(got, (tuple, list)) or len(got) != 3:
            raise Mismatch(f"{call} holds {value_text(got)}, but each point should be (step, training accuracy, validation accuracy).")

        expect_value(
            call=f"the step of {call}'s point {wanted[0]}",
            actual=got[0],
            expected=wanted[0],
        )

        for name, have, want in zip(("training", "validation"), got[1:], wanted[1:]):
            expect_close(
                call=f"the {name} accuracy at step {wanted[0]} of {call}",
                actual=have,
                expected=want,
                tolerance=1e-6,
            )


# Seed, weight decay, steps, every and stop_at. The second run stops early, at a validation accuracy it reaches.
GROK_CASES = ((730, 1.0, 60, 20, None), (731, 0.0, 60, 15, 0.3))


def small_run(seed: int) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
    inputs, answers = REFERENCE["equations"](5)
    splits = REFERENCE["split_equations"](25, 0.6, torch.Generator().manual_seed(seed))

    return inputs, answers, splits


def check_grok_curve(found: object) -> None:
    for seed, weight_decay, steps, every, stop_at in GROK_CASES:
        inputs, answers, splits = small_run(seed)
        model = small_model(seed)
        reference_model = small_model(seed)
        call = f"grok of a small GPT on mod 5, weight decay {weight_decay}, {steps} steps, every {every}, stop_at {stop_at}"
        actual = found(model, inputs, answers, splits, weight_decay, steps, every, stop_at, torch.Generator().manual_seed(seed + 1), torch.device("cpu"))
        expected = REFERENCE["grok"](reference_model, inputs, answers, splits, weight_decay, steps, every, stop_at, torch.Generator().manual_seed(seed + 1), torch.device("cpu"))
        expect_curve(
            call=call,
            actual=actual,
            expected=expected,
        )

        # Accuracies move in coarse steps, so the trained weights show a mistake in the updates the curve can hide.
        for name, weight in reference_model.named_parameters():
            expect_tensor(
                call=f"the model's {name} after {call}",
                actual=dict(model.named_parameters())[name].detach(),
                expected=weight.detach(),
            )


def check_grok_device(found: object) -> None:
    inputs, answers, splits = small_run(732)

    def scores(ids: Tensor) -> Tensor:
        return torch.zeros(ids.shape[0], ids.shape[1], 7, device="meta")

    recorder = DeviceRecorder(("inputs",), scores)
    expect_on_device(
        function="grok",
        run=lambda: found(recorder, inputs, answers, splits, 1.0, 2, 1, None, torch.Generator().manual_seed(733), torch.device("meta")),
        recorders=[recorder],
        advice="Move inputs and answers to device before the loop.",
    )


def check_grokked(found: object) -> None:
    model = paper_model(0)

    try:
        load_checkpoint(model, "paper2")
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None

    inputs, answers = REFERENCE["equations"](97)
    _, validation = REFERENCE["split_equations"](len(inputs), RUNS["FRACTION"], torch.Generator().manual_seed(0))
    accuracy = REFERENCE["answer_accuracy"](model, inputs[validation], answers[validation])

    # main stops once validation passes 99% on the GPU, and the CPU's rounding can move that a hair.
    if accuracy < 0.98:
        raise Mismatch(f"Your trained model gets {accuracy:.1%} of the held out equations right, but a model that grokked gets nearly all of them. Check grok against its other tests, then train again with run.py train paper2.")


def check_measured(name: str, steps: str) -> Callable[[object], None]:
    def check(value: object) -> None:
        expect_pair(
            call=name,
            actual=value,
            expected=REFERENCE["mean_and_spread"](RUNS[steps]),
        )

    return check


def check_paper2(checker: Checker) -> None:
    checker.test(
        function="grok",
        description="trains as the paper does and records the curve",
        run=check_grok_curve,
    )
    checker.test(
        function="grok",
        description="moves the equations to the model's device",
        run=check_grok_device,
    )
    check_on_inputs(
        checker=checker,
        function="first_step",
        description="finds the first step past the threshold, or None",
        inputs=(
            ([(100, 0.2, 0.0), (200, 0.995, 0.1), (300, 1.0, 0.99)], 1, 0.99),
            ([(100, 0.2, 0.0), (200, 0.995, 0.1), (300, 1.0, 0.99)], 2, 0.99),
            ([(50, 1.0, 0.3), (100, 1.0, 0.5)], 2, 0.99),
        ),
        reference=REFERENCE["first_step"],
        expect=expect_step,
    )
    check_on_inputs(
        checker=checker,
        function="mean_and_spread",
        description="gives the mean and the standard deviation that divides by n - 1",
        inputs=(
            ([3.0, 5.0],),
            ([400.0, 500.0, 700.0],),
        ),
        reference=REFERENCE["mean_and_spread"],
        expect=expect_pair,
    )
    checker.check_name(
        name="memorized",
        description="when the 5 seeds first passed 99% on training",
        run=check_measured("memorized", "TRAIN_STEPS"),
    )
    checker.check_name(
        name="generalized",
        description="when the 5 seeds first passed 99% on validation",
        run=check_measured("generalized", "VALIDATION_STEPS"),
    )
    checker.test(
        function="grok",
        description="trained by main, gets at least 98% of the held out equations right",
        run=check_grokked,
    )


PAPER2 = Exercise(
    name="paper2",
    chapter=PAPERS,
    title="A result reproduced small",
    lesson=PAPER2_LESSON,
    source=PAPER2_SOURCE,
    solution=PAPER2_SOLUTION,
    hints={
        "grok": "Unpack splits into train and validation. In the loop, index train with torch.randint(0, len(train), "
        "(batch,), generator=generator) to pick rows, then use inputs and answers at those rows.",
        "first_step": "Walk the points in order and return point[0] as soon as point[column] reaches threshold.",
        "mean_and_spread": "The line in the file that sets center and wobble does this for a tensor. Make a tensor "
        "from values with dtype=torch.float64 and return both numbers.",
        "memorized": "It comes from mean_and_spread. If it fails while the function passes, rerun the file.",
        "generalized": "It comes from mean_and_spread. If it fails while the function passes, rerun the file.",
    },
    introduces=(),
    check=check_paper2,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def grok_until(
    model: nn.Module,
    fraction: float,
    weight_decay: float,
    steps: int,
    seed: int,
    device: torch.device,
) -> list[tuple[int, float, float]]:
    """main's run of grok, on the data split with seed 0 and batches drawn with seed + 1, stopped once validation
    passes 99%. Stopping changes nothing before that point, as the batches are drawn the same way."""
    inputs, answers = REFERENCE["equations"](97)
    splits = REFERENCE["split_equations"](len(inputs), fraction, torch.Generator().manual_seed(0))

    return REFERENCE["grok"](model, inputs, answers, splits, weight_decay, steps, 100, 0.99, torch.Generator().manual_seed(seed + 1), device)


def started_model(
    seed: int,
    start: str,
) -> nn.Module:
    """paper2's models. "paper" is PaperGPT, "gpt2" is gpt2's GPT, tied and at 0.02, and "small" is PaperGPT with
    gpt2's 0.02 start but its own head, which splits the two changes "gpt2" makes."""
    torch.manual_seed(seed)

    if start == "gpt2":
        return REFERENCE["GPT"](99, 4, 128, 2, 4)

    model = REFERENCE["PaperGPT"](99, 4, 128, 2, 4)

    if start == "small":
        model.apply(REFERENCE["init_weights"])
        REFERENCE["scale_residuals_"](model, 2)

    return model


def measure_grokking(
    seeds: range = range(5),
    fraction: float = 0.3,
    weight_decay: float = 1.0,
    start: str = "paper",
) -> list[list[tuple[int, float, float]]]:
    """paper2's runs on the best device, each seed's curve until validation first passes 99%, or for all STEPS
    updates. The lesson's curve and seed tables come from one call, seed 0's curve and each seed's first steps."""
    device = best_device()
    curves = []

    for seed in seeds:
        curve = grok_until(started_model(seed, start).to(device), fraction, weight_decay, RUNS["STEPS"], seed, device)
        curves.append(curve)
        print(fraction, weight_decay, start, seed, REFERENCE["first_step"](curve, 1, 0.99), REFERENCE["first_step"](curve, 2, 0.99), curve[-1][2], flush=True)

    return curves
