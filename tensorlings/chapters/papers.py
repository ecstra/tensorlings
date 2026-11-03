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

        Most new modules start in a paper. This chapter reads one into code,
        Power and colleagues' grokking paper, then reproduces its result small,
        with several seeds and the spread said out loud.

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
# gpt2's GPT as the paper's model. The head gets a weight of its own, as in Nanda and
# colleagues' grokking model, and every layer keeps PyTorch's own default start.
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
    Most new modules start in a paper, so a module of your own often starts
    with reading one into code. Three moves do most of the work.

    - Shapes: list every symbol the method uses, with its shape.
    - Lines: turn each sentence of the setup into one line of code.
    - Counts: count the parameters, and check the count against the paper's.

    Power and colleagues' grokking paper shows the moves. Its data are the
    tables of an operation, such as a + b mod 97. Every number is a token of
    its own, so the model never sees digits. It learns what each number means
    only from how the numbers combine.

    - Equations: each is four tokens, a, then op, then b, then =. With p numbers the vocabulary holds p + 2 tokens.
    - Data: every pair of numbers makes one equation, p squared in all. A random fraction trains, and the rest is held out.
    - Model: a decoder only transformer of 2 layers, width 128 and 4 heads, about 400,000 weights outside the embeddings.
    - Loss: cross entropy on the answer alone, scored at the = token, never on the question.
    - Optimizer: AdamW at a rate of 0.001, betas 0.9 and 0.98, weight decay 1, warmed up over the first 10 updates.
    - Batches: 512 equations, or half the training set when that is smaller.

    gpt2's GPT at 2 layers, width 128 and 4 heads is nearly that model. Weight
    decay 1 is ten times gpt3's 0.1, and paper2 shows why the paper wanted so
    much.

    The paper never says how its weights start, or whether its head shares
    `wte`'s table. The given `PaperGPT` follows Nanda and colleagues' grokking
    model, whose head has a weight of its own. It also keeps PyTorch's own
    start for every layer, larger than gpt2's 0.02. paper2 shows what gpt2's
    two choices would change.

    The count leaves out the token and position tables and the head. It
    covers the blocks and the last LayerNorm, the part that grows with the
    model's depth.
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


    # SelfAttention, GPT2MLP and GPT2Block from block2, and init_weights,
    # scale_residuals_ and GPT from gpt2.
''' + indented(MODEL_PARTS) + '''

''' + indented(SETTINGS_PART) + '''

''' + indented(START_PART) + '''


    # TODO: Return every equation of a + b mod p, as two tensors of whole
    #   numbers. inputs has one row per pair, a then OP then b then EQUALS,
    #   where OP is p and EQUALS is p + 1, with a running slowest. answers
    #   holds (a + b) % p for each row.
    def equations(p: int) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Shuffle the numbers 0 to count - 1 with torch.randperm and
    #   generator. Return the first int(fraction * count) of them as the
    #   training rows, and the rest as the validation rows.
    def split_equations(
        count: int,
        fraction: float,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return the cross entropy of the model's scores at the last token,
    #   the = token, against answers, as a tensor.
    def answer_loss(
        model: nn.Module,
        inputs: Tensor,
        answers: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the share of answers the model gets right, its top score at
    #   the last token, as a plain number. Score in eval mode without recording
    #   for autograd, then put the model back in training mode.
    def answer_accuracy(
        model: nn.Module,
        inputs: Tensor,
        answers: Tensor,
    ) -> float:
        ...


    # TODO: Return the paper's AdamW, at a rate of 0.001, betas 0.9 and 0.98
    #   and the given weight_decay, and a LambdaLR that scales the rate by
    #   min(1.0, (step + 1) / 10), its warmup over the first 10 updates.
    def paper_optimizer(
        model: nn.Module,
        weight_decay: float,
    ) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
        ...


    # TODO: Return how many numbers the parameters of model.h and model.ln_f
    #   hold together, the weights outside the tables and the head.
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
                raise Mismatch(f"Before update {step + 1} the rate is {got:.6g}, but it should be {wanted:.6g}, rising in a line to 0.001 over the first 10 updates.")


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
        description="scores the answer at the = token alone",
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
        description="counts the weights outside the tables and the head",
        inputs=(
            (Described(REFERENCE["PaperGPT"](99, 4, 128, 2, 4), "PaperGPT(99, 4, 128, 2, 4)"),),
            (Described(REFERENCE["GPT"](20, 8, 32, 3, 4), "GPT(20, 8, 32, 3, 4)"),),
        ),
        reference=REFERENCE["non_embedding_parameters"],
        expect=expect_value,
    )
    checker.check_name(
        name="counted",
        description="the paper's model outside its embeddings",
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
        "equations": "The file's firsts and seconds lines give every pair of a and b. Stack a, a row of p, b and a row "
        "of p + 1 along dim=1, then take the sum mod p.",
        "split_equations": "The shuffled line does the shuffling. Slice it at the cut for the two parts.",
        "answer_loss": "model(inputs) has shape (batch, 4, vocabulary). The last line of the file's examples picks "
        "the scores at the last token.",
        "answer_accuracy": "argmax over the last dimension gives each answer. Compare with answers, then take the "
        "float mean and .item().",
        "paper_optimizer": "torch.optim.AdamW takes lr, betas and weight_decay. The warmup line holds the lambda the "
        "LambdaLR needs.",
        "non_embedding_parameters": "Sum numel over model.h.parameters() and model.ln_f.parameters(). PaperGPT's "
        "head has a weight of its own, so model.parameters() would count it too.",
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
    A paper's result is a claim about runs, so you check it with runs of your
    own. Power and colleagues claim three things.

    - Grokking: training accuracy reaches 100% early, and validation accuracy catches up only much later.
    - Weight decay: of every trick they tried, it helped generalization the most.
    - Data: the less of the table trains, the longer the jump takes.

    `grok` makes the run. main runs it on seed 0 with 30% of the table,
    the fraction Nanda and colleagues used.

    | Update | 100 | 300 | 1,000 | 2,000 | 3,000 | 4,000 | 4,200 | 4,400 | 4,600 |
    |---|---|---|---|---|---|---|---|---|---|
    | Training | 19% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
    | Validation | 0.1% | 5% | 13% | 15% | 21% | 36% | 59% | 95% | 99% |

    By update 300 the model knows every training equation and almost none of
    the rest. For thousands of updates little changes, then validation leaps
    from 36% to 99% in 600 updates. Training dips now and then and recovers,
    so main stops at the first point past 99%.

    One run is an anecdote. Here are 5 seeds, the update where each first
    passed 99%, with seed 0 the run above.

    | Seed | 0 | 1 | 2 | 3 | 4 |
    |---|---|---|---|---|---|
    | Training | 300 | 300 | 300 | 300 | 300 |
    | Validation | 4,600 | 3,600 | 3,200 | 3,700 | 4,900 |

    `mean_and_spread` sums them up. Validation passes at 4,000 on average,
    with a spread of 718, 13 times as late as training.

    Then what moves the jump, the update where validation first passed 99%.

    | Run | Seeds | Validation passes 99% at |
    |---|---|---|
    | 30% of the table, as above | 5 | 3,200 to 4,900 |
    | 50% | 5 | 1,300 to 1,400 |
    | 70% | 3 | 700 to 1,200 |
    | 30%, gpt3's weight decay of 0.1 | 3 | not in 6,000 updates, at 6% to 14% |
    | 30%, no weight decay | 3 | not in 6,000 updates, at 3% to 9% |
    | 30%, gpt2's 0.02 start, own head | 5 | 2,700 to 3,400 |
    | 30%, gpt2's 0.02 start, tied head | 5 | 1,200 to 2,600 |
    | 50%, gpt2's 0.02 start, tied head | 3 | 300, with training |

    All three claims hold. At gpt3's 0.1 the jump never comes, which is why
    the paper turned weight decay up to 1.

    The last three rows add a fourth. Liu and colleagues' Omnigrok traced
    grokking to weights that start large. gpt2's 0.02 start alone brings
    every seed's jump earlier, by 500 to 1,500 updates. Tying the head to
    `wte` as well brings every seed's closer still, and at 50% it removes
    the wait.

    `run.py train paper2` runs main. The check reads your model back and asks
    for nearly every held out equation.
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
    found = next(point[0] for point in pairs if point[1] >= 0.5)    # 200, the first step past 0.5
    values = torch.tensor([3.0, 5.0], dtype=torch.float64)
    center, wobble = values.mean().item(), values.std().item()     # 4.0 and 1.414..., the spread dividing by n - 1


    # The model from gpt2 and every piece of paper1.
''' + indented(MODEL_PARTS) + '''

''' + indented(SETTINGS_PART) + '''

''' + indented(START_PART) + '''

''' + indented(PAPER1_PARTS) + '''

''' + indented(RUN_PART) + '''


    # TODO: Train model as the paper does, for up to steps updates, and return
    #   its curve. Move inputs and answers to device first. Make the optimizer
    #   and scheduler with paper_optimizer and weight_decay. Each update draws
    #   a batch of min(512, len(train) // 2) training rows at random with
    #   torch.randint and generator, takes answer_loss, zeroes the old
    #   gradients, backpropagates, steps the optimizer and steps the
    #   scheduler. After each `every` updates, append (step, training
    #   accuracy, validation accuracy) with answer_accuracy on each set. Stop
    #   right after a point whose validation accuracy reaches stop_at, unless
    #   stop_at is None.
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


    # TODO: Return the step of the first point of curve whose entry at column
    #   reaches threshold, or None when none does. Column 1 is training and
    #   column 2 is validation.
    def first_step(
        curve: list[tuple[int, float, float]],
        column: int,
        threshold: float,
    ) -> int | None:
        ...


    # TODO: Return the mean of values and their spread, the standard deviation
    #   that divides by n - 1, as two plain numbers.
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
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be two plain numbers, the mean and the spread.")

    for name, got, wanted in zip(("the mean", "the spread"), actual, expected):
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
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a list of {len(expected)} points, one per {len(expected)} checks.")

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
        description="gives the mean and the spread that divides by n - 1",
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
        "mean_and_spread": "The file's center and wobble line does it for a tensor. Make the tensor from values with "
        "dtype=torch.float64 and return both numbers.",
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
