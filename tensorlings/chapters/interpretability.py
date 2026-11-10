"""107_interpretability, a small trained model opened up: hooks and probes on its hidden states, the induction head its attention patterns show, ablation and activation patching to test what it uses, and attribution by gradients and integrated gradients."""

from __future__ import annotations

import copy
import functools
import math
import textwrap
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, QUIZZES, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    reject_missing_return,
    value_text,
)
from ..text import prose, source_file

INTERPRETABILITY = Chapter(
    folder="107_interpretability",
    title="Interpretability",
    readme=source_file(
        """
        # Interpretability

        Knowing that a model works tells you nothing about how it works. This
        chapter trains a small model and looks inside it. The model has two
        layers of attention and learns to copy tokens from earlier in a
        sequence. Hooks and probes read its hidden states. Its attention
        patterns show the induction head that induction1 built by hand.
        Ablation and activation patching test which parts it actually uses.
        Attribution traces an answer back to the input tokens it came from.

        ## Further reading

        - [Olsson and colleagues, in context learning and induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
        - [Alain and Bengio, linear classifier probes](https://arxiv.org/abs/1610.01644)
        - [Heimersheim and Nanda, how to use and interpret activation patching](https://arxiv.org/abs/2404.15255)
        - [Sundararajan and colleagues, integrated gradients](https://arxiv.org/abs/1703.01365)
        """,
    ),
)

# The model every exercise of the chapter studies, given in each file.
COPIER_PART = '''
VOCABULARY = 32
STEPS = 24


# Each row repeats a segment of 6 to 12 different tokens until the row is 24 tokens long. known is
# True at the positions whose next token can be read from an earlier copy of the segment.
def repeated_sequences(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    lengths = torch.randint(6, 13, (count,), generator=generator)
    segments = torch.rand(count, VOCABULARY, generator=generator).argsort(dim=1)
    positions = torch.arange(STEPS)
    ids = segments.gather(1, positions % lengths[:, None])
    known = positions[None, :-1] + 1 >= lengths[:, None]

    return ids, known


class Attention(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.query = nn.Linear(size, size, bias=False)
        self.key = nn.Linear(size, size, bias=False)
        self.value = nn.Linear(size, size, bias=False)
        self.out = nn.Linear(size, size, bias=False)

    # Row t holds the attention weights of position t on each position up to t.
    def pattern(self, x: Tensor) -> Tensor:
        steps = x.shape[1]
        scores = self.query(x) @ self.key(x).transpose(-2, -1) / x.shape[-1] ** 0.5
        future = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=x.device), diagonal=1)

        return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.pattern(x) @ self.value(x))


class Embed(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.tokens = nn.Embedding(VOCABULARY, size)
        self.positions = nn.Embedding(STEPS, size)

    def forward(self, ids: Tensor) -> Tensor:
        return self.tokens(ids) + self.positions(torch.arange(ids.shape[1], device=ids.device))


class Block(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.attention = Attention(size)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.attention(x)


class Copier(nn.Module):
    def __init__(
        self,
        layers: int = 2,
        size: int = 64,
    ) -> None:
        super().__init__()
        self.embed = Embed(size)
        self.blocks = nn.ModuleList([Block(size) for _ in range(layers)])
        self.head = nn.Linear(size, VOCABULARY)

    def forward(self, ids: Tensor) -> Tensor:
        x = self.embed(ids)

        for block in self.blocks:
            x = block(x)

        return self.head(x)


def train_copier(
    seed: int,
    steps: int,
    layers: int = 2,
) -> Copier:
    torch.manual_seed(seed)
    model = Copier(layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed + 1)

    for _ in range(steps):
        ids, known = repeated_sequences(64, generator)
        scores = model(ids)[:, :-1]
        loss = F.cross_entropy(scores[known], ids[:, 1:][known])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model.eval()


# The fraction of known positions where model predicts the next token correctly.
def accuracy(
    model: nn.Module,
    ids: Tensor,
    known: Tensor,
) -> float:
    with torch.no_grad():
        right = model(ids)[:, :-1].argmax(dim=-1) == ids[:, 1:]

    return right[known].float().mean().item()
'''

FIT_PROBE_PART = '''
# topics2's linear probe, trained on the first half of the rows and scored on the second half.
def fit_probe(
    states: Tensor,
    targets: Tensor,
) -> float:
    torch.manual_seed(0)
    half = len(states) // 2
    probe = nn.Linear(states.shape[1], VOCABULARY)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)

    for _ in range(300):
        loss = F.cross_entropy(probe(states[:half]), targets[:half])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return (probe(states[half:]).argmax(dim=1) == targets[half:]).float().mean().item()
'''

RECORD_SOLUTION = '''
def record_outputs(
    model: nn.Module,
    modules: list[nn.Module],
    ids: Tensor,
) -> list[Tensor]:
    outputs = []
    handles = [module.register_forward_hook(lambda module, inputs, output: outputs.append(output)) for module in modules]

    try:
        with torch.no_grad():
            model(ids)
    finally:
        for handle in handles:
            handle.remove()

    return outputs
'''

PROBE_SOLUTION = '''
def probe_scores(
    model: Copier,
    ids: Tensor,
    known: Tensor,
) -> list[float]:
    outputs = record_outputs(model, [model.embed, *model.blocks], ids)

    return [fit_probe(states[:, :-1][known], ids[:, 1:][known]) for states in outputs]
'''

PATTERNS_SOLUTION = '''
def patterns(
    model: Copier,
    ids: Tensor,
) -> list[Tensor]:
    found = []

    with torch.no_grad():
        x = model.embed(ids)

        for block in model.blocks:
            found.append(block.attention.pattern(x))
            x = block(x)

    return found


def previous_token_score(pattern: Tensor) -> float:
    return torch.diagonal(pattern, offset=-1, dim1=-2, dim2=-1).mean().item()


def induction_score(
    pattern: Tensor,
    ids: Tensor,
) -> float:
    steps = ids.shape[1]
    earlier = torch.tril(torch.ones(steps, steps, dtype=torch.bool), diagonal=-1)
    matches = (ids[:, :, None] == ids[:, None, :]) & earlier
    targets = torch.zeros_like(matches)
    targets[:, :, 1:] = matches[:, :, :-1]
    has_target = targets.any(dim=-1)

    return (pattern * targets).sum(dim=-1)[has_target].mean().item()
'''

CORRUPT_PART = '''
# Finds the first copy of each row's last token and changes the token right after that copy, which
# changes the right answer at the last position. Returns the corrupted ids, the clean answers and the
# corrupted answers.
def corrupt(ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    rows = torch.arange(len(ids))
    place = (ids == ids[:, -1:]).float().argmax(dim=1) + 1
    clean_answers = ids[rows, place]
    corrupted_answers = (clean_answers + 1) % VOCABULARY
    corrupted = ids.clone()
    corrupted[rows, place] = corrupted_answers

    return corrupted, clean_answers, corrupted_answers
'''

ABLATE_SOLUTION = '''
def ablated_accuracy(
    model: nn.Module,
    ids: Tensor,
    known: Tensor,
    module: nn.Module,
) -> float:
    handle = module.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))

    try:
        return accuracy(model, ids, known)
    finally:
        handle.remove()


def patched_scores(
    model: nn.Module,
    module: nn.Module,
    ids: Tensor,
    replacement: Tensor,
) -> Tensor:
    handle = module.register_forward_hook(lambda module, inputs, output: replacement)

    try:
        with torch.no_grad():
            return model(ids)
    finally:
        handle.remove()


def logit_difference(
    scores: Tensor,
    clean_answers: Tensor,
    corrupted_answers: Tensor,
) -> float:
    last = scores[:, -1]
    rows = torch.arange(len(last))

    return (last[rows, clean_answers] - last[rows, corrupted_answers]).mean().item()
'''

ANSWER_PART = '''
# The score of each row's answer at the last position. The model runs from token_vectors instead of
# model.embed.tokens(ids), and adds the position embeddings as embed does.
def answer_scores(
    model: Copier,
    token_vectors: Tensor,
    answers: Tensor,
) -> Tensor:
    x = token_vectors + model.embed.positions(torch.arange(token_vectors.shape[1]))

    for block in model.blocks:
        x = block(x)

    return model.head(x)[torch.arange(len(answers)), -1, answers]


# Each row's answer, which is the token right after the first copy of its last token, and the position
# of that answer.
def copied_answers(ids: Tensor) -> tuple[Tensor, Tensor]:
    place = (ids == ids[:, -1:]).float().argmax(dim=1) + 1

    return ids[torch.arange(len(ids)), place], place
'''

ATTRIBUTION_SOLUTION = '''
def gradient(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
) -> Tensor:
    x = x.detach().clone().requires_grad_(True)
    f(x).sum().backward()

    return x.grad


def integrated_gradients(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    baseline: Tensor,
    steps: int,
) -> Tensor:
    total = torch.zeros_like(x)

    for k in range(1, steps + 1):
        total += gradient(f, baseline + k / steps * (x - baseline))

    return (x - baseline) * total / steps


def position_attributions(
    model: Copier,
    ids: Tensor,
    answers: Tensor,
    steps: int,
) -> Tensor:
    vectors = model.embed.tokens(ids).detach()
    found = integrated_gradients(lambda v: answer_scores(model, v, answers), vectors, torch.zeros_like(vectors), steps)

    return found.sum(dim=-1)
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "Callable": Callable,
    }
    exec(code, namespace)

    return namespace


REFERENCE = namespace_of(COPIER_PART + FIT_PROBE_PART + RECORD_SOLUTION + PROBE_SOLUTION + PATTERNS_SOLUTION + CORRUPT_PART + ABLATE_SOLUTION + ANSWER_PART + ATTRIBUTION_SOLUTION)

LESSON_STEPS = 500


@functools.cache
def trained_copier() -> nn.Module:
    """The model every file trains, made once per run of the app for the checks to call on."""
    return REFERENCE["train_copier"](0, LESSON_STEPS)


def untrained_copier(
    seed: int,
    layers: int = 2,
    size: int = 16,
) -> nn.Module:
    torch.manual_seed(seed)

    return REFERENCE["Copier"](layers, size).eval()


def sequences(
    seed: int,
    count: int,
) -> tuple[Tensor, Tensor]:
    return REFERENCE["repeated_sequences"](count, torch.Generator().manual_seed(seed))


def hooks_left(model: nn.Module) -> bool:
    return any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def reject_hooks_left(
    function: str,
    model: nn.Module,
) -> None:
    if hooks_left(model):
        raise Mismatch(f"{function} left a hook on the model. Keep each handle and call handle.remove() in a finally block.")


def expect_scores(
    call: str,
    actual: object,
    expected: list[float],
    tolerance: float,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, list) or len(actual) != len(expected):
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a list of {len(expected)} plain numbers.")

    for position, (got, wanted) in enumerate(zip(actual, expected)):
        expect_close(
            call=f"entry {position} of {call}",
            actual=got,
            expected=wanted,
            tolerance=tolerance,
        )


def expect_close_tensor(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    """A tensor summed from many gradients, where adding in another order moves the last digits."""
    if isinstance(actual, Tensor) and isinstance(expected, Tensor) and actual.shape == expected.shape:
        if torch.allclose(actual.double(), expected.double(), rtol=1e-4, atol=1e-5):
            return

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


# ---------------------------------------------------------------------------
# interp1
# ---------------------------------------------------------------------------

INTERP1_LESSON = prose(
    """
    This chapter looks inside a trained model to see how it works. The model
    is small on purpose. Two layers of attention read a row of tokens, and a
    Linear head scores every possible next token at every position.

    It trains on repeated rows. Each row repeats a segment of 6 to 12
    different tokens until the row is 24 tokens long. Within the first copy
    of the segment, the next token cannot be predicted, and a guess is right
    1 time in 32. After the first copy, the next token can be read from the
    earlier copy. `train_copier` trains a model in a few seconds, and it
    predicts 99.6% of those later positions correctly.

    A forward hook reads a module's output while the model runs, without
    changing the model's code. Each time the module runs, PyTorch calls the
    hook with the module, its inputs and its output. `register_forward_hook`
    returns a handle, and `handle.remove()` takes the hook off again.

    ```python
    handle = model.head.register_forward_hook(lambda module, inputs, output: print(output.shape))
    handle.remove()    # the hook is gone, so the next run prints nothing
    ```

    Each block adds its output to its input, so the hidden state builds up
    from block to block. The outputs of `embed` and of the two blocks show
    the hidden state at three points. A probe, like the one topics2 trained,
    is a Linear layer trained to read one fact from those states. The table
    shows how accurately probes read two facts on 500 new rows.

    | Read from | The next token, where an earlier copy shows it | The previous token, at positions 1 to 11 |
    |---|---|---|
    | `embed` | 3.3% | 3.3% |
    | `blocks[0]` | 6.9% | 100% |
    | `blocks[1]` | 99.8% | 97.5% |

    The next token can be read only after the second block. After the first
    block, a probe reads each early position's previous token perfectly. So
    the first block copies the previous token into each position early in
    the row. interp2 shows that the second block needs it there.

    A probe shows that a fact can be read from a hidden state. It cannot show
    that the model uses that fact. interp3 tests what the model uses by
    changing the model and watching how its answer changes.
    """,
)

INTERP1_SOURCE = source_file(
    '''
    # interp1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    hooked = nn.Linear(2, 2)
    handle = hooked.register_forward_hook(lambda module, inputs, output: None)    # the hook now runs after every call of hooked
    handle.remove()                                                              # and now it never runs again
    picked = torch.arange(6).view(2, 3)[torch.tensor([[True, False, True], [False, True, False]])]    # tensor([0, 2, 4])


    # The copier model this chapter studies, and the function that trains it.
''' + indented(COPIER_PART) + '''


''' + indented(FIT_PROBE_PART) + '''


    # TODO: Run model on ids once and return the output of each module in
    #   modules, as a list in the order the model runs them. Put a forward
    #   hook on each module that saves its output. Run the model inside
    #   torch.no_grad(). Remove every hook afterwards, even if the run raises
    #   an error.
    def record_outputs(
        model: nn.Module,
        modules: list[nn.Module],
        ids: Tensor,
    ) -> list[Tensor]:
        ...


    # TODO: Return how accurately a probe reads the next token from the outputs
    #   of model.embed and of each block in model.blocks, as a list of Python
    #   floats. Use only the positions where known is True. For an output of
    #   shape (batch, steps, size), the states are output[:, :-1][known] and
    #   the targets are ids[:, 1:][known]. Score each output with fit_probe.
    def probe_scores(
        model: Copier,
        ids: Tensor,
        known: Tensor,
    ) -> list[float]:
        ...


    model = train_copier(0, 500)
    ids, known = repeated_sequences(500, torch.Generator().manual_seed(99))
    scores = probe_scores(model, ids, known)
    ''',
)

INTERP1_SOLUTION = source_file(
    '''
    # interp1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    hooked = nn.Linear(2, 2)
    handle = hooked.register_forward_hook(lambda module, inputs, output: None)
    handle.remove()
    picked = torch.arange(6).view(2, 3)[torch.tensor([[True, False, True], [False, True, False]])]


''' + indented(COPIER_PART) + '''


''' + indented(FIT_PROBE_PART) + '''


''' + indented(RECORD_SOLUTION) + '''


''' + indented(PROBE_SOLUTION) + '''


    model = train_copier(0, 500)
    ids, known = repeated_sequences(500, torch.Generator().manual_seed(99))
    scores = probe_scores(model, ids, known)
    ''',
)


class Crasher(nn.Module):
    """Stands in for a block and fails, as a bug in a model would, to see whether the hooks still come off."""

    def forward(self, x: Tensor) -> Tensor:
        raise RuntimeError("this block crashes on purpose")


def check_record_outputs(found: object) -> None:
    for seed, pick in ((650, "embed and both blocks"), (651, "the second attention and the head")):
        model = untrained_copier(seed)
        ids, _ = sequences(seed + 10, 4)
        modules = [model.embed, *model.blocks] if pick == "embed and both blocks" else [model.blocks[1].attention, model.head]
        call = f"record_outputs of an untrained Copier(2, 16), {pick}, on 4 rows"
        actual = found(model, modules, ids)
        reject_hooks_left("record_outputs", model)
        expected = REFERENCE["record_outputs"](model, modules, ids)
        reject_missing_return(
            call=call,
            actual=actual,
        )

        if not isinstance(actual, list) or len(actual) != len(expected):
            raise Mismatch(f"{call} gives {value_text(actual)}, but it should be a list of {len(expected)} outputs, one per module.")

        for position, (got, wanted) in enumerate(zip(actual, expected)):
            expect_tensor(
                call=f"output {position} of {call}",
                actual=got,
                expected=wanted,
            )

            if got.requires_grad:
                raise Mismatch(f"output {position} of {call} was recorded for autograd. Run the model inside torch.no_grad().")


def check_hooks_come_off(found: object) -> None:
    model = untrained_copier(652)
    model.blocks[1] = Crasher()
    ids, _ = sequences(653, 2)

    try:
        found(model, [model.embed, model.blocks[0]], ids)
    except Exception:
        pass

    if hooks_left(model):
        raise Mismatch(
            "When the model crashed partway, record_outputs left its hooks on. Take them off in a finally block, "
            "which runs even when the code above it fails.",
        )


def check_probe_scores(found: object) -> None:
    for model, name, seed in ((trained_copier(), "the trained copier", 641), (untrained_copier(654), "an untrained Copier(2, 16)", 642)):
        ids, known = sequences(seed, 300)
        expect_scores(
            call=f"probe_scores of {name} on 300 rows",
            actual=found(model, ids, known),
            expected=REFERENCE["probe_scores"](model, ids, known),
            tolerance=0.01,
        )
        reject_hooks_left("probe_scores", model)


def check_learned_scores(value: object) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise Mismatch(f"scores is {value_text(value)}, but it should be a list of 3 plain numbers.")

    if value[2] < 0.9:
        raise Mismatch(f"After the second block a probe reads {value[2]:.1%} of next tokens correctly, but it should read more than 90%.")

    if value[0] > 0.2:
        raise Mismatch(f"From embed a probe reads {value[0]:.1%} of next tokens, but it should stay near guessing, below 20%.")


def check_interp1(checker: Checker) -> None:
    checker.test(
        function="record_outputs",
        description="saves each module's output with hooks, then removes them",
        run=check_record_outputs,
    )
    checker.test(
        function="record_outputs",
        description="removes the hooks even when the model raises an error",
        run=check_hooks_come_off,
    )
    checker.test(
        function="probe_scores",
        description="probes the next token at each point of the model",
        run=check_probe_scores,
    )
    checker.check_name(
        name="scores",
        description="a probe reads your copier's answer only after the second block",
        run=check_learned_scores,
    )


INTERP1 = Exercise(
    name="interp1",
    chapter=INTERPRETABILITY,
    title="Hooks and probes",
    lesson=INTERP1_LESSON,
    source=INTERP1_SOURCE,
    solution=INTERP1_SOLUTION,
    hints={
        "record_outputs": "Each hook can append its output argument to one shared list. Keep the handles that "
        "register_forward_hook returns. Run the model inside torch.no_grad() in a try block, and remove every "
        "handle in the finally block.",
        "probe_scores": "Calling record_outputs with [model.embed, *model.blocks] gives the three outputs. For each "
        "output, pass fit_probe the states and targets that the TODO describes.",
        "scores": "It comes from probe_scores. If it fails while probe_scores passes, rerun the file.",
    },
    introduces=(),
    check=check_interp1,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# interp2
# ---------------------------------------------------------------------------

INTERP2_LESSON = prose(
    """
    induction1 built an induction head by hand. The first layer copies each
    position's previous token into that position. The second layer's query
    looks for the current token among those copies, and takes the token that
    followed it. Did training find the same circuit? Attention patterns can
    show it.

    `pattern(x)` in each block returns the attention weights, one row per
    position. Row t holds how much weight position t puts on each position up
    to t. Two scores describe what a head does.

    - The previous token score: the average weight each position puts on the position right before it. `torch.diagonal` with `offset=-1` picks those weights out.
    - The induction score: at positions whose token appeared earlier, the weight on the position right after that earlier copy.

    Olsson and colleagues call the second one prefix matching.

    The table shows both scores for each block of the trained copier, on 500
    new rows. It also shows a copier with only one block, trained the same
    way.

    | Model and block | Previous token score | Induction score | Right where a copy shows the answer |
    |---|---|---|---|
    | Two blocks, `blocks[0]` | 0.44 | 0.09 | 99.6% |
    | Two blocks, `blocks[1]` | 0.07 | 0.96 | |
    | One block | 0.06 | 0.16 | 36.9% |

    Training found induction1's circuit. The second block is an induction
    head, and a model with one block cannot form one. Olsson and colleagues
    found the same, because an induction head needs heads in two layers
    working together.

    The first block's score of 0.44 is an average, and it hides two different
    behaviors. Its weight on the previous position is 0.99 or more at
    positions 1 to 5, and 0.62 to 0.90 at positions 6 to 11. From position 12
    on, it stays under 0.08. Every token's first copy lies in the first
    segment, within the first 12 positions. So the second block always finds
    its answer there. Training put the previous token only at the positions
    where it gets read. A pattern shows how a circuit fits its data, which a
    single final score never shows.
    """,
)

INTERP2_SOURCE = source_file(
    '''
    # interp2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    weights = torch.full((1, 3, 3), 1 / 3)
    before = torch.diagonal(weights, offset=-1, dim1=-2, dim2=-1)    # tensor([[0.3333, 0.3333]]), each row's weight on the position before it
    tokens = torch.tensor([[5, 7, 5]])
    same = tokens[:, :, None] == tokens[:, None, :]                   # same[0, 2, 0] is True, since positions 2 and 0 hold token 5


    # The copier model this chapter studies, and the function that trains it.
''' + indented(COPIER_PART) + '''


    # TODO: Return each block's attention pattern on ids, as a list in block
    #   order. A block's pattern is block.attention.pattern of that block's
    #   input. The first block's input is model.embed(ids), and each block's
    #   output is the next block's input. Work inside torch.no_grad().
    def patterns(
        model: Copier,
        ids: Tensor,
    ) -> list[Tensor]:
        ...


    # TODO: Return the previous token score of pattern, as a Python float.
    #   pattern has shape (batch, steps, steps). The score is the average of
    #   the weight that each row t puts on position t - 1, over every row that
    #   has a previous position, in every sequence.
    #   Example: previous_token_score(weights), with weights from the top of
    #   this file, returns about 0.3333.
    def previous_token_score(pattern: Tensor) -> float:
        ...


    # TODO: Return the induction score of pattern for ids, as a Python float.
    #   For row t, a target is position s + 1 wherever s < t and ids holds the
    #   same token at s and at t. Sum each row's weights on its targets. Then
    #   average those sums over all the rows, in every sequence, that have at
    #   least one target.
    def induction_score(
        pattern: Tensor,
        ids: Tensor,
    ) -> float:
        ...


    model = train_copier(0, 500)
    ids, known = repeated_sequences(500, torch.Generator().manual_seed(99))
    scores = [(previous_token_score(pattern), induction_score(pattern, ids)) for pattern in patterns(model, ids)]
    ''',
)

INTERP2_SOLUTION = source_file(
    '''
    # interp2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    weights = torch.full((1, 3, 3), 1 / 3)
    before = torch.diagonal(weights, offset=-1, dim1=-2, dim2=-1)
    tokens = torch.tensor([[5, 7, 5]])
    same = tokens[:, :, None] == tokens[:, None, :]


''' + indented(COPIER_PART) + '''


''' + indented(PATTERNS_SOLUTION) + '''


    model = train_copier(0, 500)
    ids, known = repeated_sequences(500, torch.Generator().manual_seed(99))
    scores = [(previous_token_score(pattern), induction_score(pattern, ids)) for pattern in patterns(model, ids)]
    ''',
)


def random_pattern(
    seed: int,
    batch: int,
    steps: int,
) -> Tensor:
    """Softmax rows over the places up to each row, as an attention pattern holds them."""
    scores = torch.randn(batch, steps, steps, generator=torch.Generator().manual_seed(seed)) * 2
    future = torch.triu(torch.ones(steps, steps, dtype=torch.bool), diagonal=1)

    return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)


def check_patterns(found: object) -> None:
    for model, name, seed in (
        (trained_copier(), "the trained copier", 660),
        (untrained_copier(661, layers=3), "an untrained Copier(3, 16)", 662),
    ):
        ids, _ = sequences(seed, 5)
        call = f"patterns of {name} on 5 rows"
        actual = found(model, ids)
        expected = REFERENCE["patterns"](model, ids)
        reject_missing_return(
            call=call,
            actual=actual,
        )

        if not isinstance(actual, list) or len(actual) != len(expected):
            raise Mismatch(f"{call} gives {value_text(actual)}, but it should be a list of {len(expected)} patterns, one per block.")

        for position, (got, wanted) in enumerate(zip(actual, expected)):
            expect_tensor(
                call=f"pattern {position} of {call}",
                actual=got,
                expected=wanted,
            )

            if got.requires_grad:
                raise Mismatch(f"pattern {position} of {call} was recorded for autograd. Work inside torch.no_grad().")


def repeat_with_doubles() -> Tensor:
    """Rows where a token shows up three times, so some rows have two targets to sum."""
    return torch.tensor([[1, 2, 1, 3, 1, 2], [4, 4, 5, 4, 6, 5]])


def check_interp2(checker: Checker) -> None:
    checker.test(
        function="patterns",
        description="reads each block's pattern on its own input",
        run=check_patterns,
    )
    check_on_inputs(
        checker=checker,
        function="previous_token_score",
        description="averages the weight on the previous position",
        inputs=(
            (Described(random_pattern(663, 2, 5), "a random pattern of shape (2, 5, 5)"),),
            (Described(random_pattern(664, 3, 4), "a random pattern of shape (3, 4, 4)"),),
        ),
        reference=REFERENCE["previous_token_score"],
        expect=expect_close,
    )
    ids, _ = sequences(665, 3)
    check_on_inputs(
        checker=checker,
        function="induction_score",
        description="averages the weight on the position after an earlier copy",
        inputs=(
            (Described(random_pattern(666, 3, 24), "a random pattern of shape (3, 24, 24)"), Described(ids, "3 repeated rows of 24 tokens")),
            (Described(random_pattern(667, 2, 6), "a random pattern of shape (2, 6, 6)"), repeat_with_doubles()),
        ),
        reference=REFERENCE["induction_score"],
        expect=expect_close,
    )
    checker.check_name(
        name="scores",
        description="your copier's second block is an induction head",
        run=check_learned_patterns,
    )


def check_learned_patterns(value: object) -> None:
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in value):
        raise Mismatch(f"scores is {value_text(value)}, but it should be a list of 2 pairs, one per block.")

    if value[1][1] < 0.8:
        raise Mismatch(f"The second block's induction score is {value[1][1]:.2f}, but it should pass 0.8.")

    if value[0][0] < 0.3:
        raise Mismatch(f"The first block's previous token score is {value[0][0]:.2f}, but it should pass 0.3.")


INTERP2 = Exercise(
    name="interp2",
    chapter=INTERPRETABILITY,
    title="Attention patterns and induction heads",
    lesson=INTERP2_LESSON,
    source=INTERP2_SOURCE,
    solution=INTERP2_SOLUTION,
    hints={
        "patterns": "Inside torch.no_grad(), start with the model's embed of ids as x. For each block, keep the "
        "pattern its attention gives for the current x, then run the block on x to get the next block's input.",
        "previous_token_score": "The first lines of the file show torch.diagonal with offset=-1. Take the mean of "
        "what it returns and convert it to a Python float with .item().",
        "induction_score": "Compare every pair of positions as the line that sets same does. Keep only the pairs "
        "with s before t, using a mask from torch.tril with diagonal=-1. Shift that mask one position along the last "
        "dimension so it marks s + 1. Multiply it by the pattern and sum over the last dimension.",
        "scores": "It comes from the three functions. If it fails while they pass, rerun the file.",
    },
    introduces=(),
    check=check_interp2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# interp3
# ---------------------------------------------------------------------------

INTERP3_LESSON = prose(
    """
    Probes show what information a model holds, and attention patterns show
    where it looks. Neither shows what the model actually uses. To find that
    out, change the model and watch how its answer changes.

    If a forward hook returns a tensor, that tensor replaces the module's
    output. Zero ablation uses a hook that returns zeros. That removes the
    module's contribution, and everything after it runs as usual.

    ```python
    torch.zeros_like(torch.ones(2, 3))    # tensor([[0., 0., 0.], [0., 0., 0.]])
    ```

    The table shows the trained copier's accuracy on 500 new rows, at the
    positions where an earlier copy shows the next token. Each line of the
    table zeroes the attention of a different block.

    | Attention zeroed | Right where a copy shows the answer |
    |---|---|
    | None | 99.6% |
    | `blocks[0]` | 21.6% |
    | `blocks[1]` | 5.0% |

    Both blocks matter. But zeros are far from any output the model saw in
    training, so a large drop could come from that unusual input alone. Mean
    ablation is gentler. It replaces the output with its average over many
    rows. Here it gives 21.9% and 4.5%, about the same as zero ablation, so
    the drops are real.

    Activation patching asks a sharper question. Take a clean row and a
    corrupted row that differs from it in one important token. Run the
    corrupted row, but replace one module's output with that module's output
    from the clean run. If the clean answer comes back, that module carries
    the information the answer needs. Heimersheim and Nanda call this
    direction of patching denoising.

    Here `corrupt` finds the first copy of the last token and changes the
    token right after it, so the right answer changes. The measure is the
    logit difference. It is the clean answer's score minus the corrupted
    answer's score, at the last position. Heimersheim and Nanda recommend it
    most among the measures they compare.

    | Run | Logit difference |
    |---|---|
    | Clean | 11.14 |
    | Corrupted | -9.26 |
    | Corrupted, `blocks[0]` patched from clean | -8.91 |
    | Corrupted, `blocks[1]` patched from clean | 10.92 |

    Patching the second block brings the clean answer back. Patching the
    first block does almost nothing, even though zeroing it broke the model.
    The first block does copy the changed token one position forward, as the
    next position's previous token. But the answer is never read from there.
    The second block reads the changed token directly from the embedding at
    the token's own position. So the path that carries the answer skips the
    first block.
    """,
)

INTERP3_SOURCE = source_file(
    '''
    # interp3.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    blank = torch.zeros_like(torch.ones(2, 3))    # tensor([[0., 0., 0.], [0., 0., 0.]])
    replace = lambda module, inputs, output: blank    # used as a forward hook, this replaces the module's output with blank


    # The copier model this chapter studies, and the function that trains it.
''' + indented(COPIER_PART) + '''


    # interp1's record_outputs.
''' + indented(RECORD_SOLUTION) + '''


''' + indented(CORRUPT_PART) + '''


    # TODO: Return the accuracy of model on ids and known, from the given
    #   accuracy function, while the output of module is replaced by zeros of
    #   the same shape. Remove the hook afterwards, even if the run raises an
    #   error.
    def ablated_accuracy(
        model: nn.Module,
        ids: Tensor,
        known: Tensor,
        module: nn.Module,
    ) -> float:
        ...


    # TODO: Return model's scores on ids while the output of module is
    #   replaced by replacement. Run the model inside torch.no_grad(). Remove
    #   the hook afterwards, even if the run raises an error.
    def patched_scores(
        model: nn.Module,
        module: nn.Module,
        ids: Tensor,
        replacement: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the logit difference as a Python float. scores has shape
    #   (batch, steps, vocabulary). At the last position, take each row's
    #   score for its clean answer minus its score for its corrupted answer.
    #   Return the mean over the rows.
    #   Example: logit_difference(torch.tensor([[[0.0, 0.0], [1.0, 3.0]]]),
    #   torch.tensor([1]), torch.tensor([0])) returns 2.0.
    def logit_difference(
        scores: Tensor,
        clean_answers: Tensor,
        corrupted_answers: Tensor,
    ) -> float:
        ...


    model = train_copier(0, 500)
    ids, known = repeated_sequences(500, torch.Generator().manual_seed(99))
    ablated = [ablated_accuracy(model, ids, known, block.attention) for block in model.blocks]
    corrupted, clean_answers, corrupted_answers = corrupt(ids)
    patched = [
        logit_difference(
            patched_scores(model, block.attention, corrupted, record_outputs(model, [block.attention], ids)[0]),
            clean_answers,
            corrupted_answers,
        )
        for block in model.blocks
    ]
    ''',
)

INTERP3_SOLUTION = source_file(
    '''
    # interp3.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    blank = torch.zeros_like(torch.ones(2, 3))
    replace = lambda module, inputs, output: blank


''' + indented(COPIER_PART) + '''


''' + indented(RECORD_SOLUTION) + '''


''' + indented(CORRUPT_PART) + '''


''' + indented(ABLATE_SOLUTION) + '''


    model = train_copier(0, 500)
    ids, known = repeated_sequences(500, torch.Generator().manual_seed(99))
    ablated = [ablated_accuracy(model, ids, known, block.attention) for block in model.blocks]
    corrupted, clean_answers, corrupted_answers = corrupt(ids)
    patched = [
        logit_difference(
            patched_scores(model, block.attention, corrupted, record_outputs(model, [block.attention], ids)[0]),
            clean_answers,
            corrupted_answers,
        )
        for block in model.blocks
    ]
    ''',
)


def check_ablated_accuracy(found: object) -> None:
    for model, name, seed in ((trained_copier(), "the trained copier", 670), (untrained_copier(671), "an untrained Copier(2, 16)", 672)):
        ids, known = sequences(seed, 200)

        for index in range(len(model.blocks)):
            module = model.blocks[index].attention
            expected = REFERENCE["ablated_accuracy"](model, ids, known, module)
            expect_close(
                call=f"ablated_accuracy of {name} with blocks[{index}].attention zeroed, on 200 rows",
                actual=found(model, ids, known, module),
                expected=expected,
                tolerance=1e-6,
            )
            reject_hooks_left("ablated_accuracy", model)


def check_ablation_hooks_come_off(found: object) -> None:
    model = untrained_copier(673)
    model.blocks[1] = Crasher()
    ids, known = sequences(674, 2)

    try:
        found(model, ids, known, model.blocks[0].attention)
    except Exception:
        pass

    if hooks_left(model):
        raise Mismatch("When the model crashed partway, ablated_accuracy left its hook on. Take it off in a finally block.")


def check_patched_scores(found: object) -> None:
    model = trained_copier()
    ids, _ = sequences(675, 6)
    corrupted, _, _ = REFERENCE["corrupt"](ids)

    for index in range(2):
        module = model.blocks[index].attention
        replacement = REFERENCE["record_outputs"](model, [module], ids)[0]
        call = f"patched_scores of the trained copier, blocks[{index}].attention patched from 6 clean rows"
        actual = found(model, module, corrupted, replacement)
        reject_hooks_left("patched_scores", model)
        expect_tensor(
            call=call,
            actual=actual,
            expected=REFERENCE["patched_scores"](model, module, corrupted, replacement),
        )

        if actual.requires_grad:
            raise Mismatch(f"{call} was recorded for autograd. Run the model inside torch.no_grad().")


def check_learned_patching(value: object) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise Mismatch(f"patched is {value_text(value)}, but it should be a list of 2 plain numbers.")

    if value[1] < 5:
        raise Mismatch(f"Patching the second block gives a logit difference of {value[1]:.2f}, but it should bring the clean answer back, above 5.")

    if value[0] > 0:
        raise Mismatch(f"Patching the first block gives a logit difference of {value[0]:.2f}, but it should stay with the corrupted answer, below 0.")


def check_learned_ablation(value: object) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise Mismatch(f"ablated is {value_text(value)}, but it should be a list of 2 plain numbers.")

    if max(value) > 0.5:
        raise Mismatch(f"With a block zeroed your copier still gets {max(value):.1%} right, but zeroing either block should break it, below 50%.")


def check_interp3(checker: Checker) -> None:
    checker.test(
        function="ablated_accuracy",
        description="scores the model with one module's output zeroed",
        run=check_ablated_accuracy,
    )
    checker.test(
        function="ablated_accuracy",
        description="removes the hook even when the model raises an error",
        run=check_ablation_hooks_come_off,
    )
    checker.test(
        function="patched_scores",
        description="runs the model with one module's output replaced",
        run=check_patched_scores,
    )
    scores = torch.randn(4, 5, 7, generator=torch.Generator().manual_seed(676))
    check_on_inputs(
        checker=checker,
        function="logit_difference",
        description="averages the clean answer's score minus the corrupted answer's at the last position",
        inputs=(
            (Described(scores, "random scores of shape (4, 5, 7)"), torch.tensor([0, 3, 6, 2]), torch.tensor([1, 3, 0, 5])),
            (Described(scores[:2, :3], "random scores of shape (2, 3, 7)"), torch.tensor([4, 4]), torch.tensor([2, 6])),
        ),
        reference=REFERENCE["logit_difference"],
        expect=expect_close,
    )
    checker.check_name(
        name="ablated",
        description="zeroing either block breaks your copier",
        run=check_learned_ablation,
    )
    checker.check_name(
        name="patched",
        description="patching the second block brings the clean answer back",
        run=check_learned_patching,
    )


INTERP3 = Exercise(
    name="interp3",
    chapter=INTERPRETABILITY,
    title="Ablation and activation patching",
    lesson=INTERP3_LESSON,
    source=INTERP3_SOURCE,
    solution=INTERP3_SOLUTION,
    hints={
        "ablated_accuracy": "Register a forward hook on module that returns torch.zeros_like of its output. Return "
        "the accuracy of the model inside a try block, and remove the handle in the finally block.",
        "patched_scores": "Write it like ablated_accuracy, with a hook that returns replacement, and run the model "
        "inside torch.no_grad().",
        "logit_difference": "scores[:, -1] holds the scores at the last position. Index it with torch.arange of the "
        "batch size and each set of answers, subtract, then call .mean().item().",
        "ablated": "It comes from ablated_accuracy. If it fails while the function passes, rerun the file.",
        "patched": "It comes from patched_scores and logit_difference. If it fails while they pass, rerun the file.",
    },
    introduces=(),
    check=check_interp3,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# interp4
# ---------------------------------------------------------------------------

INTERP4_LESSON = prose(
    """
    Attribution asks which inputs an answer came from. The simplest method
    uses the gradient, which says how much the output changes when an input
    changes a little.

    Sundararajan and colleagues showed where the gradient fails. Take
    `f(x) = 1 - relu(1 - x)`. As x goes from 0 to 2, the output rises from 0
    to 1, so x matters. But f is flat for x above 1, so the gradient at 2 is
    0. Their first rule, which they call sensitivity, says that an input
    that changed the output must get some credit.

    Integrated gradients fix this. They average the gradient along the
    straight line from a baseline input to the real input. Then they multiply
    that average by the input minus the baseline, which gives each input's
    credit. Their completeness property says the credits add up to
    `f(input) - f(baseline)`. With m steps, the average uses the points that
    are k / m of the way along the line, for k from 1 to m.

    ```python
    torch.linspace(0, 2, 5)[1:]    # tensor([0.5000, 1.0000, 1.5000, 2.0000]), 4 steps from 0 to 2
    ```

    The paper says 20 to 300 steps are enough, and advises checking that the
    credits add up. On the toy function, the output changes by 1. The credits
    add up to 0.9 with 20 steps and to 0.993 with 300 steps. So checking the
    sum tells you when to add more steps.

    On the copier, the output is the score at the last position for the
    right answer. The right answer is the token right after the first copy
    of the last token. The inputs are the token vectors at each position, and
    the baseline is all zeros, meaning no token at all. Each position's
    credit is the sum of the credits over its vector. On 200 rows, the score
    changes by 10.19 on average. The table shows how far the sum of the
    credits is from that change, on average.

    | Steps | Credits miss the change by |
    |---|---|
    | 1 | 3.61 |
    | 5 | 1.21 |
    | 20 | 0.31 |
    | 50 | 0.13 |
    | 300 | 0.02 |

    With one step, integrated gradients is the gradient at the input times
    the input, a common shortcut. It misses by about a third. With 50 steps,
    74% of the credit goes to three positions. They are the token after the
    copy with 28%, the copy itself with 24%, and the last token with 23%.
    Those are the positions that interp2's induction head reads.
    """,
)

INTERP4_SOURCE = source_file(
    '''
    # interp4.py

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    inputs = torch.tensor([3.0]).requires_grad_(True)
    (inputs ** 2).sum().backward()                    # inputs.grad is now tensor([6.])
    points = torch.linspace(0, 2, 5)[1:]              # tensor([0.5000, 1.0000, 1.5000, 2.0000]), 4 steps from 0 to 2


    # The copier model this chapter studies, and the function that trains it.
''' + indented(COPIER_PART) + '''


''' + indented(ANSWER_PART) + '''


    # TODO: Return the gradient of f(x).sum() with respect to x, without
    #   changing x. Work on a detached copy of x that requires a gradient.
    #   Example: gradient(lambda x: x ** 2, torch.tensor([3.0])) returns
    #   tensor([6.]).
    def gradient(
        f: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the integrated gradients of f at x from baseline, using
    #   `steps` steps. Average your gradient of f over the points
    #   baseline + (k / steps) * (x - baseline), for k from 1 to steps. Then
    #   multiply that average by x - baseline.
    #   Example: integrated_gradients(lambda x: 1 - torch.relu(1 - x),
    #   torch.tensor([2.0]), torch.zeros(1), 20) returns tensor([0.9000]).
    def integrated_gradients(
        f: Callable[[Tensor], Tensor],
        x: Tensor,
        baseline: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: Return each position's credit for each row's answer score, as a
    #   tensor of shape (batch, STEPS), one credit per position. The inputs are
    #   the token vectors model.embed.tokens(ids), detached. Take
    #   integrated_gradients of answer_scores with respect to those vectors,
    #   from a baseline of zeros, with `steps` steps. Then sum each position's
    #   credits over its vector.
    def position_attributions(
        model: Copier,
        ids: Tensor,
        answers: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    toy = integrated_gradients(lambda x: 1 - torch.relu(1 - x), torch.tensor([2.0]), torch.zeros(1), 20)
    model = train_copier(0, 500)
    ids, known = repeated_sequences(200, torch.Generator().manual_seed(99))
    answers, places = copied_answers(ids)
    credits = position_attributions(model, ids, answers, 50)
    ''',
)

INTERP4_SOLUTION = source_file(
    '''
    # interp4.py, a solution for comparison

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    inputs = torch.tensor([3.0]).requires_grad_(True)
    (inputs ** 2).sum().backward()
    points = torch.linspace(0, 2, 5)[1:]


''' + indented(COPIER_PART) + '''


''' + indented(ANSWER_PART) + '''


''' + indented(ATTRIBUTION_SOLUTION) + '''


    toy = integrated_gradients(lambda x: 1 - torch.relu(1 - x), torch.tensor([2.0]), torch.zeros(1), 20)
    model = train_copier(0, 500)
    ids, known = repeated_sequences(200, torch.Generator().manual_seed(99))
    answers, places = copied_answers(ids)
    credits = position_attributions(model, ids, answers, 50)
    ''',
)


def toy_ramp(x: Tensor) -> Tensor:
    return 1 - torch.relu(1 - x)


def square_sum(x: Tensor) -> Tensor:
    return (x**2).sum(dim=-1)


def sine_mix(x: Tensor) -> Tensor:
    return torch.sin(x[..., 0]) * x[..., 1] + x[..., 1] ** 3


def check_gradient(found: object) -> None:
    for f, text, x in (
        (square_sum, "the sum of squares of each row", torch.tensor([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]])),
        (sine_mix, "sin(x0) * x1 + x1 cubed", torch.tensor([[0.3, -1.2], [2.0, 0.7]])),
    ):
        before = x.clone()
        call = f"gradient of {text} at {value_text(x)}"
        actual = found(f, x)

        if not torch.equal(x, before) or x.requires_grad:
            raise Mismatch(f"{call} changed x itself. Work on x.detach().clone() with .requires_grad_(True).")

        expect_tensor(
            call=call,
            actual=actual,
            expected=REFERENCE["gradient"](f, before),
        )


def check_integrated_gradients(found: object) -> None:
    cases = (
        (toy_ramp, "1 - relu(1 - x)", torch.tensor([2.0]), torch.zeros(1), 20),
        (toy_ramp, "1 - relu(1 - x)", torch.tensor([2.0]), torch.zeros(1), 7),
        (sine_mix, "sin(x0) * x1 + x1 cubed", torch.tensor([[0.3, -1.2], [2.0, 0.7]]), torch.tensor([[0.0, 0.5], [-1.0, 0.0]]), 9),
    )

    for f, text, x, baseline, steps in cases:
        expect_close_tensor(
            call=f"integrated_gradients of {text} at {value_text(x)} from {value_text(baseline)}, {steps} steps",
            actual=found(f, x, baseline, steps),
            expected=REFERENCE["integrated_gradients"](f, x, baseline, steps),
        )


def check_position_attributions(found: object) -> None:
    for model, name, seed, steps in ((untrained_copier(680), "an untrained Copier(2, 16)", 681, 6), (trained_copier(), "the trained copier", 682, 4)):
        ids, _ = sequences(seed, 3)
        answers, _ = REFERENCE["copied_answers"](ids)
        expect_close_tensor(
            call=f"position_attributions of {name} on 3 rows, {steps} steps",
            actual=found(model, ids, answers, steps),
            expected=REFERENCE["position_attributions"](model, ids, answers, steps),
        )


def check_learned_toy(value: object) -> None:
    expect_close_tensor(
        call="toy",
        actual=value,
        expected=REFERENCE["integrated_gradients"](toy_ramp, torch.tensor([2.0]), torch.zeros(1), 20),
    )


def check_learned_credits(value: object) -> None:
    if not isinstance(value, Tensor) or tuple(value.shape) != (200, 24):
        raise Mismatch(f"credits is {value_text(value)}, but it should be a tensor of shape (200, 24).")

    model = trained_copier()
    ids, _ = sequences(99, 200)
    answers, _ = REFERENCE["copied_answers"](ids)

    with torch.no_grad():
        vectors = model.embed.tokens(ids)
        change = REFERENCE["answer_scores"](model, vectors, answers) - REFERENCE["answer_scores"](model, torch.zeros_like(vectors), answers)

    miss = (value.sum(dim=1) - change).abs().mean().item()

    if miss > 0.5:
        raise Mismatch(f"Your copier's credits miss the change in its answer scores by {miss:.2f} on average, but should add up to it within 0.5.")


def check_interp4(checker: Checker) -> None:
    checker.test(
        function="gradient",
        description="takes the gradient without changing x",
        run=check_gradient,
    )
    checker.test(
        function="integrated_gradients",
        description="averages the gradient along the straight line from the baseline",
        run=check_integrated_gradients,
    )
    checker.test(
        function="position_attributions",
        description="credits each position of each row",
        run=check_position_attributions,
    )
    checker.check_name(
        name="toy",
        description="integrated gradients give the toy's input the credit its gradient misses",
        run=check_learned_toy,
    )
    checker.check_name(
        name="credits",
        description="your copier's credits add up to the change in its answer",
        run=check_learned_credits,
    )


INTERP4 = Exercise(
    name="interp4",
    chapter=INTERPRETABILITY,
    title="Attribution",
    lesson=INTERP4_LESSON,
    source=INTERP4_SOURCE,
    solution=INTERP4_SOLUTION,
    hints={
        "gradient": "Detach x, clone it, and call requires_grad_(True), so autograd tracks the copy as a new leaf "
        "tensor. Call .backward() on the sum of f of the copy, and return the copy's .grad.",
        "integrated_gradients": "Loop k over range(1, steps + 1) and add up gradient at each point. Divide the total "
        "by steps, and multiply by x - baseline.",
        "position_attributions": "Pass integrated_gradients a lambda that runs answer_scores on the vectors it "
        "receives, and torch.zeros_like of the vectors as the baseline. Then take .sum(dim=-1) of the result.",
        "toy": "It comes from integrated_gradients. If it fails while the function passes, rerun the file.",
        "credits": "It comes from position_attributions. If it fails while the function passes, rerun the file.",
    },
    introduces=(),
    check=check_interp4,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# quiz17
# ---------------------------------------------------------------------------

QUIZ17_LESSON = prose(
    """
    This is the course's last quiz. Each part names a problem and a property
    that your answer must have. No part says which layer to use, and nothing
    in the file shows one. Choosing the layer is the task.

    | Name | Problem | What the check tests |
    |---|---|---|
    | `train_step` | one clipped step of training a classifier, in micro batches | the loss and the weights after two calls, against a reference step |
    | `ZeroStartBlock` | a block for a deep stack | it returns exactly its input at the start, and every weight gets a gradient after one step |
    | `LocalDetector` | a pattern on a ring of readings, wherever it appears | rotating the ring rotates the scores, and each score depends on only 5 steps |
    | `ReadingsEncoder` | readings with some missing | a missing reading changes nothing |
    | `SetCounter` | how many items in a set are marked | the items' order, any number of items, every item listed twice, and the trained count |
    | `TriangleScorer` | a triangle's shape and size | moving it, rotating it or listing its corners in another order |
    | `Recall` | the value stored with an asked key | the answer once trained, in sequences longer than any it trained on |

    The check builds each module with fixed seeds and tests its property
    directly, as design5 did. Every module must have weights to learn. Parts
    that need training are trained with the check's own recipe, which their
    TODO describes. When a part fails, the message says which property broke
    and by how much.
    """,
)

QUIZ17_SOURCE = source_file(
    '''
    # quiz17.py
    #
    # The course's last quiz. Each part names a problem and the property your
    # answer must have. Choosing the layer that fits is up to you.

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Run one training step of model on x and its class labels y. Split
    #   the batch into micro_batches equal pieces and run them one after
    #   another, as ddp2 did. Backpropagate each piece's cross entropy divided
    #   by micro_batches, so the gradients add up to the whole batch's
    #   gradient. Then clip that total gradient to a norm of max_norm and step
    #   the optimizer once. No gradient from an earlier call may carry over.
    #   Return the sum of the divided losses as a Python float.
    def train_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        x: Tensor,
        y: Tensor,
        micro_batches: int,
        max_norm: float,
    ) -> float:
        ...


    # TODO: Finish both methods. The block returns x plus a branch. The branch
    #   is a LayerNorm of width, a Linear from width to 4 * width, a GELU, and
    #   a Linear back to width. Choose the starting weights so the block
    #   returns exactly x at the start, and every weight still gets a gradient
    #   after one step.
    class ZeroStartBlock(nn.Module):
        def __init__(
            self,
            width: int,
        ) -> None:
            super().__init__()
            ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Finish both methods. x holds readings of shape (batch, channels,
    #   steps), taken around a ring, so the last step is next to the first.
    #   Return one score per step, of shape (batch, steps). A pattern should
    #   get the same score wherever it appears on the ring. Each score may
    #   depend only on the 5 steps centered on its own step. hidden is the
    #   width to use for the layers inside.
    class LocalDetector(nn.Module):
        def __init__(
            self,
            channels: int,
            hidden: int,
        ) -> None:
            super().__init__()
            ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Finish both methods. readings has shape (batch, steps, features).
    #   present has shape (batch, steps) and is False where a reading is
    #   missing. Return one vector of width numbers per row, of shape (batch,
    #   width). A missing reading must change nothing. Its numbers must not
    #   affect the result. And a row whose last readings are missing must give
    #   the same vector as that row cut off before them. Every row has at
    #   least one reading present.
    class ReadingsEncoder(nn.Module):
        def __init__(
            self,
            features: int,
            width: int,
        ) -> None:
            super().__init__()
            ...

        def forward(
            self,
            readings: Tensor,
            present: Tensor,
        ) -> Tensor:
            ...


    # TODO: Finish both methods. items has shape (batch, items, features),
    #   with one set of items per row. An item is marked when its first
    #   feature is 1. Return one number per row, of shape (batch,). The order
    #   of the items must not matter, and any number of items must work.
    #   Listing every item twice doubles the number of marked items, so it
    #   must change the result. The check trains the module for 800 steps of
    #   Adam at 0.003 on the squared error, over sets of 1 to 12 items. After
    #   that training, the number should be the count of marked items.
    class SetCounter(nn.Module):
        def __init__(
            self,
            features: int,
            hidden: int,
        ) -> None:
            super().__init__()
            ...

        def forward(self, items: Tensor) -> Tensor:
            ...


    # TODO: Finish both methods. corners has shape (batch, 3, 2), the three
    #   corners of a triangle on a plane. Return one number per triangle, of
    #   shape (batch,), that depends only on its shape and size. Moving the
    #   triangle, rotating it, or listing its corners in another order must
    #   not change the number. Different triangles should get different
    #   numbers.
    class TriangleScorer(nn.Module):
        def __init__(
            self,
            hidden: int,
        ) -> None:
            super().__init__()
            ...

        def forward(self, corners: Tensor) -> Tensor:
            ...


    # TODO: Finish both methods. tokens has shape (batch, steps, keys +
    #   values). Every step except the last holds one pair, the one hot vector
    #   of a key followed by the one hot vector of its value. No key repeats.
    #   The last step holds the one hot vector of the asked key, followed by
    #   zeros. Return logits over the values for the asked key's value, of
    #   shape (batch, values). The check trains the module for 300 steps of
    #   Adam at 0.01 on sequences of 8 pairs. After that, it should also
    #   answer sequences of 24 pairs.
    class Recall(nn.Module):
        def __init__(
            self,
            keys: int,
            values: int,
            width: int,
        ) -> None:
            super().__init__()
            ...

        def forward(self, tokens: Tensor) -> Tensor:
            ...
    ''',
)

QUIZ17_PARTS = '''
def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    micro_batches: int,
    max_norm: float,
) -> float:
    optimizer.zero_grad()
    total = 0.0

    for micro_x, micro_y in zip(x.chunk(micro_batches), y.chunk(micro_batches)):
        loss = F.cross_entropy(model(micro_x), micro_y) / micro_batches
        loss.backward()
        total += loss.item()

    # Clip only after the last piece, because one piece's gradient alone is not the whole batch's gradient.
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()

    return total


# Setting the branch's last layer to zero makes the block return x itself, and the layers before it still get
# gradients.
class ZeroStartBlock(nn.Module):
    def __init__(
        self,
        width: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

        with torch.no_grad():
            self.down.weight.zero_()
            self.down.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.down(F.gelu(self.up(self.norm(x))))


# A convolution's output shifts when its input shifts, and circular padding makes that hold at the ring's
# ends too. A kernel of 5, followed by a layer with kernel size 1, sees exactly 5 steps.
class LocalDetector(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, hidden, 5, padding=2, padding_mode="circular")
        self.score = nn.Conv1d(hidden, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.score(F.relu(self.conv(x))).squeeze(1)


# Encode each reading on its own, then average over the present readings only, so a missing one counts for
# nothing.
class ReadingsEncoder(nn.Module):
    def __init__(
        self,
        features: int,
        width: int,
    ) -> None:
        super().__init__()
        self.each = nn.Sequential(nn.Linear(features, width), nn.ReLU(), nn.Linear(width, width))

    def forward(
        self,
        readings: Tensor,
        present: Tensor,
    ) -> Tensor:
        encoded = torch.where(present.unsqueeze(-1), self.each(readings), 0.0)

        return encoded.sum(dim=1) / present.sum(dim=1, keepdim=True)


# A Deep Set. A sum over the items keeps how many there are, which a mean or a max would lose.
class SetCounter(nn.Module):
    def __init__(
        self,
        features: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Linear(hidden, 1)

    def forward(self, items: Tensor) -> Tensor:
        return self.rho(self.phi(items).sum(dim=1)).squeeze(-1)


# Side lengths don't change when the triangle moves or rotates, and a sum over the sides ignores their order.
class TriangleScorer(nn.Module):
    def __init__(
        self,
        hidden: int,
    ) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(1, hidden), nn.ReLU())
        self.rho = nn.Linear(hidden, 1)

    def forward(self, corners: Tensor) -> Tensor:
        sides = (corners - corners.roll(1, dims=1)).norm(dim=-1, keepdim=True)

        return self.rho(self.phi(sides).sum(dim=1)).squeeze(-1)


# Attention from the last step finds the pair by its key, wherever the pair is. With no position
# embeddings, any length works.
class Recall(nn.Module):
    def __init__(
        self,
        keys: int,
        values: int,
        width: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(keys + values, width)
        self.qkv = nn.Linear(width, 3 * width)
        self.head = nn.Linear(width, values)

    def forward(self, tokens: Tensor) -> Tensor:
        q, k, v = self.qkv(self.embed(tokens)).chunk(3, dim=-1)

        return self.head(F.scaled_dot_product_attention(q[:, -1:], k, v).squeeze(1))
'''

QUIZ17_SOLUTION = source_file(
    '''
    # quiz17.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


''' + indented(QUIZ17_PARTS) + '''
    ''',
)

QUIZ17_REFERENCE = namespace_of(QUIZ17_PARTS)

SET_FEATURES = 4
RECALL_KEYS = 32
RECALL_VALUES = 16


def quiz17_module(
    found: type,
    call: str,
    seed: int,
    *arguments: int,
) -> nn.Module:
    torch.manual_seed(seed)
    module = found(*arguments)

    if not isinstance(module, nn.Module):
        raise Mismatch(f"{call} is {value_text(module)}. It should be a module.")

    return module


def expect_rows(
    call: str,
    out: object,
    shape: tuple[int, ...],
) -> Tensor:
    reject_missing_return(
        call=call,
        actual=out,
    )

    if not isinstance(out, Tensor) or tuple(out.shape) != shape:
        raise Mismatch(f"{call} gives {value_text(out)}, but it should give shape {shape}.")

    return out


def largest_gap(
    first: Tensor,
    second: Tensor,
) -> float:
    return (first - second).abs().max().item()


def scale_of(out: Tensor) -> float:
    """How far apart two outputs may sit and still count as equal, float rounding and no more."""
    return 1e-5 * (1 + out.abs().max().item())


def check_has_weights(
    name: str,
    arguments: tuple[int, ...],
) -> Callable[[type], None]:
    def run(found: type) -> None:
        call = f"{name}{arguments}"
        module = quiz17_module(found, call, 1700, *arguments)

        if not any(parameter.requires_grad for parameter in module.parameters()):
            raise Mismatch(f"{call} holds no weights to learn. Build it from layers with weights.")

    return run


def check_train_step(found: object) -> None:
    # The first clips hard, so a clip of each piece alone shows. The second never clips, so the sum alone shows.
    for seed, max_norm, rows, micro_batches in ((1701, 0.05, 12, 3), (1702, 100.0, 8, 4)):
        torch.manual_seed(seed)
        mine = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        theirs = copy.deepcopy(mine)
        my_optimizer = torch.optim.SGD(mine.parameters(), lr=0.5, momentum=0.9)
        their_optimizer = torch.optim.SGD(theirs.parameters(), lr=0.5, momentum=0.9)
        generator = torch.Generator().manual_seed(seed)

        for number in (1, 2):
            x = 3 * torch.randn(rows, 4, generator=generator)
            y = torch.randint(0, 3, (rows,), generator=generator)
            call = (
                f"train_step call {number}, on {rows} rows in {micro_batches} micro batches, with a Linear 4 to 8, a "
                f"ReLU and a Linear 8 to 3, SGD at 0.5 and momentum 0.9, and max_norm {max_norm}"
            )
            loss = found(mine, my_optimizer, x, y, micro_batches, max_norm)
            wanted = QUIZ17_REFERENCE["train_step"](theirs, their_optimizer, x, y, micro_batches, max_norm)

            reject_missing_return(
                call=call,
                actual=loss,
            )

            if isinstance(loss, Tensor):
                raise Mismatch(f"{call} gives {value_text(loss)}, a tensor, but it should be a plain number. Use .item() on it.")

            expect_close(
                call=call,
                actual=loss,
                expected=wanted,
                tolerance=1e-5,
            )

            for (name, parameter), expected in zip(mine.named_parameters(), theirs.parameters()):
                if not torch.allclose(parameter, expected, atol=1e-5):
                    raise Mismatch(
                        f"After {call}, the weight {name} is off by up to {largest_gap(parameter, expected):.3g} from a "
                        "step that clears the old gradients, adds up every piece's divided gradient, then clips the "
                        f"total to a norm of {max_norm} and steps, once.",
                    )


def silent_weights(
    block: nn.Module,
    x: Tensor,
    target: Tensor,
) -> list[str]:
    """The weights with no gradient after one step of SGD, as design4's silent_after_one_step finds them."""
    copied = copy.deepcopy(block)
    optimizer = torch.optim.SGD(copied.parameters(), lr=0.1)
    F.mse_loss(copied(x), target).backward()
    optimizer.step()
    optimizer.zero_grad()
    F.mse_loss(copied(x), target).backward()

    return [name for name, parameter in copied.named_parameters() if parameter.grad is None or not parameter.grad.any()]


def check_zero_start(found: type) -> None:
    for width in (8, 12):
        call = f"ZeroStartBlock({width})"
        block = quiz17_module(found, call, 1710 + width, width)
        x = 2 * torch.randn(5, width, generator=torch.Generator().manual_seed(1720 + width))
        out = expect_rows(f"{call}(x of shape (5, {width}))", block(x), (5, width))

        if largest_gap(out, x) > 1e-6:
            raise Mismatch(f"{call} changes its input by up to {largest_gap(out, x):.3g} at the start, but it should return exactly x at the start.")


def check_zero_start_learns(found: type) -> None:
    call = "ZeroStartBlock(8)"
    block = quiz17_module(found, call, 1730, 8)
    generator = torch.Generator().manual_seed(1731)
    x = torch.randn(16, 8, generator=generator)
    target = torch.randn(16, 8, generator=generator)

    if not list(block.parameters()):
        raise Mismatch(f"{call} holds no weights to learn. Give it the branch's layers.")

    silent = silent_weights(block, x, target)

    if silent:
        raise Mismatch(
            f"After one step of SGD, these weights of {call} still get no gradient: {', '.join(silent)}. A layer at "
            "zero passes no gradient back to the layers before it.",
        )


def ring_readings(
    rows: int,
    steps: int,
    seed: int,
) -> Tensor:
    return torch.randn(rows, 3, steps, generator=torch.Generator().manual_seed(seed))


def check_detector_shape(found: type) -> None:
    detector = quiz17_module(found, "LocalDetector(3, 16)", 1740, 3, 16)

    for rows, steps in ((2, 12), (1, 31)):
        expect_rows(
            f"LocalDetector(3, 16)(x of shape ({rows}, 3, {steps}))",
            detector(ring_readings(rows, steps, 1741 + steps)),
            (rows, steps),
        )


def check_detector_shift(found: type) -> None:
    detector = quiz17_module(found, "LocalDetector(3, 16)", 1742, 3, 16)
    x = ring_readings(2, 16, 1743)
    out = expect_rows("LocalDetector(3, 16)(x of shape (2, 3, 16))", detector(x), (2, 16))

    for turn in (3, -5):
        turned = detector(torch.roll(x, turn, dims=2))
        gap = largest_gap(turned, torch.roll(out, turn, dims=1))

        if gap > scale_of(out):
            raise Mismatch(
                f"Rotating the ring by {turn} steps should rotate the scores by {turn} steps too, but they differ by up "
                f"to {gap:.3g}. A pattern should get the same score wherever it appears, the ring's ends included.",
            )


def check_detector_reach(found: type) -> None:
    detector = quiz17_module(found, "LocalDetector(3, 16)", 1744, 3, 16)
    x = ring_readings(1, 16, 1745)
    changed = x.clone()
    changed[:, :, 8] += 2 * torch.randn(3, generator=torch.Generator().manual_seed(1746))
    moved = (detector(changed) - detector(x)).abs()[0]

    for step in range(16):
        distance = min(abs(step - 8), 16 - abs(step - 8))

        if distance > 2 and moved[step] > 1e-6:
            raise Mismatch(
                f"Changing step 8 moved the score at step {step}, {distance} steps away, by {moved[step]:.3g}. Each "
                "score should see only the 5 steps centered on its own.",
            )

    for step in (6, 10):
        if moved[step] <= 1e-6:
            raise Mismatch(
                f"Changing step 8 left the score at step {step}, 2 steps away, where it was. Each score should see "
                "the 5 steps centered on its own, 2 on each side.",
            )


def readings_with_gaps(seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    readings = torch.randn(4, 7, 5, generator=generator)
    present = torch.rand(4, 7, generator=generator) < 0.6
    present[:, 0] = True

    return readings, present


def check_readings_shape(found: type) -> None:
    encoder = quiz17_module(found, "ReadingsEncoder(5, 12)", 1750, 5, 12)
    readings, present = readings_with_gaps(1751)
    expect_rows("ReadingsEncoder(5, 12)(readings of shape (4, 7, 5), present)", encoder(readings, present), (4, 12))
    expect_rows(
        "ReadingsEncoder(5, 12)(readings of shape (2, 1, 5), both present)",
        encoder(readings[:2, :1], present[:2, :1]),
        (2, 12),
    )


def check_readings_ignored(found: type) -> None:
    encoder = quiz17_module(found, "ReadingsEncoder(5, 12)", 1752, 5, 12)
    readings, present = readings_with_gaps(1753)
    out = expect_rows("ReadingsEncoder(5, 12)(readings of shape (4, 7, 5), present)", encoder(readings, present), (4, 12))
    noise = 100 * torch.randn(readings.shape, generator=torch.Generator().manual_seed(1754))
    filled = torch.where(present.unsqueeze(-1), readings, noise)
    gap = largest_gap(encoder(filled, present), out)

    if gap > scale_of(out):
        raise Mismatch(
            f"Changing the numbers of the missing readings moved the result by up to {gap:.3g}. A missing reading's "
            "numbers shouldn't count at all.",
        )


def check_readings_shorter(found: type) -> None:
    encoder = quiz17_module(found, "ReadingsEncoder(5, 12)", 1755, 5, 12)
    generator = torch.Generator().manual_seed(1756)
    readings = torch.randn(3, 6, 5, generator=generator)
    lengths = (2, 6, 4)
    present = torch.arange(6).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    out = expect_rows("ReadingsEncoder(5, 12)(readings of shape (3, 6, 5), present)", encoder(readings, present), (3, 12))

    for row, length in enumerate(lengths):
        alone = encoder(readings[row : row + 1, :length], present[row : row + 1, :length])
        gap = largest_gap(alone[0], out[row])

        if gap > scale_of(out):
            raise Mismatch(
                f"Row {row} has {length} readings present and the rest missing at the end. Its result differs by up to "
                f"{gap:.3g} from the same row cut to its {length} readings. Missing readings should count for nothing.",
            )


def marked_items(
    rows: int,
    items: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    x = torch.randn(rows, items, SET_FEATURES, generator=generator)
    x[..., 0] = torch.randint(0, 2, (rows, items), generator=generator).float()

    return x, x[..., 0].sum(dim=1)


def check_counter_sizes(found: type) -> None:
    counter = quiz17_module(found, "SetCounter(4, 32)", 1760, SET_FEATURES, 32)
    generator = torch.Generator().manual_seed(1761)

    for items in (5, 1, 20):
        expect_rows(
            f"SetCounter(4, 32)(items of shape (3, {items}, 4))",
            counter(marked_items(3, items, generator)[0]),
            (3,),
        )


def check_counter_order(found: type) -> None:
    counter = quiz17_module(found, "SetCounter(4, 32)", 1762, SET_FEATURES, 32)
    items, _ = marked_items(3, 6, torch.Generator().manual_seed(1763))
    out = expect_rows("SetCounter(4, 32)(items of shape (3, 6, 4))", counter(items), (3,))
    gap = largest_gap(counter(items[:, torch.tensor([4, 2, 0, 5, 1, 3])]), out)

    if gap > scale_of(out):
        raise Mismatch(f"Putting the items in another order moved the result by up to {gap:.3g}. A set's items have no order.")


def counting_error(
    counter: nn.Module,
    items: int,
    held: torch.Generator,
) -> float:
    sets, counts = marked_items(200, items, held)

    with torch.no_grad():
        return (counter(sets) - counts).abs().mean().item()


def check_counter_counts(found: type) -> None:
    counter = quiz17_module(found, "SetCounter(4, 32)", 1764, SET_FEATURES, 32)
    generator = torch.Generator().manual_seed(1765)
    optimizer = torch.optim.Adam(counter.parameters(), lr=0.003)
    expect_rows("SetCounter(4, 32)(items of shape (2, 3, 4))", counter(marked_items(2, 3, generator)[0]), (2,))

    for step in range(800):
        items, counts = marked_items(64, 1 + step % 12, generator)
        loss = F.mse_loss(counter(items), counts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    held = torch.Generator().manual_seed(1766)
    error = sum(counting_error(counter, items, held) for items in range(1, 13)) / 12

    # A loose bar on purpose: a right module from other starting weights can land near 0.8 after 800 steps.
    # check_counter_doubled is the sharp test of what joins the items.
    if error > 1:
        raise Mismatch(
            f"After the check's 800 steps, the counts are off by {error:.2f} on average for sets of 1 to 12 items, "
            "but they should be off by under 1. The number has to be able to reach every count, so nothing after "
            "the join may squash it.",
        )


def check_counter_doubled(found: type) -> None:
    counter = quiz17_module(found, "SetCounter(4, 32)", 1767, SET_FEATURES, 32)
    items, _ = marked_items(3, 6, torch.Generator().manual_seed(1768))
    out = expect_rows("SetCounter(4, 32)(items of shape (3, 6, 4))", counter(items), (3,))
    doubled = expect_rows(
        "SetCounter(4, 32)(the same items listed twice, shape (3, 12, 4))",
        counter(torch.cat([items, items], dim=1)),
        (3,),
    )
    gap = largest_gap(doubled, out)

    if gap <= scale_of(out):
        raise Mismatch(
            f"Listing every item twice moved the numbers by only {gap:.3g}, float rounding. That doubles the marked "
            "items, so the count should change. What joins the items has to keep how many there are.",
        )


def triangles(seed: int) -> Tensor:
    return torch.randn(4, 3, 2, generator=torch.Generator().manual_seed(seed))


def turned(
    corners: Tensor,
    angle: float,
) -> Tensor:
    turn = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    return corners @ turn.T


def check_triangle_shape(found: type) -> None:
    scorer = quiz17_module(found, "TriangleScorer(16)", 1770, 16)
    expect_rows("TriangleScorer(16)(corners of shape (4, 3, 2))", scorer(triangles(1771)), (4,))


def check_triangle_moves(found: type) -> None:
    scorer = quiz17_module(found, "TriangleScorer(16)", 1772, 16)
    corners = triangles(1773)
    out = expect_rows("TriangleScorer(16)(corners of shape (4, 3, 2))", scorer(corners), (4,))
    offsets = 3 * torch.randn(4, 1, 2, generator=torch.Generator().manual_seed(1774))

    for change, moved in (
        ("Moving each triangle", corners + offsets),
        ("Rotating each triangle by 0.7", turned(corners, 0.7)),
        ("Rotating each triangle by 2.5", turned(corners, 2.5)),
    ):
        gap = largest_gap(scorer(moved), out)

        # Moving and turning round the corners themselves, so this allows ten times the usual rounding.
        if gap > 10 * scale_of(out):
            raise Mismatch(f"{change} moved its number by up to {gap:.3g}. The number should depend on the shape and size alone.")


def check_triangle_relisted(found: type) -> None:
    scorer = quiz17_module(found, "TriangleScorer(16)", 1775, 16)
    corners = triangles(1776)
    out = expect_rows("TriangleScorer(16)(corners of shape (4, 3, 2))", scorer(corners), (4,))

    for order in ([2, 0, 1], [1, 0, 2]):
        gap = largest_gap(scorer(corners[:, order]), out)

        if gap > scale_of(out):
            raise Mismatch(
                f"Listing the corners in the order {order} moved the number by up to {gap:.3g}. The same triangle "
                "should get the same number whichever corner comes first.",
            )


def check_triangle_apart(found: type) -> None:
    scorer = quiz17_module(found, "TriangleScorer(16)", 1777, 16)
    corners = triangles(1778)
    out = expect_rows("TriangleScorer(16)(corners of shape (4, 3, 2))", scorer(corners), (4,))
    doubled = scorer(2 * corners)

    if (out[1:] - out[0]).abs().min().item() < 1e-4:
        raise Mismatch(f"Four different triangles got the numbers {value_text(out)}. Different triangles should get different numbers.")

    if (doubled - out).abs().min().item() < 1e-4:
        raise Mismatch("Doubling each triangle's size left a number where it was. Size should change the number too.")


def recall_batch(
    rows: int,
    pairs: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    keys = torch.rand(rows, RECALL_KEYS, generator=generator).argsort(dim=1)[:, :pairs]
    values = torch.randint(0, RECALL_VALUES, (rows, pairs), generator=generator)
    asked = torch.randint(0, pairs, (rows, 1), generator=generator)
    tokens = torch.zeros(rows, pairs + 1, RECALL_KEYS + RECALL_VALUES)
    tokens[:, :pairs] = torch.cat([F.one_hot(keys, RECALL_KEYS), F.one_hot(values, RECALL_VALUES)], dim=-1).float()
    tokens[:, pairs, :RECALL_KEYS] = F.one_hot(keys.gather(1, asked).squeeze(1), RECALL_KEYS).float()

    return tokens, values.gather(1, asked).squeeze(1)


def longer_sequences(
    recall: nn.Module,
    tokens: Tensor,
) -> object:
    """Recall run on sequences longer than any it trained on, where a layer sized to one length crashes."""
    try:
        with torch.no_grad():
            return recall(tokens)
    except (RuntimeError, IndexError) as error:
        raise Mismatch(
            f"Recall crashed on sequences of {tokens.shape[1] - 1} pairs: {error}. It should take sequences of any "
            "length, so no layer in it can be sized to one length, such as a table of positions.",
        ) from None


def check_recall_shape(found: type) -> None:
    recall = quiz17_module(found, "Recall(32, 16, 32)", 1780, RECALL_KEYS, RECALL_VALUES, 32)
    generator = torch.Generator().manual_seed(1781)
    expect_rows(
        "Recall(32, 16, 32)(tokens of 8 pairs and the asked key, shape (3, 9, 48))",
        recall(recall_batch(3, 8, generator)[0]),
        (3, RECALL_VALUES),
    )
    expect_rows(
        "Recall(32, 16, 32)(tokens of 24 pairs and the asked key, shape (3, 25, 48))",
        longer_sequences(recall, recall_batch(3, 24, generator)[0]),
        (3, RECALL_VALUES),
    )


def check_recall_answers(found: type) -> None:
    recall = quiz17_module(found, "Recall(32, 16, 32)", 1782, RECALL_KEYS, RECALL_VALUES, 32)
    generator = torch.Generator().manual_seed(1783)
    optimizer = torch.optim.Adam(recall.parameters(), lr=0.01)
    expect_rows("Recall(32, 16, 32)(tokens of shape (2, 9, 48))", recall(recall_batch(2, 8, generator)[0]), (2, RECALL_VALUES))

    for _ in range(300):
        tokens, answers = recall_batch(64, 8, generator)
        loss = F.cross_entropy(recall(tokens), answers)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    held = torch.Generator().manual_seed(1784)

    for pairs in (8, 24):
        tokens, answers = recall_batch(500, pairs, held)
        right = (longer_sequences(recall, tokens).argmax(dim=1) == answers).float().mean().item()

        if right < 0.9:
            raise Mismatch(
                f"After the check's 300 steps, Recall answers {right:.0%} of new sequences of {pairs} pairs right, but "
                "it should answer at least 90%. The answer is found by its key, wherever the pair sits.",
            )


# Each module part's arguments for the weights test, then its tests of forward.
QUIZ17_MODULE_TESTS: dict[str, tuple[tuple[int, ...], tuple[tuple[str, Callable[[type], None]], ...]]] = {
    "LocalDetector": (
        (3, 16),
        (
            ("gives one score per step, at any length", check_detector_shape),
            ("rotates its scores when the ring rotates", check_detector_shift),
            ("sees the 5 steps centered on each score", check_detector_reach),
        ),
    ),
    "ReadingsEncoder": (
        (5, 12),
        (
            ("gives one vector per row", check_readings_shape),
            ("ignores the numbers of missing readings", check_readings_ignored),
            ("treats missing readings at the end as a shorter row", check_readings_shorter),
        ),
    ),
    "SetCounter": (
        (SET_FEATURES, 32),
        (
            ("gives one number per set, for any number of items", check_counter_sizes),
            ("ignores the order of the items", check_counter_order),
            ("changes when every item is listed twice", check_counter_doubled),
            ("counts the marked items once trained", check_counter_counts),
        ),
    ),
    "TriangleScorer": (
        (16,),
        (
            ("gives one number per triangle", check_triangle_shape),
            ("keeps its number when a triangle moves or rotates", check_triangle_moves),
            ("keeps its number when the corners are relisted", check_triangle_relisted),
            ("tells different triangles apart", check_triangle_apart),
        ),
    ),
    "Recall": (
        (RECALL_KEYS, RECALL_VALUES, 32),
        (
            ("gives logits over the values, at any length", check_recall_shape),
            ("finds the asked value once trained, in longer sequences too", check_recall_answers),
        ),
    ),
}


def check_quiz17(checker: Checker) -> None:
    checker.test(
        function="train_step",
        description="adds up the micro batches, then clips and steps once, and gives the loss",
        run=check_train_step,
    )
    checker.test_method(
        class_name="ZeroStartBlock",
        method="__init__",
        description="starts as exactly its input",
        run=check_zero_start,
    )
    checker.test_method(
        class_name="ZeroStartBlock",
        method="forward",
        description="gets a gradient to every weight after one step",
        run=check_zero_start_learns,
    )

    for name, (arguments, forward_tests) in QUIZ17_MODULE_TESTS.items():
        checker.test_method(
            class_name=name,
            method="__init__",
            description="holds weights to learn",
            run=check_has_weights(name, arguments),
        )

        for description, run in forward_tests:
            checker.test_method(
                class_name=name,
                method="forward",
                description=description,
                run=run,
            )


QUIZ17 = Exercise(
    name="quiz17",
    chapter=QUIZZES,
    title="The whole course, a module for each problem",
    lesson=QUIZ17_LESSON,
    source=QUIZ17_SOURCE,
    solution=QUIZ17_SOLUTION,
    hints={
        "train_step": "Zero the gradients once, before the first piece. Loop over matching chunks of x and y as "
        "ddp2 did, and add each divided loss's .item() to a running total. Call nn.utils.clip_grad_norm_ and step "
        "after the loop, because only then do the gradients cover the whole batch.",
        "ZeroStartBlock.__init__": "A branch whose last layer is all zeros adds nothing to x. Zero that one layer's "
        "weight and bias under torch.no_grad(), and leave the layers before it with their random starting values.",
        "ZeroStartBlock.forward": "Return x plus the branch, with its layers in the order the TODO lists them. A "
        "layer at zero blocks the gradient to the layers before it, so set only the last layer to zero.",
        "LocalDetector.__init__": "Give it a layer that slides the same weights along the steps.",
        "LocalDetector.forward": "A Conv1d with kernel_size 5 sees 2 steps on each side. Padding of 2 with "
        "padding_mode circular wraps around the ring. A second Conv1d with kernel_size 1 after it keeps each score's "
        "view at 5 steps.",
        "ReadingsEncoder.__init__": "Give it layers that read one reading at a time.",
        "ReadingsEncoder.forward": "Encode every reading, then set the missing ones to 0 with torch.where. Divide "
        "the sum over the steps by how many readings are present, not by the number of steps.",
        "SetCounter.__init__": "Give it one network to run on each item and another to run on the combined result.",
        "SetCounter.forward": "Run a network on every item, then sum over the items. A sum is the same in any order "
        "and grows with the number of items. A mean or a max would hide that number.",
        "TriangleScorer.__init__": "Give it a small network to run on a quantity computed from the corners.",
        "TriangleScorer.forward": "The lengths of the three sides stay the same when a triangle moves or rotates. "
        "Run a network on each length and sum the three results, so their order does not matter.",
        "Recall.__init__": "Give it a Linear layer into width, the query, key and value projections that attention "
        "needs, and a Linear layer to the values.",
        "Recall.forward": "Compare the last step's query with every step's key, and take the sum of the values "
        "weighted by those attention weights. Leave out positions, so any length works.",
    },
    introduces=(),
    check=check_quiz17,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_interpretability() -> dict[str, object]:
    """Every number in the four lessons, on the CPU with 4 threads, on the model every file trains."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        return measured_numbers()
    finally:
        torch.set_num_threads(threads)


def measured_numbers() -> dict[str, object]:
    reference = REFERENCE
    model = reference["train_copier"](0, LESSON_STEPS)
    one = reference["train_copier"](0, LESSON_STEPS, 1)
    ids, known = sequences(99, 500)
    results: dict[str, object] = {
        "right": reference["accuracy"](model, ids, known),
        "one block right": reference["accuracy"](one, ids, known),
    }

    outputs = reference["record_outputs"](model, [model.embed, *model.blocks], ids)
    results["next token probes"] = reference["probe_scores"](model, ids, known)
    results["previous token probes, positions 1 to 11"] = [
        reference["fit_probe"](states[:, 1:12].flatten(0, 1), ids[:, :11].flatten()) for states in outputs
    ]

    found = reference["patterns"](model, ids)
    results["scores by block"] = [(reference["previous_token_score"](p), reference["induction_score"](p, ids)) for p in found]
    single = reference["patterns"](one, ids)[0]
    results["one block scores"] = (reference["previous_token_score"](single), reference["induction_score"](single, ids))
    results["first block weight before, by position"] = torch.diagonal(found[0], offset=-1, dim1=-2, dim2=-1).mean(dim=0).tolist()

    results["zeroed"] = [reference["ablated_accuracy"](model, ids, known, block.attention) for block in model.blocks]
    means = []

    for block in model.blocks:
        average = reference["record_outputs"](model, [block.attention], ids)[0].mean(dim=(0, 1))
        means.append(mean_ablated(model, ids, known, block.attention, average))

    results["mean ablated"] = means
    corrupted, clean_answers, corrupted_answers = reference["corrupt"](ids)

    with torch.no_grad():
        results["clean difference"] = reference["logit_difference"](model(ids), clean_answers, corrupted_answers)
        results["corrupted difference"] = reference["logit_difference"](model(corrupted), clean_answers, corrupted_answers)

    results["patched"] = [
        reference["logit_difference"](
            reference["patched_scores"](model, block.attention, corrupted, reference["record_outputs"](model, [block.attention], ids)[0]),
            clean_answers,
            corrupted_answers,
        )
        for block in model.blocks
    ]

    results["toy, 20 steps"] = reference["integrated_gradients"](toy_ramp, torch.tensor([2.0]), torch.zeros(1), 20).item()
    results["toy, 300 steps"] = reference["integrated_gradients"](toy_ramp, torch.tensor([2.0]), torch.zeros(1), 300).item()
    results.update(measured_credits(model))

    return results


def mean_ablated(
    model: nn.Module,
    ids: Tensor,
    known: Tensor,
    module: nn.Module,
    average: Tensor,
) -> float:
    handle = module.register_forward_hook(lambda module, inputs, output: average.expand_as(output))

    try:
        return REFERENCE["accuracy"](model, ids, known)
    finally:
        handle.remove()


def measured_credits(model: nn.Module) -> dict[str, object]:
    reference = REFERENCE
    ids, _ = sequences(99, 200)
    answers, places = reference["copied_answers"](ids)
    rows = torch.arange(len(ids))

    with torch.no_grad():
        vectors = model.embed.tokens(ids)
        change = reference["answer_scores"](model, vectors, answers) - reference["answer_scores"](model, torch.zeros_like(vectors), answers)

    results: dict[str, object] = {"score change": change.mean().item()}

    for steps in (1, 5, 20, 50, 300):
        credits = reference["position_attributions"](model, ids, answers, steps)
        results[f"miss, {steps} steps"] = (credits.sum(dim=1) - change).abs().mean().item()

        if steps != 50:
            continue

        total = credits.sum(dim=1)
        answer_share = (credits[rows, places] / total).mean().item()
        match_share = (credits[rows, places - 1] / total).mean().item()
        last_share = (credits[:, -1] / total).mean().item()
        results["shares, answer place, copy, last"] = (answer_share, match_share, last_share, answer_share + match_share + last_share)

    vectors = model.embed.tokens(ids).detach()
    shortcut = (reference["gradient"](lambda v: reference["answer_scores"](model, v, answers), vectors) * vectors).sum(dim=(1, 2))
    results["gradient times input miss"] = (shortcut - change).abs().mean().item()

    return results
