"""107_interpretability, a small trained model opened up: hooks and probes on its hidden states, the induction head its attention patterns show, ablation and activation patching to test what it uses, and attribution by gradients and integrated gradients."""

from __future__ import annotations

import functools
import textwrap
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
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

        A model that works says nothing about how it works. This chapter
        trains a small one, two layers of attention that copy from earlier in
        a sequence, and opens it up. Hooks and probes read its hidden states,
        attention patterns show the induction head that induction1 built by
        hand, ablation and patching test what it uses, and attribution traces
        an answer back to its inputs.

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


# Each row repeats a segment of 6 to 12 different tokens until it is 24 long. known marks the
# positions whose next token an earlier copy gives away.
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

    # Row t holds how much position t takes from each position up to t.
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


# The share of the known positions whose next token model gets right.
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
# topics2's linear probe, trained on the first half of the rows and scored on the second.
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
# Swaps the token right after the last token's first copy, so the right answer at the last position
# changes. Gives the corrupted ids, the clean answers and the corrupted ones.
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
# The score of each row's answer at the last position, run from token vectors in place of
# model.embed.tokens(ids), with the positions added as embed adds them.
def answer_scores(
    model: Copier,
    token_vectors: Tensor,
    answers: Tensor,
) -> Tensor:
    x = token_vectors + model.embed.positions(torch.arange(token_vectors.shape[1]))

    for block in model.blocks:
        x = block(x)

    return model.head(x)[torch.arange(len(answers)), -1, answers]


# Each row's answer, the token right after the last token's first copy, and that place.
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
    This chapter opens up a trained model to see how it works. The model is
    small on purpose. Two layers of attention read a row of tokens, and a
    Linear head scores the next token at every position.

    It learns from repeated rows. Each row repeats a segment of 6 to 12
    different tokens until it is 24 long. Within the first copy, the next token
    can't be known, and guessing gets 1 in 32. After that, an earlier copy
    gives it away. `train_copier` trains one in a few seconds, and it gets
    99.6% of those positions right.

    A forward hook reads a module's output while the model runs, with no
    change to the model's code. PyTorch calls the hook with the module, its
    inputs and its output each time the module runs. A handle takes it off.

    ```python
    handle = model.head.register_forward_hook(lambda module, inputs, output: print(output.shape))
    handle.remove()    # gone, so the next run prints nothing
    ```

    Each block adds its output to what came before. So the outputs of `embed`
    and the two blocks show the model's state at three points. A probe, as
    topics2 trained one, asks whether a Linear can read a fact from those
    states. Here is what probes read from 500 new rows.

    | Read from | The next token, where a copy gives it | The previous token, at positions 1 to 11 |
    |---|---|---|
    | `embed` | 3.3% | 3.3% |
    | `blocks[0]` | 6.9% | 100% |
    | `blocks[1]` | 99.8% | 97.5% |

    The answer shows up only after the second block. The first block writes
    each position's previous token into it early in the row, where interp2
    shows the second block needs it.

    A probe shows that a fact is there to read. It can't show that the model
    uses it. interp3 tests that by changing the model and watching the answer.
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
    handle = hooked.register_forward_hook(lambda module, inputs, output: None)    # runs after every call of hooked
    handle.remove()                                                              # and now never again
    picked = torch.arange(6).view(2, 3)[torch.tensor([[True, False, True], [False, True, False]])]    # tensor([0, 2, 4])


    # The copier this chapter studies, and its training.
''' + indented(COPIER_PART) + '''


''' + indented(FIT_PROBE_PART) + '''


    # TODO: Return the output of each module in modules, in the order the
    #   model runs them, from one run of model on ids. Put a forward hook on
    #   each module that keeps its output, run the model without recording
    #   for autograd, and take every hook off again, even when the run crashes.
    def record_outputs(
        model: nn.Module,
        modules: list[nn.Module],
        ids: Tensor,
    ) -> list[Tensor]:
        ...


    # TODO: Return how well a probe reads the next token from the outputs of
    #   model.embed and of each block in model.blocks, as a list of plain
    #   numbers. Read only the positions known marks. From outputs of shape
    #   (batch, steps, size), the states are outputs[:, :-1][known] and the
    #   targets ids[:, 1:][known]. Score each with fit_probe.
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
        raise Mismatch(f"After the second block a probe reads {value[2]:.1%} of next tokens, but it should pass 90%.")

    if value[0] > 0.2:
        raise Mismatch(f"From embed a probe reads {value[0]:.1%} of next tokens, but it should stay near guessing, below 20%.")


def check_interp1(checker: Checker) -> None:
    checker.test(
        function="record_outputs",
        description="keeps each module's output with hooks, then takes them off",
        run=check_record_outputs,
    )
    checker.test(
        function="record_outputs",
        description="takes the hooks off even when the model crashes",
        run=check_hooks_come_off,
    )
    checker.test(
        function="probe_scores",
        description="probes the next token at each point of the model",
        run=check_probe_scores,
    )
    checker.check_name(
        name="scores",
        description="your copier's answer shows up only after the second block",
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
        "record_outputs": "Each hook can append its output argument to one list. Keep the handles that "
        "register_forward_hook gives, run the model inside torch.no_grad() within try, and remove every "
        "handle in finally.",
        "probe_scores": "record_outputs with [model.embed, *model.blocks] gives the three outputs. For each, "
        "pass fit_probe the states and targets the TODO names.",
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
    position's previous token into it. The second layer's query looks for the
    current token among those copies, and takes the token that followed. Did
    training find the same circuit? Attention patterns can show it.

    `pattern(x)` in each block is the attention's weights, one row per
    position. Row t says how much position t takes from each position up to
    t. Two scores read a head.

    - The previous token score: the average weight on the place right before, which `torch.diagonal` with `offset=-1` picks out.
    - The induction score: on positions whose token showed up earlier, the weight on the place right after that earlier copy.

    Olsson and colleagues call the second one prefix matching.

    Here are both scores for each block of the trained copier, on 500 new
    rows, beside a copier with one block, trained the same way.

    | Model and block | Previous token score | Induction score | Right where a copy gives it |
    |---|---|---|---|
    | Two blocks, `blocks[0]` | 0.44 | 0.09 | 99.6% |
    | Two blocks, `blocks[1]` | 0.07 | 0.96 | |
    | One block | 0.06 | 0.16 | 36.9% |

    Training found induction1's circuit. The second block is an induction
    head, and one block alone can't make one. Olsson and colleagues say the
    same, since an induction head needs heads in two layers working together.

    The first block's score of 0.44 is an average, and by position it splits
    in two. Its weight on the place before is 0.99 or more at places 1 to 5
    and 0.62 to 0.90 at places 6 to 11. From place 12 on it stays under 0.08.
    Every token's first copy sits in the first segment, within the first 12
    places. So the second block can always find its answer there, and
    training put the previous token only where it is read. A pattern shows
    how a circuit fits its data, which a final score never will.
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
    before = torch.diagonal(weights, offset=-1, dim1=-2, dim2=-1)    # tensor([[0.3333, 0.3333]]), each row's weight on the place before
    tokens = torch.tensor([[5, 7, 5]])
    same = tokens[:, :, None] == tokens[:, None, :]                   # same[0, 2, 0] is True, since places 2 and 0 hold token 5


    # The copier this chapter studies, and its training.
''' + indented(COPIER_PART) + '''


    # TODO: Return each block's attention pattern on ids, in order, from
    #   block.attention.pattern of the block's own input. The first block reads
    #   model.embed(ids), and each block's output is the next one's input.
    #   Record nothing for autograd.
    def patterns(
        model: Copier,
        ids: Tensor,
    ) -> list[Tensor]:
        ...


    # TODO: Return the previous token score of pattern, of shape (batch, steps,
    #   steps), as a plain number. It is the average weight each row puts on
    #   the place right before its own.
    def previous_token_score(pattern: Tensor) -> float:
        ...


    # TODO: Return the induction score of pattern for ids, as a plain number.
    #   A target of row t is the place s + 1, wherever ids holds the same token
    #   at s and at t, with s before t. Sum each row's weight on its targets,
    #   then average over the rows that have at least one target.
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
        description="averages the weight on the place before",
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
        description="averages the weight on the place after an earlier copy",
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
        "patterns": "Start from x = model.embed(ids) inside torch.no_grad(). For each block, keep "
        "block.attention.pattern(x) in a list, then move on with x = block(x).",
        "previous_token_score": "The file's first lines show torch.diagonal with offset=-1. Average what it gives "
        "and turn it into a plain number.",
        "induction_score": "Compare every pair of places as the file's same line does, and keep the pairs with s "
        "before t using torch.tril with diagonal=-1. Shift that mask one place along the last dimension to land on "
        "s + 1, multiply by the pattern, and sum the last dimension.",
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
    Probes show what a model holds, and patterns show where it looks. Neither
    shows what it uses. For that, change the model and watch the answer.

    A forward hook that returns a tensor replaces the module's output with
    it. Zero ablation returns zeros, so the module's work is gone and
    everything after it runs as usual.

    ```python
    torch.zeros_like(torch.ones(2, 3))    # tensor([[0., 0., 0.], [0., 0., 0.]])
    ```

    With either block's attention zeroed, the trained copier gets this share
    of the knowable next tokens right, on 500 new rows.

    | Attention zeroed | Right where a copy gives it |
    |---|---|
    | None | 99.6% |
    | `blocks[0]` | 21.6% |
    | `blocks[1]` | 5.0% |

    Both blocks matter. Zeros are far from anything the model saw in
    training, though, so a large drop can come from the shock alone.
    Replacing the output with its average over many rows, mean ablation, is
    gentler. It gives 21.9% and 4.5% here, about the same, so the drops are
    real.

    Activation patching asks a sharper question. Take a clean row and a
    corrupted one that differs in one key token. Run the corrupted row, but
    swap one module's output for its value from the clean run. If the clean
    answer comes back, that module carries it. Heimersheim and Nanda call
    this direction denoising.

    Here `corrupt` swaps the token after the last token's first copy, so the
    right answer changes. The measure is the logit difference, the clean
    answer's score minus the corrupted one's at the last position. Heimersheim
    and Nanda trust it most among the measures they compare.

    | Run | Logit difference |
    |---|---|
    | Clean | 11.14 |
    | Corrupted | -9.26 |
    | Corrupted, `blocks[0]` patched from clean | -8.91 |
    | Corrupted, `blocks[1]` patched from clean | 10.92 |

    Patching the second block brings the clean answer back. Patching the
    first does nothing, though zeroing it broke the model. The first block
    does copy the swapped token one place forward, as the next position's
    previous token. No answer reads it there. The second block reads the
    swapped token straight from the embedding at its own place. So the path
    that carries the answer skips the first block.
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
    replace = lambda module, inputs, output: blank    # as a forward hook, this swaps the output for blank


    # The copier this chapter studies, and its training.
''' + indented(COPIER_PART) + '''


    # interp1's record_outputs.
''' + indented(RECORD_SOLUTION) + '''


''' + indented(CORRUPT_PART) + '''


    # TODO: Return accuracy of model on ids and known with the output of
    #   module replaced by zeros of the same shape. Take the hook off again
    #   afterwards, even when the run crashes.
    def ablated_accuracy(
        model: nn.Module,
        ids: Tensor,
        known: Tensor,
        module: nn.Module,
    ) -> float:
        ...


    # TODO: Return model's scores on ids with the output of module replaced
    #   by replacement. Record nothing for autograd, and take the hook off
    #   again afterwards, even when the run crashes.
    def patched_scores(
        model: nn.Module,
        module: nn.Module,
        ids: Tensor,
        replacement: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the logit difference as a plain number. At the last
    #   position of scores, of shape (batch, steps, vocabulary), take each
    #   row's score for its clean answer minus its score for its corrupted
    #   answer, and average over the rows.
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
        description="takes the hook off even when the model crashes",
        run=check_ablation_hooks_come_off,
    )
    checker.test(
        function="patched_scores",
        description="runs the model with one module's output swapped in",
        run=check_patched_scores,
    )
    scores = torch.randn(4, 5, 7, generator=torch.Generator().manual_seed(676))
    check_on_inputs(
        checker=checker,
        function="logit_difference",
        description="averages the clean answer's lead at the last position",
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
        "ablated_accuracy": "Register a forward hook on module that returns torch.zeros_like of its output, "
        "return accuracy of the model inside try, and remove the handle in finally.",
        "patched_scores": "The same shape as ablated_accuracy, with a hook that returns replacement, and the model "
        "run inside torch.no_grad().",
        "logit_difference": "scores[:, -1] holds the last position. Index it with torch.arange of the batch and each "
        "set of answers, subtract, then .mean().item().",
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
    Attribution asks which inputs an answer came from. The plainest way is
    the gradient. It says how far the output moves when an input moves a
    little.

    Sundararajan and colleagues showed where that fails. Take f(x) = 1 -
    ReLU(1 - x). From 0 to 2 the output climbs from 0 to 1, so x mattered.
    But f is flat past 1, so the gradient at 2 is 0. Their first rule, which
    they call sensitivity, says an input that changed the output needs some
    credit.

    Integrated gradients average the gradient along the straight path from a
    baseline to the input, then multiply by how far the input moved. Their
    completeness property says the credits add up to f(input) - f(baseline).
    With m steps, the average uses the points at k / m of the way, for k from
    1 to m.

    ```python
    torch.linspace(0, 2, 5)[1:]    # tensor([0.5000, 1.0000, 1.5000, 2.0000]), 4 steps from 0 to 2
    ```

    The paper says 20 to 300 steps are enough, and to check that the credits
    add up. On the toy, 20 steps give 0.9 and 300 give 0.993 of the change
    of 1, so the check says when to add steps.

    On the copier, the answer is the last position's score for the token
    after the last token's first copy. The inputs are each position's token
    vector, the baseline is no token at all, and each position's credit is
    the sum over its vector. On 200 rows, the credits miss the score's change
    by this much on average, out of a change of 10.19.

    | Steps | Credits miss the change by |
    |---|---|
    | 1 | 3.61 |
    | 5 | 1.21 |
    | 20 | 0.31 |
    | 50 | 0.13 |
    | 300 | 0.02 |

    One step is the gradient at the input times the input, a common shortcut,
    and it misses by a third. With 50 steps, 74% of the credit lands on three
    places. They are the token after the copy with 28%, the copy itself with
    24%, and the last token with 23%. Those are the places interp2's
    induction head reads.
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


    # The copier this chapter studies, and its training.
''' + indented(COPIER_PART) + '''


''' + indented(ANSWER_PART) + '''


    # TODO: Return the gradient of f(x).sum() with respect to x, without
    #   changing x. Work on a detached copy of x that requires a gradient.
    def gradient(
        f: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the integrated gradients of f at x from baseline, with steps
    #   steps. Average gradient of f at the points baseline + k / steps * (x -
    #   baseline), for k from 1 to steps, then multiply by x - baseline.
    def integrated_gradients(
        f: Callable[[Tensor], Tensor],
        x: Tensor,
        baseline: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: Return each position's credit for each row's answer score, of
    #   shape (batch, steps). Take integrated_gradients of answer_scores over
    #   the token vectors model.embed.tokens(ids), detached, from a baseline of
    #   zeros, then sum each position's credits over its vector.
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
        description="averages the gradient along the straight path",
        run=check_integrated_gradients,
    )
    checker.test(
        function="position_attributions",
        description="credits each position of each row",
        run=check_position_attributions,
    )
    checker.check_name(
        name="toy",
        description="integrated gradients give the toy the credit its gradient denies",
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
        "gradient": "Make x = x.detach().clone().requires_grad_(True), call .backward() on f(x).sum(), and return "
        "x.grad.",
        "integrated_gradients": "Loop k over range(1, steps + 1), add up gradient at each point, divide by steps, "
        "and multiply by x - baseline.",
        "position_attributions": "Pass integrated_gradients a lambda that runs answer_scores on the vectors it gets, "
        "torch.zeros_like of the vectors as the baseline, then .sum(dim=-1).",
        "toy": "It comes from integrated_gradients. If it fails while the function passes, rerun the file.",
        "credits": "It comes from position_attributions. If it fails while the function passes, rerun the file.",
    },
    introduces=(),
    check=check_interp4,
    time_limit=60,
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
