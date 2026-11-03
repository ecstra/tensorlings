"""110_own_module, Section 14's last project, a module of your own planned with the design method, its property tests,
and its result against a baseline, with a learned cellular automaton rule as the worked example."""

from __future__ import annotations

import contextlib
import copy
import textwrap
from collections.abc import Callable, Iterator
from typing import NamedTuple

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
from .linear_layers import expect_module

OWN_MODULE = Chapter(
    folder="110_own_module",
    title="Project: a module of your own",
    readme=source_file(
        """
        # Project: a module of your own

        The course's last project. You plan a module for a problem with the
        design method, build it, test the property its structure promises,
        and measure it against a baseline. The worked example learns the rule
        of a cellular automaton from its rows. Keep it, or swap in a problem of
        your own and keep the same names.

        ## Further reading

        - [Wikipedia, elementary cellular automata and the Wolfram code](https://en.wikipedia.org/wiki/Elementary_cellular_automaton)
        - [Wikipedia, Rule 110 and Cook's proof](https://en.wikipedia.org/wiki/Rule_110)
        - [Karpathy, a recipe for training neural networks](https://karpathy.github.io/2019/04/25/recipe/)
        """,
    ),
)

MEASURE_THREADS = 4


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """The CPU's thread count changes its sums' rounding, so the lessons' runs pin it."""
    threads = torch.get_num_threads()
    torch.set_num_threads(MEASURE_THREADS)

    try:
        yield
    finally:
        torch.set_num_threads(threads)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


SETTINGS_PART = '''
RULE = 110     # the worked example's rule, by its Wolfram number
WIDTH = 16     # cells in each training row, on a ring
'''

PLAN_CLASS_PART = '''
# A record of the plan, its fields filled by name.
class Plan(NamedTuple):
    problem: str
    skill: str
    structure: str
    layer: str
    shown_in: str
    baseline: str
    property: str
    margin: float
'''

SHIFT_PART = '''
# design5's shift, given.
def shifted(
    x: Tensor,
    steps: int,
) -> Tensor:
    return torch.roll(x, steps, dims=1)
'''

CHANGE_PART = '''
# A change the answer must follow. One function changes the input and the answer alike. A pair
# gives the input's change and then the answer's, for an answer that moves another way or not at all.
Change = Callable[[Tensor], Tensor] | tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]
'''

PLAN_SOLUTION = '''
PLAN = Plan(
    problem="predict the next row of an elementary cellular automaton on a ring",
    skill="one local rule, applied the same way at every cell",
    structure="each cell's next state reads only itself and its two neighbours, the same everywhere on the ring",
    layer="a convolution of width 3 with circular padding, then a 1 by 1 convolution",
    shown_in="conv4",
    baseline="an MLP over the whole row",
    property="shifting a row around the ring shifts its next row the same way",
    margin=0.2,
)
'''

EXACT_SOLUTION = '''
def exact(rows: Tensor) -> Tensor:
    left = torch.roll(rows, 1, dims=1)
    right = torch.roll(rows, -1, dims=1)
    patterns = (4 * left + 2 * rows + right).long()
    table = torch.tensor([(RULE >> bit) & 1 for bit in range(8)], dtype=rows.dtype)

    return table[patterns]
'''

BATCH_SOLUTION = '''
def make_batch(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    rows = torch.randint(0, 2, (count, WIDTH), generator=generator).float()

    return rows, exact(rows)
'''

TRANSFORMS_SOLUTION = '''
TRANSFORMS: dict[str, Change] = {
    "shift by 1": lambda rows: shifted(rows, 1),
    "shift by 5": lambda rows: shifted(rows, 5),
}
'''

FLOOR_SOLUTION = '''
def floor_score(
    rows: Tensor,
    next_rows: Tensor,
) -> float:
    return (rows == next_rows).float().mean().item()
'''

# own1's whole file past its imports, which own2 hands the learner as given code.
OWN1_PARTS = (
    SETTINGS_PART + PLAN_CLASS_PART + SHIFT_PART + CHANGE_PART + PLAN_SOLUTION + EXACT_SOLUTION + BATCH_SOLUTION + TRANSFORMS_SOLUTION + FLOOR_SOLUTION
)

GAP_PART = '''
# design5's gaps, given, with a pair's first change run on the input and its second on the answer.
def property_gaps(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    transforms: dict[str, Change],
) -> dict[str, float]:
    gaps = {}

    with torch.no_grad():
        for name, change in transforms.items():
            change_input, change_answer = change if isinstance(change, tuple) else (change, change)
            gaps[name] = (f(change_input(x)) - change_answer(f(x))).abs().max().item()

    return gaps
'''

SILENT_PART = '''
# design4's check for parameters that never learn, given.
def silent_after_one_step(
    model: nn.Module,
    loss_of: Callable[[nn.Module], Tensor],
    lr: float,
) -> list[str]:
    copied = copy.deepcopy(model)
    optimizer = torch.optim.SGD(copied.parameters(), lr=lr)
    loss_of(copied).backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_of(copied).backward()

    return [name for name, parameter in copied.named_parameters() if parameter.grad is None or not parameter.grad.any()]
'''

BASELINE_PART = '''
# The plan's baseline, an MLP over the whole row, given.
class Baseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(WIDTH, 64), nn.ReLU(), nn.Linear(64, WIDTH))

    def forward(self, rows: Tensor) -> Tensor:
        return self.layers(rows)
'''

MODULE_SOLUTION = '''
class OwnModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.see = nn.Conv1d(1, 16, 3, padding=1, padding_mode="circular")
        self.decide = nn.Conv1d(16, 1, 1)

    def forward(self, rows: Tensor) -> Tensor:
        return self.decide(torch.relu(self.see(rows.unsqueeze(1)))).squeeze(1)
'''

LOSS_SOLUTION = '''
def loss_of(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    return F.binary_cross_entropy_with_logits(model(inputs), targets)
'''

TRAIN_SOLUTION = '''
def train(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    steps: int,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(steps):
        loss = loss_of(model, inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
'''

SCORE_SOLUTION = '''
def score(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
) -> float:
    model.eval()

    with torch.no_grad():
        right = ((model(inputs) > 0).float() == targets).float().mean().item()

    model.train()

    return right
'''

RECIPE_PART = '''
# The run the checks repeat with your own functions, both models from seed 0. Raise these if your
# problem needs more rows or more steps to learn.
TRAIN_ROWS = 64
TRAIN_STEPS = 500
'''


def namespace_of(
    code: str,
    rule: int = 110,
    width: int = 16,
) -> dict[str, object]:
    """The parts run as one file, with the rule and the width a learner's file may have changed."""
    namespace: dict[str, object] = {
        "copy": copy,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "Callable": Callable,
        "NamedTuple": NamedTuple,
    }
    exec(code, namespace)
    namespace["RULE"] = rule
    namespace["WIDTH"] = width

    return namespace


REFERENCE = namespace_of(
    OWN1_PARTS + GAP_PART + SILENT_PART + BASELINE_PART + MODULE_SOLUTION + LOSS_SOLUTION + TRAIN_SOLUTION + SCORE_SOLUTION + RECIPE_PART,
)

TRAIN_ROWS = REFERENCE["TRAIN_ROWS"]
TRAIN_STEPS = REFERENCE["TRAIN_STEPS"]
HELD_ROWS = 1000
GAP_ROWS = 64


def rule_reference(module: object) -> dict[str, object] | None:
    """The reference with the learner's rule and width, or None when the file has set the worked example aside."""
    rule = getattr(module, "RULE", None)
    width = getattr(module, "WIDTH", None)

    if not isinstance(rule, int) or not isinstance(width, int):
        return None

    return namespace_of(
        OWN1_PARTS + GAP_PART + MODULE_SOLUTION + LOSS_SOLUTION + SCORE_SOLUTION,
        rule=rule,
        width=width,
    )


def random_rows(
    count: int,
    width: int,
    seed: int,
) -> Described:
    return Described(
        value=torch.randint(0, 2, (count, width), generator=torch.Generator().manual_seed(seed)).float(),
        text=f"{count} random rows of {width} cells drawn with seed {seed}",
    )


def expect_share(
    call: str,
    actual: object,
    expected: float,
) -> None:
    expect_close(
        call=call,
        actual=actual,
        expected=expected,
        tolerance=1e-6,
    )


def expect_plain_share(
    call: str,
    value: object,
) -> float:
    reject_missing_return(
        call=call,
        actual=value,
    )

    if isinstance(value, Tensor):
        raise Mismatch(f"{call} is {value_text(value)}, a tensor, but it should be a plain number. Use .item() on it.")

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise Mismatch(f"{call} is {value_text(value)}, but it should be a share between 0 and 1.")

    return float(value)


# ---------------------------------------------------------------------------
# own1
# ---------------------------------------------------------------------------

OWN1_LESSON = prose(
    """
    This last project is yours. Pick a problem, plan a module for it with
    the design method, then build it in own2. The file holds a worked
    example to keep or replace.

    The plan comes before any code. `Plan` is a named tuple, a record whose
    fields you fill by name. Each field is one step of the method.

    - problem: what the module must do.
    - skill: the one skill a toy task isolates, as design1's recall did.
    - structure and layer: what shape the data has, and the layer that fits it from combine5's guide.
    - shown_in: the exercise that showed that layer.
    - baseline: the simplest model to beat, as design2 set up.
    - property: the change the answer must follow, as design5 tested.
    - margin: how far above the baseline the module must score.

    The worked example is a cellular automaton. A row of cells, each 0 or 1,
    sits on a ring. Each step, every cell reads itself and its two
    neighbours, and a rule gives its new state.

    Three cells make 8 patterns, so a rule is 8 bits. Stephen Wolfram
    numbered the rules by them. Read each pattern as a 3 bit number, and bit
    k of the rule's number is the new state of pattern k. In 2004 Matthew
    Cook published a proof that Rule 110 can compute anything a computer can.

    ```python
    [(110 >> k) & 1 for k in range(8)]    # [0, 1, 1, 1, 0, 1, 1, 0], pattern 000 first
    ```

    `exact` is the rule's exact program, as design1's `exact_recall` was.
    `make_batch` draws random rows and gives each one's next row, so every
    answer is known and the data never runs out.

    The property belongs to the problem before any module has it. Shift a
    row around the ring, and its next row shifts the same way. So the check
    tests your `TRANSFORMS` on `exact` first. A gap there means the property
    is wrong, not the module.

    A `Change` that is one function changes the input and the answer
    alike. When the answer moves another way, give a pair, the input's
    change and then the answer's. Reversing a row, for one, leaves its
    majority as it was.

    ```python
    "reverse": (lambda rows: rows.flip(1), lambda answer: answer),
    ```

    A change must also do the same thing every time it runs, so a shuffle
    draws its order once, outside the function.

    `floor_score` is what doing nothing scores, every cell keeping its
    state. On Rule 110 it gets 62% of cells right, so a model has learned
    something only above that.

    For a problem of your own, keep these names and change what they hold.
    The checks then hold for your problem. The ones tied to the rule switch
    off once `RULE` is gone. Three problems to start from:

    - Conway's Game of Life on a torus: a local rule in two directions, with shifts both ways as its property.
    - Ranks in a set: the rank of each number among the rest, where shuffling the set shuffles the ranks.
    - A running maximum: the largest value so far in a stream, where doubling every value doubles every answer.
    """,
)

OWN1_SOURCE = source_file(
    '''
    # own1.py

    from collections.abc import Callable
    from typing import NamedTuple

    import torch
    from torch import Tensor

''' + indented(SETTINGS_PART) + '''


''' + indented(PLAN_CLASS_PART) + '''


''' + indented(SHIFT_PART) + '''


''' + indented(CHANGE_PART) + '''


    # TODO: Plan the module before writing it. Fill each field in plain words.
    #   shown_in is the name of the exercise that showed the layer you pick,
    #   as combine5's guide lists them. margin is how much higher the module's
    #   share right must be than the baseline's, between 0 and 1.
    PLAN = Plan(
        problem=...,
        skill=...,
        structure=...,
        layer=...,
        shown_in=...,
        baseline=...,
        property=...,
        margin=...,
    )


    bits = [(110 >> k) & 1 for k in range(8)]                     # [0, 1, 1, 1, 0, 1, 1, 0], pattern 000 first
    ring = torch.roll(torch.tensor([[1.0, 0.0, 0.0]]), 1, dims=1)    # tensor([[0., 1., 0.]]), the last cell wraps to the front
    picked = torch.tensor([5.0, 7.0])[torch.tensor([[1, 0]])]        # tensor([[7., 5.]]), a table read at many places at once


    # TODO: Return each row's next row under RULE, of the same shape. A cell's
    #   pattern is its left neighbour, itself and its right neighbour, read as
    #   a 3 bit number. On the ring, the first cell's left neighbour is the
    #   last cell. Its new state is that bit of RULE, as 0.0 or 1.0.
    def exact(rows: Tensor) -> Tensor:
        ...


    # TODO: Draw count random rows of WIDTH cells, each 0.0 or 1.0, with
    #   generator, and return them with their next rows from exact.
    def make_batch(
        count: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Name the changes the property promises the answer will follow,
    #   each a Change of a batch of rows. Each must do the same thing every
    #   time it runs. For the rule, give a shift by 1 and a shift by 5 around
    #   the ring.
    TRANSFORMS: dict[str, Change] = ...


    # TODO: Return the share of cells right when every cell keeps its state, as
    #   a plain number. That is the floor any model has to beat.
    def floor_score(
        rows: Tensor,
        next_rows: Tensor,
    ) -> float:
        ...


    floor = floor_score(*make_batch(1000, torch.Generator().manual_seed(2)))
    ''',
)

OWN1_SOLUTION = source_file(
    '''
    # own1.py, a solution for comparison

    from collections.abc import Callable
    from typing import NamedTuple

    import torch
    from torch import Tensor

''' + indented(SETTINGS_PART) + '''


''' + indented(PLAN_CLASS_PART) + '''


''' + indented(SHIFT_PART) + '''


''' + indented(CHANGE_PART) + '''


''' + indented(PLAN_SOLUTION) + '''


    bits = [(110 >> k) & 1 for k in range(8)]
    ring = torch.roll(torch.tensor([[1.0, 0.0, 0.0]]), 1, dims=1)
    picked = torch.tensor([5.0, 7.0])[torch.tensor([[1, 0]])]


''' + indented(EXACT_SOLUTION) + '''


''' + indented(BATCH_SOLUTION) + '''


''' + indented(TRANSFORMS_SOLUTION) + '''


''' + indented(FLOOR_SOLUTION) + '''


    floor = floor_score(*make_batch(1000, torch.Generator().manual_seed(2)))
    ''',
)

PLAN_TEXT_FIELDS = ("problem", "skill", "structure", "layer", "shown_in", "baseline", "property")


def check_plan(value: object) -> None:
    from . import EXERCISES

    for field in (*PLAN_TEXT_FIELDS, "margin"):
        if not hasattr(value, field):
            raise Mismatch(f"PLAN is {value_text(value)}, but it should be a Plan with every field filled, {field} among them.")

        if getattr(value, field) is Ellipsis:
            raise Mismatch(f"PLAN.{field} is still ..., so it is not written yet.")

    for field in PLAN_TEXT_FIELDS:
        text = getattr(value, field)

        if not isinstance(text, str) or not text.strip():
            raise Mismatch(f"PLAN.{field} is {value_text(text)}, but it should say it in words.")

    names = [exercise.name for exercise in EXERCISES]
    earlier = names[: names.index("own1")] if "own1" in names else names

    if value.shown_in not in earlier:
        raise Mismatch(
            f"PLAN.shown_in is {value.shown_in!r}, but it should be the name of an exercise before this one that "
            "showed your layer, such as conv4 or attention6 from combine5's guide.",
        )

    margin = value.margin

    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 < margin < 1:
        raise Mismatch(f"PLAN.margin is {value_text(margin)}, but it should be a share between 0 and 1, such as 0.2.")


def generic_batch(module: object) -> tuple[Tensor, Tensor]:
    make_batch = getattr(module, "make_batch", None)

    if not callable(make_batch):
        raise Mismatch("This test draws a batch with your make_batch, so write that first.")

    return make_batch(8, torch.Generator().manual_seed(21))


def check_exact_runs(module: object) -> Callable[[object], None]:
    """With the worked example set aside, exact can only be held to giving the same answer every time."""

    def run(found: object) -> None:
        inputs, _ = generic_batch(module)
        first = found(inputs.clone())
        second = found(inputs.clone())

        reject_missing_return(
            call="exact(a batch from make_batch)",
            actual=first,
        )

        if isinstance(first, Tensor) and not torch.equal(first, second):
            raise Mismatch("exact gave two different answers for the same inputs. The exact program has one answer.")

    return run


def check_make_batch(module: object) -> Callable[[object], None]:
    def run(found: object) -> None:
        torch.manual_seed(1)
        first = found(5, torch.Generator().manual_seed(11))
        torch.manual_seed(2)
        again = found(5, torch.Generator().manual_seed(11))
        other = found(5, torch.Generator().manual_seed(12))

        reject_missing_return(
            call="make_batch(5, a generator seeded with 11)",
            actual=first,
        )

        if not isinstance(first, tuple) or len(first) != 2:
            raise Mismatch(f"make_batch(5, a generator seeded with 11) is {value_text(first)}, but it should be a pair, the inputs and their targets.")

        inputs, targets = first

        if len(inputs) != 5 or len(targets) != 5:
            raise Mismatch(f"make_batch(5, ...) gave {len(inputs)} inputs and {len(targets)} targets, but it should give 5 of each.")

        if not all(torch.equal(a, b) for a, b in zip(first, again)):
            raise Mismatch(
                "make_batch gave different batches from two generators with the same seed. Draw every random number "
                "with generator, so the same seed gives the same batch.",
            )

        if all(torch.equal(a, b) for a, b in zip(first, other)):
            raise Mismatch("make_batch gave the same batch for seeds 11 and 12. Draw the inputs at random with generator.")

        expect_tensor(
            call="the targets of make_batch(5, a generator seeded with 11), against exact of its inputs",
            actual=targets,
            expected=module.exact(inputs.clone()),
        )

        reference = rule_reference(module)

        if reference is None:
            return

        if inputs.shape != (5, reference["WIDTH"]) or not inputs.is_floating_point():
            raise Mismatch(
                f"make_batch's rows are {value_text(inputs)}, but they should be floats of shape (5, WIDTH), each 0.0 or 1.0.",
            )

        if not ((inputs == 0) | (inputs == 1)).all():
            raise Mismatch("make_batch's rows hold values other than 0.0 and 1.0. Each cell is 0 or 1.")

    return run


def is_change(change: object) -> bool:
    if isinstance(change, tuple):
        return len(change) == 2 and all(callable(part) for part in change)

    return callable(change)


def changed_twice(
    name: str,
    change: Callable[[Tensor], Tensor],
    batch: Tensor,
    what: str,
) -> Tensor:
    """The change run twice on copies of one batch, which must agree, since a gap compares two runs of it."""
    changed = change(batch.clone())
    again = change(batch.clone())

    if not isinstance(changed, Tensor) or changed.shape != batch.shape:
        raise Mismatch(f"TRANSFORMS[{name!r}] gave {value_text(changed)} for the {what}, but it should keep their shape.")

    if not torch.equal(changed, again):
        raise Mismatch(
            f"TRANSFORMS[{name!r}] gave two different results for the same {what}. A change must do the same thing "
            "every time it runs, so draw a shuffle's order once, outside the function.",
        )

    return changed


def check_transforms(module: object) -> Callable[[object], None]:
    def run(value: object) -> None:
        if not isinstance(value, dict) or not value or not all(is_change(change) for change in value.values()):
            raise Mismatch(
                f"TRANSFORMS is {value_text(value)}, but it should be a dict of names to changes, each a function of a "
                "batch or a pair of them.",
            )

        inputs, _ = generic_batch(module)
        exact = getattr(module, "exact", None)

        if not callable(exact):
            raise Mismatch("This test runs your exact through each change, so write exact first.")

        answers = exact(inputs.clone())

        for name, change in value.items():
            change_input, change_answer = change if isinstance(change, tuple) else (change, change)

            if torch.equal(changed_twice(name, change_input, inputs, "inputs"), inputs):
                raise Mismatch(f"TRANSFORMS[{name!r}] leaves the batch as it was, so it tests nothing. Make it change the input.")

            try:
                changed_twice(name, change_answer, answers, "answers from exact")
            except (IndexError, RuntimeError) as error:
                raise Mismatch(
                    f"TRANSFORMS[{name!r}] crashed on exact's answers: {error}. When the answer doesn't move the way "
                    "the input does, give a pair, the input's change and then the answer's.",
                ) from None

        gaps = REFERENCE["property_gaps"](exact, inputs, value)

        for name, gap in gaps.items():
            if gap > 1e-6:
                raise Mismatch(
                    f"exact, the exact program, misses by {gap:.3g} under {name!r}, so that change isn't a property of "
                    "the problem. A property the answer itself breaks can't be asked of the module.",
                )

    return run


def check_floor(module: object) -> Callable[[object], None]:
    def run(value: object) -> None:
        share = expect_plain_share(
            call="floor",
            value=value,
        )
        reference = rule_reference(module)

        if reference is None:
            return

        expected = reference["floor_score"](*reference["make_batch"](1000, torch.Generator().manual_seed(2)))

        # Shown to 2 digits, fewer than the tolerance needs, so a number copied from this message doesn't pass.
        if abs(share - expected) > 1e-6:
            raise Mismatch(f"floor is {value_text(value)}, but it should be about {expected:.2f}.")

    return run


def check_own1(checker: Checker) -> None:
    module = checker.module
    reference = rule_reference(module)

    checker.check_name(
        name="PLAN",
        description="names the problem, the skill, the structure and layer, the baseline, the property and a margin",
        run=check_plan,
    )

    if reference is None:
        checker.test(
            function="exact",
            description="gives the same answer every time",
            run=check_exact_runs(module),
        )
    else:
        check_on_inputs(
            checker=checker,
            function="exact",
            description="gives each row's next row under RULE, on a ring",
            inputs=(
                (random_rows(3, 5, 31),),
                (random_rows(4, 9, 32),),
            ),
            reference=reference["exact"],
        )

    checker.test(
        function="make_batch",
        description="draws a batch with its generator, its targets from exact",
        run=check_make_batch(module),
    )
    checker.check_name(
        name="TRANSFORMS",
        description="names changes that exact itself follows",
        run=check_transforms(module),
    )

    if reference is None:
        checker.test(
            function="floor_score",
            description="gives a share as a plain number",
            run=lambda found: expect_plain_share(
                call="floor_score(a batch from make_batch)",
                value=found(*generic_batch(module)),
            ),
        )
    else:
        exact = reference["exact"]
        rows = [random_rows(6, 8, 33).value, random_rows(5, 12, 34).value]
        check_on_inputs(
            checker=checker,
            function="floor_score",
            description="gives the share of cells that keep their state, as a plain number",
            inputs=tuple(
                (
                    Described(value=row, text=f"{len(row)} random rows of {row.shape[1]} cells"),
                    Described(value=exact(row), text="their next rows"),
                )
                for row in rows
            ),
            reference=reference["floor_score"],
            expect=expect_share,
        )

    checker.check_name(
        name="floor",
        description="is the floor on 1,000 rows",
        run=check_floor(module),
    )


OWN1 = Exercise(
    name="own1",
    chapter=OWN_MODULE,
    title="Project: plan your own module",
    lesson=OWN1_LESSON,
    source=OWN1_SOURCE,
    solution=OWN1_SOLUTION,
    hints={
        "PLAN": "Answer each field in a few words. For the rule, the structure is local and the same at every cell, "
        "which combine5's guide gives to a convolution, shown in conv4. A margin of 0.2 asks for 20 points over the "
        "baseline.",
        "exact": "torch.roll by 1 and by -1 along dim 1 gives each cell's left and right neighbour. Weigh them 4, 2 "
        "and 1 to get the pattern, .long() it, and index a tensor of the rule's 8 bits with it.",
        "make_batch": "torch.randint(0, 2, (count, WIDTH), generator=generator) gives whole numbers. Make them "
        "floats with .float(), then return them with exact of them.",
        "TRANSFORMS": "A dict of two lambdas, each calling shifted on its rows, one with 1 and one with 5. A shift "
        "moves the next row the same way, so each entry is one function, not a pair.",
        "floor_score": "Compare rows with next_rows, turn the matches into floats, and take their mean as a plain "
        "number.",
        "floor": "It comes from floor_score and make_batch. Get those right first.",
    },
    introduces=(),
    check=check_own1,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# own2
# ---------------------------------------------------------------------------

OWN2_LESSON = prose(
    """
    Now the module, built and judged the way the method says.

    The plan picked two layers, each for a reason it can name.

    | Layer | Why |
    |---|---|
    | `nn.Conv1d` of width 3, circular padding | a cell reads itself and two neighbours, the same everywhere, and design5's table shows circular padding keeps shifts exactly |
    | `nn.Conv1d` of width 1 | each cell's 16 readings become one logit for its next state, the same per cell |

    A 1 by 1 convolution is a Linear run on each place on its own. No layer
    reads the row's length, so the module takes a row of any width.

    `train` doesn't depend on the problem, so it serves any module of
    yours. It runs Adam at 0.01 on one fixed batch. `loss_of` and `score`
    fit the rule, whose cells are 0 or 1, with binary cross entropy and a
    logit cut at 0. A problem with other answers, such as ranks, writes
    both its own way, a mean squared error and a share within a tolerance,
    for example.

    The last lines run the method's checks in order.

    - Before training: design4's `silent_after_one_step` must list nothing, every parameter learning.
    - After training: design5's `property_gaps` must be float rounding, below 1e-4.
    - Against the baseline: both train on the same `TRAIN_ROWS` rows for `TRAIN_STEPS` steps, then are scored on 1,000 rows they never saw, as design2 did.

    The checks repeat that run with your own functions, so raise
    `TRAIN_ROWS` or `TRAIN_STEPS` if your problem needs more. Here is the
    run with 64 rows and 500 steps, the range over 3 seeds.

    | Model | Training rows | New rows | New rows of 256 cells |
    |---|---|---|---|
    | the floor, no model | | 62% | |
    | your module | 100% | 100% | 100% |
    | the baseline | 100% | 65.9% to 68.7% | can't run |

    The baseline fit every training row and missed a third of the new cells,
    design2's "failed to generalize". Every input cell reaches every output
    through weights of its own, so it learned 64 rows and not the rule.

    Your module learned the 8 patterns themselves, so a row it never saw is
    no harder. Fed its own rows back 100 times on a ring of 80 cells, it
    matched the rule on every cell, on all 3 seeds.

    Structure buys data. Your module is exact from 16 training rows. The
    baseline needs 256 rows to reach 89% to 90% and 1,024 to reach 100%, and
    it still can't take a row of another width.

    Once your module wins, interp's tools show how. A probe, as interp1 fit
    one, reads what each layer holds, and patching, as interp3 did, tests
    which layer the answer runs through.
    """,
)

OWN2_SOURCE = source_file(
    '''
    # own2.py

    import copy
    from collections.abc import Callable
    from typing import NamedTuple

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # own1's plan and toy task, given. If you planned a problem of your own,
    # paste your own1 here in their place, and write a Baseline for it below.
''' + indented(OWN1_PARTS) + '''


''' + indented(GAP_PART) + '''


''' + indented(SILENT_PART) + '''


''' + indented(BASELINE_PART) + '''


    wraps = nn.Conv1d(1, 1, 3, padding=1, padding_mode="circular")    # its ends read each other, as the ring does
    per_cell = nn.Conv1d(16, 1, 1)                                     # one Linear from 16 channels to 1, run at every cell
    read = per_cell(torch.zeros(2, 16, 7))                             # shape (2, 1, 7), one value per cell


    # TODO: Build the module the plan names. For the rule, rows of shape
    #   (count, width) become one channel, a convolution of width 3 with
    #   circular padding reads each cell's neighbourhood into 16 channels, a
    #   ReLU follows, and a convolution of width 1 gives one logit per cell.
    #   forward returns logits of shape (count, width).
    class OwnModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ...

        def forward(self, rows: Tensor) -> Tensor:
            ...


    # TODO: Return the loss of model on inputs against targets, as a tensor.
    #   For the rule, binary cross entropy on the logits, from
    #   F.binary_cross_entropy_with_logits.
    def loss_of(
        model: nn.Module,
        inputs: Tensor,
        targets: Tensor,
    ) -> Tensor:
        ...


    # TODO: Train model for steps steps with Adam at a rate of 0.01, each step
    #   on the whole batch of inputs and targets, with loss_of. Return the
    #   last loss as a plain number.
    def train(
        model: nn.Module,
        inputs: Tensor,
        targets: Tensor,
        steps: int,
    ) -> float:
        ...


    # TODO: Return the share of targets model gets right on inputs, as a plain
    #   number, a logit above 0 meaning 1. Score in eval mode without recording
    #   for autograd, then put model back in training mode.
    def score(
        model: nn.Module,
        inputs: Tensor,
        targets: Tensor,
    ) -> float:
        ...


''' + indented(RECIPE_PART) + '''
    train_rows, train_next = make_batch(TRAIN_ROWS, torch.Generator().manual_seed(0))
    held_rows, held_next = make_batch(1000, torch.Generator().manual_seed(1000))

    torch.manual_seed(0)
    own = OwnModule()
    torch.manual_seed(0)
    baseline = Baseline()


    # TODO: Run the method's checks. Before training, set silent to
    #   silent_after_one_step of own, with loss_of on the training rows and a
    #   rate of 0.1. Then train own and baseline TRAIN_STEPS steps each on the
    #   training rows. Set own_score and baseline_score to their scores on the
    #   held rows, and gaps to property_gaps of own on the first 64 held rows.
    silent = ...
    own_score = ...
    baseline_score = ...
    gaps = ...
    ''',
)

OWN2_SOLUTION = source_file(
    '''
    # own2.py, a solution for comparison

    import copy
    from collections.abc import Callable
    from typing import NamedTuple

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


''' + indented(OWN1_PARTS) + '''


''' + indented(GAP_PART) + '''


''' + indented(SILENT_PART) + '''


''' + indented(BASELINE_PART) + '''


    wraps = nn.Conv1d(1, 1, 3, padding=1, padding_mode="circular")
    per_cell = nn.Conv1d(16, 1, 1)
    read = per_cell(torch.zeros(2, 16, 7))


''' + indented(MODULE_SOLUTION) + '''


''' + indented(LOSS_SOLUTION) + '''


''' + indented(TRAIN_SOLUTION) + '''


''' + indented(SCORE_SOLUTION) + '''


''' + indented(RECIPE_PART) + '''
    train_rows, train_next = make_batch(TRAIN_ROWS, torch.Generator().manual_seed(0))
    held_rows, held_next = make_batch(1000, torch.Generator().manual_seed(1000))

    torch.manual_seed(0)
    own = OwnModule()
    torch.manual_seed(0)
    baseline = Baseline()


    silent = silent_after_one_step(own, lambda model: loss_of(model, train_rows, train_next), 0.1)
    train(own, train_rows, train_next, TRAIN_STEPS)
    train(baseline, train_rows, train_next, TRAIN_STEPS)
    own_score = score(own, held_rows, held_next)
    baseline_score = score(baseline, held_rows, held_next)
    gaps = property_gaps(own, held_rows[:64], TRANSFORMS)
    ''',
)


def needed(
    module: object,
    name: str,
) -> object:
    found = getattr(module, name, None)

    if found is None or found is Ellipsis:
        raise Mismatch(f"This test uses your {name}, so write that first.")

    return found


def seeded_module(
    found: type,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)

    return expect_module(
        call=f"{found.__name__}()",
        made=found(),
    )


def module_batch(
    module: object,
    count: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    return needed(module, "make_batch")(count, torch.Generator().manual_seed(seed))


def check_own_forward(module: object) -> Callable[[type], None]:
    def run(found: type) -> None:
        model = seeded_module(found, 0)
        inputs, targets = module_batch(module, 5, 41)
        out = model(inputs.clone())

        reject_missing_return(
            call="OwnModule()(a batch of 5 from make_batch)",
            actual=out,
        )

        if not isinstance(out, Tensor) or out.shape != targets.shape:
            raise Mismatch(
                f"OwnModule()(a batch of 5) gives {value_text(out)}, but it should give one output per target, of shape "
                f"{tuple(targets.shape)}.",
            )

        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise Mismatch("OwnModule() has no weights to train. Build it from layers the check can train.")

        if rule_reference(module) is None:
            return

        try:
            wider = model(random_rows(3, 9, 42).value)
        except RuntimeError as error:
            raise Mismatch(
                f"OwnModule crashed on rows of 9 cells: {error}. The rule reads only neighbours, so the module should "
                "take a row of any width, with no layer tied to WIDTH.",
            ) from None

        if wider.shape != (3, 9):
            raise Mismatch(f"OwnModule() gives shape {tuple(wider.shape)} on 3 rows of 9 cells, but it should give (3, 9).")

    return run


def check_loss_of(module: object) -> Callable[[object], None]:
    def run(found: object) -> None:
        model = seeded_module(needed(module, "OwnModule"), 0)
        inputs, targets = module_batch(module, 6, 43)
        loss = found(model, inputs, targets)

        reject_missing_return(
            call="loss_of(OwnModule(), a batch of 6)",
            actual=loss,
        )

        if not isinstance(loss, Tensor) or loss.numel() != 1 or not loss.requires_grad:
            raise Mismatch(
                f"loss_of(OwnModule(), a batch of 6) is {value_text(loss)}, but it should be a tensor of one number "
                "that training can backpropagate through.",
            )

        reference = rule_reference(module)

        if reference is None:
            return

        for seed in (44, 45):
            inputs, targets = reference["make_batch"](4, torch.Generator().manual_seed(seed))
            expect_close(
                call=f"loss_of(OwnModule(), 4 rows drawn with seed {seed})",
                actual=found(model, inputs, targets).item(),
                expected=F.binary_cross_entropy_with_logits(model(inputs), targets).item(),
                tolerance=1e-5,
            )

    return run


def reference_train(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    steps: int,
    loss_of: Callable[[nn.Module, Tensor, Tensor], Tensor],
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(steps):
        loss = loss_of(model, inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def check_train(module: object) -> Callable[[object], None]:
    def run(found: object) -> None:
        own_class = needed(module, "OwnModule")
        loss_of = needed(module, "loss_of")

        for seed, steps in ((46, 2), (47, 3)):
            inputs, targets = module_batch(module, 8, seed)
            mine = seeded_module(own_class, seed)

            if not list(mine.parameters()):
                raise Mismatch("This test trains your OwnModule, so give it its layers first.")

            theirs = copy.deepcopy(mine)
            last = found(mine, inputs, targets, steps)
            wanted = reference_train(theirs, inputs, targets, steps, loss_of)
            call = f"train(OwnModule(), a batch of 8, {steps})"

            expect_close(
                call=call,
                actual=last,
                expected=wanted,
                tolerance=1e-5,
            )

            for (name, parameter), (_, expected) in zip(mine.named_parameters(), theirs.named_parameters()):
                if not torch.allclose(parameter, expected, atol=1e-5):
                    raise Mismatch(
                        f"After {call}, {name} differs from {steps} steps of Adam at 0.01 on loss_of. Zero the old "
                        "gradients each step, and step once per step.",
                    )

    return run


def check_score(module: object) -> Callable[[object], None]:
    def run(found: object) -> None:
        model = seeded_module(needed(module, "OwnModule"), 0)
        seen: list[tuple[bool, bool]] = []
        hook = model.register_forward_pre_hook(lambda layer, args: seen.append((layer.training, torch.is_grad_enabled())))
        inputs, targets = module_batch(module, 6, 48)

        try:
            share = expect_plain_share(
                call="score(OwnModule(), a batch of 6)",
                value=found(model, inputs, targets),
            )
        finally:
            hook.remove()

        if not seen:
            raise Mismatch("score never ran the model. Score its outputs on inputs.")

        if any(training for training, _ in seen):
            raise Mismatch("score ran the model in training mode. Call model.eval() first.")

        if any(recording for _, recording in seen):
            raise Mismatch("score recorded for autograd. Run the model inside torch.no_grad().")

        if not model.training:
            raise Mismatch("score left the model in eval mode. Put it back with model.train() before returning.")

        if rule_reference(module) is None:
            return

        # A trained module's logits sit far from 0, so these put some between 0 and 0.5, where a wrong cut shows.
        for logits, targets in (
            (
                torch.tensor([[0.3, -0.2, 0.7, 0.1], [-0.4, 0.2, 0.05, -1.0]]),
                torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]]),
            ),
            (
                torch.tensor([[0.45, 0.6, -0.1], [0.2, -0.3, 0.9]]),
                torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
            ),
        ):
            stand_in = FixedLogits(logits)
            expect_close(
                call=f"score(a model whose logits are {value_text(logits)}, targets {value_text(targets)})",
                actual=found(stand_in, torch.zeros_like(logits), targets),
                expected=REFERENCE["score"](FixedLogits(logits), torch.zeros_like(logits), targets),
                tolerance=1e-6,
            )

        del share

    return run


class FixedLogits(nn.Module):
    """A stand in model that gives the same logits whatever it reads, with one weight so eval and train mean something."""

    def __init__(self, logits: Tensor) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.logits = logits

    def forward(self, inputs: Tensor) -> Tensor:
        return self.logits * self.scale


def recipe(module: object) -> tuple[int, int]:
    """The file's TRAIN_ROWS and TRAIN_STEPS, which a learner may raise for a problem that needs more."""
    found = []

    for name in ("TRAIN_ROWS", "TRAIN_STEPS"):
        value = getattr(module, name, None)

        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise Mismatch(f"{name} is {value_text(value)}, but it should be a whole number above 0.")

        found.append(value)

    return found[0], found[1]


def trained_pair(module: object) -> tuple[nn.Module, nn.Module, tuple[Tensor, Tensor]]:
    """The file's own run done again with its own functions: both models from seed 0, on the same rows."""
    rows, steps = recipe(module)
    train = needed(module, "train")
    train_rows, train_next = module_batch(module, rows, 0)
    held = module_batch(module, HELD_ROWS, 1000)
    own = seeded_module(needed(module, "OwnModule"), 0)
    baseline = seeded_module(needed(module, "Baseline"), 0)
    train(own, train_rows, train_next, steps)
    train(baseline, train_rows, train_next, steps)

    return own, baseline, held


def check_silent(module: object) -> Callable[[object], None]:
    def run(value: object) -> None:
        if value != []:
            raise Mismatch(
                f"silent is {value_text(value)}, but it should be [], every parameter learning. A parameter listed "
                "never moves, so the module isn't the one you planned.",
            )

        train_rows, train_next = module_batch(module, recipe(module)[0], 0)
        loss_of = needed(module, "loss_of")
        found = REFERENCE["silent_after_one_step"](
            seeded_module(needed(module, "OwnModule"), 0),
            lambda model: loss_of(model, train_rows, train_next),
            0.1,
        )

        if found:
            raise Mismatch(f"silent_after_one_step of a fresh OwnModule lists {found}, so silent can't be []. Find why they never learn.")

    return run


def check_scores(module: object) -> Callable[[object], None]:
    def run(value: object) -> None:
        own_score = expect_plain_share(
            call="own_score",
            value=getattr(module, "own_score", None),
        )
        baseline_score = expect_plain_share(
            call="baseline_score",
            value=getattr(module, "baseline_score", None),
        )
        rows, steps = recipe(module)
        own, baseline, held = trained_pair(module)
        score = needed(module, "score")
        expected_own = score(own, *held)
        expected_baseline = score(baseline, *held)

        for name, actual, expected in (
            ("own_score", own_score, expected_own),
            ("baseline_score", baseline_score, expected_baseline),
        ):
            if abs(actual - expected) > 1e-4:
                raise Mismatch(
                    f"{name} is {value_text(actual)}, but the same run gives about {expected:.2f}. Train both from seed "
                    f"0 for TRAIN_STEPS steps on TRAIN_ROWS training rows, {steps} and {rows} in your file, then score "
                    "them on the held rows.",
                )

        margin = needed(module, "PLAN").margin

        if expected_own < expected_baseline + margin:
            raise Mismatch(
                f"Your module scores {expected_own:.3f} on new rows and the baseline {expected_baseline:.3f}, short "
                f"of the plan's margin of {margin}. Read the curves as design2 did, then change the module, not the "
                "margin.",
            )

        del value

    return run


def check_gaps(module: object) -> Callable[[object], None]:
    def run(value: object) -> None:
        transforms = needed(module, "TRANSFORMS")

        if not isinstance(value, dict) or set(value) != set(transforms):
            raise Mismatch(f"gaps is {value_text(value)}, but it should be a dict with the names {sorted(transforms)}.")

        own, _, held = trained_pair(module)
        found = REFERENCE["property_gaps"](own, held[0][:GAP_ROWS], transforms)

        for name, gap in found.items():
            if gap > 1e-4:
                raise Mismatch(
                    f"The trained module misses by {gap:.3g} under {name!r}, so its structure doesn't keep the plan's "
                    "property. Pick layers that keep it by design, as design5's table shows.",
                )

    return run


def check_own2(checker: Checker) -> None:
    module = checker.module

    checker.test_method(
        class_name="OwnModule",
        method="forward",
        description="gives one output per target, and takes rows of any width",
        run=check_own_forward(module),
    )
    checker.test(
        function="loss_of",
        description="gives the loss as a tensor training can follow",
        run=check_loss_of(module),
    )
    checker.test(
        function="train",
        description="steps Adam at 0.01 on loss_of and gives the last loss",
        run=check_train(module),
    )
    checker.test(
        function="score",
        description="scores in eval mode without autograd, then puts the model back",
        run=check_score(module),
    )
    checker.check_name(
        name="silent",
        description="is design4's check before training, with every parameter learning",
        run=check_silent(module),
    )
    checker.check_name(
        name="own_score",
        description="beats the baseline on new rows by the plan's margin",
        run=check_scores(module),
    )
    checker.check_name(
        name="baseline_score",
        description="is the baseline's score on new rows",
        run=check_scores(module),
    )
    checker.check_name(
        name="gaps",
        description="is design5's property test of the trained module",
        run=check_gaps(module),
    )


OWN2 = Exercise(
    name="own2",
    chapter=OWN_MODULE,
    title="Project: build it, test it, beat the baseline",
    lesson=OWN2_LESSON,
    source=OWN2_SOURCE,
    solution=OWN2_SOLUTION,
    hints={
        "OwnModule.forward": "In __init__ make the two Conv1d layers from the file's examples, 1 to 16 channels and "
        "16 to 1. In forward, unsqueeze(1) the rows into one channel, run the first, a ReLU and the second, then "
        "squeeze(1).",
        "loss_of": "Run model on inputs and hand the logits and targets to F.binary_cross_entropy_with_logits.",
        "train": "Make one torch.optim.Adam over model.parameters() before the loop. Each step, take loss_of, zero "
        "the gradients, backpropagate and step. Return loss.item() after the loop.",
        "score": "model.eval(), then inside torch.no_grad() compare (model(inputs) > 0).float() with targets and take "
        "the mean's .item(). Call model.train() before returning it.",
        "silent": "Pass own, a lambda that takes a model and returns loss_of of it on train_rows and train_next, and "
        "0.1.",
        "own_score": "Call train on own and on baseline with the training rows and TRAIN_STEPS, then score each on "
        "held_rows and held_next.",
        "baseline_score": "It comes from the same lines as own_score. Get those right first.",
        "gaps": "Call property_gaps with own, held_rows[:64] and TRANSFORMS, after training.",
    },
    introduces=(),
    check=check_own2,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def wide_rows(
    count: int,
    width: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    rows = torch.randint(0, 2, (count, width), generator=torch.Generator().manual_seed(seed)).float()

    return rows, REFERENCE["exact"](rows)


def rollout_match(
    model: nn.Module,
    width: int,
    steps: int,
    seed: int,
) -> float:
    """The share of cells the module gets right when it feeds its own rows back for steps steps, against the rule."""
    truth = torch.randint(0, 2, (1, width), generator=torch.Generator().manual_seed(seed)).float()
    guess = truth
    matched = []

    with torch.no_grad():
        for _ in range(steps):
            truth = REFERENCE["exact"](truth)
            guess = (model(guess) > 0).float()
            matched.append((truth == guess).float().mean().item())

    return sum(matched) / len(matched)


def trained_reference(
    model_class: type,
    rows: int,
    seed: int,
) -> tuple[nn.Module, tuple[Tensor, Tensor]]:
    batch = REFERENCE["make_batch"](rows, torch.Generator().manual_seed(seed))
    torch.manual_seed(seed)
    model = model_class()
    REFERENCE["train"](model, *batch, TRAIN_STEPS)

    return model, batch


def measure_own_module(seeds: range = range(3)) -> dict[str, object]:
    """own1's floor and own2's table, rollouts and runs with fewer and more rows, on Rule 110 with 4 threads."""
    score = REFERENCE["score"]

    with pinned_threads():
        held = REFERENCE["make_batch"](HELD_ROWS, torch.Generator().manual_seed(1000))
        wide = wide_rows(HELD_ROWS, 256, 1001)
        results: dict[str, object] = {
            "floor": REFERENCE["floor_score"](*REFERENCE["make_batch"](HELD_ROWS, torch.Generator().manual_seed(2))),
            "own": [],
            "baseline": [],
            "own rollout": [],
            "own from 16 rows": [],
            "baseline from 256 rows": [],
            "baseline from 1024 rows": [],
        }

        for seed in seeds:
            own, batch = trained_reference(REFERENCE["OwnModule"], TRAIN_ROWS, seed)
            baseline, _ = trained_reference(REFERENCE["Baseline"], TRAIN_ROWS, seed)
            results["own"].append((score(own, *batch), score(own, *held), score(own, *wide)))
            results["baseline"].append((score(baseline, *batch), score(baseline, *held)))
            results["own rollout"].append(rollout_match(own, 80, 100, 2000 + seed))
            results["own from 16 rows"].append(score(trained_reference(REFERENCE["OwnModule"], 16, seed)[0], *held))
            results["baseline from 256 rows"].append(score(trained_reference(REFERENCE["Baseline"], 256, seed)[0], *held))
            results["baseline from 1024 rows"].append(score(trained_reference(REFERENCE["Baseline"], 1024, seed)[0], *held))

    return results
