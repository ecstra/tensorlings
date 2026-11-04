"""102_outputs, what a model writes and the head that fits it: classes, tags and values from one trunk, a mixture of normals for several right answers, constrained decoding for valid JSON, and actions and signals written as tokens."""

from __future__ import annotations

import contextlib
import json
import math
import string
import textwrap
from collections.abc import Callable, Iterator

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
    expect_value,
    reject_missing_return,
    seeded,
    small_tensor,
    value_text,
)
from ..text import prose, source_file
from .fusion import FUSE2_REFERENCE
from .linear_layers import expect_module
from .vqvae import QUANTIZE_PART

OUTPUTS = Chapter(
    folder="102_outputs",
    title="Outputs",
    readme=source_file(
        """
        # Outputs

        A model's last layer decides what it can say. One class, several tags,
        a number, a choice between two right answers, a valid JSON object, a
        robot's next move and a signal each need their own head, or their own
        way of writing tokens. This chapter builds them and measures what the
        wrong head gets wrong.

        ## Further reading

        - [Bishop, mixture density networks](https://publications.aston.ac.uk/id/eprint/373)
        - [Willard and Louf, efficient guided generation](https://arxiv.org/abs/2307.09702)
        - [Brohan and colleagues, RT-2](https://arxiv.org/abs/2307.15818)
        """,
    ),
)


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """The CPU's thread count moves the last digits of a run, so the measurements pin it."""
    before = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        yield
    finally:
        torch.set_num_threads(before)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "json": json,
        "string": string,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(code, namespace)

    return namespace


def seeded_model(
    make: Callable[..., nn.Module],
    seed: int,
    *arguments: object,
) -> nn.Module:
    torch.manual_seed(seed)

    return make(*arguments)


# ---------------------------------------------------------------------------
# out1
# ---------------------------------------------------------------------------

HEADS_INIT = '''
class Heads(nn.Module):
    def __init__(
        self,
        features: int = 8,
        classes: int = 4,
        tags: int = 5,
        values: int = 2,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(features, hidden), nn.ReLU())
        self.class_head = nn.Linear(hidden, classes)
        self.tag_head = nn.Linear(hidden, tags)
        self.value_head = nn.Linear(hidden, values)
'''

HEADS_FORWARD = '''
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        shared = self.trunk(x)

        return self.class_head(shared), self.tag_head(shared), self.value_head(shared)
'''

HEADS_LOSS_SOLUTION = '''
def heads_loss(
    outputs: tuple[Tensor, Tensor, Tensor],
    classes: Tensor,
    tags: Tensor,
    values: Tensor,
) -> Tensor:
    class_scores, tag_scores, guessed = outputs
    one_class = F.cross_entropy(class_scores, classes)
    every_tag = F.binary_cross_entropy_with_logits(tag_scores, tags)

    return one_class + every_tag + F.mse_loss(guessed, values)
'''

READ_SOLUTION = '''
def read_heads(outputs: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    class_scores, tag_scores, guessed = outputs

    return class_scores.argmax(dim=-1), torch.sigmoid(tag_scores) > 0.5, guessed
'''

OUT1_REFERENCE = namespace_of(HEADS_INIT + HEADS_FORWARD + HEADS_LOSS_SOLUTION + READ_SOLUTION)

FEATURES = 8
TAGS = 5
TAG_WEIGHTS = torch.randn(TAGS, FEATURES, generator=torch.Generator().manual_seed(7))


def tag_data(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Inputs of 8 numbers and 5 tags, each on where its own weighted sum passes 0.4, so many inputs hold several."""
    x = torch.randn(count, FEATURES, generator=generator)

    return x, (x @ TAG_WEIGHTS.T > 0.4).float()


OUT1_LESSON = prose(
    """
    activations2 matched the output to the answer's form. A model often has
    several answers to give at once. One trunk reads the input, and a small
    head per answer reads the trunk.

    | Answer | Head gives | Loss |
    |---|---|---|
    | one class of several | a score per class | `F.cross_entropy`, softmax inside |
    | any number of tags | a score per tag | `F.binary_cross_entropy_with_logits`, a sigmoid per tag |
    | several numbers | one number each | `F.mse_loss` |

    A softmax's chances add up to 1, so it bets on one answer. A sigmoid per
    tag says yes or no to each tag on its own, so a document can be news and
    sport at once. Reading a tag means its sigmoid passes 0.5.

    Here each input of 8 numbers carried 5 tags, each on when its own
    weighted sum passed 0.4. 67% to 69% of inputs carried two or more, and
    4.5% to 5.5% none. Two heads trained 1,500 steps on three seeds, scored
    on the share of inputs whose whole set of tags came out right.

    | Head | Whole set right |
    |---|---|
    | softmax, its top class taken as the one tag | 25% to 27% |
    | a sigmoid per tag | 95% to 96% |

    The softmax head names exactly one tag, so it fails every input with
    none or with more. It got right every input with exactly one tag, the
    most it could. The losses of several heads add up into one, and one
    backward pass trains the trunk for all of them.
    """,
)

OUT1_SOURCE = source_file(
    '''
    # out1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    torch.sigmoid(torch.tensor([2.0, -1.0, 0.0])) > 0.5    # tensor([ True, False, False])
    F.binary_cross_entropy_with_logits(torch.tensor([3.0]), torch.tensor([1.0]))    # tensor(0.0486)


    # TODO: Finish forward. Run x through trunk, then each of the three heads
    #   on the trunk's output. Return the class scores, the tag scores and
    #   the values, in that order.
''' + indented(HEADS_INIT) + '''

        def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            ...


    # TODO: Return one loss for all three heads, as a tensor. outputs is what
    #   Heads gives. Add the cross entropy of the class scores against
    #   classes, the binary cross entropy of the tag scores against tags,
    #   from the raw scores, and the mean squared error against values.
    def heads_loss(
        outputs: tuple[Tensor, Tensor, Tensor],
        classes: Tensor,
        tags: Tensor,
        values: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the answers outputs gives: each row's class, its tags as
    #   True where the tag's sigmoid passes 0.5, and the values as they are.
    def read_heads(outputs: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        ...
    ''',
)

OUT1_SOLUTION = source_file(
    '''
    # out1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    torch.sigmoid(torch.tensor([2.0, -1.0, 0.0])) > 0.5
    F.binary_cross_entropy_with_logits(torch.tensor([3.0]), torch.tensor([1.0]))


''' + indented(HEADS_INIT + HEADS_FORWARD) + '''


''' + indented(HEADS_LOSS_SOLUTION) + '''


''' + indented(READ_SOLUTION) + '''
    ''',
)


def expect_triple(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor, Tensor],
    names: tuple[str, str, str],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, tuple) or len(actual) != 3:
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be three tensors, {', '.join(names)}.")

    for name, found, wanted in zip(names, actual, expected):
        expect_tensor(
            call=f"the {name} from {call}",
            actual=found,
            expected=wanted,
        )


def check_heads_forward(found: type) -> None:
    for seed, batch in ((1301, 3), (1302, 2)):
        model = expect_module("Heads(6, 3, 4, 2, 8)", seeded_model(found, seed, 6, 3, 4, 2, 8))
        x = small_tensor(seed, batch, 6) / 3

        with torch.no_grad():
            expected = seeded_model(OUT1_REFERENCE["Heads"], seed, 6, 3, 4, 2, 8)(x)
            actual = model(x)

        expect_triple(
            call=f"Heads(6, 3, 4, 2, 8) on {batch} rows",
            actual=actual,
            expected=expected,
            names=("class scores", "tag scores", "values"),
        )


def head_outputs(
    seed: int,
    batch: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return small_tensor(seed, batch, 3) / 2, small_tensor(seed + 1, batch, 4) / 2, small_tensor(seed + 2, batch, 2) / 2


def check_heads_loss(found: object) -> None:
    for seed, batch in ((1303, 3), (1304, 4)):
        outputs = head_outputs(seed, batch)
        generator = torch.Generator().manual_seed(seed)
        classes = torch.randint(0, 3, (batch,), generator=generator)
        tags = (torch.rand(batch, 4, generator=generator) > 0.5).float()
        values = torch.randn(batch, 2, generator=generator)
        call = f"heads_loss on {batch} rows of 3 classes, 4 tags and 2 values"
        actual = found(tuple(part.clone() for part in outputs), classes, tags, values)
        reject_missing_return(
            call=call,
            actual=actual,
        )

        if not isinstance(actual, Tensor):
            raise Mismatch(f"{call} is {value_text(actual)}, but it should be a tensor, so backward can run through it.")

        expect_tensor(
            call=call,
            actual=actual,
            expected=OUT1_REFERENCE["heads_loss"](outputs, classes, tags, values),
        )


def expect_read(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor, Tensor],
) -> None:
    expect_triple(
        call=call,
        actual=actual,
        expected=expected,
        names=("classes", "tags", "values"),
    )

    if actual[0].dtype != torch.long:
        raise Mismatch(f"The classes from {call} hold {actual[0].dtype} numbers, but they should be whole numbers, positions from argmax.")

    if actual[1].dtype != torch.bool:
        raise Mismatch(f"The tags from {call} hold {actual[1].dtype} numbers, but they should be True and False. Return the comparison itself.")


def check_out1(checker: Checker) -> None:
    checker.test_method(
        class_name="Heads",
        method="forward",
        description="runs the trunk once and each head on it",
        run=check_heads_forward,
    )
    checker.test(
        function="heads_loss",
        description="adds the cross entropy, the binary cross entropy and the squared error",
        run=check_heads_loss,
    )
    check_on_inputs(
        checker=checker,
        function="read_heads",
        description="reads one class, the tags past 0.5 and the values",
        inputs=(
            (Described(head_outputs(1305, 3), "the three heads' outputs for 3 rows"),),
            (Described(head_outputs(1306, 5), "the three heads' outputs for 5 rows"),),
        ),
        reference=OUT1_REFERENCE["read_heads"],
        expect=expect_read,
    )


OUT1 = Exercise(
    name="out1",
    chapter=OUTPUTS,
    title="Heads for classes, tags and values",
    lesson=OUT1_LESSON,
    source=OUT1_SOURCE,
    solution=OUT1_SOLUTION,
    hints={
        "Heads.forward": "Keep self.trunk(x) in a name, then return a tuple of self.class_head, self.tag_head and "
        "self.value_head each run on it.",
        "heads_loss": "Unpack outputs into three names. F.cross_entropy, F.binary_cross_entropy_with_logits and F.mse_loss, "
        "each with its own target, added together.",
        "read_heads": "argmax over the last dimension for the class. torch.sigmoid of the tag scores compared with 0.5 for "
        "the tags. The values come back unchanged.",
    },
    introduces=(),
    check=check_out1,
)


# ---------------------------------------------------------------------------
# out2
# ---------------------------------------------------------------------------

SPLIT_SOLUTION = '''
def split_mixture(
    out: Tensor,
    components: int,
) -> tuple[Tensor, Tensor, Tensor]:
    weights, means, log_spreads = out.split(components, dim=-1)

    return weights, means, log_spreads
'''

NLL_SOLUTION = '''
def mixture_nll(
    weights: Tensor,
    means: Tensor,
    log_spreads: Tensor,
    y: Tensor,
) -> Tensor:
    gap = (y.unsqueeze(-1) - means) / log_spreads.exp()
    log_densities = -0.5 * gap**2 - log_spreads - 0.5 * math.log(2 * math.pi)

    return -torch.logsumexp(F.log_softmax(weights, dim=-1) + log_densities, dim=-1).mean()
'''

SAMPLE_SOLUTION = '''
def sample_mixture(
    weights: Tensor,
    means: Tensor,
    log_spreads: Tensor,
    generator: torch.Generator,
) -> Tensor:
    picked = torch.multinomial(torch.softmax(weights, dim=-1), 1, generator=generator)
    mean = means.gather(-1, picked).squeeze(-1)
    spread = log_spreads.exp().gather(-1, picked).squeeze(-1)

    return mean + spread * torch.randn(mean.shape, generator=generator)
'''

OUT2_REFERENCE = namespace_of(SPLIT_SOLUTION + NLL_SOLUTION + SAMPLE_SOLUTION)


def branch_data(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """x from 0 to 1, and y on either of two branches, 0.3 + x or its negative, with a little noise."""
    x = torch.rand(count, 1, generator=generator)
    sign = torch.randint(0, 2, (count, 1), generator=generator).float() * 2 - 1
    y = sign * (0.3 + x) + 0.05 * torch.randn(count, 1, generator=generator)

    return x, y.squeeze(1)


OUT2_LESSON = prose(
    """
    Some inputs have more than one right answer. An arm can reach the same
    point with its elbow up or down. A road forks, and either branch is fine.

    Bishop showed in 1994 that a head trained on the squared error learns the
    average answer for each input. With two right answers, the average can be
    neither of them, a hand halfway between up and down.

    His **mixture density network** gives a whole distribution instead. For
    each input, the head gives K normals, each with a weight, a mean and a
    spread. The weights go through a softmax, and the spreads are kept as
    logs so they stay positive.

    The mixture's density at y is the weighted sum of its normals'
    densities. The loss is likelihood3's negative log of that density.
    Adding chances in log space is likelihood2's logsumexp.

    ```text
    loss = -logsumexp(log_softmax(weights) + log N(y; mean, spread))
    ```

    To answer, pick one normal by its weight, then draw from it.

    Here y lies on one of two branches, `0.3 + x` or `-(0.3 + x)`, for x from
    0 to 1. Each head trained 2,000 steps on three seeds. An answer counts as
    right within 0.15 of a branch.

    | Head | Answers right | Answers above 0 |
    |---|---|---|
    | one number, squared error | 0% | 0% or 100% |
    | a mixture of 2 normals, sampled | 99.4% to 99.9% | 47% to 53% |

    The plain head settles near 0, between the branches, and is never right.
    Each of its runs put every answer on the same side of 0. The mixture
    puts one normal on each branch and picks them evenly.
    """,
)

OUT2_SOURCE = source_file(
    '''
    # out2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    torch.arange(6.0).split(2, dim=-1)    # (tensor([0., 1.]), tensor([2., 3.]), tensor([4., 5.]))
    torch.tensor([[5.0, 6.0]]).gather(-1, torch.tensor([[1]]))    # tensor([[6.]])


    # TODO: Return the weights, the means and the log spreads, the head's out
    #   cut into three parts of components numbers each along the last
    #   dimension.
    def split_mixture(
        out: Tensor,
        components: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ...


    # TODO: Return the average negative log likelihood of y under each row's
    #   mixture, as a tensor. The log density of a normal at y is minus half
    #   the square of (the gap divided by the spread), minus the log spread,
    #   minus half of log 2 pi. Add each normal's log softmax weight and
    #   logsumexp them.
    def mixture_nll(
        weights: Tensor,
        means: Tensor,
        log_spreads: Tensor,
        y: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return one draw per row. First pick a normal per row by its
    #   softmax weight with torch.multinomial and the generator. Then return
    #   its mean plus its spread times one torch.randn draw from the generator.
    def sample_mixture(
        weights: Tensor,
        means: Tensor,
        log_spreads: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        ...
    ''',
)

OUT2_SOLUTION = source_file(
    '''
    # out2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    torch.arange(6.0).split(2, dim=-1)
    torch.tensor([[5.0, 6.0]]).gather(-1, torch.tensor([[1]]))


''' + indented(SPLIT_SOLUTION) + '''


''' + indented(NLL_SOLUTION) + '''


''' + indented(SAMPLE_SOLUTION) + '''
    ''',
)


def expect_split(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor, Tensor],
) -> None:
    expect_triple(
        call=call,
        actual=actual,
        expected=expected,
        names=("weights", "means", "log spreads"),
    )


def mixture_inputs(
    seed: int,
    batch: int,
    components: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)

    return (
        torch.randn(batch, components, generator=generator),
        torch.randn(batch, components, generator=generator),
        torch.randn(batch, components, generator=generator) * 0.3,
        torch.randn(batch, generator=generator),
    )


def expect_loss_tensor(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, Tensor):
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a tensor, so backward can run through it.")

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def check_out2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="split_mixture",
        description="cuts the head's output into weights, means and log spreads",
        inputs=(
            (small_tensor(1401, 2, 6), 2),
            (small_tensor(1402, 3, 9), 3),
        ),
        reference=OUT2_REFERENCE["split_mixture"],
        expect=expect_split,
    )
    check_on_inputs(
        checker=checker,
        function="mixture_nll",
        description="gives the average negative log likelihood under each row's mixture",
        inputs=(
            mixture_inputs(1403, 4, 2),
            mixture_inputs(1404, 3, 3),
        ),
        reference=OUT2_REFERENCE["mixture_nll"],
        expect=expect_loss_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="sample_mixture",
        description="picks a normal by its weight, then draws from it",
        inputs=(
            (*mixture_inputs(1405, 6, 2)[:3], seeded(1405)),
            (*mixture_inputs(1406, 4, 3)[:3], seeded(1406)),
        ),
        reference=OUT2_REFERENCE["sample_mixture"],
    )


OUT2 = Exercise(
    name="out2",
    chapter=OUTPUTS,
    title="A mixture for several right answers",
    lesson=OUT2_LESSON,
    source=OUT2_SOURCE,
    solution=OUT2_SOLUTION,
    hints={
        "split_mixture": "out.split(components, dim=-1) gives the three parts as a tuple, in order.",
        "mixture_nll": "Give y a last dimension with .unsqueeze(-1) so it lines up with every normal. log_spreads.exp() "
        "is the spread. F.log_softmax of the weights, added to the log densities, then torch.logsumexp over dim=-1, "
        "minus, and the mean.",
        "sample_mixture": "torch.multinomial(torch.softmax(weights, dim=-1), 1, generator=generator) picks a column per row. "
        ".gather(-1, picked) reads that column of the means and the spreads. Then one torch.randn of that shape.",
    },
    introduces=("mixture density network",),
    check=check_out2,
)


# ---------------------------------------------------------------------------
# out3
# ---------------------------------------------------------------------------

SHAPE_PART = '''
CHARACTERS = list('{}":, ' + string.ascii_lowercase + string.digits) + ["\\n"]
END = "\\n"

# The shape every answer must take: {"name": "<1 to 6 letters>", "age": <1 to 2 digits>} and a newline.
PARTS = (
    ("literal", '{"name": "'),
    ("letters", 1, 6),
    ("literal", '", "age": '),
    ("digits", 1, 2),
    ("literal", "}\\n"),
)


def allowed(state: tuple[int, int]) -> str:
    """The characters that may come next, where state is the part we are in and how far into it."""
    part, used = state
    kind = PARTS[part][0]

    if kind == "literal":
        return PARTS[part][1][used]

    pool = string.ascii_lowercase if kind == "letters" else string.digits

    if kind == "digits" and used == 0:
        pool = string.digits[1:]

    least, most = PARTS[part][1], PARTS[part][2]
    following = PARTS[part + 1][1][0]

    return (pool if used < most else "") + (following if used >= least else "")


def step(
    state: tuple[int, int],
    character: str,
) -> tuple[int, int]:
    """The state after writing one allowed character."""
    part, used = state
    kind = PARTS[part][0]

    if kind == "literal":
        used += 1

        return (part + 1, 0) if used == len(PARTS[part][1]) else (part, used)

    pool = string.ascii_lowercase if kind == "letters" else string.digits

    if character in pool:
        return (part, used + 1)

    return step((part + 1, 0), character)


def is_valid(text: str) -> bool:
    if not text.endswith(END):
        return False

    try:
        value = json.loads(text[:-1])
    except json.JSONDecodeError:
        return False

    return isinstance(value, dict) and list(value) == ["name", "age"] and isinstance(value["name"], str) and type(value["age"]) is int
'''

MASK_SOLUTION = '''
def mask_scores(
    scores: Tensor,
    allowed_characters: str,
) -> Tensor:
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep[[CHARACTERS.index(character) for character in allowed_characters]] = True

    return scores.masked_fill(~keep, float("-inf"))
'''

CONSTRAINED_SOLUTION = '''
def constrained_text(
    next_scores: Callable[[str], Tensor],
    generator: torch.Generator,
    limit: int = 40,
) -> str:
    text = ""
    state = (0, 0)

    for _ in range(limit):
        scores = mask_scores(next_scores(text), allowed(state))
        character = CHARACTERS[int(torch.multinomial(torch.softmax(scores, dim=-1), 1, generator=generator))]
        text += character

        if character == END:
            return text

        state = step(state, character)

    return text
'''

SHARE_SOLUTION = '''
def valid_share(texts: list[str]) -> float:
    return sum(is_valid(text) for text in texts) / len(texts)
'''

OUT3_REFERENCE = namespace_of("from collections.abc import Callable\n" + SHAPE_PART + MASK_SOLUTION + CONSTRAINED_SOLUTION + SHARE_SOLUTION)
CHARACTERS: list[str] = OUT3_REFERENCE["CHARACTERS"]
END: str = OUT3_REFERENCE["END"]

OUT3_LESSON = prose(
    """
    A program that reads a model's answer needs it in a fixed form, like
    JSON. A model writing one character at a time can slip, and one wrong
    character breaks the whole answer.

    The form says which characters may come next. After `{"name": "` comes a
    letter. After the age's digits come more digits or `}`. Before each
    choice, set every other character's score to minus infinity, as gpt4's
    `top_k_filter` did. The softmax gives those exactly 0.

    ```python
    torch.tensor([2.0, 1.0, 3.0]).masked_fill(torch.tensor([False, True, True]), float("-inf"))
    # tensor([2., -inf, -inf])
    ```

    This is **constrained decoding**. The given `allowed` and `step` follow
    one fixed shape. Willard and Louf's Outlines library builds the same
    allowed sets from a regular expression or a grammar. It runs them as a
    finite state machine, with a stack added for a grammar, at little cost
    per token. RT-2 restricts its output
    the same way, to its action tokens.

    Here a small GRU learned the shape from examples like
    `{"name": "ada", "age": 36}`, for 100, 300 or 1,000 steps on three seeds.
    Each wrote 300 answers, free and masked.

    | Training | Valid, free | Valid, masked |
    |---|---|---|
    | 100 steps | 15% to 45% | 100% |
    | 300 steps | 91% to 94% | 100% |
    | 1,000 steps | 98% to 99.7% | 100% |

    The mask fixes the form at any stage of training, so the reading program
    never fails. What the model puts in the form, the name and the age,
    still comes from the model.
    """,
)

OUT3_SOURCE = source_file(
    '''
    # out3.py

    import json
    import string
    from collections.abc import Callable

    import torch
    from torch import Tensor

    blocked = torch.tensor([2.0, 1.0, 3.0]).masked_fill(torch.tensor([False, True, True]), float("-inf"))    # tensor([2., -inf, -inf])
    torch.softmax(blocked, dim=-1)    # tensor([1., 0., 0.])

''' + indented(SHAPE_PART) + '''


    # TODO: Return scores, one per character of CHARACTERS, with every
    #   character not in allowed_characters set to minus infinity.
    def mask_scores(
        scores: Tensor,
        allowed_characters: str,
    ) -> Tensor:
        ...


    # TODO: Return one answer written a character at a time, starting from the
    #   empty text and state (0, 0). Each step, mask next_scores(text) to what
    #   allowed(state) gives, pick one character with torch.multinomial of the
    #   softmax and the generator, and add it. Stop after END. Otherwise move
    #   the state on with step. Give up after limit characters.
    def constrained_text(
        next_scores: Callable[[str], Tensor],
        generator: torch.Generator,
        limit: int = 40,
    ) -> str:
        ...


    # TODO: Return the share of texts that is_valid accepts, as a plain number.
    def valid_share(texts: list[str]) -> float:
        ...
    ''',
)

OUT3_SOLUTION = source_file(
    '''
    # out3.py, a solution for comparison

    import json
    import string
    from collections.abc import Callable

    import torch
    from torch import Tensor

    blocked = torch.tensor([2.0, 1.0, 3.0]).masked_fill(torch.tensor([False, True, True]), float("-inf"))
    torch.softmax(blocked, dim=-1)

''' + indented(SHAPE_PART) + '''


''' + indented(MASK_SOLUTION) + '''


''' + indented(CONSTRAINED_SOLUTION) + '''


''' + indented(SHARE_SOLUTION) + '''
    ''',
)


def random_scorer(offset: int) -> Callable[[str], Tensor]:
    """A stand in for a model, the same scores for the same text however often it is asked."""

    def next_scores(text: str) -> Tensor:
        seed = offset + len(text) * 7919 + sum(ord(character) * (index + 1) for index, character in enumerate(text))

        return torch.randn(len(CHARACTERS), generator=torch.Generator().manual_seed(seed)) * 3

    return next_scores


def check_constrained_text(found: object) -> None:
    for offset in (1501, 1502):
        scorer = random_scorer(offset)

        for draw in range(4):
            seed = offset * 10 + draw
            actual = found(scorer, torch.Generator().manual_seed(seed), 40)
            expected = OUT3_REFERENCE["constrained_text"](scorer, torch.Generator().manual_seed(seed), 40)
            call = f"constrained_text with a model of random scores and a generator seeded with {seed}"
            reject_missing_return(
                call=call,
                actual=actual,
            )

            if not isinstance(actual, str):
                raise Mismatch(f"{call} is {value_text(actual)}, but it should be the text, a string.")

            if not OUT3_REFERENCE["is_valid"](actual):
                raise Mismatch(f"{call} wrote {actual!r}, which breaks the shape. Mask before every pick, with the state moved on after each character.")

            if actual != expected:
                raise Mismatch(f"{call} wrote {actual!r}, but it should write {expected!r}. Draw once per character, with the mask applied first.")


def check_out3(checker: Checker) -> None:
    for seed, allowed_characters in ((1503, "ab}"), (1504, '"0123456789')):
        check_on_inputs(
            checker=checker,
            function="mask_scores",
            description="sets every character not allowed to minus infinity",
            inputs=((torch.randn(len(CHARACTERS), generator=torch.Generator().manual_seed(seed)), allowed_characters),),
            reference=OUT3_REFERENCE["mask_scores"],
        )

    checker.test(
        function="constrained_text",
        description="writes only valid answers, masking before every pick",
        run=check_constrained_text,
    )
    check_on_inputs(
        checker=checker,
        function="valid_share",
        description="gives the share of valid answers",
        inputs=(
            (['{"name": "ab", "age": 7}\n', '{"name": "ab, "age": 7}\n', '{"name": "c", "age": 12}\n'],),
            (['{"name": "x", "age": 5}\n', '{"name": "x", "age": 5}', '{"age": 5, "name": "x"}\n', '{"name": "x", "age": 5.0}\n'],),
        ),
        reference=OUT3_REFERENCE["valid_share"],
        expect=expect_close,
    )


OUT3 = Exercise(
    name="out3",
    chapter=OUTPUTS,
    title="Constrained decoding",
    lesson=OUT3_LESSON,
    source=OUT3_SOURCE,
    solution=OUT3_SOLUTION,
    hints={
        "mask_scores": "Make a mask of False, shaped like scores, with torch.zeros_like and dtype=torch.bool. Set True at "
        "CHARACTERS.index of each allowed character, then masked_fill where the mask is False.",
        "constrained_text": "A for loop over range(limit). mask_scores of next_scores(text) and allowed(state), then "
        "torch.multinomial of the softmax with 1 draw and the generator. CHARACTERS of that position is the character.",
        "valid_share": "Count the texts is_valid accepts with sum, then divide by len(texts).",
    },
    introduces=("constrained decoding",),
    check=check_out3,
)


# ---------------------------------------------------------------------------
# out4
# ---------------------------------------------------------------------------

BINS_SOLUTION = '''
def to_bins(
    values: Tensor,
    low: float,
    high: float,
    bins: int,
) -> Tensor:
    return ((values - low) / (high - low) * bins).floor().long().clamp(0, bins - 1)


def from_bins(
    ids: Tensor,
    low: float,
    high: float,
    bins: int,
) -> Tensor:
    return low + (ids.float() + 0.5) * (high - low) / bins
'''

NEAREST_PART = QUANTIZE_PART.split("\n\n\ndef quantize(")[0]

CODEBOOK_SOLUTION = '''
def signal_ids(
    signal: Tensor,
    codebook: Tensor,
) -> Tensor:
    return nearest_codes(signal.view(-1, codebook.shape[1]), codebook)


def write_signal(
    ids: Tensor,
    codebook: Tensor,
) -> Tensor:
    return codebook[ids].flatten()
'''

OUT4_REFERENCE = namespace_of(BINS_SOLUTION + NEAREST_PART + CODEBOOK_SOLUTION)
BINS = 256


def action_data(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """A state of 2 numbers and a steer of 0.6 left or right around an obstacle, either one fine."""
    x = torch.rand(count, 2, generator=generator)
    side = torch.randint(0, 2, (count,), generator=generator).float() * 2 - 1

    return x, side * 0.6 + 0.02 * torch.randn(count, generator=generator)


def sine_signals(
    count: int,
    generator: torch.Generator,
    length: int = 256,
) -> Tensor:
    """Signals of 256 steps, each the sum of three sines of random frequency, phase and size."""
    t = torch.arange(length).float()
    frequency = torch.rand(count, 3, 1, generator=generator) * 0.2 + 0.01
    phase = torch.rand(count, 3, 1, generator=generator) * 2 * math.pi
    size = torch.rand(count, 3, 1, generator=generator)

    return (size * torch.sin(frequency * t + phase)).sum(dim=1)


OUT4_LESSON = prose(
    """
    A language model writes tokens. To make it act, RT-2 wrote a robot's
    moves as tokens too. Each move is 8 numbers, a stop flag and 7 for how
    the arm moves and grips. The 7 continuous ones are cut into 256 even
    bins, and each bin is a token. PaLI-X already had a token for each whole
    number to 1,000. PaLM-E had none, so its 256 rarest tokens were taken.

    ```text
    bin = floor((value - low) / (high - low) * bins), kept from 0 to bins - 1
    value back = low + (bin + 0.5) * (high - low) / bins
    ```

    Reading a bin back gives its middle, at most half a bin from the true
    value. With 256 bins over -1 to 1 that is 0.0039.

    Tokens also keep out2's two right answers. Here a cart steers 0.6 left or
    right around an obstacle, either one fine. A head of one number and a
    head of 256 bin tokens trained 1,500 steps on three seeds. A steer counts
    within 0.08 of a fine one.

    | Head | Steers fine |
    |---|---|
    | one number, squared error | 0% |
    | 256 bin tokens, the likeliest | 100% |

    The number head steers straight at the obstacle. The tokens' softmax
    holds both sides, and the likeliest bin is always one of them.

    Any signal can be written the same way through vq1's codebook. Cut it
    into pieces, write each piece as its nearest code's id, and read the ids
    back as the codes. Here codebooks were fitted to sums of sines by
    k-means, each code moved to the mean of its pieces. The error left is
    the share of the signal's sum of squares that the codes miss.

    | Codes | Error left |
    |---|---|
    | 8 | 9.5% to 10.7% |
    | 32 | 3.1% to 3.3% |
    | 128 | 1.0% |

    Then a model writes a signal as it writes text. Here a wave runs 32
    steps, then rises or falls, either one right. out3's GRU learned such
    series as ids, 8 steps to a piece and 128 codes, and wrote each future
    by sampling 4 ids. A head of 32 numbers learned the future by squared
    error. Each trained 6,000 steps on three seeds.

    | Head | Futures nearer a right one than the two's average |
    |---|---|
    | 32 numbers, squared error | 0% |
    | 4 codebook ids, sampled | 99.7% to 99.8% |

    The number head writes nearer the average, the wave with neither rise
    nor fall. The sampled ids pick a side, rising in 50% to 54% of series.
    """,
)

OUT4_SOURCE = source_file(
    '''
    # out4.py

    import torch
    from torch import Tensor

    torch.tensor([-1.0, 0.3, 1.0]).floor().long().clamp(0, 1)    # tensor([0, 0, 1])
    torch.tensor([[1.0, 2.0], [3.0, 4.0]])[torch.tensor([1, 1, 0])].flatten()    # tensor([3., 4., 3., 4., 1., 2.])


    # TODO: Return each value's bin, a whole number from 0 to bins - 1, for
    #   bins even bins from low to high. A value at high goes in the last bin.
    def to_bins(
        values: Tensor,
        low: float,
        high: float,
        bins: int,
    ) -> Tensor:
        ...


    # TODO: Return the middle of each bin of ids, as floats, for the same bins.
    def from_bins(
        ids: Tensor,
        low: float,
        high: float,
        bins: int,
    ) -> Tensor:
        ...


    # vq1's distances and nearest codes.
''' + indented(NEAREST_PART) + '''


    # TODO: signal_ids returns the id of the nearest code for each piece of
    #   signal, cut into pieces as long as a code. write_signal returns the
    #   signal the ids spell, their codes one after another, flat.
    def signal_ids(
        signal: Tensor,
        codebook: Tensor,
    ) -> Tensor:
        ...


    def write_signal(
        ids: Tensor,
        codebook: Tensor,
    ) -> Tensor:
        ...
    ''',
)

OUT4_SOLUTION = source_file(
    '''
    # out4.py, a solution for comparison

    import torch
    from torch import Tensor

    torch.tensor([-1.0, 0.3, 1.0]).floor().long().clamp(0, 1)
    torch.tensor([[1.0, 2.0], [3.0, 4.0]])[torch.tensor([1, 1, 0])].flatten()


''' + indented(BINS_SOLUTION) + '''


''' + indented(NEAREST_PART) + '''


''' + indented(CODEBOOK_SOLUTION) + '''
    ''',
)


def expect_whole(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )

    if actual.dtype != torch.long:
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but ids are whole numbers. Turn them into whole numbers with .long().")


def check_out4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="to_bins",
        description="gives each value its bin, the edges kept inside",
        inputs=(
            (torch.tensor([-1.0, -0.6, 0.0, 0.3, 1.0]), -1.0, 1.0, 4),
            (torch.tensor([[0.0, 2.0], [9.4, 10.0]]), 0.0, 10.0, 8),
        ),
        reference=OUT4_REFERENCE["to_bins"],
        expect=expect_whole,
    )
    check_on_inputs(
        checker=checker,
        function="from_bins",
        description="gives each bin's middle",
        inputs=(
            (torch.tensor([0, 1, 3]), -1.0, 1.0, 4),
            (torch.tensor([[0, 7], [3, 4]]), 0.0, 10.0, 8),
        ),
        reference=OUT4_REFERENCE["from_bins"],
    )
    codebook_a = small_tensor(1601, 4, 3) / 2
    codebook_b = small_tensor(1602, 6, 2) / 2
    check_on_inputs(
        checker=checker,
        function="signal_ids",
        description="writes each piece of the signal as its nearest code's id",
        inputs=(
            (small_tensor(1603, 12) / 2, codebook_a),
            (small_tensor(1604, 10) / 2, codebook_b),
        ),
        reference=OUT4_REFERENCE["signal_ids"],
        expect=expect_whole,
    )
    check_on_inputs(
        checker=checker,
        function="write_signal",
        description="reads the ids back as their codes, flat",
        inputs=(
            (torch.tensor([2, 0, 2]), codebook_a),
            (torch.tensor([5, 1, 1, 4]), codebook_b),
        ),
        reference=OUT4_REFERENCE["write_signal"],
    )


OUT4 = Exercise(
    name="out4",
    chapter=OUTPUTS,
    title="Actions and signals as tokens",
    lesson=OUT4_LESSON,
    source=OUT4_SOURCE,
    solution=OUT4_SOLUTION,
    hints={
        "to_bins": "Scale the values to run from 0 to bins, then .floor() and .long() for whole numbers. .clamp(0, bins - 1) "
        "keeps a value at high in the last bin.",
        "from_bins": "Turn ids into floats, add 0.5 for the middle, then scale back by the bin width (high - low) / bins "
        "and add low.",
        "signal_ids": "signal.view(-1, codebook.shape[1]) cuts it into pieces as long as a code. nearest_codes does the rest.",
        "write_signal": "Index codebook with the ids, then .flatten().",
    },
    introduces=(),
    check=check_out4,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_tags(seeds: range = range(3)) -> dict[str, list[float]]:
    """out1's table: whole tag sets right with one softmax class and with a sigmoid per tag, and the shares of inputs
    with no tag, with exactly one, and with 2 or more."""
    results: dict[str, list[float]] = {"softmax": [], "sigmoid": [], "none": [], "one": [], "several": []}

    with pinned_threads():
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)
            x, tags = tag_data(4000, torch.Generator().manual_seed(seed + 50))
            results["none"].append((tags.sum(dim=1) == 0).float().mean().item())
            results["one"].append((tags.sum(dim=1) == 1).float().mean().item())
            results["several"].append((tags.sum(dim=1) >= 2).float().mean().item())

            for kind in ("softmax", "sigmoid"):
                torch.manual_seed(seed)
                model = nn.Sequential(nn.Linear(FEATURES, 32), nn.ReLU(), nn.Linear(32, TAGS))
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

                for _ in range(1500):
                    batch_x, batch_tags = tag_data(128, generator)
                    scores = model(batch_x)

                    if kind == "softmax":
                        # The tags spread evenly over the softmax, the best one class head can learn from a set.
                        share = batch_tags / batch_tags.sum(dim=1, keepdim=True).clamp(min=1)
                        loss = -(share * F.log_softmax(scores, dim=1)).sum(dim=1).mean()
                    else:
                        loss = F.binary_cross_entropy_with_logits(scores, batch_tags)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    scores = model(x)

                if kind == "softmax":
                    guess = F.one_hot(scores.argmax(dim=1), TAGS).float()
                else:
                    guess = (torch.sigmoid(scores) > 0.5).float()

                results[kind].append((guess == tags).all(dim=1).float().mean().item())

    return results


def measure_mixture(seeds: range = range(3)) -> dict[str, list[float]]:
    """out2's table: answers within 0.15 of a branch, and above 0, for the plain head and the sampled mixture."""
    results: dict[str, list[float]] = {"plain right": [], "plain up": [], "mixture right": [], "mixture up": []}

    with pinned_threads():
        for seed in seeds:
            x, _ = branch_data(2000, torch.Generator().manual_seed(seed + 50))
            branch = 0.3 + x.squeeze(1)

            def right(guess: Tensor) -> float:
                return (((guess - branch).abs() < 0.15) | ((guess + branch).abs() < 0.15)).float().mean().item()

            for kind in ("plain", "mixture"):
                generator = torch.Generator().manual_seed(seed)
                torch.manual_seed(seed)
                model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1 if kind == "plain" else 6))
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

                for _ in range(2000):
                    batch_x, batch_y = branch_data(128, generator)
                    out = model(batch_x)

                    if kind == "plain":
                        loss = F.mse_loss(out.squeeze(1), batch_y)
                    else:
                        loss = OUT2_REFERENCE["mixture_nll"](*OUT2_REFERENCE["split_mixture"](out, 2), batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    out = model(x)

                if kind == "plain":
                    guess = out.squeeze(1)
                else:
                    guess = OUT2_REFERENCE["sample_mixture"](*OUT2_REFERENCE["split_mixture"](out, 2), torch.Generator().manual_seed(seed))

                results[f"{kind} right"].append(right(guess))
                results[f"{kind} up"].append((guess > 0).float().mean().item())

    return results


class CharModel(nn.Module):
    """out3's writer, a GRU over the characters so far, with one extra id to start from."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(len(CHARACTERS) + 1, 16)
        self.gru = nn.GRU(16, 64, batch_first=True)
        self.head = nn.Linear(64, len(CHARACTERS))

    def forward(self, ids: Tensor) -> Tensor:
        out, _ = self.gru(self.embed(ids))

        return self.head(out)


def shaped_example(generator: torch.Generator) -> str:
    letters = int(torch.randint(1, 7, (1,), generator=generator))
    name = "".join(string.ascii_lowercase[index] for index in torch.randint(0, 26, (letters,), generator=generator).tolist())
    age = int(torch.randint(1, 100, (1,), generator=generator))

    return json.dumps({"name": name, "age": age}) + END


def shaped_batch(
    generator: torch.Generator,
    count: int = 64,
) -> tuple[Tensor, Tensor]:
    texts = [shaped_example(generator) for _ in range(count)]
    longest = max(len(text) for text in texts)
    x = torch.full((count, longest), len(CHARACTERS))
    y = torch.full((count, longest), -100)

    for row, text in enumerate(texts):
        ids = [CHARACTERS.index(character) for character in text]
        x[row, 1 : len(ids)] = torch.tensor(ids[:-1])
        y[row, : len(ids)] = torch.tensor(ids)

    return x, y


def trained_writer(
    seed: int,
    steps: int,
) -> CharModel:
    torch.manual_seed(seed)
    model = CharModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        x, y = shaped_batch(generator)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def writer_scores(model: CharModel) -> Callable[[str], Tensor]:
    def next_scores(text: str) -> Tensor:
        ids = torch.tensor([[len(CHARACTERS)] + [CHARACTERS.index(character) for character in text]])

        with torch.no_grad():
            return model(ids)[0, -1]

    return next_scores


def free_text(
    next_scores: Callable[[str], Tensor],
    generator: torch.Generator,
    limit: int = 40,
) -> str:
    """out3's comparison, the same loop with no mask."""
    text = ""

    for _ in range(limit):
        character = CHARACTERS[int(torch.multinomial(torch.softmax(next_scores(text), dim=-1), 1, generator=generator))]
        text += character

        if character == END:
            break

    return text


def measure_masking(
    seeds: range = range(3),
    answers: int = 300,
) -> dict[int, list[tuple[float, float]]]:
    """out3's table: the share of valid answers, free and masked, after 100, 300 and 1,000 steps."""
    results: dict[int, list[tuple[float, float]]] = {}

    with pinned_threads():
        for steps in (100, 300, 1000):
            results[steps] = []

            for seed in seeds:
                scores = writer_scores(trained_writer(seed, steps))
                free = [free_text(scores, torch.Generator().manual_seed(seed * 1000 + index)) for index in range(answers)]
                masked = [
                    OUT3_REFERENCE["constrained_text"](scores, torch.Generator().manual_seed(seed * 1000 + index)) for index in range(answers)
                ]
                results[steps].append((OUT3_REFERENCE["valid_share"](free), OUT3_REFERENCE["valid_share"](masked)))

    return results


def measure_actions(seeds: range = range(3)) -> dict[str, list[float]]:
    """out4's first table: steers within 0.08 of a fine one, from a number head and from the likeliest bin token."""
    results: dict[str, list[float]] = {"regression": [], "tokens": []}

    with pinned_threads():
        for seed in seeds:
            x, _ = action_data(2000, torch.Generator().manual_seed(seed + 9))

            for kind in ("regression", "tokens"):
                generator = torch.Generator().manual_seed(seed)
                torch.manual_seed(seed)
                model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1 if kind == "regression" else BINS))
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

                for _ in range(1500):
                    batch_x, steer = action_data(128, generator)

                    if kind == "regression":
                        loss = F.mse_loss(model(batch_x).squeeze(1), steer)
                    else:
                        loss = F.cross_entropy(model(batch_x), OUT4_REFERENCE["to_bins"](steer, -1.0, 1.0, BINS))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    out = model(x)

                guess = out.squeeze(1) if kind == "regression" else OUT4_REFERENCE["from_bins"](out.argmax(dim=1), -1.0, 1.0, BINS)
                results[kind].append((((guess - 0.6).abs() < 0.08) | ((guess + 0.6).abs() < 0.08)).float().mean().item())

    return results


def fitted_codebook(
    pieces: Tensor,
    codes: int,
    generator: torch.Generator,
    rounds: int = 20,
) -> Tensor:
    """k-means, as vq1's codebooks settle: start from random pieces, then move each code to the mean of its pieces."""
    codebook = pieces[torch.randperm(len(pieces), generator=generator)[:codes]].clone()

    for _ in range(rounds):
        nearest = OUT4_REFERENCE["nearest_codes"](pieces, codebook)

        for code in range(codes):
            members = pieces[nearest == code]

            if len(members):
                codebook[code] = members.mean(dim=0)

    return codebook


def forked_series(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """Series of 64 steps, a wave of random speed, phase and size, that rises or falls from step 32 on. The series
    come back with both right futures, the wave rising and the wave falling."""
    t = torch.arange(64).float()
    speed = torch.rand(count, 1, generator=generator) * 0.2 + 0.1
    phase = torch.rand(count, 1, generator=generator) * 2 * math.pi
    size = torch.rand(count, 1, generator=generator) * 0.5 + 0.5
    wave = size * torch.sin(speed * t + phase)
    ramp = 0.04 * (t - 31).clamp(min=0)
    side = torch.randint(0, 2, (count, 1), generator=generator).float() * 2 - 1

    return wave + side * ramp, wave + ramp, wave - ramp


class IdWriter(nn.Module):
    """out3's writer over codebook ids in place of characters, with one extra id to start from."""

    def __init__(self, codes: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(codes + 1, 32)
        self.gru = nn.GRU(32, 128, batch_first=True)
        self.head = nn.Linear(128, codes)

    def forward(self, ids: Tensor) -> Tensor:
        out, _ = self.gru(self.embed(ids))

        return self.head(out)


def series_ids(
    series: Tensor,
    codebook: Tensor,
) -> Tensor:
    """Each series of a batch written as ids, one per piece, through signal_ids."""
    return OUT4_REFERENCE["signal_ids"](series.flatten(), codebook).view(len(series), -1)


def measure_forks(
    seeds: range = range(3),
    steps: int = 6000,
    codes: int = 128,
) -> dict[str, list[float]]:
    """out4's last table: the share of futures nearer one of the two right futures than to their average, written by
    a head of 32 numbers trained on the squared error and by an IdWriter sampling 4 codebook ids, and the share of the
    sampled futures that rise."""
    results: dict[str, list[float]] = {"regression": [], "ids": [], "ids rising": []}

    with pinned_threads():
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)
            series, _, _ = forked_series(4000, generator)
            codebook = fitted_codebook(series.reshape(-1, 8), codes, generator)
            torch.manual_seed(seed)
            writer = IdWriter(codes)
            regression = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 32))

            for model in (writer, regression):
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

                for _ in range(steps):
                    batch, _, _ = forked_series(64, generator)

                    if model is writer:
                        ids = series_ids(batch, codebook)
                        x = torch.cat([torch.full((64, 1), codes), ids[:, :-1]], dim=1)
                        loss = F.cross_entropy(writer(x).flatten(0, 1), ids.flatten())
                    else:
                        loss = F.mse_loss(regression(batch[:, :32]), batch[:, 32:])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            tests, rising, falling = forked_series(1000, torch.Generator().manual_seed(seed + 77))
            sampler = torch.Generator().manual_seed(seed)

            with torch.no_grad():
                ids = torch.cat([torch.full((1000, 1), codes), series_ids(tests[:, :32], codebook)], dim=1)

                for _ in range(4):
                    ids = torch.cat([ids, torch.multinomial(torch.softmax(writer(ids)[:, -1], dim=-1), 1, generator=sampler)], dim=1)

                written = OUT4_REFERENCE["write_signal"](ids[:, 5:].flatten(), codebook).view(1000, -1)
                averaged = regression(tests[:, :32])

            def distance(guess: Tensor, future: Tensor) -> Tensor:
                return ((guess - future[:, 32:]) ** 2).mean(dim=1)

            middle = (rising + falling) / 2

            for name, guess in (("regression", averaged), ("ids", written)):
                nearest = torch.minimum(distance(guess, rising), distance(guess, falling))
                results[name].append((nearest < distance(guess, middle)).float().mean().item())

            results["ids rising"].append((distance(written, rising) < distance(written, falling)).float().mean().item())

    return results


def measure_codebooks(
    seeds: range = range(3),
    piece: int = 8,
) -> dict[int, list[float]]:
    """out4's second table: the share of a signal's energy left as error after writing it through codebooks of 8, 32 and 128."""
    results: dict[int, list[float]] = {}

    with pinned_threads():
        for codes in (8, 32, 128):
            results[codes] = []

            for seed in seeds:
                generator = torch.Generator().manual_seed(seed)
                training = sine_signals(200, generator).view(-1, piece)
                codebook = fitted_codebook(training, codes, generator)
                tests = sine_signals(50, torch.Generator().manual_seed(seed + 77))
                written = torch.stack(
                    [OUT4_REFERENCE["write_signal"](OUT4_REFERENCE["signal_ids"](signal, codebook), codebook) for signal in tests],
                )
                results[codes].append(((written - tests) ** 2).sum().item() / (tests**2).sum().item())

    return results


# ---------------------------------------------------------------------------
# quiz16
# ---------------------------------------------------------------------------

QUIZ16_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines moves
    from Section 13. If a step is unclear, the exercise in the table shows it.

    | Function | Moves | Shown in |
    |---|---|---|
    | `row_tokens` | each number times its column's weight plus its bias, then each category looked up in its column's table | table1 |
    | `normalized_patches` | each series against its own mean and spread, padded at the end, then cut into patches | series1, series2 |
    | `residual_ids` | each codebook's nearest entry to what the levels before it left over | sound2 |
    | `clip_loss` | CLIP's loss both ways, and the share of rows whose own partner scores highest | clip1, clip2 |
    | `gated_cross` | cross attention through a tanh gate that starts closed, skipped where the memory is missing | fuse2, fuse3 |
    | `prefix_inputs` | the projected features in front of the text's embeddings, then positions over the whole row | bridge1 |
    | `action_values` | scores masked to the action ids, the likeliest taken, then read back as its bin's middle | out3, out4 |
    """,
)

QUIZ16_SOURCE = source_file(
    '''
    # quiz16.py
    #
    # No examples and no new tools. Each function combines moves from Section 13.

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Return each row's tokens, of shape (rows, columns of numbers plus
    #   columns of categories, size). A number's token is the number times its
    #   column's row of weight, plus its column's row of bias. A category's
    #   token is its id's row in its column's own table in tables. The number
    #   tokens come first.
    def row_tokens(
        numbers: Tensor,
        weight: Tensor,
        bias: Tensor,
        categories: Tensor,
        tables: list[Tensor],
    ) -> Tensor:
        ...


    # TODO: Normalize each row of series, less its own mean, over its `.std()`
    #   plus 1e-5. Pad the row's end with `stride` copies of its last value, then
    #   cut it into patches of `length` steps, a new one every `stride` steps.
    #   Return the patches, of shape (rows, patches, length).
    def normalized_patches(
        series: Tensor,
        length: int,
        stride: int,
    ) -> Tensor:
        ...


    # TODO: Return the ids residual quantization gives each row of z, of shape
    #   (rows, levels). codebooks has shape (levels, entries, size). Each level
    #   picks its codebook's entry nearest to what is left of z, by squared
    #   distance, and takes that entry off what is left.
    def residual_ids(
        z: Tensor,
        codebooks: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return CLIP's loss for a batch of pairs, where row i of a goes with
    #   row i of b, and the share of rows of a whose best scoring row of b is
    #   their own partner, as a plain number. Score every row of a against
    #   every row of b by cosine similarity over temperature. The loss averages
    #   the cross entropy that picks each row's partner from a to b and from b
    #   to a.
    def clip_loss(
        a: Tensor,
        b: Tensor,
        temperature: float,
    ) -> tuple[Tensor, float]:
        ...


    # TODO: Return x after gated cross attention, skipping the rows with no
    #   memory. cross is fuse2's MemoryAttention, called as cross(x, memory).
    #   Scale its output by the tanh of gate. Add that to x in the rows where
    #   has_memory is True, and leave x as it is where it is False.
    def gated_cross(
        x: Tensor,
        memory: Tensor,
        has_memory: Tensor,
        cross: nn.Module,
        gate: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return what a GPT reads for features from another modality followed
    #   by text ids. Run features through projector and put the result in
    #   front of wte's embeddings of ids. Then add wpe's embedding of every
    #   place in the joined row.
    def prefix_inputs(
        features: Tensor,
        projector: nn.Module,
        ids: Tensor,
        wte: nn.Embedding,
        wpe: nn.Embedding,
    ) -> Tensor:
        ...


    # TODO: Return the value a model's scores pick, one per row, as RT-2 reads
    #   an action. Only ids `first` to `first + bins - 1` stand for actions, so
    #   set every other id's score to minus infinity. Take each row's likeliest
    #   id, and return the middle of its bin, with the range from low to high
    #   cut into `bins` equal bins and id `first` the lowest bin.
    def action_values(
        scores: Tensor,
        low: float,
        high: float,
        bins: int,
        first: int,
    ) -> Tensor:
        ...
    ''',
)

QUIZ16_PARTS = '''
def row_tokens(
    numbers: Tensor,
    weight: Tensor,
    bias: Tensor,
    categories: Tensor,
    tables: list[Tensor],
) -> Tensor:
    number_part = numbers.unsqueeze(-1) * weight + bias
    category_part = torch.stack([table[categories[:, column]] for column, table in enumerate(tables)], dim=1)

    return torch.cat([number_part, category_part], dim=1)


def normalized_patches(
    series: Tensor,
    length: int,
    stride: int,
) -> Tensor:
    mean = series.mean(dim=1, keepdim=True)
    spread = series.std(dim=1, keepdim=True) + 1e-5
    normal = (series - mean) / spread
    padded = torch.cat([normal, normal[:, -1:].expand(-1, stride)], dim=1)

    return padded.unfold(1, length, stride)


def residual_ids(
    z: Tensor,
    codebooks: Tensor,
) -> Tensor:
    left = z
    ids = []

    for codebook in codebooks:
        chosen = ((left.unsqueeze(1) - codebook.unsqueeze(0)) ** 2).sum(dim=2).argmin(dim=1)
        ids.append(chosen)
        left = left - codebook[chosen]

    return torch.stack(ids, dim=1)


def clip_loss(
    a: Tensor,
    b: Tensor,
    temperature: float,
) -> tuple[Tensor, float]:
    scores = F.normalize(a, dim=1) @ F.normalize(b, dim=1).T / temperature
    targets = torch.arange(len(scores))
    loss = (F.cross_entropy(scores, targets) + F.cross_entropy(scores.T, targets)) / 2

    return loss, (scores.argmax(dim=1) == targets).float().mean().item()


def gated_cross(
    x: Tensor,
    memory: Tensor,
    has_memory: Tensor,
    cross: nn.Module,
    gate: Tensor,
) -> Tensor:
    attended = torch.tanh(gate) * cross(x, memory)

    return x + torch.where(has_memory.view(-1, 1, 1), attended, torch.zeros_like(attended))


def prefix_inputs(
    features: Tensor,
    projector: nn.Module,
    ids: Tensor,
    wte: nn.Embedding,
    wpe: nn.Embedding,
) -> Tensor:
    joined = torch.cat([projector(features), wte(ids)], dim=1)

    return joined + wpe(torch.arange(joined.shape[1]))


def action_values(
    scores: Tensor,
    low: float,
    high: float,
    bins: int,
    first: int,
) -> Tensor:
    is_action = torch.zeros(scores.shape[-1], dtype=torch.bool)
    is_action[first : first + bins] = True
    picked = scores.masked_fill(~is_action, float("-inf")).argmax(dim=-1)

    return low + (picked - first + 0.5) * (high - low) / bins
'''

QUIZ16_SOLUTION = source_file(
    '''
    # quiz16.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


''' + indented(QUIZ16_PARTS) + '''
    ''',
)

QUIZ16_REFERENCE = namespace_of(QUIZ16_PARTS)


def random_input(
    seed: int,
    *shape: int,
) -> Described:
    """Random numbers have no ties, so a nearest entry is the same however the distances are worked out."""
    return Described(
        value=torch.randn(shape, generator=torch.Generator().manual_seed(seed)),
        text=f"a random tensor of shape {shape}",
    )


def seeded_input(
    seed: int,
    build: Callable[[], nn.Module],
    text: str,
) -> Described:
    torch.manual_seed(seed)

    return Described(
        value=build(),
        text=text,
    )


def category_tables(
    seed: int,
    size: int,
    *choices: int,
) -> Described:
    tables = [small_tensor(seed + column, count, size) for column, count in enumerate(choices)]

    return Described(
        value=tables,
        text="[" + ", ".join(value_text(table) for table in tables) + "]",
    )


def expect_token_ids(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    if isinstance(actual, Tensor) and actual.dtype != torch.long:
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but token ids are whole numbers, torch.long, as an nn.Embedding wants them.")

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def partner_pair(
    seed: int,
    rows: int,
    size: int,
) -> tuple[Described, Described]:
    """Rows of b near their partners in a, so some partners score highest and some don't."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(rows, size, generator=generator)
    b = a + torch.randn(rows, size, generator=generator)

    return Described(a, f"a random tensor of shape {(rows, size)}"), Described(b, f"a plus noise, of shape {(rows, size)}")


def expect_loss_and_share(
    call: str,
    actual: object,
    expected: tuple[Tensor, float],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a pair, the loss and the share of rows whose partner scores highest.")

    expect_tensor(
        call=f"the loss from {call}",
        actual=actual[0],
        expected=expected[0],
    )
    expect_close(
        call=f"the share from {call}",
        actual=actual[1],
        expected=expected[1],
    )


def action_scores(
    seed: int,
    rows: int,
    vocabulary: int,
    planted: dict[int, int],
) -> Described:
    """Random scores with the highest of some rows planted outside the action ids, so a pick with no mask goes wrong."""
    scores = torch.randn(rows, vocabulary, generator=torch.Generator().manual_seed(seed))

    for row, place in planted.items():
        scores[row, place] = 6.0

    return Described(scores, f"scores of shape {(rows, vocabulary)}, some rows highest outside the action ids")


def check_quiz16(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="row_tokens",
        description="gives each column its own weight and bias, or its own table",
        inputs=(
            (small_tensor(1600, 3, 2), small_tensor(1601, 2, 4), small_tensor(1602, 2, 4), torch.tensor([[0, 2], [1, 0], [2, 1]]), category_tables(1603, 4, 3, 3)),
            (small_tensor(1606, 2, 1), small_tensor(1607, 1, 3), small_tensor(1608, 1, 3), torch.tensor([[1, 3, 0], [0, 2, 2]]), category_tables(1609, 3, 2, 4, 3)),
        ),
        reference=QUIZ16_REFERENCE["row_tokens"],
    )
    check_on_inputs(
        checker=checker,
        function="normalized_patches",
        description="measures each row against itself, pads its end and cuts it into patches",
        inputs=(
            (small_tensor(1610, 2, 10), 4, 2),
            (small_tensor(1611, 3, 13), 6, 3),
        ),
        reference=QUIZ16_REFERENCE["normalized_patches"],
    )
    check_on_inputs(
        checker=checker,
        function="residual_ids",
        description="matches each level against what the levels before it left",
        inputs=(
            (random_input(1620, 4, 2), random_input(1621, 3, 5, 2)),
            (random_input(1622, 6, 3), random_input(1623, 2, 4, 3)),
        ),
        reference=QUIZ16_REFERENCE["residual_ids"],
        expect=expect_token_ids,
    )
    check_on_inputs(
        checker=checker,
        function="clip_loss",
        description="averages the cross entropy both ways, and finds each row's partner",
        inputs=(
            (*partner_pair(1630, 6, 3), 0.07),
            (*partner_pair(1632, 8, 5), 0.5),
        ),
        reference=QUIZ16_REFERENCE["clip_loss"],
        expect=expect_loss_and_share,
    )
    check_on_inputs(
        checker=checker,
        function="gated_cross",
        description="adds the gated cross attention to x where the memory is there",
        inputs=tuple(
            (
                random_input(seed, 3, 3, 8),
                random_input(seed + 1, 3, 5, 6),
                torch.tensor([True, False, True]),
                seeded_input(seed + 2, lambda: FUSE2_REFERENCE["MemoryAttention"](8, 6, 2), "fuse2's MemoryAttention(8, 6, 2)"),
                torch.tensor([gate]),
            )
            for seed, gate in ((1640, 0.0), (1643, 0.9), (1646, -0.4))
        ),
        reference=QUIZ16_REFERENCE["gated_cross"],
    )
    check_on_inputs(
        checker=checker,
        function="prefix_inputs",
        description="puts the projected features first and gives every place its position",
        inputs=(
            (
                random_input(1650, 2, 3, 5),
                seeded_input(1651, lambda: nn.Sequential(nn.Linear(5, 6), nn.GELU(), nn.Linear(6, 6)), "nn.Sequential(nn.Linear(5, 6), nn.GELU(), nn.Linear(6, 6))"),
                torch.tensor([[1, 4], [0, 2]]),
                seeded_input(1652, lambda: nn.Embedding(7, 6), "nn.Embedding(7, 6)"),
                seeded_input(1653, lambda: nn.Embedding(8, 6), "nn.Embedding(8, 6)"),
            ),
            (
                random_input(1654, 1, 2, 4),
                seeded_input(1655, lambda: nn.Linear(4, 3), "nn.Linear(4, 3)"),
                torch.tensor([[2, 0, 1]]),
                seeded_input(1656, lambda: nn.Embedding(3, 3), "nn.Embedding(3, 3)"),
                seeded_input(1657, lambda: nn.Embedding(6, 3), "nn.Embedding(6, 3)"),
            ),
        ),
        reference=QUIZ16_REFERENCE["prefix_inputs"],
    )
    check_on_inputs(
        checker=checker,
        function="action_values",
        description="masks the scores to the action ids, takes the likeliest and reads its bin's middle",
        inputs=(
            (action_scores(1660, 5, 12, {0: 0, 1: 10, 2: 3, 3: 11}), -1.0, 1.0, 6, 4),
            (action_scores(1661, 3, 9, {0: 4, 1: 8}), 0.0, 10.0, 4, 0),
        ),
        reference=QUIZ16_REFERENCE["action_values"],
    )


QUIZ16 = Exercise(
    name="quiz16",
    chapter=QUIZZES,
    title="Modalities, start to finish",
    lesson=QUIZ16_LESSON,
    source=QUIZ16_SOURCE,
    solution=QUIZ16_SOLUTION,
    hints={
        "row_tokens": "numbers.unsqueeze(-1) gives each number a last axis of size 1, which broadcasts against weight. "
        "Column j's category tokens are tables[j] indexed by categories[:, j]. Stack those on dim 1, then join both parts.",
        "normalized_patches": "mean and std over dim 1 with keepdim=True keep one number per row. Repeat the "
        "normalized row's last column with .expand, join it on, and .unfold on dim 1 cuts the patches.",
        "residual_ids": "Keep what is left of z, starting with z. For each codebook, compare it with every entry "
        "through unsqueeze, take the argmin of the summed squares, and take the chosen entries off.",
        "clip_loss": "F.normalize with dim=1 makes every row length 1, so a matrix multiply gives the cosines. Row "
        "i's right class is i, from torch.arange. The other direction scores the transpose. For the share, compare "
        "each row's argmax with those classes, as floats, and take the mean with .item().",
        "gated_cross": "cross(x, memory) gives an output shaped like x. torch.where lines up with x once has_memory has "
        "shape (rows, 1, 1), and torch.zeros_like gives the rows that add nothing.",
        "prefix_inputs": "torch.cat on dim 1 joins the prefix and the text. The positions run over the joined length, "
        "so look up torch.arange of that length in wpe.",
        "action_values": "A mask of False over the scores' last dimension, True from first up to first + bins, marks "
        "the action ids. masked_fill the rest with float(\"-inf\") and take argmax over the last dimension. Take "
        "first off for the bin, then read its middle, each bin (high - low) / bins wide.",
    },
    introduces=(),
    check=check_quiz16,
)
