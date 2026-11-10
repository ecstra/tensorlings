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

        A model's last layer decides what kind of answer it can give. Each
        kind of answer needs its own head, or its own way of writing tokens.
        The kinds here are one class, several tags, a number, a choice between
        two right answers, a valid JSON object, a robot's next move, and a
        signal. This chapter builds a head for each one and measures the
        mistakes a mismatched head makes.

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
    activations2 matched a model's output layer to the form of the answer.
    Often a model has to give several answers at once. Then one shared
    network, called the trunk, reads the input. A small head for each answer
    reads the trunk's output.

    | Answer | Head gives | Loss |
    |---|---|---|
    | one class of several | a score per class | `F.cross_entropy`, which applies a softmax inside |
    | any number of tags | a score per tag | `F.binary_cross_entropy_with_logits`, a sigmoid per tag |
    | several numbers | one number each | `F.mse_loss` |

    The probabilities from a softmax add up to 1, so the softmax favors one
    answer over the others. A separate sigmoid per tag gives each tag its own
    probability, independent of the others. So a document can be tagged both
    news and sport. A tag counts as present when its sigmoid is above 0.5.

    Here each input is 8 numbers, and there are 5 possible tags. Each tag is
    present when its own weighted sum of the input is above 0.4. Of the
    inputs, 67% to 69% had two or more tags, and 4.5% to 5.5% had none. Two
    heads trained for 1,500 steps on three seeds. Each was scored on the
    share of inputs whose whole set of tags came out right.

    | Head | Whole set right |
    |---|---|
    | softmax, its top class taken as the one tag | 25% to 27% |
    | a sigmoid per tag | 95% to 96% |

    The softmax head names exactly one tag, so it fails on every input with
    no tags or with more than one. It got every input with exactly one tag
    right, which is the best it could do. When a model has several heads,
    their losses are added into one loss. Then one backward pass trains the
    trunk for all the heads at once.
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


    # TODO: Finish forward. Run x through trunk once, then run each of the
    #   three heads on the trunk's output. Return the class scores, the tag
    #   scores and the values, as a tuple in that order.
''' + indented(HEADS_INIT) + '''

        def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            ...


    # TODO: Return one loss for all three heads, as a tensor. outputs is the
    #   tuple Heads returns, (class scores, tag scores, values). Add three
    #   losses. The first is the cross entropy of the class scores against
    #   classes. The second is the binary cross entropy of the raw tag scores
    #   against tags, computed with F.binary_cross_entropy_with_logits. The
    #   third is the mean squared error of the predicted values against
    #   values.
    def heads_loss(
        outputs: tuple[Tensor, Tensor, Tensor],
        classes: Tensor,
        tags: Tensor,
        values: Tensor,
    ) -> Tensor:
        ...


    # TODO: Turn outputs into answers, and return three tensors. The first is
    #   each row's class, the index of its highest class score. The second is
    #   the tags, True where the tag's sigmoid is above 0.5. The third is the
    #   values, unchanged.
    #   Example: for class scores [[0.1, 2.0, -1.0]] and tag scores
    #   [[2.0, -1.0]], the class is tensor([1]) and the tags are
    #   tensor([[ True, False]]).
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
        raise Mismatch(f"The classes from {call} hold {actual[0].dtype} numbers, but they should be whole numbers, the indices from argmax.")

    if actual[1].dtype != torch.bool:
        raise Mismatch(f"The tags from {call} hold {actual[1].dtype} numbers, but they should be booleans, True and False. Return the comparison result without converting it.")


def check_out1(checker: Checker) -> None:
    checker.test_method(
        class_name="Heads",
        method="forward",
        description="runs the trunk once and each head on its output",
        run=check_heads_forward,
    )
    checker.test(
        function="heads_loss",
        description="adds the cross entropy, the binary cross entropy and the mean squared error",
        run=check_heads_loss,
    )
    check_on_inputs(
        checker=checker,
        function="read_heads",
        description="reads one class, the tags whose sigmoid is above 0.5, and the values",
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
        "Heads.forward": "Store self.trunk(x) in a variable. Then return a tuple of self.class_head, self.tag_head and "
        "self.value_head, each applied to that variable.",
        "heads_loss": "Unpack outputs into three variables. Then add F.cross_entropy, "
        "F.binary_cross_entropy_with_logits and F.mse_loss, each with its own target.",
        "read_heads": "Take argmax over the last dimension for the class. For the tags, compare torch.sigmoid of the "
        "tag scores with 0.5. Return the values unchanged.",
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
    average of the right answers for each input. With two right answers, the
    average can be neither of them. For the arm, it would put the elbow
    halfway between up and down.

    His **mixture density network** outputs a whole probability distribution
    instead. For each input, the head outputs K normal distributions, called
    components. Each component has a weight, a mean and a standard deviation.
    The code calls the standard deviation the spread.

    The weights go through a softmax. The head outputs the log of each
    spread, `log_spreads`, and the spread is `exp` of that. This keeps every
    spread positive.

    The density of the mixture at y is the weighted sum of its components'
    densities. The loss is the negative log of that density, as in
    likelihood3. To add probabilities that are stored as logs, use
    logsumexp, as in likelihood2.

    ```text
    loss = -logsumexp(log_softmax(weights) + log N(y; mean, spread))
    ```

    To produce an answer, pick one component at random, with probabilities
    given by the softmax of the weights. Then draw a sample from that
    component's normal distribution.

    Here y lies on one of two branches, `0.3 + x` or `-(0.3 + x)`, for x from
    0 to 1. Each head trained for 2,000 steps on three seeds. An answer counts
    as right if it is within 0.15 of a branch.

    | Head | Answers right | Answers above 0 |
    |---|---|---|
    | one number, squared error | 0% | 0% or 100% |
    | a mixture of 2 normals, sampled | 99.4% to 99.9% | 47% to 53% |

    The plain head's answers end up near 0, between the branches, so they are
    never right. In each run, all of its answers were on the same side of 0.
    The mixture puts one component on each branch and picks each one about
    half the time.
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


    # TODO: Split the head's output out into three parts along the last
    #   dimension, each of size components. Return them in order as the
    #   weights, the means and the log spreads.
    #   Example: for out = torch.arange(6.0) and components = 2, return
    #   (tensor([0., 1.]), tensor([2., 3.]), tensor([4., 5.])).
    def split_mixture(
        out: Tensor,
        components: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ...


    # TODO: Return the negative log likelihood of y under each row's mixture,
    #   averaged over the rows, as a tensor. weights, means and log_spreads
    #   have shape (batch, K), and y has shape (batch,). For one component,
    #   with spread = exp(log_spread), the log density of its normal at y is
    #   -0.5 * ((y - mean) / spread) ** 2 - log_spread - 0.5 * log(2 * pi).
    #   Add the log softmax of the weights to these log densities, take
    #   logsumexp over the K components, negate, and average over the rows.
    #   Example: for one row of two components with weights 0, means 0 and
    #   log spreads 0, and y = 0, the result is tensor(0.9189).
    def mixture_nll(
        weights: Tensor,
        means: Tensor,
        log_spreads: Tensor,
        y: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return one sample per row, a tensor of shape (batch,). First pick
    #   one component per row with torch.multinomial and the generator, using
    #   the softmax of the weights as the probabilities. Then return that
    #   component's mean plus its spread, exp(log_spread), times a standard
    #   normal draw from torch.randn with the generator.
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
        description="splits the head's output into weights, means and log spreads",
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
        description="returns the negative log likelihood under each row's mixture, averaged over the rows",
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
        description="picks a component by its weight, then draws a sample from it",
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
        "mixture_nll": "Give y a last dimension with .unsqueeze(-1), so it lines up with every component. "
        "log_spreads.exp() is the spread. Add F.log_softmax of the weights to the log densities, then take "
        "torch.logsumexp over dim=-1. Negate that and take the mean.",
        "sample_mixture": "torch.multinomial(torch.softmax(weights, dim=-1), 1, generator=generator) picks a column per row. "
        "Then .gather(-1, picked) reads that column of the means and of the spreads. Last, draw one torch.randn of "
        "that shape with the generator.",
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

# Every answer must have the form {"name": "<1 to 6 letters>", "age": <1 to 2 digits>} followed by a newline.
PARTS = (
    ("literal", '{"name": "'),
    ("letters", 1, 6),
    ("literal", '", "age": '),
    ("digits", 1, 2),
    ("literal", "}\\n"),
)


def allowed(state: tuple[int, int]) -> str:
    """The characters allowed next. state is the index of the current part and how many of its characters are written."""
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
    A program that reads a model's answer needs the answer in a fixed format,
    such as JSON. A model that writes one character at a time can make a
    mistake, and one wrong character makes the whole answer unreadable.

    At each point, the format says which characters are allowed next. After
    `{"name": "` only a letter is allowed. After the age's digits, only
    another digit or `}` is allowed. Before picking each character, set the
    score of every character that is not allowed to minus infinity, as
    gpt4's `top_k_filter` did. The softmax then gives those characters a
    probability of exactly 0.

    ```python
    torch.tensor([2.0, 1.0, 3.0]).masked_fill(torch.tensor([False, True, True]), float("-inf"))
    # tensor([2., -inf, -inf])
    ```

    This is called **constrained decoding**. In this exercise, the given
    functions `allowed` and `step` track one fixed format. Willard and Louf's
    Outlines library builds the same sets of allowed characters from a
    regular expression or a grammar. It tracks them with a finite state
    machine, adding a stack for a grammar, so each token costs little extra.
    RT-2 restricts its output in the same way, to its action tokens.

    Here a small GRU learned the format from examples like
    `{"name": "ada", "age": 36}`. It trained for 100, 300 or 1,000 steps, on
    three seeds. Each trained model wrote 300 answers without the mask and
    300 with it. The table shows the share of valid answers.

    | Training | Valid without mask | Valid with mask |
    |---|---|---|
    | 100 steps | 15% to 45% | 100% |
    | 300 steps | 91% to 94% | 100% |
    | 1,000 steps | 98% to 99.7% | 100% |

    The mask guarantees the format at any stage of training, so the program
    reading the answers never fails. The content inside the format, the name
    and the age, still comes from the model.
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


    # TODO: scores holds one score per character of CHARACTERS. Return new
    #   scores where every character not in allowed_characters has its score
    #   set to minus infinity. The scores of allowed characters stay as they
    #   are.
    #   Example: mask_scores(torch.zeros(len(CHARACTERS)), "{") returns 0 at
    #   index 0, the index of "{", and minus infinity everywhere else.
    def mask_scores(
        scores: Tensor,
        allowed_characters: str,
    ) -> Tensor:
        ...


    # TODO: Write one answer a character at a time and return it as a string.
    #   Start with an empty text and the state (0, 0). At each step, mask the
    #   scores next_scores(text) so only the characters in allowed(state)
    #   remain. Pick one character with torch.multinomial on the softmax of
    #   the masked scores and the generator, and append it to the text. If
    #   the character is END, return the text. Otherwise update the state
    #   with step. After limit characters, return the text as it is.
    def constrained_text(
        next_scores: Callable[[str], Tensor],
        generator: torch.Generator,
        limit: int = 40,
    ) -> str:
        ...


    # TODO: Return the fraction of texts that is_valid accepts, as a Python
    #   float. Example: for one valid text and one invalid text, return 0.5.
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
                raise Mismatch(f"{call} wrote {actual!r}, which does not match the format. Mask the scores before every pick, and update the state with step after each character.")

            if actual != expected:
                raise Mismatch(f"{call} wrote {actual!r}, but it should write {expected!r}. Draw exactly once per character, after applying the mask.")


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
        description="returns the fraction of valid answers",
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
        "mask_scores": "Make a boolean mask of False values shaped like scores, with torch.zeros_like and "
        "dtype=torch.bool. Set it to True at CHARACTERS.index of each allowed character. Then use masked_fill where "
        "the mask is False.",
        "constrained_text": "Use a for loop over range(limit). Inside, call mask_scores on next_scores(text) and "
        "allowed(state). Then call torch.multinomial on the softmax with 1 draw and the generator. The character is "
        "CHARACTERS at the index it returns.",
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
    A language model writes tokens. To make a language model control a
    robot, RT-2 wrote the robot's actions as tokens too. Each action is 8
    numbers. One is a flag that says whether to stop, and 7 say how the arm
    moves and grips. The range of each of those 7 continuous numbers is cut
    into 256 equal bins, and each bin is a token. PaLI-X already had a token
    for each whole number up to 1,000. PaLM-E had no such tokens, so RT-2
    used its 256 least used tokens for the bins.

    ```text
    bin = floor((value - low) / (high - low) * bins), clamped to 0 to bins - 1
    value back = low + (bin + 0.5) * (high - low) / bins
    ```

    Converting a bin back to a value gives the middle of the bin. That is at
    most half a bin width from the true value. With 256 bins from -1 to 1,
    half a bin width is 0.0039.

    Tokens also handle out2's problem of two right answers. Here a cart
    steers 0.6 left or right around an obstacle, and either way is fine. Two
    heads trained for 1,500 steps on three seeds. One outputs a single
    number, and the other outputs scores for 256 bin tokens. A steer counts
    as fine if it is within 0.08 of a fine steer.

    | Head | Steers fine |
    |---|---|
    | one number, squared error | 0% |
    | 256 bin tokens, most likely bin taken | 100% |

    The number head outputs the average and steers straight at the obstacle.
    The softmax over the bin tokens puts high probability on both sides, and
    the most likely bin is always on one of them.

    Any signal can also be written as tokens, through a codebook like the
    one in vq1. Cut the signal into pieces, and write each piece as the id of
    its nearest code. To read the signal back, replace each id with its code.
    Here the codebooks were fitted to sums of sines by k-means. K-means
    repeatedly moves each code to the mean of the pieces nearest to it. The
    error left is the fraction of the signal's sum of squares that the codes
    miss.

    | Codes | Error left |
    |---|---|
    | 8 | 9.5% to 10.7% |
    | 32 | 3.1% to 3.3% |
    | 128 | 1.0% |

    Then a model can write a signal the same way it writes text. Here a wave
    runs for 32 steps, then either rises or falls, and both are right. out3's
    GRU learned such series as ids, with 8 steps per piece and 128 codes. It
    wrote each future, the next 32 steps, by sampling 4 ids. For comparison,
    a head of 32 numbers learned the future with the squared error. Each
    trained for 6,000 steps on three seeds.

    | Head | Futures closer to a right future than to the average of the two |
    |---|---|
    | 32 numbers, squared error | 0% |
    | 4 codebook ids, sampled | 99.7% to 99.8% |

    The number head writes a future close to the average of the two, a wave
    that neither rises nor falls. The sampled ids pick one side, rising in
    50% to 54% of series.
    """,
)

OUT4_SOURCE = source_file(
    '''
    # out4.py

    import torch
    from torch import Tensor

    torch.tensor([-1.0, 0.3, 1.0]).floor().long().clamp(0, 1)    # tensor([0, 0, 1])
    torch.tensor([[1.0, 2.0], [3.0, 4.0]])[torch.tensor([1, 1, 0])].flatten()    # tensor([3., 4., 3., 4., 1., 2.])


    # TODO: The range from low to high is cut into `bins` equal bins. Return
    #   the bin index of each value, a whole number from 0 to bins - 1, with
    #   the formula from the lesson. A value equal to high goes in the last
    #   bin.
    #   Example: to_bins(torch.tensor([-1.0, 0.0, 1.0]), -1.0, 1.0, 4)
    #   returns tensor([0, 2, 3]).
    def to_bins(
        values: Tensor,
        low: float,
        high: float,
        bins: int,
    ) -> Tensor:
        ...


    # TODO: Return the middle value of each bin index in ids, as floats, for
    #   the same bins as to_bins.
    #   Example: from_bins(torch.tensor([0, 3]), -1.0, 1.0, 4) returns
    #   tensor([-0.7500,  0.7500]).
    def from_bins(
        ids: Tensor,
        low: float,
        high: float,
        bins: int,
    ) -> Tensor:
        ...


    # The distance and nearest code functions from vq1.
''' + indented(NEAREST_PART) + '''


    # TODO: Finish both functions. codebook has shape (codes, piece). In
    #   signal_ids, cut the 1D signal into pieces of length piece, and return
    #   the id of the nearest code for each piece. In write_signal, replace
    #   each id with its code, and return the codes joined end to end as one
    #   flat 1D tensor.
    #   Example: with codebook [[0., 0.], [1., 1.]], signal_ids of
    #   tensor([0.9, 1.1, 0.1, -0.1]) returns tensor([1, 0]), and
    #   write_signal of tensor([1, 0]) returns tensor([1., 1., 0., 0.]).
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
        description="returns each value's bin, with values at the edges kept inside 0 to bins - 1",
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
        description="returns the middle of each bin",
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
        description="replaces the ids with their codes, as one flat tensor",
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
        "to_bins": "Scale the values so they run from 0 to bins, then use .floor() and .long() to get whole numbers. "
        "Then .clamp(0, bins - 1) keeps a value at high in the last bin.",
        "from_bins": "Turn ids into floats, add 0.5 for the middle, then multiply by the bin width (high - low) / bins "
        "and add low.",
        "signal_ids": "signal.view(-1, codebook.shape[1]) cuts the signal into pieces as long as a code. Then pass the "
        "pieces and the codebook to nearest_codes.",
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
    This quiz has no examples and no new tools. Each function combines steps
    you learned in Section 13. If a step is unclear, go back to the exercise
    the table names for it.

    | Function | Steps | Shown in |
    |---|---|---|
    | `row_tokens` | each number times its column's weight plus its column's bias, then each category looked up in its column's table | table1 |
    | `normalized_patches` | each series normalized by its own mean and standard deviation, padded at the end, then cut into patches | series1, series2 |
    | `residual_ids` | at each level, the codebook entry nearest to what the earlier levels left over | sound2 |
    | `clip_loss` | CLIP's loss in both directions, and the fraction of rows whose own partner scores highest | clip1, clip2 |
    | `gated_cross` | cross attention scaled by the tanh of a gate that starts at 0, skipped in rows where the memory is missing | fuse2, fuse3 |
    | `prefix_inputs` | the projected features in front of the text's embeddings, then position embeddings over the whole joined row | bridge1 |
    | `action_values` | scores masked to the action ids, the highest scoring id taken, then turned into the middle of its bin | out3, out4 |
    """,
)

QUIZ16_SOURCE = source_file(
    '''
    # quiz16.py
    #
    # No examples and no new tools. Each function combines steps from Section 13.

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Return the tokens of each table row, a tensor of shape
    #   (rows, number columns + category columns, size). numbers has shape
    #   (rows, number columns), and weight and bias have shape
    #   (number columns, size). The token of a number in column j is the
    #   number times row j of weight, plus row j of bias. categories has
    #   shape (rows, category columns), and tables[j] is the embedding table
    #   of category column j. The token of a category in column j is the row
    #   of tables[j] at that category's id. Put the number tokens first.
    def row_tokens(
        numbers: Tensor,
        weight: Tensor,
        bias: Tensor,
        categories: Tensor,
        tables: list[Tensor],
    ) -> Tensor:
        ...


    # TODO: Normalize each row of series by its own statistics, computing
    #   (row - row mean) / (row.std() + 1e-5). Pad the end of each normalized
    #   row with `stride` copies of its last value. Then cut it into patches
    #   of `length` steps, starting a new patch every `stride` steps. Return
    #   the patches, of shape (rows, patches, length).
    def normalized_patches(
        series: Tensor,
        length: int,
        stride: int,
    ) -> Tensor:
        ...


    # TODO: Return the ids that residual quantization gives each row of z, a
    #   tensor of shape (rows, levels). codebooks has shape (levels, entries,
    #   size). Start with the residual equal to z. At each level, pick the
    #   entry of that level's codebook nearest to the residual, by squared
    #   distance, and subtract the chosen entry from the residual.
    def residual_ids(
        z: Tensor,
        codebooks: Tensor,
    ) -> Tensor:
        ...


    # TODO: Row i of a and row i of b form a pair. Return two things. The
    #   first is CLIP's loss, a tensor. The second is the fraction of rows of
    #   a whose highest scoring row of b is their own partner, as a Python
    #   float. The score of row i of a against row j of b is their cosine
    #   similarity divided by temperature. The loss is the average of two
    #   cross entropies. One picks the partner of each row of a among the
    #   rows of b, and the other picks the partner of each row of b among the
    #   rows of a.
    def clip_loss(
        a: Tensor,
        b: Tensor,
        temperature: float,
    ) -> tuple[Tensor, float]:
        ...


    # TODO: Return x after gated cross attention, skipping the rows with no
    #   memory. x has shape (rows, steps, size) and has_memory has shape
    #   (rows,). cross is fuse2's MemoryAttention, called as cross(x, memory).
    #   Scale its output by the tanh of gate. Add that to x in the rows where
    #   has_memory is True, and leave x unchanged where it is False.
    def gated_cross(
        x: Tensor,
        memory: Tensor,
        has_memory: Tensor,
        cross: nn.Module,
        gate: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the input vectors a GPT reads for features from another
    #   modality followed by the text ids. Run features through projector,
    #   and put the result in front of wte(ids) along the positions. Then add
    #   the position embedding from wpe for every position of the joined row.
    def prefix_inputs(
        features: Tensor,
        projector: nn.Module,
        ids: Tensor,
        wte: nn.Embedding,
        wpe: nn.Embedding,
    ) -> Tensor:
        ...


    # TODO: Return the action value the scores pick, one per row, the way
    #   RT-2 reads an action. Only the ids from `first` to `first + bins - 1`
    #   stand for actions, so set every other id's score to minus infinity.
    #   Take each row's highest scoring id. The range from low to high is cut
    #   into `bins` equal bins, and id `first` stands for the lowest bin.
    #   Return the middle value of the chosen id's bin.
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
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a pair, the loss and the fraction of rows whose partner scores highest.")

    expect_tensor(
        call=f"the loss from {call}",
        actual=actual[0],
        expected=expected[0],
    )
    expect_close(
        call=f"the fraction from {call}",
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
        description="normalizes each row by its own mean and standard deviation, pads its end and cuts it into patches",
        inputs=(
            (small_tensor(1610, 2, 10), 4, 2),
            (small_tensor(1611, 3, 13), 6, 3),
        ),
        reference=QUIZ16_REFERENCE["normalized_patches"],
    )
    check_on_inputs(
        checker=checker,
        function="residual_ids",
        description="matches each level against the residual the earlier levels left",
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
        description="averages the cross entropy in both directions, and the fraction of rows matched to their partner",
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
        description="adds the gated cross attention to x where the memory is present",
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
        description="puts the projected features first and adds a position embedding at every position",
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
        description="masks the scores to the action ids, takes the highest and returns the middle of its bin",
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
        "Column j's category tokens are tables[j] indexed by categories[:, j]. Stack those on dim 1, then "
        "concatenate the number tokens and the category tokens on dim 1.",
        "normalized_patches": "Take mean and std over dim 1 with keepdim=True, which keeps one number per row. "
        "Repeat the normalized rows' last column with .expand and concatenate it at the end. Then .unfold on dim 1 "
        "cuts the patches.",
        "residual_ids": "Keep the residual in a variable, starting with z. For each codebook, compare the residual "
        "with every entry by broadcasting with unsqueeze. Take the argmin of the summed squared differences, then "
        "subtract the chosen entries from the residual.",
        "clip_loss": "F.normalize with dim=1 makes every row length 1, so a matrix multiply gives the cosine "
        "similarities. The right class for row i is i, from torch.arange. For the other direction, use the "
        "transpose of the scores. For the fraction, compare each row's argmax with those classes, convert to "
        "floats, and take the mean with .item().",
        "gated_cross": "cross(x, memory) returns an output shaped like x. torch.where broadcasts against x once "
        "has_memory has shape (rows, 1, 1). torch.zeros_like gives the zeros to add in rows with no memory.",
        "prefix_inputs": "torch.cat on dim 1 joins the prefix and the text. The positions run over the joined length, "
        "so look up torch.arange of that length in wpe.",
        "action_values": "Make a boolean mask over the scores' last dimension, True from first up to first + bins "
        "and False elsewhere. Use masked_fill to set the rest to float(\"-inf\"), then take argmax over the last "
        "dimension. Subtract first to get the bin, then compute its middle. Each bin is (high - low) / bins wide.",
    },
    introduces=(),
    check=check_quiz16,
)
