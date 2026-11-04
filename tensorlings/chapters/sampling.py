"""65_sampling, drawing from a categorical distribution by hand, top p beside temperature and top k, and the reparameterization trick that lets gradients through a Gaussian draw."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
    seeded,
)
from ..text import prose, source_file

SAMPLING = Chapter(
    folder="65_sampling",
    title="Sampling",
    readme=source_file(
        """
        # Sampling

        Section 9 is about models that make new data, and every one of them
        draws random numbers. This chapter opens up the draws the course has
        used so far. It picks from a categorical distribution by hand, shapes
        the draw with temperature, top k and top p, and draws from a Gaussian
        in a way that lets gradients through.

        ## Further reading

        - [Holtzman and colleagues, the curious case of neural text degeneration](https://arxiv.org/abs/1904.09751)
        - [Kingma and Welling, auto-encoding variational Bayes](https://arxiv.org/abs/1312.6114)
        - [PyTorch, torch.multinomial](https://docs.pytorch.org/docs/2.14/generated/torch.multinomial.html)
        """,
    ),
)


def reference_draw_by_hand(
    probabilities: Tensor,
    u: Tensor,
) -> Tensor:
    ends = probabilities.cumsum(0)

    return (u.unsqueeze(1) >= ends).sum(dim=1).clamp(max=len(probabilities) - 1)


def reference_draw_many(
    probabilities: Tensor,
    count: int,
    generator: torch.Generator,
) -> Tensor:
    return torch.multinomial(probabilities, count, replacement=True, generator=generator)


def reference_frequencies(
    draws: Tensor,
    choices: int,
) -> Tensor:
    return torch.bincount(draws, minlength=choices).float() / len(draws)


def reference_largest_gap(
    probabilities: Tensor,
    count: int,
    generator: torch.Generator,
) -> float:
    seen = reference_frequencies(reference_draw_many(probabilities, count, generator), len(probabilities))

    return (seen - probabilities).abs().max().item()


def measure_largest_gaps(seed: int = 0) -> dict[int, float]:
    """sampling1's table for one seed, with a fresh generator for each count. The lesson gives seeds 0 to 2. Not run
    by any check."""
    return {
        count: reference_largest_gap(
            probabilities=torch.tensor([0.5, 0.3, 0.2]),
            count=count,
            generator=torch.Generator().manual_seed(seed),
        )
        for count in (100, 10_000, 1_000_000)
    }


# ---------------------------------------------------------------------------
# sampling1
# ---------------------------------------------------------------------------

SAMPLING1_LESSON = prose(
    """
    A model's output is often a probability distribution over choices, like
    the 65 characters of gpt1. Sampling picks one at random, each with its own
    chance. names3 and gpt4 used `torch.multinomial` for it, and this exercise
    opens it up.

    Lay the probabilities end to end from 0 to 1, so each choice owns a
    stretch as long as its probability. Pick a point anywhere from 0 to 1,
    evenly at random. The stretch it lands in is the draw.

    ```python
    probabilities = torch.tensor([0.5, 0.3, 0.2])
    probabilities.cumsum(0)    # tensor([0.5000, 0.8000, 1.0000]), where each stretch ends
    ```

    A point at 0.65 has passed one end, 0.5, so it draws choice 1. The number
    of ends a point has passed is its choice.

    `torch.multinomial(probabilities, count)` draws without replacement by
    default. It never picks the same choice twice, and fails when count is
    more than the choices. Draws that may repeat need `replacement=True`.

    The share of each choice over many draws comes close to its probability.
    Here is the largest gap over the three choices above, on 3 seeds.

    | Draws | 100 | 10,000 | 1,000,000 |
    |---|---|---|---|
    | Largest gap | 0.03 to 0.13 | 0.0030 to 0.0047 | 0.00003 to 0.00091 |

    A hundred times more draws make the gap about ten times smaller. That
    slow shrinking is why a sampled estimate needs many draws to be exact.
    """,
)

SAMPLING1_SOURCE = source_file(
    '''
    # sampling1.py

    import torch
    from torch import Tensor

    ends = torch.tensor([0.5, 0.3, 0.2]).cumsum(0)    # tensor([0.5000, 0.8000, 1.0000])
    passed = (torch.tensor(0.65) >= ends).sum()       # tensor(1), the choice a point at 0.65 draws


    # TODO: Return the choice that each point of u, numbers from 0 to 1, lands
    #   on. Count the stretch ends of probabilities at or below the point. When
    #   rounding leaves the last end below 1, a point past it still draws the
    #   last choice.
    def draw_by_hand(
        probabilities: Tensor,
        u: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return count draws from probabilities that may repeat, with
    #   torch.multinomial and generator.
    def draw_many(
        probabilities: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return, for each choice from 0 to choices - 1, the share of draws
    #   that picked it, as floats, a choice never drawn included.
    def frequencies(
        draws: Tensor,
        choices: int,
    ) -> Tensor:
        ...


    # TODO: Return the largest gap, as a plain number, between probabilities
    #   and the frequencies of count draws from them, using draw_many and
    #   frequencies.
    def largest_gap(
        probabilities: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> float:
        ...
    ''',
)

SAMPLING1_SOLUTION = source_file(
    '''
    # sampling1.py, a solution for comparison

    import torch
    from torch import Tensor

    ends = torch.tensor([0.5, 0.3, 0.2]).cumsum(0)
    passed = (torch.tensor(0.65) >= ends).sum()


    def draw_by_hand(
        probabilities: Tensor,
        u: Tensor,
    ) -> Tensor:
        ends = probabilities.cumsum(0)

        return (u.unsqueeze(1) >= ends).sum(dim=1).clamp(max=len(probabilities) - 1)


    def draw_many(
        probabilities: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        return torch.multinomial(probabilities, count, replacement=True, generator=generator)


    def frequencies(
        draws: Tensor,
        choices: int,
    ) -> Tensor:
        return torch.bincount(draws, minlength=choices).float() / len(draws)


    def largest_gap(
        probabilities: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> float:
        seen = frequencies(draw_many(probabilities, count, generator), len(probabilities))

        return (seen - probabilities).abs().max().item()
    ''',
)


def check_sampling1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="draw_by_hand",
        description="gives the stretch each point lands in",
        inputs=(
            (torch.tensor([0.1, 0.4, 0.2, 0.3]), torch.tensor([0.0, 0.05, 0.1, 0.49, 0.5, 0.71, 0.99])),
            (torch.tensor([0.3, 0.3, 0.4 - 1e-6]), torch.tensor([0.2, 0.6, 0.9999999])),
        ),
        reference=reference_draw_by_hand,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="draw_many",
        description="draws with replacement",
        inputs=(
            (torch.tensor([0.5, 0.3, 0.2]), 12, seeded(650)),
            (torch.tensor([0.9, 0.1]), 5, seeded(651)),
        ),
        reference=reference_draw_many,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="frequencies",
        description="gives each choice's share, unseen ones too",
        inputs=(
            (torch.tensor([0, 2, 2, 1, 2, 0]), 3),
            (torch.tensor([1, 1, 3]), 5),
        ),
        reference=reference_frequencies,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="largest_gap",
        description="gives the largest gap after count draws",
        inputs=(
            (torch.tensor([0.5, 0.3, 0.2]), 100, seeded(652)),
            (torch.tensor([0.25, 0.25, 0.25, 0.25]), 1000, seeded(653)),
            # Its largest gap is a choice drawn too rarely, so a gap taken without its size comes out smaller.
            (torch.tensor([0.5, 0.3, 0.2]), 100, seeded(655)),
        ),
        reference=reference_largest_gap,
        expect=expect_close,
    )


SAMPLING1 = Exercise(
    name="sampling1",
    chapter=SAMPLING,
    title="A draw by hand",
    lesson=SAMPLING1_LESSON,
    source=SAMPLING1_SOURCE,
    solution=SAMPLING1_SOLUTION,
    hints={
        "draw_by_hand": "u.unsqueeze(1) >= ends compares every point with every end. Sum over dim=1, then "
        ".clamp(max=len(probabilities) - 1).",
        "draw_many": "torch.multinomial with replacement=True and generator=generator.",
        "frequencies": "torch.bincount with minlength=choices counts each choice. Turn it into floats and divide by "
        "len(draws).",
        "largest_gap": "frequencies of draw_many's draws, minus probabilities, .abs().max(), then .item().",
    },
    introduces=(),
    check=check_sampling1,
)


# ---------------------------------------------------------------------------
# sampling2
# ---------------------------------------------------------------------------


def reference_top_p_filter(
    logits: Tensor,
    p: float,
) -> Tensor:
    sorted_logits, order = logits.sort(descending=True)
    chances = torch.softmax(sorted_logits, dim=0)
    sorted_keep = chances.cumsum(0) - chances < p
    keep = torch.zeros_like(sorted_keep)
    keep[order] = sorted_keep

    return logits.masked_fill(~keep, float("-inf"))


def reference_kept_count(
    logits: Tensor,
    p: float,
) -> int:
    return int(torch.isfinite(reference_top_p_filter(logits, p)).sum())


def reference_sample_with(
    logits: Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
) -> int:
    scores = logits / temperature
    kth = scores.topk(top_k).values[-1]
    scores = scores.masked_fill(scores < kth, float("-inf"))
    scores = reference_top_p_filter(scores, top_p)

    return torch.multinomial(torch.softmax(scores, dim=0), 1, generator=generator).item()


PEAKED = torch.tensor([6.0, 3.0, 2.0, 1.0, 0.5, 0.0, 0.0, -1.0, -1.0, -2.0])
FLAT = torch.tensor([1.0, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3])


def measure_top_p_table() -> dict[str, tuple[float, int, float]]:
    """sampling2's table, the top chance, the count top p of 0.9 keeps and the chance in the top 3 of each set of
    scores. It draws nothing, so it has no seed. Not run by any check."""
    table: dict[str, tuple[float, int, float]] = {}

    for name, logits in (("peaked", PEAKED), ("flat", FLAT)):
        chances = torch.softmax(logits, dim=0)
        table[name] = (
            chances.max().item(),
            reference_kept_count(logits, 0.9),
            chances.topk(3).values.sum().item(),
        )

    return table


SAMPLING2_LESSON = prose(
    """
    gpt4 shaped its draws with two knobs, temperature and top k. Top k keeps
    a fixed count of choices, whatever the model believes, and that is its
    weakness.

    When the model is sure, a few choices hold almost all the chance, and
    keeping more lets in nonsense. When it is unsure, many choices are fair,
    and keeping few cuts good ones. Top p, which Holtzman and colleagues call
    nucleus sampling, keeps the smallest set of likeliest choices whose
    chances add up to at least p.

    Two sets of scores over 10 choices show the difference.

    | Scores | Top chance | Kept by top p of 0.9 | Chance in the top 3 |
    |---|---|---|---|
    | peaked | 0.921 | 1 | 0.983 |
    | flat | 0.139 | 9 | 0.378 |

    Top 3 keeps two needless choices on the peaked scores and cuts most of the
    chance on the flat ones. Top p follows the model's own certainty.

    To find the set, sort the chances from largest down and add them up with
    `cumsum`. A choice stays when the chances before it add up to less than p,
    so the likeliest one always stays.

    ```python
    sorted_chances, order = torch.tensor([0.1, 0.6, 0.3]).sort(descending=True)
    sorted_chances    # tensor([0.6000, 0.3000, 0.1000])
    order             # tensor([1, 2, 0]), where each one came from
    ```

    Writing into a tensor at `order` puts each sorted entry back in its own
    place, so `keep[order] = sorted_keep` undoes the sort.

    Hugging Face's transformers library applies the knobs in a row,
    temperature first, then top k, then top p, and draws from what is left.
    """,
)

SAMPLING2_SOURCE = source_file(
    '''
    # sampling2.py

    import torch
    from torch import Tensor

    sorted_chances, order = torch.tensor([0.1, 0.6, 0.3]).sort(descending=True)
    placed = torch.zeros(3)
    placed[order] = sorted_chances                    # tensor([0.1000, 0.6000, 0.3000]), back in place


    # TODO: Return logits, scores over choices of shape (choices,), with every
    #   choice outside top p set to minus infinity. A choice stays when the
    #   chances of the likelier ones before it add up to less than p.
    def top_p_filter(
        logits: Tensor,
        p: float,
    ) -> Tensor:
        ...


    # TODO: Return how many choices top_p_filter keeps for logits at p, as a
    #   plain int.
    def kept_count(
        logits: Tensor,
        p: float,
    ) -> int:
        ...


    # TODO: Return one choice drawn from logits, as a plain int. Divide by
    #   temperature, keep the top_k scores as gpt4 did, then apply top_p_filter,
    #   and draw from the softmax with torch.multinomial and generator.
    def sample_with(
        logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        generator: torch.Generator,
    ) -> int:
        ...
    ''',
)

SAMPLING2_SOLUTION = source_file(
    '''
    # sampling2.py, a solution for comparison

    import torch
    from torch import Tensor

    sorted_chances, order = torch.tensor([0.1, 0.6, 0.3]).sort(descending=True)
    placed = torch.zeros(3)
    placed[order] = sorted_chances


    def top_p_filter(
        logits: Tensor,
        p: float,
    ) -> Tensor:
        sorted_logits, order = logits.sort(descending=True)
        chances = torch.softmax(sorted_logits, dim=0)
        sorted_keep = chances.cumsum(0) - chances < p
        keep = torch.zeros_like(sorted_keep)
        keep[order] = sorted_keep

        return logits.masked_fill(~keep, float("-inf"))


    def kept_count(
        logits: Tensor,
        p: float,
    ) -> int:
        return int(torch.isfinite(top_p_filter(logits, p)).sum())


    def sample_with(
        logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        generator: torch.Generator,
    ) -> int:
        scores = logits / temperature
        kth = scores.topk(top_k).values[-1]
        scores = scores.masked_fill(scores < kth, float("-inf"))
        scores = top_p_filter(scores, top_p)

        return torch.multinomial(torch.softmax(scores, dim=0), 1, generator=generator).item()
    ''',
)


def check_sample_with(found: object) -> None:
    for logits, temperature, top_k, top_p, seed in (
        (FLAT, 0.7, 6, 0.6, 654),
        (PEAKED, 2.0, 4, 0.95, 655),
        (torch.tensor([0.2, -0.4, 1.3, 0.9, 0.0]), 1.0, 5, 0.6, 656),
    ):
        call = f"sample_with(scores, {temperature}, {top_k}, {top_p}, a generator seeded with {seed})"
        drawn = [found(logits, temperature, top_k, top_p, torch.Generator().manual_seed(seed + trial)) for trial in range(40)]
        expected = [reference_sample_with(logits, temperature, top_k, top_p, torch.Generator().manual_seed(seed + trial)) for trial in range(40)]

        if not all(isinstance(choice, int) for choice in drawn):
            raise Mismatch(f"{call} should give a plain int. Take .item() of the draw.")

        if drawn != expected:
            raise Mismatch(f"Over 40 seeds, {call} drew {drawn[:12]}..., but it should draw {expected[:12]}....")


def check_sampling2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="top_p_filter",
        description="keeps the smallest likeliest set reaching p",
        inputs=(
            (FLAT, 0.5),
            (PEAKED, 0.95),
            (torch.tensor([0.2, -0.4, 1.3, 0.9, 0.0]), 0.6),
            (torch.tensor([3.0, 1.0, 2.0]), 0.01),
        ),
        reference=reference_top_p_filter,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="kept_count",
        description="counts the choices top p keeps",
        inputs=(
            (PEAKED, 0.9),
            (FLAT, 0.9),
            (FLAT, 0.45),
        ),
        reference=reference_kept_count,
        expect=expect_value,
    )
    checker.test(
        function="sample_with",
        description="applies temperature, top k and top p, then draws",
        run=check_sample_with,
    )


SAMPLING2 = Exercise(
    name="sampling2",
    chapter=SAMPLING,
    title="Top p",
    lesson=SAMPLING2_LESSON,
    source=SAMPLING2_SOURCE,
    solution=SAMPLING2_SOLUTION,
    hints={
        "top_p_filter": "Sort the logits largest first and take their softmax. A sorted choice stays where "
        "chances.cumsum(0) - chances < p. Put that back in place through order, then masked_fill the rest with "
        "float(\"-inf\").",
        "kept_count": "torch.isfinite of top_p_filter's result marks what stays. Sum it and turn it into an int.",
        "sample_with": "scores.topk(top_k).values[-1] is the k-th largest. masked_fill the scores below it, then "
        "top_p_filter, torch.softmax with dim=0, torch.multinomial and .item().",
    },
    introduces=(),
    check=check_sampling2,
)


# ---------------------------------------------------------------------------
# sampling3
# ---------------------------------------------------------------------------


def reference_reparameterize(
    mean: Tensor,
    log_variance: Tensor,
    noise: Tensor,
) -> Tensor:
    return mean + torch.exp(0.5 * log_variance) * noise


def reference_draw_codes(
    mean: Tensor,
    log_variance: Tensor,
    count: int,
    generator: torch.Generator,
) -> Tensor:
    noise = torch.randn(count, *mean.shape, generator=generator)

    return reference_reparameterize(mean, log_variance, noise)


def reference_square_gradients(
    mean: float,
    log_variance: float,
    count: int,
    generator: torch.Generator,
) -> tuple[float, float]:
    mean_leaf = torch.tensor(mean, requires_grad=True)
    log_variance_leaf = torch.tensor(log_variance, requires_grad=True)
    (reference_draw_codes(mean_leaf, log_variance_leaf, count, generator) ** 2).mean().backward()

    return mean_leaf.grad.item(), log_variance_leaf.grad.item()


def score_function_gradients(
    mean: float,
    log_variance: float,
    count: int,
    generator: torch.Generator,
) -> tuple[float, float]:
    """The same two gradients by the score function estimator, which Kingma and Welling set the trick against. The
    draws carry no gradient, and each one's z squared weights the gradient of its own log density."""
    mean_leaf = torch.tensor(mean, requires_grad=True)
    log_variance_leaf = torch.tensor(log_variance, requires_grad=True)
    z = reference_draw_codes(torch.tensor(mean), torch.tensor(log_variance), count, generator)
    log_density = -0.5 * (math.log(2 * math.pi) + log_variance_leaf + (z - mean_leaf) ** 2 / log_variance_leaf.exp())
    (z**2 * log_density).mean().backward()

    return mean_leaf.grad.item(), log_variance_leaf.grad.item()


def measure_square_gradients(seed: int = 0) -> dict[int, tuple[float, float]]:
    """sampling3's first table for one seed, with a fresh generator for each count. The lesson gives seeds 0 to 2.
    Not run by any check."""
    return {
        count: reference_square_gradients(
            mean=1.5,
            log_variance=math.log(0.25),
            count=count,
            generator=torch.Generator().manual_seed(seed),
        )
        for count in (10, 1000, 100_000)
    }


def measure_gradient_spreads(seed: int = 0) -> dict[str, tuple[float, float]]:
    """sampling3's second table for one seed, the spread of each estimator's two gradients over 2,000 estimates of 10
    draws. Both estimators read the same noise. The lesson gives seeds 0 to 2. Not run by any check."""
    spreads: dict[str, tuple[float, float]] = {}

    for name, estimate in (("trick", reference_square_gradients), ("score function", score_function_gradients)):
        generator = torch.Generator().manual_seed(seed)
        estimates = torch.tensor([estimate(1.5, math.log(0.25), 10, generator) for _ in range(2000)])
        spreads[name] = (estimates[:, 0].std().item(), estimates[:, 1].std().item())

    return spreads


SAMPLING3_LESSON = prose(
    """
    A VAE, in the next chapter, draws a random code while it trains. The loss
    then needs gradients back into the mean and the spread the code was drawn
    from, and a plain draw passes none.

    ```python
    mean = torch.tensor(1.5, requires_grad=True)
    spread = torch.tensor(0.5, requires_grad=True)
    torch.distributions.Normal(mean, spread).sample().requires_grad    # False
    ```

    `torch.normal(mean, spread)` does record a step, but it hands the mean and
    the spread a gradient of 0.

    The reparameterization trick moves the randomness out of the way. Draw
    noise from a standard normal, likelihood3's Gaussian with mean 0 and
    spread 1. Then shift and stretch it, `z = mean + spread * noise`. z has
    the same distribution as a direct draw. It is a plain sum and product of mean and spread, so autograd
    follows it. `torch.distributions.Normal(...).rsample()` does exactly this.

    Models output the log of the variance instead of the spread. A Linear can
    give any number, a spread must be positive, and `exp` of anything is
    positive. The spread is then `exp(log_variance / 2)`.

    Gradients through the trick are estimates, averaged over draws. Take the
    average of z squared, with z drawn at a mean of 1.5 and a spread of 0.5.
    It comes out at the mean squared plus the variance, 2.5. So its true
    gradient is 2 times the mean, 3, for the mean, and the variance, 0.25,
    for the log variance. Here are the estimates on 3 seeds.

    | Draws | 10 | 1,000 | 100,000 |
    |---|---|---|---|
    | Gradient for the mean | 2.73 to 2.87 | 3.01 to 3.05 | 3.00 |
    | Gradient for the log variance | -0.01 to 0.10 | 0.27 to 0.29 | 0.25 |

    Kingma and Welling call the older way, the score function estimator, too
    noisy to use. It needs no trick. It weights each draw's z squared by how
    a nudge to the mean or the log variance raises that draw's log density.
    Both average to the true gradient. Here is how far each strays, as the
    spread of 2,000 estimates from 10 draws each, on 3 seeds.

    | Estimate | Spread for the mean | Spread for the log variance |
    |---|---|---|
    | The trick | 0.31 to 0.32 | 0.26 to 0.27 |
    | Score function | 2.30 to 2.37 | 1.04 to 1.11 |

    The trick strays about 7 times less for the mean and 4 times less for the
    log variance. So far fewer draws give a usable gradient.
    """,
)

SAMPLING3_SOURCE = source_file(
    '''
    # sampling3.py

    import math

    import torch
    from torch import Tensor

    log_variance = torch.tensor(math.log(0.25))
    spread = torch.exp(0.5 * log_variance)    # tensor(0.5000), the square root of the variance


    # TODO: Return mean plus the spread times noise, the spread coming from
    #   log_variance. All three have the same shape.
    def reparameterize(
        mean: Tensor,
        log_variance: Tensor,
        noise: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return count draws for mean and log_variance through
    #   reparameterize, of shape (count, *mean.shape). Draw the noise for all
    #   of them at once with torch.randn and generator.
    def draw_codes(
        mean: Tensor,
        log_variance: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the gradients of the average of z squared, over count draws
    #   of z from draw_codes, for the mean and for the log variance, as two
    #   plain numbers. Make mean and log_variance tensors that record gradients
    #   first.
    def square_gradients(
        mean: float,
        log_variance: float,
        count: int,
        generator: torch.Generator,
    ) -> tuple[float, float]:
        ...
    ''',
)

SAMPLING3_SOLUTION = source_file(
    '''
    # sampling3.py, a solution for comparison

    import math

    import torch
    from torch import Tensor

    log_variance = torch.tensor(math.log(0.25))
    spread = torch.exp(0.5 * log_variance)


    def reparameterize(
        mean: Tensor,
        log_variance: Tensor,
        noise: Tensor,
    ) -> Tensor:
        return mean + torch.exp(0.5 * log_variance) * noise


    def draw_codes(
        mean: Tensor,
        log_variance: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        noise = torch.randn(count, *mean.shape, generator=generator)

        return reparameterize(mean, log_variance, noise)


    def square_gradients(
        mean: float,
        log_variance: float,
        count: int,
        generator: torch.Generator,
    ) -> tuple[float, float]:
        mean_leaf = torch.tensor(mean, requires_grad=True)
        log_variance_leaf = torch.tensor(log_variance, requires_grad=True)
        (draw_codes(mean_leaf, log_variance_leaf, count, generator) ** 2).mean().backward()

        return mean_leaf.grad.item(), log_variance_leaf.grad.item()
    ''',
)


def expect_two_numbers(
    call: str,
    actual: object,
    expected: tuple[float, float],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should give the two gradients as a tuple of two numbers.")

    for name, number, wanted in zip(("mean", "log variance"), actual, expected):
        expect_close(
            call=f"the gradient for the {name} from {call}",
            actual=number,
            expected=wanted,
            tolerance=1e-5,
        )


def check_draws_pass_gradients(found: object) -> None:
    mean = torch.tensor([0.5, -1.0], requires_grad=True)
    log_variance = torch.tensor([0.0, -0.5], requires_grad=True)
    drawn = found(mean, log_variance, 3, torch.Generator().manual_seed(657))

    if not isinstance(drawn, torch.Tensor) or drawn.grad_fn is None:
        raise Mismatch("draw_codes gave draws that record nothing, so no gradient could reach the mean.")

    drawn.sum().backward()

    if mean.grad is None or torch.all(mean.grad == 0):
        raise Mismatch("draw_codes passes no gradient to the mean. Draw the noise, then shift and stretch it.")


def check_sampling3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="reparameterize",
        description="shifts and stretches the noise",
        inputs=(
            (torch.tensor([0.5, -1.0, 2.0]), torch.tensor([0.0, math.log(0.25), 1.0]), torch.tensor([1.0, -2.0, 0.5])),
            (torch.zeros(2, 3), torch.full((2, 3), -1.0), torch.ones(2, 3)),
        ),
        reference=reference_reparameterize,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="draw_codes",
        description="gives count draws of the mean's shape",
        inputs=(
            (torch.tensor([0.5, -1.0]), torch.tensor([0.0, -0.5]), 4, seeded(658)),
            (torch.zeros(2, 3), torch.zeros(2, 3), 2, seeded(659)),
        ),
        reference=reference_draw_codes,
        expect=expect_tensor,
    )
    checker.test(
        function="draw_codes",
        description="passes gradients back to the mean",
        run=check_draws_pass_gradients,
    )
    check_on_inputs(
        checker=checker,
        function="square_gradients",
        description="estimates the gradients through the draws",
        inputs=(
            (1.5, math.log(0.25), 1000, seeded(660)),
            (-0.5, 0.3, 50, seeded(661)),
        ),
        reference=reference_square_gradients,
        expect=expect_two_numbers,
    )


SAMPLING3 = Exercise(
    name="sampling3",
    chapter=SAMPLING,
    title="The reparameterization trick",
    lesson=SAMPLING3_LESSON,
    source=SAMPLING3_SOURCE,
    solution=SAMPLING3_SOLUTION,
    hints={
        "reparameterize": "The spread is torch.exp(0.5 * log_variance). Multiply the noise by it and add the mean.",
        "draw_codes": "torch.randn(count, *mean.shape, generator=generator) gives all the noise at once. Pass it to "
        "reparameterize.",
        "square_gradients": "torch.tensor(mean, requires_grad=True), and the same for log_variance. Take the mean of "
        "draw_codes' draws squared, call .backward(), and return each .grad.item().",
    },
    introduces=(),
    check=check_sampling3,
)
