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

        Section 9 is about models that generate new data, and all of them draw
        random numbers. This chapter shows how the random draws the course has
        used so far actually work. You sample from a categorical distribution
        by hand, and control the samples with temperature, top k and top p.
        Then you sample from a Gaussian in a way that lets gradients flow back
        through the sample.

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
    the 65 characters of gpt1. Sampling picks one choice at random, so that
    each choice is picked with its own probability. Each random pick is
    called a draw, or a sample. names3 and gpt4 used `torch.multinomial` to
    sample, and this exercise shows how it works.

    Lay the probabilities end to end along the line from 0 to 1. Each choice
    then covers an interval as long as its probability. Pick a point between
    0 and 1 uniformly at random, so every point is equally likely. The choice
    whose interval contains the point is the draw. The running total from
    `cumsum` gives the end of each interval.

    ```python
    probabilities = torch.tensor([0.5, 0.3, 0.2])
    probabilities.cumsum(0)    # tensor([0.5000, 0.8000, 1.0000]), where each interval ends
    ```

    A point at 0.65 is past one interval end, 0.5, but not past 0.8. So it
    falls in the interval of choice 1. In general, the number of interval
    ends at or below the point is the index of the choice it picks.

    `torch.multinomial(probabilities, count)` draws without replacement by
    default. That means it never picks the same choice twice, and it fails
    when count is larger than the number of choices. For draws that may
    repeat a choice, pass `replacement=True`.

    Over many draws, the fraction of draws that pick each choice comes close
    to its probability. The table shows the largest gap between a choice's
    fraction and its probability, over the three choices above. Each entry
    covers 3 seeds.

    | Draws | 100 | 10,000 | 1,000,000 |
    |---|---|---|---|
    | Largest gap | 0.03 to 0.13 | 0.0030 to 0.0047 | 0.00003 to 0.00091 |

    A hundred times more draws make the gap only about ten times smaller.
    That slow shrinking is why an estimate from random draws needs many draws
    to be accurate.
    """,
)

SAMPLING1_SOURCE = source_file(
    '''
    # sampling1.py

    import torch
    from torch import Tensor

    ends = torch.tensor([0.5, 0.3, 0.2]).cumsum(0)    # tensor([0.5000, 0.8000, 1.0000])
    passed = (torch.tensor(0.65) >= ends).sum()       # tensor(1), the choice that a point at 0.65 picks


    # TODO: Return the choice that each point in u picks, as a tensor of indices
    #   with the shape of u. u holds numbers from 0 to 1. The choice for a point
    #   is the number of interval ends, from probabilities.cumsum(0), that are
    #   at or below the point. Rounding can leave the last end slightly below
    #   1. A point past it must still pick the last choice.
    #   Example: draw_by_hand(torch.tensor([0.5, 0.3, 0.2]),
    #   torch.tensor([0.1, 0.65, 0.9])) returns tensor([0, 1, 2]).
    def draw_by_hand(
        probabilities: Tensor,
        u: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return count draws from probabilities, with repeats allowed, using
    #   torch.multinomial and generator. The result is a tensor of count
    #   indices.
    def draw_many(
        probabilities: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return a float tensor of length choices. Entry i is the fraction of
    #   draws that picked choice i, and it is 0 for a choice never drawn.
    #   Example: frequencies(torch.tensor([1, 1, 3]), 5) returns
    #   tensor([0.0000, 0.6667, 0.0000, 0.3333, 0.0000]).
    def frequencies(
        draws: Tensor,
        choices: int,
    ) -> Tensor:
        ...


    # TODO: Make count draws from probabilities with draw_many, and compute
    #   their frequencies. Return the largest absolute difference between a
    #   frequency and its probability, as a Python float.
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
        description="returns the interval each point falls in",
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
        description="returns each choice's fraction, including choices never drawn",
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
        description="returns the largest gap after count draws",
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
        "draw_by_hand": "Compute the ends with cumsum. Then u.unsqueeze(1) >= ends compares every point with every end. "
        "Sum over dim=1, then cap the result with .clamp(max=len(probabilities) - 1).",
        "draw_many": "Call torch.multinomial with replacement=True and generator=generator.",
        "frequencies": "torch.bincount with minlength=choices counts each choice. Convert the counts to floats and "
        "divide by len(draws).",
        "largest_gap": "Subtract probabilities from the frequencies of draw_many's draws. Then take .abs().max() and "
        "convert it with .item().",
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
    gpt4 controlled its draws with two settings, temperature and top k. Top k
    always keeps the same number of choices, however confident the model is.
    That is its weakness.

    When the model is confident, a few choices hold almost all the
    probability, and keeping more choices lets in nonsense. When it is
    unsure, many choices are reasonable, and keeping only a few cuts out good
    ones. Top p, which Holtzman and colleagues call nucleus sampling, adapts
    to this. It keeps the smallest set of most likely choices whose
    probabilities add up to at least p.

    Two sets of logits over 10 choices show the difference.

    | Scores | Top probability | Kept by top p of 0.9 | Probability in the top 3 |
    |---|---|---|---|
    | peaked | 0.921 | 1 | 0.983 |
    | flat | 0.139 | 9 | 0.378 |

    On the peaked scores, top 3 keeps two choices that are not needed. On the
    flat scores, it cuts out most of the probability. Top p adjusts to how
    confident the model is.

    To find the set, sort the probabilities from largest to smallest and take
    their running total with `cumsum`. A choice stays if the probabilities
    before it add up to less than p. So the most likely choice always stays,
    since nothing comes before it.

    ```python
    sorted_chances, order = torch.tensor([0.1, 0.6, 0.3]).sort(descending=True)
    sorted_chances    # tensor([0.6000, 0.3000, 0.1000])
    order             # tensor([1, 2, 0]), the original index of each sorted entry
    ```

    Assigning into a tensor at the indices `order` puts each sorted entry back
    at its original index. So `keep[order] = sorted_keep` undoes the sort.

    Hugging Face's transformers library applies the settings in order.
    Temperature comes first, then top k, then top p, and it draws from the
    choices that are left.
    """,
)

SAMPLING2_SOURCE = source_file(
    '''
    # sampling2.py

    import torch
    from torch import Tensor

    sorted_chances, order = torch.tensor([0.1, 0.6, 0.3]).sort(descending=True)
    placed = torch.zeros(3)
    placed[order] = sorted_chances                    # tensor([0.1000, 0.6000, 0.3000]), back in the original order


    # TODO: Return a copy of logits, of shape (choices,), with every choice
    #   outside the top p set to minus infinity. Take the softmax of the logits
    #   sorted from largest to smallest. A choice stays if the probabilities of
    #   the more likely choices before it add up to less than p.
    #   Example: top_p_filter(torch.tensor([3.0, 1.0, 2.0]), 0.7) returns
    #   tensor([3., -inf, 2.]).
    def top_p_filter(
        logits: Tensor,
        p: float,
    ) -> Tensor:
        ...


    # TODO: Return how many choices top_p_filter keeps for logits at p, as a
    #   Python int.
    def kept_count(
        logits: Tensor,
        p: float,
    ) -> int:
        ...


    # TODO: Return one choice drawn from logits, as a Python int. Divide the
    #   logits by temperature. Keep the top_k largest scores as gpt4 did, by
    #   setting the others to minus infinity. Then apply top_p_filter, take the
    #   softmax, and draw one choice with torch.multinomial and generator.
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
            raise Mismatch(f"{call} should return a Python int. Call .item() on the draw.")

        if drawn != expected:
            raise Mismatch(f"Over 40 seeds, {call} drew {drawn[:12]}..., but it should draw {expected[:12]}....")


def check_sampling2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="top_p_filter",
        description="keeps the smallest set of most likely choices whose probabilities reach p",
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
        "chances.cumsum(0) - chances < p. Put those True and False values back in the original order through order. "
        "Then masked_fill the other choices with float(\"-inf\").",
        "kept_count": "torch.isfinite of top_p_filter's result is True for the choices that stay. Sum it and convert "
        "the sum to an int.",
        "sample_with": "scores.topk(top_k).values[-1] is the k-th largest score. masked_fill the scores below it with "
        "minus infinity. Then apply top_p_filter, torch.softmax with dim=0, torch.multinomial and .item().",
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
    A VAE, in the next chapter, draws a random code while it trains. The code
    is drawn from a Gaussian whose mean and standard deviation the model
    computes. So the loss needs gradients for that mean and standard
    deviation. But an ordinary draw passes no gradient back to them. In the
    code below, `spread` is the standard deviation.

    ```python
    mean = torch.tensor(1.5, requires_grad=True)
    spread = torch.tensor(0.5, requires_grad=True)
    torch.distributions.Normal(mean, spread).sample().requires_grad    # False
    ```

    `torch.normal(mean, spread)` does record a step for autograd. But it
    gives the mean and the standard deviation a gradient of 0.

    The reparameterization trick moves the randomness into a separate input.
    First draw noise from a standard normal, the Gaussian from likelihood3
    with mean 0 and standard deviation 1. Then scale the noise and shift it.

    ```text
    z = mean + spread * noise
    ```

    z has the same distribution as a direct draw. It is computed from mean
    and spread with ordinary addition and multiplication, so autograd can
    follow it back to both. `torch.distributions.Normal(...).rsample()` does
    exactly this.

    Models usually output the log of the variance instead of the standard
    deviation. A Linear layer can output any number, but a standard deviation
    must be positive, and `exp` of any number is positive. The variance is
    the standard deviation squared, so the standard deviation is
    `exp(log_variance / 2)`.

    Gradients through the trick are estimates, because they are averages over
    a finite number of draws. Take the average of `z ** 2`, with z drawn from
    a mean of 1.5 and a standard deviation of 0.5. Its expected value is
    `mean ** 2 + variance`, which is 2.5. So the true gradient for the mean
    is `2 * mean`, which is 3. The true gradient for the log variance is the
    variance, 0.25. Here are the estimates on 3 seeds.

    | Draws | 10 | 1,000 | 100,000 |
    |---|---|---|---|
    | Gradient for the mean | 2.73 to 2.87 | 3.01 to 3.05 | 3.00 |
    | Gradient for the log variance | -0.01 to 0.10 | 0.27 to 0.29 | 0.25 |

    The older method is the score function estimator, and Kingma and Welling
    call it too noisy to use. It needs no trick. For each draw, it multiplies
    `z ** 2` by the gradient of that draw's log density with respect to the
    mean or the log variance. Both methods average to the true gradient. The
    table shows how much each one varies. Each entry is the standard
    deviation of 2,000 estimates, each from 10 draws, on 3 seeds.

    | Estimate | Standard deviation for the mean | Standard deviation for the log variance |
    |---|---|---|
    | The trick | 0.31 to 0.32 | 0.26 to 0.27 |
    | Score function | 2.30 to 2.37 | 1.04 to 1.11 |

    The trick's estimates vary about 7 times less for the mean, and 4 times
    less for the log variance. So far fewer draws are enough for a usable
    gradient.
    """,
)

SAMPLING3_SOURCE = source_file(
    '''
    # sampling3.py

    import math

    import torch
    from torch import Tensor

    log_variance = torch.tensor(math.log(0.25))
    spread = torch.exp(0.5 * log_variance)    # tensor(0.5000), the standard deviation, which is the square root of the variance


    # TODO: Return mean + spread * noise, where spread is the standard deviation
    #   computed from log_variance. All three inputs have the same shape.
    #   Example: reparameterize(torch.tensor([1.0]), torch.tensor([0.0]),
    #   torch.tensor([2.0])) returns tensor([3.]).
    def reparameterize(
        mean: Tensor,
        log_variance: Tensor,
        noise: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return count draws of z, made with reparameterize, as a tensor of
    #   shape (count, *mean.shape). Draw the noise for all of them at once with
    #   torch.randn and generator, then pass it to reparameterize with mean and
    #   log_variance.
    def draw_codes(
        mean: Tensor,
        log_variance: Tensor,
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Estimate the gradients of the average of z ** 2 from the lesson.
    #   First turn mean and log_variance into tensors with requires_grad=True.
    #   Make count draws of z with draw_codes, average z ** 2 over them, and
    #   call backward. Return the gradient for the mean and the gradient for
    #   the log variance, as a tuple of two Python floats.
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
        raise Mismatch(f"{call} should return the two gradients as a tuple of two numbers.")

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
        raise Mismatch("draw_codes returned draws that autograd did not record, so no gradient can reach the mean.")

    drawn.sum().backward()

    if mean.grad is None or torch.all(mean.grad == 0):
        raise Mismatch(
            "draw_codes passes no gradient to the mean. Draw the noise first, then scale it by the standard deviation "
            "and add the mean.",
        )


def check_sampling3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="reparameterize",
        description="scales the noise by the standard deviation and adds the mean",
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
        description="returns count draws, each with the mean's shape",
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
        "reparameterize": "The standard deviation is torch.exp(0.5 * log_variance). Multiply the noise by it and add "
        "the mean.",
        "draw_codes": "torch.randn(count, *mean.shape, generator=generator) gives all the noise at once. Pass it to "
        "reparameterize.",
        "square_gradients": "Create torch.tensor(mean, requires_grad=True), and the same for log_variance. Take the "
        "mean of draw_codes' draws squared, call .backward(), and return the .grad.item() of each tensor.",
    },
    introduces=(),
    check=check_sampling3,
)
