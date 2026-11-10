"""96_audio, sound turned into tokens: log mel frames, the tokens speech models read, and a residual quantizer's
few streams of codebook ids, the tokens of a neural codec."""

from __future__ import annotations

import contextlib
import functools
import math
import textwrap
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
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
    value_text,
)
from ..text import prose, source_file

AUDIO = Chapter(
    folder="96_audio",
    title="Audio as tokens",
    readme=source_file(
        """
        # Audio as tokens

        A second of sound is thousands of samples, far too many to use as
        tokens in a transformer. This chapter turns sound into the two kinds
        of tokens that audio models read. The first kind is log mel frames,
        which speech recognizers such as Whisper read. There are 100 frames a
        second, and Whisper's first layers turn them into 50 tokens. The
        second kind is codebook ids from a residual quantizer, in a few
        parallel streams. Neural codecs such as SoundStream and EnCodec make
        these, so a model can write sound as well as read it.

        ## Further reading

        - [Radford and colleagues, Whisper](https://arxiv.org/abs/2212.04356)
        - [Zeghidour and colleagues, SoundStream](https://arxiv.org/abs/2107.03312)
        - [Défossez and colleagues, EnCodec](https://arxiv.org/abs/2210.13438)
        - [torchaudio, melscale_fbanks](https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.melscale_fbanks.html)
        """,
    ),
)

# Whisper's settings, from its audio.py: 16,000 samples a second, windows of 400 samples, a frame every 160.
SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 80

MEASURE_THREADS = 4

MEL_PART = '''
def hz_to_mel(hertz: Tensor) -> Tensor:
    return 2595 * torch.log10(1 + hertz / 700)


def mel_to_hz(mels: Tensor) -> Tensor:
    return 700 * (10 ** (mels / 2595) - 1)


def mel_filters(
    n_freqs: int,
    n_mels: int,
    sample_rate: int,
) -> Tensor:
    freqs = torch.linspace(0, sample_rate / 2, n_freqs)
    top = hz_to_mel(torch.tensor(sample_rate / 2))
    points = mel_to_hz(torch.linspace(0, top.item(), n_mels + 2))
    left, center, right = points[:-2], points[1:-1], points[2:]
    up = (freqs.unsqueeze(1) - left) / (center - left)
    down = (right - freqs.unsqueeze(1)) / (right - center)

    return torch.clamp(torch.minimum(up, down), min=0)


def log_mel(
    signal: Tensor,
    n_fft: int,
    hop: int,
    filters: Tensor,
) -> Tensor:
    window = torch.hann_window(n_fft)
    spectrum = torch.stft(signal, n_fft=n_fft, hop_length=hop, window=window, center=False, return_complex=True)
    power = spectrum.abs() ** 2

    return torch.clamp(filters.T @ power, min=1e-10).log10()


def loudest_channel(log_mels: Tensor) -> int:
    return int(log_mels.mean(dim=1).argmax())
'''

TONE_PART = '''
def tone(
    frequency: float,
    seconds: float,
    sample_rate: int,
) -> Tensor:
    t = torch.arange(int(seconds * sample_rate)) / sample_rate

    return torch.sin(2 * math.pi * frequency * t)
'''

SOUNDS_PART = '''
def made_up_sound(
    seconds: float,
    sample_rate: int,
    generator: torch.Generator,
) -> Tensor:
    """One to three tones that start and stop, a rising chirp in the second half, a burst of noise and a faint hiss."""
    t = torch.arange(int(seconds * sample_rate)) / sample_rate
    sound = torch.zeros(len(t))

    def draw() -> float:
        return torch.rand(1, generator=generator).item()

    for _ in range(int(torch.randint(1, 4, (1,), generator=generator))):
        frequency = 100 * 2 ** (6 * draw())
        start = draw() * seconds / 2
        length = 0.2 + draw() * seconds / 2
        playing = ((t >= start) & (t < start + length)).float()
        sound += playing * torch.sin(2 * math.pi * frequency * t) * (0.2 + draw())

    low, high = 200 + 1000 * draw(), 200 + 3000 * draw()
    sound += 0.5 * torch.sin(2 * math.pi * (low * t + (high - low) * t**2 / (2 * seconds))) * (t > seconds / 2).float()
    burst = 0.8 * seconds * draw()
    sound += 0.3 * torch.randn(len(t), generator=generator) * ((t > burst) & (t < burst + 0.1)).float()

    return sound + 0.001 * torch.randn(len(t), generator=generator)
'''

GIVEN_CODES_PART = '''
def nearest_codes(
    z: Tensor,
    codebook: Tensor,
) -> Tensor:
    return ((z.unsqueeze(1) - codebook.unsqueeze(0)) ** 2).sum(dim=2).argmin(dim=1)


def kmeans(
    vectors: Tensor,
    entries: int,
    steps: int,
    generator: torch.Generator,
) -> Tensor:
    codebook = vectors[torch.randperm(len(vectors), generator=generator)[:entries]].clone()

    for _ in range(steps):
        ids = nearest_codes(vectors, codebook)
        sums = torch.zeros_like(codebook).index_add_(0, ids, vectors)
        counts = torch.bincount(ids, minlength=entries).unsqueeze(1)
        codebook = torch.where(counts > 0, sums / counts.clamp(min=1), codebook)

    return codebook


def relative_error(
    z: Tensor,
    quantized: Tensor,
) -> float:
    return ((z - quantized) ** 2).sum().item() / (z**2).sum().item()
'''

RVQ_PART = '''
def residual_quantize(
    z: Tensor,
    codebooks: Tensor,
) -> tuple[Tensor, Tensor]:
    residual = z
    quantized = torch.zeros_like(z)
    ids = []

    for codebook in codebooks:
        chosen = nearest_codes(residual, codebook)
        ids.append(chosen)
        quantized = quantized + codebook[chosen]
        residual = residual - codebook[chosen]

    return torch.stack(ids, dim=1), quantized


def decode(
    ids: Tensor,
    codebooks: Tensor,
) -> Tensor:
    return torch.stack([codebook[ids[:, level]] for level, codebook in enumerate(codebooks)]).sum(dim=0)


def bits_per_second(
    frames_per_second: float,
    codebooks: int,
    entries: int,
) -> float:
    return frames_per_second * codebooks * math.log2(entries)


def fit_codebooks(
    vectors: Tensor,
    levels: int,
    entries: int,
    steps: int,
    generator: torch.Generator,
) -> Tensor:
    residual = vectors
    codebooks = []

    for _ in range(levels):
        codebook = kmeans(residual, entries, steps, generator)
        residual = residual - codebook[nearest_codes(residual, codebook)]
        codebooks.append(codebook)

    return torch.stack(codebooks)
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "F": F,
        "Tensor": Tensor,
    }
    exec(code, namespace)

    return namespace


REFERENCE = namespace_of(MEL_PART + TONE_PART + SOUNDS_PART + GIVEN_CODES_PART + RVQ_PART)


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(MEASURE_THREADS)

    try:
        yield
    finally:
        torch.set_num_threads(threads)


def noise(
    seed: int,
    length: int,
) -> Tensor:
    return torch.randn(length, generator=torch.Generator().manual_seed(seed))


def two_tones(
    first: float,
    second: float,
    length: int,
    sample_rate: int,
) -> Tensor:
    t = torch.arange(length) / sample_rate

    return torch.sin(2 * math.pi * first * t) + 0.3 * torch.sin(2 * math.pi * second * t)


# ---------------------------------------------------------------------------
# sound1
# ---------------------------------------------------------------------------

TONES = (200.0, 300.0, 6000.0, 6100.0)

SOUND1_LESSON = prose(
    """
    Whisper is a speech recognizer. It first resamples all audio to 16,000
    samples a second. Then it reads an 80 channel log mel spectrogram,
    computed from windows of 25 milliseconds taken every 10 milliseconds.
    That is the spectrogram from stft1 with two changes.

    - The **mel scale** spaces frequencies the way human hearing does, with fine steps at low pitches and coarse steps at high ones.
    - Taking the log turns loudness, which spans many powers of 10, into a small range that a network handles well.

    The HTK formula converts a frequency of `f` hertz to `m` mels. The
    second line below is its inverse, from mels back to hertz.

    ```text
    m = 2595 * log10(1 + f / 700)
    f = 700 * (10 ** (m / 2595) - 1)
    ```

    At low frequencies the mel scale is close to a straight line, and at high
    frequencies it grows like a logarithm.

    ```python
    torch.log10(torch.tensor([10.0, 1000.0]))    # tensor([1., 3.])
    ```

    A filter bank turns the frequencies of a spectrum into mel channels. To
    build one, place `n_mels + 2` points evenly spaced in mels, from 0 up to
    the mels of half the sample rate. Then convert the points back to hertz.
    Filter i is a triangle over the frequencies. It is 0 at point i, rises to
    1 at point i + 1, and falls back to 0 at point i + 2. torchaudio's
    `melscale_fbanks` builds its default filters this way. At each
    frequency, the triangle is the smaller of a rising line and a falling
    line, clamped at 0.

    ```python
    up = torch.tensor([-0.5, 0.5, 1.5])
    down = torch.tensor([1.5, 0.5, -0.5])
    torch.clamp(torch.minimum(up, down), min=0)    # tensor([0.0000, 0.5000, 0.0000])
    ```

    Whisper's own filters come from librosa, which uses the Slaney scale
    instead. The Slaney scale is a straight line below 1,000 Hz, and its
    triangles all have the same area. This file keeps Whisper's sample rate,
    window, hop length and 80 channels, but uses the HTK filters.

    The power of a frequency in a window is its magnitude squared. The
    magnitude is the `.abs()` of the STFT, as in stft1. Multiplying the power
    by the filters gives one number per channel. Whisper's code raises any
    value below 1e-10 up to 1e-10, then takes the log in base 10.

    With these filters, channel 0 covers 45 Hz and channel 79 covers 525 Hz.
    Here is the loudest channel for four pure tones, each one second long.

    | Tone | Loudest channel |
    |---|---|
    | 200 Hz | 7 |
    | 300 Hz | 11 |
    | 6,000 Hz | 72 |
    | 6,100 Hz | 72 |

    At low frequencies, two tones 100 Hz apart land four channels apart. At
    high frequencies they land in the same channel, which matches how hard
    it is for a listener to tell them apart. A second of sound becomes 98
    frames of 80 numbers. Whisper pads both ends of the audio and drops the
    last frame, which gives exactly 100 frames.

    Whisper's first layers are two convolutions, and the second has a stride
    of 2. Together they turn the 100 frames of each second into 50 token
    vectors. The raw audio of that second held 16,000 samples.
    """,
)

SOUND1_SOURCE = source_file(
    '''
    # sound1.py

    import math

    import torch
    from torch import Tensor

    lows = torch.log10(torch.tensor([10.0, 1000.0]))       # tensor([1., 3.])
    points = torch.linspace(0, 10, 5)                       # tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])
    peaks = torch.clamp(torch.tensor([-1.0, 2.0]), min=0)   # tensor([0., 2.])


    # TODO: Convert each frequency in hertz to mels with the HTK formula,
    #   m = 2595 * log10(1 + f / 700). hertz is a tensor of any shape.
    #   Example: hz_to_mel(torch.tensor([700.0])) returns tensor([781.1729]).
    def hz_to_mel(hertz: Tensor) -> Tensor:
        ...


    # TODO: Convert each value in mels back to hertz. This is the inverse of
    #   hz_to_mel, f = 700 * (10 ** (m / 2595) - 1).
    def mel_to_hz(mels: Tensor) -> Tensor:
        ...


    # TODO: Return the mel filters, of shape (n_freqs, n_mels). The STFT's
    #   frequencies are torch.linspace(0, sample_rate / 2, n_freqs). Place
    #   n_mels + 2 points evenly in mels from 0 to the mels of sample_rate / 2,
    #   and turn them back into hertz. Filter i rises from 0 at point i to 1 at
    #   point i + 1 and falls to 0 at point i + 2, and is 0 everywhere else.
    #   Column i of the result is filter i, so entry [j, i] is the value of
    #   filter i at frequency j.
    def mel_filters(
        n_freqs: int,
        n_mels: int,
        sample_rate: int,
    ) -> Tensor:
        ...


    # TODO: Return the log mel frames of signal, of shape (n_mels, frames).
    #   Compute torch.stft with n_fft=n_fft, hop_length=hop, a Hann window of
    #   length n_fft, center=False and return_complex=True. The power is the
    #   magnitude squared, of shape (n_fft // 2 + 1, frames). Combine the power
    #   with the filters by matrix multiplication to get (n_mels, frames).
    #   Raise every value below 1e-10 up to 1e-10, then take the base 10 log.
    def log_mel(
        signal: Tensor,
        n_fft: int,
        hop: int,
        filters: Tensor,
    ) -> Tensor:
        ...


    # TODO: log_mels has shape (n_mels, frames). Return the index of the
    #   channel whose average over the frames is highest, as a plain Python int.
    #   Example: loudest_channel(torch.tensor([[1.0, 2.0], [4.0, 0.0],
    #   [2.0, 2.5]])) returns 2, since the averages are 1.5, 2.0 and 2.25.
    def loudest_channel(log_mels: Tensor) -> int:
        ...


    def tone(
        frequency: float,
        seconds: float,
        sample_rate: int,
    ) -> Tensor:
        t = torch.arange(int(seconds * sample_rate)) / sample_rate

        return torch.sin(2 * math.pi * frequency * t)


    # Whisper's sample rate, window, hop length and 80 channels, with the HTK filters. tone_channels holds the
    # loudest channel for tones of 200, 300, 6000 and 6100 Hz.
    filter_bank = mel_filters(201, 80, 16000)
    tone_channels = [loudest_channel(log_mel(tone(f, 1.0, 16000), 400, 160, filter_bank)) for f in (200, 300, 6000, 6100)]
    ''',
)

SOUND1_SOLUTION = source_file(
    '''
    # sound1.py, a solution for comparison

    import math

    import torch
    from torch import Tensor

    lows = torch.log10(torch.tensor([10.0, 1000.0]))
    points = torch.linspace(0, 10, 5)
    peaks = torch.clamp(torch.tensor([-1.0, 2.0]), min=0)


''' + indented(MEL_PART) + '''


''' + indented(TONE_PART) + '''


    filter_bank = mel_filters(201, 80, 16000)
    tone_channels = [loudest_channel(log_mel(tone(f, 1.0, 16000), 400, 160, filter_bank)) for f in (200, 300, 6000, 6100)]
    ''',
)


@functools.cache
def reference_tone_channels() -> list[int]:
    filters = REFERENCE["mel_filters"](N_FFT // 2 + 1, N_MELS, SAMPLE_RATE)

    return [
        REFERENCE["loudest_channel"](REFERENCE["log_mel"](REFERENCE["tone"](frequency, 1.0, SAMPLE_RATE), N_FFT, HOP, filters))
        for frequency in TONES
    ]


def check_tone_channels(value: object) -> None:
    expected = reference_tone_channels()

    if not isinstance(value, list):
        raise Mismatch(f"tone_channels is {value_text(value)}, but it should be a list of four channels.")

    if value != expected:
        raise Mismatch(f"tone_channels is {value_text(value)}, but it should be {expected}. Check mel_filters, log_mel and loudest_channel.")


def check_sound1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="hz_to_mel",
        description="converts hertz to mels",
        inputs=(
            (torch.tensor([0.0, 700.0, 1000.0, 8000.0]),),
            (torch.tensor([[100.0, 4000.0], [50.0, 300.0]]),),
        ),
        reference=REFERENCE["hz_to_mel"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="mel_to_hz",
        description="converts mels back to hertz",
        inputs=(
            (torch.tensor([0.0, 781.0, 2840.0]),),
            (torch.tensor([[150.0, 1500.0], [400.0, 2000.0]]),),
        ),
        reference=REFERENCE["mel_to_hz"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="mel_filters",
        description="builds triangles evenly spaced in mels",
        inputs=(
            (33, 6, 8000),
            (65, 12, 16000),
            (201, 20, 16000),
        ),
        reference=REFERENCE["mel_filters"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="log_mel",
        description="returns the base 10 log of each channel's power, one column per window",
        inputs=(
            (
                Described(noise(960, 800), "800 samples of noise"),
                64,
                32,
                Described(REFERENCE["mel_filters"](33, 8, 8000), "mel_filters(33, 8, 8000)"),
            ),
            (
                Described(two_tones(440.0, 2500.0, 1600, 8000), "tones of 440 and 2,500 Hz, 1,600 samples"),
                128,
                64,
                Described(REFERENCE["mel_filters"](65, 12, 8000), "mel_filters(65, 12, 8000)"),
            ),
        ),
        reference=REFERENCE["log_mel"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="loudest_channel",
        description="returns the channel with the highest average",
        inputs=(
            (torch.tensor([[1.0, 2.0], [4.0, 0.0], [2.0, 2.5]]),),
            (Described(REFERENCE["log_mel"](two_tones(2500.0, 440.0, 1600, 8000), 128, 64, REFERENCE["mel_filters"](65, 12, 8000)), "the log mel frames of a 2,500 Hz tone"),),
        ),
        reference=REFERENCE["loudest_channel"],
        expect=expect_value,
    )
    checker.check_name(
        name="tone_channels",
        description="is the loudest channel for each of the four tones",
        run=check_tone_channels,
    )


SOUND1 = Exercise(
    name="sound1",
    chapter=AUDIO,
    title="Log mel frames",
    lesson=SOUND1_LESSON,
    source=SOUND1_SOURCE,
    solution=SOUND1_SOLUTION,
    hints={
        "hz_to_mel": "Divide hertz by 700, add 1, take torch.log10 and multiply by 2595.",
        "mel_to_hz": "Undo the steps of hz_to_mel in reverse order. Divide by 2595, raise 10 to that power, subtract "
        "1, and multiply by 700.",
        "mel_filters": "points[:-2], points[1:-1] and points[2:] hold each filter's left edge, peak and right edge. The "
        "rising side is (frequency minus left) divided by (peak minus left), with the frequencies as a column so the "
        "result broadcasts to (n_freqs, n_mels). The falling side mirrors it. Take the smaller of the two sides, then "
        "clamp at 0.",
        "log_mel": "The power is the .abs() of the stft, squared, with shape (n_fft // 2 + 1, frames). Matrix multiply "
        "filters.T by the power to get (n_mels, frames). Then torch.clamp with min=1e-10, then .log10().",
        "loudest_channel": "The mean over dim=1 gives one number per channel. Return the argmax of those means, turned "
        "into an int.",
    },
    introduces=(),
    check=check_sound1,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# sound2
# ---------------------------------------------------------------------------

LEVELS = (1, 2, 4, 8)
ENTRIES = 64

SOUND2_LESSON = prose(
    """
    A neural codec such as SoundStream or EnCodec turns sound into codebook
    ids, so a model can write sound as well as read it. For audio at 24,000
    samples a second, the codec's encoder outputs 75 vectors a second.

    The quantizer from vq1 gives each vector one id from one codebook.
    SoundStream points out that 6,000 bits a second at 75 vectors a second is
    80 bits per vector. A single codebook would need `2 ** 80` entries to
    spend that many bits.

    A **residual quantizer** uses several small codebooks one after another.
    The first codebook picks the entry nearest to the vector. What is left
    over, the vector minus that entry, is called the residual. The second
    codebook picks the entry nearest to the residual, and so on. Each
    codebook's ids form one stream. Decoding adds up the chosen entries from
    all the codebooks.

    ```python
    z = torch.tensor([2.6])
    first = torch.tensor([3.0])    # the nearest entry in codebook 1
    z - first                       # tensor([-0.4000]), the residual that codebook 2 quantizes
    ```

    Every codebook of K entries adds `log2(K)` bits per vector. EnCodec's
    codebooks hold 1,024 entries, so each one adds 10 bits, and EnCodec uses
    up to 32 of them.

    SoundStream initializes its codebooks with k-means on its first batch,
    then keeps updating them during training. The file gives you k-means. It
    starts with entries picked at random from the data. Then it repeatedly
    moves each entry to the average of the vectors nearest to it. Here each
    codebook is fitted with k-means on the residuals that the codebooks
    before it left over.

    The vectors here are sound1's log mel frames of made up sounds, which
    mix tones, a rising chirp and a burst of noise. Each channel is
    standardized with the mean and standard deviation of the training frames.
    The codebooks have 64 entries each. They are fitted on 40 sounds and
    scored on 10 others, with 3 seeds. The relative error is the squared
    error that is left, divided by the squared size of the frames.

    | Codebooks | Bits per frame | Relative error |
    |---|---|---|
    | 1 | 6 | 18.7% to 21.8% |
    | 2 | 12 | 11.6% to 14.5% |
    | 4 | 24 | 8.2% to 9.3% |
    | 8 | 48 | 5.6% to 6.1% |

    The file fits its codebooks on 20 sounds so it runs fast, and its errors
    still fall inside each range. Every added codebook lowers the error. A
    single codebook for 48 bits would need `2 ** 48` entries, about 281
    trillion. SoundStream also trains with a random number of codebooks for
    each example, which it calls quantizer dropout. So one model works at
    every bitrate from 3 to 18 kilobits a second.
    """,
)

SOUND2_SOURCE = source_file(
    '''
    # sound2.py

    import math

    import torch
    from torch import Tensor

    rest = torch.tensor([2.6]) - torch.tensor([3.0])               # tensor([-0.4000])
    picked = torch.tensor([[1.0], [5.0]])[torch.tensor([1, 0, 1])]   # tensor([[5.], [1.], [5.]])
    streams = torch.stack([torch.tensor([3, 1]), torch.tensor([0, 2])], dim=1)    # tensor([[3, 0], [1, 2]])
    bits = math.log2(1024)                                          # 10.0


    # The log mel code from sound1.
''' + indented(MEL_PART) + '''


    # Made up sounds, with tones that start and stop, a rising chirp and a burst of noise.
''' + indented(SOUNDS_PART) + '''


    # nearest_codes from vq1, k-means, and relative_error, which divides the squared error by the squared size of z.
''' + indented(GIVEN_CODES_PART) + '''


    # TODO: Quantize z, of shape (n, d), with each codebook in turn. codebooks
    #   has shape (levels, K, d). The residual starts as z. At each level, find
    #   the nearest_codes of the residual in that level's codebook, then
    #   subtract the chosen entries from the residual. Return the ids, of shape
    #   (n, levels), and the quantized vectors, which are the sum of every
    #   level's chosen entries, of shape (n, d).
    def residual_quantize(
        z: Tensor,
        codebooks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return the vectors that ids stands for. ids has shape (n, levels),
    #   and column l holds indices into codebooks[l]. Each vector is the sum
    #   over the levels of the chosen entries, so the result has shape (n, d).
    def decode(
        ids: Tensor,
        codebooks: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return how many bits a second the ids take, as a plain number.
    #   There are `frames_per_second` frames a second. Each frame has one id
    #   from each of `codebooks` codebooks, and each id takes log2(entries)
    #   bits.
    #   Example: bits_per_second(75, 32, 1024) returns 24000.0.
    def bits_per_second(
        frames_per_second: float,
        codebooks: int,
        entries: int,
    ) -> float:
        ...


    # TODO: Fit `levels` codebooks of `entries` entries each, one after
    #   another. The residual starts as `vectors`. Each codebook comes from
    #   kmeans(..., entries, steps, generator) run on the residual. Before the
    #   next level, subtract each vector's nearest entry in the new codebook
    #   from the residual. Return the codebooks stacked, of shape
    #   (levels, entries, d).
    def fit_codebooks(
        vectors: Tensor,
        levels: int,
        entries: int,
        steps: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # Fits 8 codebooks on 20 made up sounds, then measures the relative error on 10 other sounds with the first 1,
    # 2, 4 and 8 codebooks.
    filter_bank = mel_filters(201, 80, 16000)
    sounds = torch.Generator().manual_seed(0)
    frames = torch.cat([log_mel(made_up_sound(1.0, 16000, sounds), 400, 160, filter_bank).T for _ in range(30)])
    frames = (frames - frames[:1960].mean(dim=0)) / frames[:1960].std(dim=0)
    codebooks = fit_codebooks(frames[:1960], 8, 64, 15, torch.Generator().manual_seed(1))
    errors = [relative_error(frames[1960:], residual_quantize(frames[1960:], codebooks[:levels])[1]) for levels in (1, 2, 4, 8)]
    ''',
)

SOUND2_SOLUTION = source_file(
    '''
    # sound2.py, a solution for comparison

    import math

    import torch
    from torch import Tensor

    rest = torch.tensor([2.6]) - torch.tensor([3.0])
    picked = torch.tensor([[1.0], [5.0]])[torch.tensor([1, 0, 1])]
    streams = torch.stack([torch.tensor([3, 1]), torch.tensor([0, 2])], dim=1)
    bits = math.log2(1024)


''' + indented(MEL_PART) + '''


''' + indented(SOUNDS_PART) + '''


''' + indented(GIVEN_CODES_PART) + '''


''' + indented(RVQ_PART) + '''


    filter_bank = mel_filters(201, 80, 16000)
    sounds = torch.Generator().manual_seed(0)
    frames = torch.cat([log_mel(made_up_sound(1.0, 16000, sounds), 400, 160, filter_bank).T for _ in range(30)])
    frames = (frames - frames[:1960].mean(dim=0)) / frames[:1960].std(dim=0)
    codebooks = fit_codebooks(frames[:1960], 8, 64, 15, torch.Generator().manual_seed(1))
    errors = [relative_error(frames[1960:], residual_quantize(frames[1960:], codebooks[:levels])[1]) for levels in (1, 2, 4, 8)]
    ''',
)

CODEBOOKS = torch.tensor(
    [
        [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]],
        [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]],
        [[0.25, 0.0], [0.0, 0.25], [-0.25, 0.0], [0.0, -0.25]],
    ],
)


def random_codebooks(
    seed: int,
    levels: int,
    entries: int,
    size: int,
) -> Tensor:
    spread = 0.5 ** torch.arange(levels).float()

    return torch.randn(levels, entries, size, generator=torch.Generator().manual_seed(seed)) * spread.view(-1, 1, 1)


def expect_ids_and_vectors(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise Mismatch(f"{call} should return two things, the ids and the quantized vectors, but it returns {value_text(actual)}.")

    ids, quantized = actual

    if isinstance(ids, Tensor) and ids.is_floating_point():
        raise Mismatch(f"The ids from {call} are floats. Ids index a codebook, so they must be integers, as nearest_codes returns them.")

    expect_tensor(
        call=f"the ids from {call}",
        actual=ids,
        expected=expected[0],
    )
    expect_tensor(
        call=f"the quantized vectors from {call}",
        actual=quantized,
        expected=expected[1],
    )


@functools.cache
def reference_errors() -> tuple[float, ...]:
    filters = REFERENCE["mel_filters"](N_FFT // 2 + 1, N_MELS, SAMPLE_RATE)
    sounds = torch.Generator().manual_seed(0)
    frames = torch.cat([REFERENCE["log_mel"](REFERENCE["made_up_sound"](1.0, SAMPLE_RATE, sounds), N_FFT, HOP, filters).T for _ in range(30)])
    frames = (frames - frames[:1960].mean(dim=0)) / frames[:1960].std(dim=0)
    codebooks = REFERENCE["fit_codebooks"](frames[:1960], 8, ENTRIES, 15, torch.Generator().manual_seed(1))

    return tuple(
        REFERENCE["relative_error"](frames[1960:], REFERENCE["residual_quantize"](frames[1960:], codebooks[:levels])[1])
        for levels in LEVELS
    )


def check_errors(value: object) -> None:
    expected = reference_errors()

    if not isinstance(value, list) or len(value) != len(expected):
        raise Mismatch(f"errors is {value_text(value)}, but it should be a list of {len(expected)} relative errors.")

    for levels, found, wanted in zip(LEVELS, value, expected):
        expect_close(
            call=f"the relative error with {levels} codebooks",
            actual=found,
            expected=wanted,
            tolerance=1e-5,
        )


def check_sound2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="residual_quantize",
        description="quantizes the residual each level leaves for the next",
        inputs=(
            (torch.tensor([[2.6, 0.4], [-1.7, -2.4], [0.1, 1.2], [2.2, -0.6]]), Described(CODEBOOKS, "three codebooks of 4 entries")),
            (Described(torch.randn(7, 3, generator=torch.Generator().manual_seed(961)), "7 random vectors of 3"), Described(random_codebooks(962, 2, 5, 3), "two random codebooks of 5 entries")),
        ),
        reference=REFERENCE["residual_quantize"],
        expect=expect_ids_and_vectors,
    )
    check_on_inputs(
        checker=checker,
        function="decode",
        description="adds up each level's chosen entry",
        inputs=(
            (torch.tensor([[0, 1, 2], [3, 3, 0]]), Described(CODEBOOKS, "three codebooks of 4 entries")),
            (torch.tensor([[4, 0], [1, 2], [2, 4]]), Described(random_codebooks(963, 2, 5, 3), "two random codebooks of 5 entries")),
        ),
        reference=REFERENCE["decode"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="bits_per_second",
        description="multiplies frames a second, codebooks and bits per codebook",
        inputs=(
            (75, 32, 1024),
            (75, 2, 1024),
            (50, 8, 64),
        ),
        reference=REFERENCE["bits_per_second"],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="fit_codebooks",
        description="fits each codebook on the residual the ones before it left",
        inputs=(
            (Described(torch.randn(40, 3, generator=torch.Generator().manual_seed(964)), "40 random vectors of 3"), 3, 4, 5, seeded(965)),
            (Described(torch.randn(30, 2, generator=torch.Generator().manual_seed(966)), "30 random vectors of 2"), 2, 6, 3, seeded(967)),
        ),
        reference=REFERENCE["fit_codebooks"],
        expect=expect_tensor,
    )
    checker.check_name(
        name="errors",
        description="is the relative error with 1, 2, 4 and 8 codebooks",
        run=check_errors,
    )


SOUND2 = Exercise(
    name="sound2",
    chapter=AUDIO,
    title="Sound as codebook ids",
    lesson=SOUND2_LESSON,
    source=SOUND2_SOURCE,
    solution=SOUND2_SOLUTION,
    hints={
        "residual_quantize": "Keep the residual in a variable that starts as z, and a running sum that starts as "
        "zeros. For each codebook, nearest_codes of the residual gives the ids. Add the chosen entries to the sum and "
        "subtract them from the residual. torch.stack the ids along dim=1.",
        "decode": "For each level, codebook[ids[:, level]] picks its entries. Stack them and sum over the levels.",
        "bits_per_second": "Each codebook adds math.log2(entries) bits to every frame. Multiply that by the number of "
        "codebooks and by the frames per second.",
        "fit_codebooks": "Loop levels times. kmeans on the residual gives a codebook. Before the next level, subtract "
        "each vector's nearest entry in that codebook from the residual. torch.stack the codebooks.",
    },
    introduces=(),
    check=check_sound2,
    time_limit=60,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_mel() -> dict[str, object]:
    """sound1's numbers, with Whisper's rate, window and hop and the HTK filters: each channel's width in hertz and
    each tone's loudest channel."""
    points = REFERENCE["mel_to_hz"](torch.linspace(0, REFERENCE["hz_to_mel"](torch.tensor(SAMPLE_RATE / 2)).item(), N_MELS + 2))
    filters = REFERENCE["mel_filters"](N_FFT // 2 + 1, N_MELS, SAMPLE_RATE)
    frames = REFERENCE["log_mel"](REFERENCE["tone"](440.0, 1.0, SAMPLE_RATE), N_FFT, HOP, filters)

    return {
        "first width": (points[2] - points[0]).item(),
        "last width": (points[-1] - points[-3]).item(),
        "tone channels": dict(zip(TONES, reference_tone_channels())),
        "frames in a second": frames.shape[1],
    }


def measure_residual(seeds: range = range(3)) -> dict[int, list[float]]:
    """sound2's table: the share of the frames left as error with 1, 2, 4 and 8 codebooks of 64 entries, fitted on
    40 made up sounds and scored on 10 others, for each seed, on 4 threads."""
    filters = REFERENCE["mel_filters"](N_FFT // 2 + 1, N_MELS, SAMPLE_RATE)
    results: dict[int, list[float]] = {}

    with pinned_threads():
        for seed in seeds:
            sounds = torch.Generator().manual_seed(100 + seed)
            frames = torch.cat([REFERENCE["log_mel"](REFERENCE["made_up_sound"](1.0, SAMPLE_RATE, sounds), N_FFT, HOP, filters).T for _ in range(50)])
            train, test = frames[: 40 * 98], frames[40 * 98 :]
            mean, spread = train.mean(dim=0), train.std(dim=0)
            train, test = (train - mean) / spread, (test - mean) / spread
            codebooks = REFERENCE["fit_codebooks"](train, max(LEVELS), ENTRIES, 15, torch.Generator().manual_seed(seed))
            results[seed] = [
                REFERENCE["relative_error"](test, REFERENCE["residual_quantize"](test, codebooks[:levels])[1])
                for levels in LEVELS
            ]

    return results
