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

        A second of sound is thousands of samples, far too many tokens for a
        transformer. This chapter turns sound into the two kinds of tokens
        audio models read. Log mel frames are what speech recognizers such as
        Whisper read, 100 a second, which Whisper's first layers turn into 50
        tokens. A residual quantizer gives a few streams of codebook ids,
        which neural codecs such as SoundStream and EnCodec make so a model
        can write sound as well as read it.

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
    Whisper, a speech recognizer, resamples all audio to 16,000 samples a
    second. It then reads an 80 channel log mel spectrogram, from windows of
    25 milliseconds taken every 10 milliseconds. That is stft1's spectrogram
    with two changes.

    - The **mel scale** spaces frequencies the way hearing does, fine at low pitches and coarse at high ones.
    - A log turns loudness, which spans many powers of 10, into a range a network handles well.

    The HTK formula maps a frequency of f hertz to m = 2595 log10(1 + f /
    700) mels. It runs near a straight line at low frequencies and bends into
    a log above. Its inverse is f = 700 (10 to the power m / 2595, minus 1).

    ```python
    torch.log10(torch.tensor([10.0, 1000.0]))    # tensor([1., 3.])
    ```

    A filter bank turns a spectrum's frequencies into channels. Place n_mels
    plus 2 points evenly in mels, from 0 to half the sample rate, then turn
    them back into hertz. Channel i is a triangle over the frequencies. It is
    0 at point i, rises to 1 at point i + 1, and falls to 0 at point i + 2.
    torchaudio's melscale_fbanks builds its default filters this way.

    ```python
    up = torch.tensor([-0.5, 0.5, 1.5])
    down = torch.tensor([1.5, 0.5, -0.5])
    torch.clamp(torch.minimum(up, down), min=0)    # tensor([0.0000, 0.5000, 0.0000])
    ```

    Whisper's own filters come from librosa, which uses the Slaney scale,
    straight below 1,000 Hz, and gives every triangle the same area. This
    file keeps Whisper's rate, window, hop and 80 channels with the HTK
    filters.

    Each window's power, the strength squared, times the filters gives one
    number per channel. Whisper's code floors it at 1e-10 and takes the log
    in base 10.

    With these filters, channel 0 spans 45 Hz and channel 79 spans 525 Hz.
    Here is the loudest channel for four pure tones, a second each.

    | Tone | Loudest channel |
    |---|---|
    | 200 Hz | 7 |
    | 300 Hz | 11 |
    | 6,000 Hz | 72 |
    | 6,100 Hz | 72 |

    100 Hz apart is four channels down low and the same channel up high, as
    for a listener. A second of sound becomes 98 frames of 80 numbers.
    Whisper pads both ends and drops the last frame, for exactly 100.

    Whisper's first layers are two convolutions, the second with a stride
    of 2. They turn the 100 frames a second into 50 token vectors, where the
    raw audio held 16,000 samples.
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


    # TODO: Return each frequency in hertz turned into mels, 2595 times the
    #   base 10 log of 1 plus hertz over 700.
    def hz_to_mel(hertz: Tensor) -> Tensor:
        ...


    # TODO: Return each number of mels turned back into hertz, the inverse of
    #   hz_to_mel.
    def mel_to_hz(mels: Tensor) -> Tensor:
        ...


    # TODO: Return the mel filters, of shape (n_freqs, n_mels). The STFT's
    #   frequencies are torch.linspace(0, sample_rate / 2, n_freqs). Place
    #   n_mels + 2 points evenly in mels from 0 to the mels of sample_rate / 2,
    #   and turn them back into hertz. Filter i rises from 0 at point i to 1 at
    #   point i + 1 and falls to 0 at point i + 2, and is 0 everywhere else.
    def mel_filters(
        n_freqs: int,
        n_mels: int,
        sample_rate: int,
    ) -> Tensor:
        ...


    # TODO: Return the log mel frames of signal, of shape (n_mels, frames).
    #   Take torch.stft with n_fft=n_fft, hop_length=hop, a Hann window of n_fft,
    #   center=False and return_complex=True. Square its strengths for the
    #   power, multiply by the filters, floor at 1e-10 and take the base 10 log.
    def log_mel(
        signal: Tensor,
        n_fft: int,
        hop: int,
        filters: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the channel whose average over the frames is highest, as a
    #   plain whole number.
    def loudest_channel(log_mels: Tensor) -> int:
        ...


    def tone(
        frequency: float,
        seconds: float,
        sample_rate: int,
    ) -> Tensor:
        t = torch.arange(int(seconds * sample_rate)) / sample_rate

        return torch.sin(2 * math.pi * frequency * t)


    # Whisper's rate, window, hop and 80 channels, with the HTK filters, and the loudest channel for tones of 200,
    # 300, 6000 and 6100 Hz.
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
        raise Mismatch(f"tone_channels is {value_text(value)}, but it should be {expected}. Check log_mel and loudest_channel.")


def check_sound1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="hz_to_mel",
        description="turns hertz into mels",
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
        description="turns mels back into hertz",
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
        description="gives the base 10 log of each channel's power, one column per window",
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
        description="gives the channel with the highest average",
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
        "mel_to_hz": "Undo each step of hz_to_mel in reverse: divide by 2595, raise 10 to that power, take away 1, "
        "multiply by 700.",
        "mel_filters": "points[:-2], points[1:-1] and points[2:] are each filter's left, top and right. The rising "
        "side is (freq minus left) over (top minus left), with the frequencies as a column so it broadcasts to (n_freqs, "
        "n_mels). The falling side mirrors it. Take the smaller of the two, then clamp at 0.",
        "log_mel": "The power is the stft's .abs() squared, of shape (n_fft // 2 + 1, frames). filters.T times it gives "
        "(n_mels, frames). torch.clamp with min=1e-10, then .log10().",
        "loudest_channel": "The mean over dim=1 gives one number per channel. Its argmax, turned into an int.",
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
    ids, so a model can write sound as well as read it. Its encoder gives 75
    vectors a second at 24,000 samples a second.

    vq1's quantizer gives each vector one id from one codebook. SoundStream
    works out that 6,000 bits a second at 75 vectors a second is 80 bits per
    vector. One codebook would need 2 to the 80 entries for that.

    A **residual quantizer** uses several small codebooks in a row. The first
    picks the nearest entry to the vector. The second picks the nearest entry
    to what is left over, and so on. Each codebook's ids form one stream, and
    decoding adds the chosen entries back up.

    ```python
    z = torch.tensor([2.6])
    first = torch.tensor([3.0])    # the nearest entry in codebook 1
    z - first                       # tensor([-0.4000]), what codebook 2 quantizes
    ```

    Every codebook of K entries adds log2(K) bits per vector. EnCodec's
    codebooks hold 1,024 entries, 10 bits each, and it uses up to 32 of them.

    SoundStream starts its codebooks with k-means on its first batch, then
    keeps updating them. k-means, given in the file, starts from vectors
    picked at random from the data. It then moves each entry to the average
    of the vectors nearest it, again and again. Here each codebook is fitted
    with k-means on what the codebooks before it left over.

    These are the log mel frames of sound1 for made up sounds: tones, a
    chirp, a burst of noise. Each channel is standardized with the training
    frames. The codebooks of 64 entries are fitted on 40 sounds and scored on
    10 others, on 3 seeds.

    | Codebooks | Bits per frame | Share of the frames left as error |
    |---|---|---|
    | 1 | 6 | 18.7% to 21.8% |
    | 2 | 12 | 11.6% to 14.5% |
    | 4 | 24 | 8.2% to 9.3% |
    | 8 | 48 | 5.6% to 6.1% |

    The file fits on 20 sounds so it runs fast, and its errors land inside
    each range. Every codebook added cuts what is left. One codebook for 48
    bits would need 2 to the 48 entries, about 281 trillion. SoundStream also
    trains with a random number of codebooks per example, its quantizer
    dropout. So one model serves every bitrate from 3 to 18 kilobits a second.
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


    # sound1's log mel frames.
''' + indented(MEL_PART) + '''


    # Made up sounds: tones that start and stop, a chirp, a burst of noise.
''' + indented(SOUNDS_PART) + '''


    # vq1's nearest code, k-means, and the error left as a share of the frames.
''' + indented(GIVEN_CODES_PART) + '''


    # TODO: Quantize z, of shape (n, d), with each codebook in turn. codebooks
    #   has shape (levels, K, d). Each level picks the nearest_codes of what is
    #   left, then takes those entries away from it. Return the ids, of shape
    #   (n, levels), and the sum of every level's entries, of shape (n, d).
    def residual_quantize(
        z: Tensor,
        codebooks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return the vectors that ids, of shape (n, levels), stand for, the
    #   sum over the levels of each codebook's chosen entries.
    def decode(
        ids: Tensor,
        codebooks: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the bits a second that `codebooks` codebooks of `entries`
    #   entries each send at `frames_per_second`, as a plain number.
    def bits_per_second(
        frames_per_second: float,
        codebooks: int,
        entries: int,
    ) -> float:
        ...


    # TODO: Fit `levels` codebooks of `entries` entries each, one after
    #   another. Each comes from kmeans(..., entries, steps, generator) on what
    #   the ones before left of `vectors`. Return them stacked, of shape
    #   (levels, entries, d).
    def fit_codebooks(
        vectors: Tensor,
        levels: int,
        entries: int,
        steps: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # Codebooks fitted on 20 made up sounds, then the error left on 10 others with 1, 2, 4 and 8 of them.
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
        raise Mismatch(f"{call} should give two things, the ids and the quantized vectors, but it gives {value_text(actual)}.")

    ids, quantized = actual

    if isinstance(ids, Tensor) and ids.is_floating_point():
        raise Mismatch(f"The ids from {call} hold decimals. Ids index a codebook, so they are whole numbers, as nearest_codes gives them.")

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
        raise Mismatch(f"errors is {value_text(value)}, but it should be a list of {len(expected)} shares.")

    for levels, found, wanted in zip(LEVELS, value, expected):
        expect_close(
            call=f"the error left with {levels} codebooks",
            actual=found,
            expected=wanted,
            tolerance=1e-5,
        )


def check_sound2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="residual_quantize",
        description="quantizes what each level leaves for the next",
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
        description="fits each codebook on what the ones before left over",
        inputs=(
            (Described(torch.randn(40, 3, generator=torch.Generator().manual_seed(964)), "40 random vectors of 3"), 3, 4, 5, seeded(965)),
            (Described(torch.randn(30, 2, generator=torch.Generator().manual_seed(966)), "30 random vectors of 2"), 2, 6, 3, seeded(967)),
        ),
        reference=REFERENCE["fit_codebooks"],
        expect=expect_tensor,
    )
    checker.check_name(
        name="errors",
        description="is the error left with 1, 2, 4 and 8 codebooks",
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
        "residual_quantize": "Keep what is left in a variable that starts as z, and a running sum that starts as "
        "zeros. For each codebook, nearest_codes of what is left gives the ids. Add those entries to the sum, take them "
        "from what is left. torch.stack the ids along dim=1.",
        "decode": "For each level, codebook[ids[:, level]] picks its entries. Stack them and sum over the levels.",
        "bits_per_second": "Each codebook adds math.log2(entries) bits per frame.",
        "fit_codebooks": "Loop levels times. kmeans on what is left gives a codebook. Take each vector's nearest entry "
        "away from what is left before the next level. torch.stack the codebooks.",
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
