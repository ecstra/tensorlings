"""98_echo_canceller, Section 13's first project, an echo canceller for spoken digits: the classic NLMS filter
against a filter per frequency on STFT frames, designed with 95_design_method's steps, and where each one wins."""

from __future__ import annotations

import contextlib
import functools
import math
import textwrap
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import load_spoken_digits
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
    reject_missing_return,
    seeded,
    value_text,
)
from ..text import prose, source_file
from .linear_layers import expect_module

ECHO_CANCELLER = Chapter(
    folder="98_echo_canceller",
    title="Project: an echo canceller",
    readme=source_file(
        """
        # Project: an echo canceller

        Section 13's first project. On a call, the far end's voice plays from
        your speaker, bounces around the room, and reaches your microphone. An
        echo canceller guesses that echo and takes it out. You build the room
        and its echo from real spoken digits, run the classic NLMS filter, then
        design a filter per frequency on STFT frames with the design method.
        Last, you measure where each one wins.

        The recordings come from the Free Spoken Digit Dataset, shared under
        CC BY-SA 4.0. Download them once with `.venv/bin/python run.py data
        spoken_digits`.

        ## Further reading

        - [Wikipedia, echo suppression and cancellation](https://en.wikipedia.org/wiki/Echo_suppression_and_cancellation)
        - [Wikipedia, least mean squares filter and NLMS](https://en.wikipedia.org/wiki/Least_mean_squares_filter)
        - [Avargel and Cohen, the multiplicative transfer function in the STFT domain](https://webee.technion.ac.il/Sites/People/IsraelCohen/Publications/SPL_May2007.pdf)
        - [Zhang and colleagues, a linear canceller followed by a neural network](https://arxiv.org/abs/2202.06850)
        - [Jakobovski, the Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
        """,
    ),
)

RATE = 8000
N_FFT = 256
HOP = 64
TAPS = 256
STEP = 0.5
EPS = 1e-3
FRAMES_BACK = 8
FIT_STEPS = 300
FIT_RATE = 0.05
FAR_SPEAKERS = ["george", "jackson"]
NEAR_SPEAKERS = ["theo"]
CASES = ("single talk", "double talk", "room changes", "long tail")


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


SETTINGS_PART = '''
RATE = 8000     # samples a second
N_FFT = 256     # samples in each STFT window, 32 milliseconds
HOP = 64        # a window starts every 64 samples, so neighbors overlap by three quarters
'''

SPEECH_PART = '''
# Spoken digits joined end to end until the clip is long enough, each one scaled to the same loudness.
def speech_clip(
    speakers: list[str],
    seconds: float,
    generator: torch.Generator,
) -> Tensor:
    recordings = load_spoken_digits()
    pool = [recording for speaker in speakers for recording in recordings[speaker]]
    pieces = []
    total = 0

    while total < seconds * RATE:
        recording = pool[torch.randint(0, len(pool), (1,), generator=generator).item()]
        pieces.append(recording / recording.pow(2).mean().sqrt())
        total += len(recording)

    return torch.cat(pieces)[: int(seconds * RATE)]
'''

ROOM_SOLUTION = '''
def room_response(
    length: int,
    decay: float,
    generator: torch.Generator,
) -> Tensor:
    t = torch.arange(length, dtype=torch.float32)
    response = torch.randn(length, generator=generator) * torch.exp(-t / decay)

    return response / response.norm()


def echo_of(
    far: Tensor,
    response: Tensor,
) -> Tensor:
    length = len(far) + len(response) - 1
    spectrum = torch.fft.rfft(far, n=length) * torch.fft.rfft(response, n=length)

    return torch.fft.irfft(spectrum, n=length)[: len(far)]


def microphone(
    echo: Tensor,
    near: Tensor,
    noise_level: float,
    generator: torch.Generator,
) -> Tensor:
    return echo + near + noise_level * torch.randn(len(echo), generator=generator)
'''

FRAMES_SOLUTION = '''
def frames_of(signal: Tensor) -> Tensor:
    window = torch.hann_window(N_FFT, dtype=signal.dtype)

    return torch.stft(signal, n_fft=N_FFT, hop_length=HOP, window=window, pad_mode="constant", return_complex=True)


def room_spectrum(response: Tensor) -> Tensor:
    return torch.fft.rfft(response, n=N_FFT)


def product_error(
    echo_frames: Tensor,
    far_frames: Tensor,
    spectrum: Tensor,
) -> float:
    miss = echo_frames - spectrum.unsqueeze(1) * far_frames

    return 10 * math.log10(miss.abs().pow(2).sum().item() / echo_frames.abs().pow(2).sum().item())
'''

NLMS_SOLUTION = '''
def nlms_step(
    weights: Tensor,
    recent: Tensor,
    mic_sample: Tensor,
    step: float,
    eps: float,
) -> tuple[Tensor, Tensor]:
    error = mic_sample - weights @ recent

    return weights + step * error * recent / (recent @ recent + eps), error


def nlms(
    far: Tensor,
    mic: Tensor,
    taps: int,
    step: float,
    eps: float,
) -> tuple[Tensor, Tensor]:
    weights = torch.zeros(taps, dtype=far.dtype)
    padded = torch.cat([torch.zeros(taps - 1, dtype=far.dtype), far])
    residual = torch.zeros(len(mic), dtype=far.dtype)

    for n in range(len(mic)):
        recent = padded[n : n + taps].flip(0)
        weights, residual[n] = nlms_step(weights, recent, mic[n], step, eps)

    return residual, weights


def misalignment_db(
    weights: Tensor,
    response: Tensor,
) -> float:
    length = max(len(weights), len(response))
    gap = F.pad(weights, (0, length - len(weights))) - F.pad(response, (0, length - len(response)))

    return 20 * math.log10(gap.norm().item() / response.norm().item())
'''

FILTER_INIT = '''
class SubbandFilter(nn.Module):
    def __init__(
        self,
        frequencies: int,
        taps: int,
    ) -> None:
        super().__init__()
        self.taps = taps
        # Each frequency holds taps complex weights, kept as a real part and an imaginary part. All start at 0.
        self.real = nn.Parameter(torch.zeros(frequencies, taps))
        self.imag = nn.Parameter(torch.zeros(frequencies, taps))
'''

FILTER_FORWARD = '''
    def forward(self, far_frames: Tensor) -> Tensor:
        weights = torch.complex(self.real, self.imag)
        count = far_frames.shape[1]
        padded = F.pad(far_frames, (self.taps - 1, 0))
        estimate = torch.zeros_like(far_frames)

        for k in range(self.taps):
            start = self.taps - 1 - k
            estimate = estimate + weights[:, k : k + 1] * padded[:, start : start + count]

        return estimate
'''

FIT_SOLUTION = '''
def fit_filter(
    model: nn.Module,
    far_frames: Tensor,
    mic_frames: Tensor,
    steps: int,
    lr: float,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        loss = (mic_frames - model(far_frames)).abs().pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def cancel(
    model: nn.Module,
    far: Tensor,
    mic: Tensor,
) -> Tensor:
    with torch.no_grad():
        estimate = model(frames_of(far))

    window = torch.hann_window(N_FFT, dtype=far.dtype)

    return mic - torch.istft(estimate, n_fft=N_FFT, hop_length=HOP, window=window, length=len(mic))
'''

PROPERTY_SOLUTION = '''
def is_linear(
    model: nn.Module,
    first: Tensor,
    second: Tensor,
) -> bool:
    with torch.no_grad():
        together = model(2 * first + second)
        apart = 2 * model(first) + model(second)

    return torch.allclose(together, apart, atol=1e-4)


def is_causal(
    model: nn.Module,
    far_frames: Tensor,
    cut: int,
) -> bool:
    changed = far_frames.clone()
    changed[:, cut:] = 0

    with torch.no_grad():
        return torch.allclose(model(far_frames)[:, :cut], model(changed)[:, :cut], atol=1e-6)
'''

CASES_PART = '''
CASES = ("single talk", "double talk", "room changes", "long tail")


# Eight seconds of far end speech in one of the four cases, its echo, and what the microphone hears.
def case(
    name: str,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    far = speech_clip(["george", "jackson"], 8.0, generator)
    length, decay = (1024, 200.0) if name == "long tail" else (256, 40.0)
    echo = echo_of(far, room_response(length, decay, generator))

    # From halfway on, the echo comes through another room, as when someone moves the laptop.
    if name == "room changes":
        moved = echo_of(far, room_response(length, decay, generator))
        echo = torch.cat([echo[: len(far) // 2], moved[len(far) // 2 :]])

    near = 0.5 * speech_clip(["theo"], 8.0, generator) if name == "double talk" else torch.zeros(len(far))

    return far, echo, microphone(echo, near, 0.01, generator)
'''

SCORE_SOLUTION = '''
def erle_db(
    echo: Tensor,
    residual_echo: Tensor,
) -> float:
    return 10 * math.log10(echo.pow(2).sum().item() / residual_echo.pow(2).sum().item())


def score_both(
    far: Tensor,
    echo: Tensor,
    mic: Tensor,
) -> tuple[float, float]:
    half = len(far) // 2
    residual, _ = nlms(far, mic, 256, 0.5, 1e-3)
    nlms_score = erle_db(echo[half:], (echo - (mic - residual))[half:])

    model = SubbandFilter(N_FFT // 2 + 1, 8)
    fit_filter(model, frames_of(far[:half]), frames_of(mic[:half]), 300, 0.05)
    learned = mic - cancel(model, far, mic)
    learned_score = erle_db(echo[half:], (echo - learned)[half:])

    return nlms_score, learned_score
'''


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "load_spoken_digits": load_spoken_digits,
    }
    exec(code, namespace)

    return namespace


ALL_PARTS = (
    SETTINGS_PART
    + SPEECH_PART
    + ROOM_SOLUTION
    + FRAMES_SOLUTION
    + NLMS_SOLUTION
    + FILTER_INIT
    + FILTER_FORWARD
    + FIT_SOLUTION
    + PROPERTY_SOLUTION
    + CASES_PART
    + SCORE_SOLUTION
)
REFERENCE = namespace_of(ALL_PARTS)

FILE_HEADER = '''
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_spoken_digits
'''


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """CPU results can move with the number of threads, so the runs behind the lessons pin it."""
    before = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        yield
    finally:
        torch.set_num_threads(before)


def noise(
    seed: int,
    length: int,
) -> Tensor:
    return torch.randn(length, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def colored_noise(
    seed: int,
    length: int,
) -> Tensor:
    """Noise smoothed by a short average, so most of its energy sits at low frequencies, as speech's does."""
    white = noise(seed, length + 7)

    return white.unfold(0, 8, 1).mean(dim=1)[:length] * 2.8


def complex_frames(
    seed: int,
    frequencies: int,
    count: int,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(frequencies, count, generator=generator)
    imag = torch.randn(frequencies, count, generator=generator)

    return torch.complex(real, imag)


# ---------------------------------------------------------------------------
# echo1
# ---------------------------------------------------------------------------

ECHO1_LESSON = prose(
    """
    On a call, the far end's voice plays from your speaker, bounces around
    the room, and reaches your microphone a moment later. So the far end
    hears itself. An echo canceller guesses that echo and takes it out of
    the microphone. This project builds two and compares them.

    A room acts on sound as a convolution. Its response is what the
    microphone hears after a single click from the speaker, a quick arrival
    and then a fading tail. Here a room is noise that fades away, a stand in
    for a measured one, scaled to a length of 1. The echo is the far end's
    speech convolved with it, which fft1's spectra compute fast.

    The microphone also hears the near end, the person in the room, and a
    little noise. The canceller knows only the far end's speech and the
    microphone. Whatever it does, the near end's voice has to come through.

    The far end is real speech, spoken digits from the Free Spoken Digit
    Dataset at 8,000 samples a second. Download them once with `.venv/bin/python
    run.py data spoken_digits`.

    stft1 cut a signal into windows. Here each window holds 256 samples, and
    a new one starts every 64, so neighbors overlap by three quarters.
    `center=True` adds zeros at both ends, so the first window centers on the
    first sample, and `pad_mode="constant"` makes those added samples zeros.

    fft1's convolution theorem turns a convolution into a product of spectra.
    On frames it holds only roughly. When the room is far shorter than a
    window, each echo frame is about the room's spectrum times the far frame,
    frequency by frequency. A longer room spills from one window into the
    next, and the product misses.

    | Room | The product's miss, 4 seconds of speech, 3 seeds |
    |---|---|
    | 64 samples, 8 milliseconds | -18.2 to -17.4 dB |
    | 256 samples, 32 milliseconds | -9.8 to -5.9 dB |
    | 1,024 samples, 128 milliseconds | -1.4 to +0.2 dB |

    The miss is in **decibels**, 10 times the log of an energy ratio. So -20
    dB is a hundredth of the echo's energy, and 0 dB is all of it. On the
    long room the product misses about as much as the echo holds.

    Real rooms ring for a tenth of a second and more. So echo3's design has
    to reach past a single frame. Avargel and Cohen studied this product and
    when it holds, under the name multiplicative transfer function.
    """,
)

ECHO1_NAMES = '''
    generator = torch.Generator().manual_seed(0)
    far = speech_clip(["george", "jackson"], 4.0, generator)
    short_room = room_response(64, 10.0, generator)
    long_room = room_response(1024, 200.0, generator)
    short_error = product_error(frames_of(echo_of(far, short_room)), frames_of(far), room_spectrum(short_room))
    long_error = product_error(frames_of(echo_of(far, long_room)), frames_of(far), room_spectrum(long_room))
'''

ECHO1_SOURCE = source_file(
    """
    # echo1.py
    """
    + FILE_HEADER
    + """
    t = torch.arange(4.0)
    fading = torch.exp(-t / 2.0)                           # tensor([1.0000, 0.6065, 0.3679, 0.2231])
    unit = torch.tensor([3.0, 4.0]) / torch.tensor([3.0, 4.0]).norm()    # tensor([0.6000, 0.8000])
    product = torch.fft.rfft(torch.tensor([1.0, 2.0]), n=4)              # tensor([3.+0.j, 1.-2.j, -1.+0.j])
    decibels = 10 * math.log10(0.01)                       # -20.0, a hundred times less energy


    # The recordings, joined into speech. Run .venv/bin/python run.py data spoken_digits once first.
"""
    + indented(SETTINGS_PART)
    + "\n\n"
    + indented(SPEECH_PART)
    + '''


    # TODO: Return a room's response, length samples of noise drawn with
    #   torch.randn(length, generator=generator), each sample times
    #   exp(-t / decay) at its time t from 0, then scaled to a length of 1
    #   with .norm(), as unit is above.
    def room_response(
        length: int,
        decay: float,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return far convolved with response, the echo, as many samples long
    #   as far. Use fft1's spectra: rfft both with n=len(far) + len(response) - 1,
    #   multiply, irfft with the same n, and keep the first len(far) samples.
    def echo_of(
        far: Tensor,
        response: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return what the microphone hears: echo plus near plus noise_level
    #   times torch.randn(len(echo), generator=generator).
    def microphone(
        echo: Tensor,
        near: Tensor,
        noise_level: float,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the STFT frames of signal from torch.stft, with n_fft=N_FFT,
    #   hop_length=HOP, a Hann window of N_FFT in signal's dtype,
    #   pad_mode="constant" and return_complex=True. Leave center at its
    #   default, True. The shape is (N_FFT // 2 + 1, frames).
    def frames_of(signal: Tensor) -> Tensor:
        ...


    # TODO: Return the room's spectrum at the STFT's frequencies, the rfft of
    #   response with n=N_FFT.
    def room_spectrum(response: Tensor) -> Tensor:
        ...


    # TODO: Return how far the echo's frames are from spectrum times the far
    #   frames, frequency by frequency, in decibels: 10 * log10 of the energy
    #   missed over the energy of the echo's frames, as a plain number. The
    #   energy of complex numbers is the sum of .abs() squared.
    def product_error(
        echo_frames: Tensor,
        far_frames: Tensor,
        spectrum: Tensor,
    ) -> float:
        ...


    # Your own measurement, a room of 8 milliseconds and one of 128.
'''
    + ECHO1_NAMES,
)

ECHO1_SOLUTION = source_file(
    """
    # echo1.py, a solution for comparison
    """
    + FILE_HEADER
    + """
    t = torch.arange(4.0)
    fading = torch.exp(-t / 2.0)
    unit = torch.tensor([3.0, 4.0]) / torch.tensor([3.0, 4.0]).norm()
    product = torch.fft.rfft(torch.tensor([1.0, 2.0]), n=4)
    decibels = 10 * math.log10(0.01)


"""
    + indented(SETTINGS_PART)
    + "\n\n"
    + indented(SPEECH_PART)
    + "\n\n"
    + indented(ROOM_SOLUTION)
    + "\n\n"
    + indented(FRAMES_SOLUTION)
    + "\n\n"
    + ECHO1_NAMES,
)


def product_case(
    seed: int,
    frequencies: int,
    count: int,
) -> tuple[Described, Described, Described]:
    far = complex_frames(seed, frequencies, count).to(torch.complex128)
    spectrum = complex_frames(seed + 1, frequencies, 1).squeeze(1).to(torch.complex128)
    echo = spectrum.unsqueeze(1) * far + 0.3 * complex_frames(seed + 2, frequencies, count).to(torch.complex128)

    return (
        Described(echo, f"echo frames of shape ({frequencies}, {count})"),
        Described(far, f"far frames of shape ({frequencies}, {count})"),
        Described(spectrum, f"a spectrum of {frequencies} frequencies"),
    )


def reference_names() -> dict[str, float]:
    namespace = dict(REFERENCE)
    exec(textwrap.dedent(ECHO1_NAMES), namespace)

    return {name: namespace[name] for name in ("short_error", "long_error")}


def check_named_error(name: str) -> Callable[[object], None]:
    def run(value: object) -> None:
        expect_close(
            call=name,
            actual=value,
            expected=reference_names()[name],
            tolerance=1e-3,
        )

    return run


def expect_complex(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if isinstance(actual, Tensor) and not actual.is_complex():
        raise Mismatch(f"{call} holds real numbers, but frames and spectra hold complex ones.")

    expect_tensor(
        call=call,
        actual=torch.view_as_real(actual) if isinstance(actual, Tensor) else actual,
        expected=torch.view_as_real(expected),
    )


def check_echo1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="room_response",
        description="fades noise away and scales it to a length of 1",
        inputs=(
            (6, 2.0, seeded(980)),
            (10, 4.0, seeded(981)),
        ),
        reference=REFERENCE["room_response"],
    )
    check_on_inputs(
        checker=checker,
        function="echo_of",
        description="convolves the far end with the room, as long as the far end",
        inputs=(
            (Described(noise(982, 12), "12 samples of noise"), Described(noise(983, 3), "a room of 3 samples")),
            (Described(noise(984, 40), "40 samples of noise"), Described(noise(985, 9), "a room of 9 samples")),
        ),
        reference=REFERENCE["echo_of"],
    )
    check_on_inputs(
        checker=checker,
        function="microphone",
        description="adds the echo, the near end and noise",
        inputs=(
            (Described(noise(986, 5), "an echo of 5"), Described(noise(987, 5), "a near end of 5"), 0.1, seeded(988)),
            (Described(noise(989, 7), "an echo of 7"), Described(torch.zeros(7, dtype=torch.float64), "silence"), 0.5, seeded(990)),
        ),
        reference=REFERENCE["microphone"],
    )
    check_on_inputs(
        checker=checker,
        function="frames_of",
        description="gives the STFT frames, one column per window",
        inputs=(
            (Described(noise(991, 600), "600 samples of noise"),),
            (Described(noise(992, 1000), "1,000 samples of noise"),),
        ),
        reference=REFERENCE["frames_of"],
        expect=expect_complex,
    )
    check_on_inputs(
        checker=checker,
        function="room_spectrum",
        description="gives the room's spectrum at the STFT's frequencies",
        inputs=(
            (Described(noise(993, 64), "a room of 64 samples"),),
            (Described(noise(994, 300), "a room of 300 samples"),),
        ),
        reference=REFERENCE["room_spectrum"],
        expect=expect_complex,
    )
    check_on_inputs(
        checker=checker,
        function="product_error",
        description="measures the miss of a product per frequency in decibels",
        inputs=(
            product_case(995, 4, 6),
            product_case(996, 7, 9),
        ),
        reference=REFERENCE["product_error"],
        expect=expect_close,
    )

    for name in ("short_error", "long_error"):
        checker.check_name(
            name=name,
            description="your own measurement on spoken digits",
            run=check_named_error(name),
        )


ECHO1 = Exercise(
    name="echo1",
    chapter=ECHO_CANCELLER,
    title="Project: a room and its echo",
    lesson=ECHO1_LESSON,
    source=ECHO1_SOURCE,
    solution=ECHO1_SOLUTION,
    hints={
        "room_response": "t is torch.arange(length) as floats. Multiply torch.randn(length, generator=generator) by "
        "torch.exp(-t / decay), then divide the result by its own .norm().",
        "echo_of": "With length = len(far) + len(response) - 1, multiply torch.fft.rfft(far, n=length) by the same of "
        "response, then torch.fft.irfft(..., n=length) and slice [: len(far)].",
        "microphone": "Add the three: echo, near, and noise_level times the randn draw.",
        "frames_of": "Make the window with torch.hann_window(N_FFT, dtype=signal.dtype), then call torch.stft with the "
        "arguments the TODO names.",
        "room_spectrum": "torch.fft.rfft with n=N_FFT.",
        "product_error": "The miss is echo_frames minus spectrum.unsqueeze(1) times far_frames. Take the sum of "
        ".abs().pow(2) of each, divide, and give 10 * math.log10 of the plain number.",
        "short_error": "This line runs your functions on spoken digits. If it crashed on the data, download it first.",
        "long_error": "This line runs your functions on spoken digits. If it crashed on the data, download it first.",
    },
    introduces=(),
    check=check_echo1,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# echo2
# ---------------------------------------------------------------------------

ECHO2_LESSON = prose(
    """
    The classic canceller copies the room. It keeps a guess of the room's
    response as weights, one per recent far sample, and guesses the echo as
    their weighted sum. Its error is the microphone minus that guess, which
    is also what the far end gets to hear.

    After every sample it nudges the weights. LMS, the least mean squares
    filter of Widrow and Hoff, moves them by the error times the recent far
    samples. That is one step of gradient descent on that sample's squared
    error. NLMS, the normalized version, divides the step by the recent
    samples' energy, so loud speech and quiet speech learn at one pace.

    ```text
    error = mic sample - weights @ recent
    weights = weights + step * error * recent / (recent @ recent + eps)
    ```

    Any step between 0 and 2 keeps NLMS stable, and eps keeps a silent
    stretch from dividing by 0. Here the step is 0.5, and 256 weights cover 32
    milliseconds of room.

    The misalignment says how far the weights are from the true room, in
    decibels. Here is NLMS on a room of 256 samples, with a little noise at
    the microphone, over three seeds.

    | Far end | Below -10 dB | Below -20 dB | After 1 second | After 4 seconds |
    |---|---|---|---|---|
    | Noise | at 0.05 seconds | at 0.15 seconds | -44.2 to -44.0 dB | -45.0 to -44.6 dB |
    | Speech | at 1.1 to 1.95 s | not in 4 seconds | -9.4 to -6.7 dB | -19.6 to -17.9 dB |

    Noise teaches it the room in a tenth of a second, and speech takes
    seconds. LMS learns each direction of its input at a pace set by the
    energy the input puts there. White noise spreads its energy evenly, so
    every part of the room is learned at once.

    Speech doesn't. Below 1,000 Hz, a quarter of its frequencies, sit 81 to
    88 percent of these clips' energy, against 24 to 25 percent for noise. The
    weights that matter for high frequencies barely move. Speech also pauses,
    and a silent far end teaches nothing.

    Each sample costs 256 multiplications for the guess and 256 more for the
    update, which any phone can afford. echo4 shows what breaks it.
    """,
)

ECHO2_NAMES = '''
    generator = torch.Generator().manual_seed(0)
    room = room_response(256, 40.0, generator)
    noise = torch.randn(RATE, generator=generator)
    speech = speech_clip(["george", "jackson"], 1.0, generator)
    _, noise_weights = nlms(noise, echo_of(noise, room), 256, 0.5, 1e-3)
    _, speech_weights = nlms(speech, echo_of(speech, room), 256, 0.5, 1e-3)
    after_noise = misalignment_db(noise_weights, room)
    after_speech = misalignment_db(speech_weights, room)
'''

ECHO2_SOURCE = source_file(
    """
    # echo2.py
    """
    + FILE_HEADER
    + """
    weights = torch.tensor([0.5, 0.0])
    recent = torch.tensor([1.0, 2.0])                       # the newest far sample first
    guess = weights @ recent                                # tensor(0.5000), the echo NLMS expects
    nudge = 0.5 * (1.5 - guess) * recent / (recent @ recent + 1e-3)    # tensor([0.1000, 0.2000])
    newest_first = torch.arange(5.0)[1:4].flip(0)           # tensor([3., 2., 1.])


    # The settings, the speech and echo1's room.
"""
    + indented(SETTINGS_PART)
    + "\n\n"
    + indented(SPEECH_PART)
    + "\n\n"
    + indented(ROOM_SOLUTION)
    + '''


    # TODO: Take one NLMS step. The error is mic_sample minus weights @ recent.
    #   Return the new weights, weights plus step times the error times recent
    #   over (recent @ recent + eps), and the error.
    def nlms_step(
        weights: Tensor,
        recent: Tensor,
        mic_sample: Tensor,
        step: float,
        eps: float,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Run NLMS along mic, one sample at a time. weights start as
    #   torch.zeros(taps, dtype=far.dtype). At sample n, recent holds far's
    #   last taps samples up to n, the newest first, with zeros before far
    #   starts. Keep each step's error. Return the errors, the residual the
    #   far end hears, and the last weights.
    def nlms(
        far: Tensor,
        mic: Tensor,
        taps: int,
        step: float,
        eps: float,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return how far weights are from response in decibels, 20 * log10
    #   of the length of their difference over the length of response, as a
    #   plain number. Pad the shorter one with zeros at its end first.
    def misalignment_db(
        weights: Tensor,
        response: Tensor,
    ) -> float:
        ...


    # Your own run, one second of noise and one of speech through the same room.
'''
    + ECHO2_NAMES,
)

ECHO2_SOLUTION = source_file(
    """
    # echo2.py, a solution for comparison
    """
    + FILE_HEADER
    + """
    weights = torch.tensor([0.5, 0.0])
    recent = torch.tensor([1.0, 2.0])
    guess = weights @ recent
    nudge = 0.5 * (1.5 - guess) * recent / (recent @ recent + 1e-3)
    newest_first = torch.arange(5.0)[1:4].flip(0)


"""
    + indented(SETTINGS_PART)
    + "\n\n"
    + indented(SPEECH_PART)
    + "\n\n"
    + indented(ROOM_SOLUTION)
    + "\n\n"
    + indented(NLMS_SOLUTION)
    + "\n\n"
    + ECHO2_NAMES,
)


def nlms_case(
    seed: int,
    length: int,
    room: int,
) -> tuple[Described, Described]:
    far = noise(seed, length)
    response = noise(seed + 1, room) * torch.exp(-torch.arange(room, dtype=torch.float64) / 2)
    mic = REFERENCE["echo_of"](far, response) + 0.01 * noise(seed + 2, length)

    return Described(far, f"{length} samples of noise"), Described(mic, f"its echo through a room of {room}")


def reference_echo2_names() -> dict[str, float]:
    namespace = dict(REFERENCE)
    exec(textwrap.dedent(ECHO2_NAMES), namespace)

    return {name: namespace[name] for name in ("after_noise", "after_speech")}


def check_echo2_name(name: str) -> Callable[[object], None]:
    def run(value: object) -> None:
        expect_close(
            call=name,
            actual=value,
            expected=reference_echo2_names()[name],
            tolerance=0.5,
        )

    return run


def check_echo2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="nlms_step",
        description="nudges the weights by the error, scaled by the far end's power",
        inputs=(
            (
                Described(noise(1000, 4), "4 weights"),
                Described(noise(1001, 4), "4 recent samples"),
                Described(torch.tensor(0.7, dtype=torch.float64), "tensor(0.7)"),
                0.5,
                1e-3,
            ),
            (
                Described(noise(1002, 6), "6 weights"),
                Described(noise(1003, 6), "6 recent samples"),
                Described(torch.tensor(-1.3, dtype=torch.float64), "tensor(-1.3)"),
                1.0,
                0.1,
            ),
        ),
        reference=REFERENCE["nlms_step"],
        expect=expect_parts("the new weights", "the error"),
    )
    check_on_inputs(
        checker=checker,
        function="nlms",
        description="runs NLMS along the microphone, newest far sample first",
        inputs=(
            (*nlms_case(1004, 60, 5), 8, 0.5, 1e-3),
            (*nlms_case(1007, 90, 3), 4, 1.0, 1e-2),
        ),
        reference=REFERENCE["nlms"],
        expect=expect_parts("the residual", "the last weights"),
    )
    check_on_inputs(
        checker=checker,
        function="misalignment_db",
        description="compares the weights with the room in decibels",
        inputs=(
            (Described(noise(1010, 3), "3 weights"), Described(noise(1011, 5), "a room of 5")),
            (Described(noise(1012, 8), "8 weights"), Described(noise(1013, 6), "a room of 6")),
        ),
        reference=REFERENCE["misalignment_db"],
        expect=expect_close,
    )

    for name in ("after_noise", "after_speech"):
        checker.check_name(
            name=name,
            description="your own run on one second of noise or speech",
            run=check_echo2_name(name),
        )


ECHO2 = Exercise(
    name="echo2",
    chapter=ECHO_CANCELLER,
    title="Project: the classic canceller",
    lesson=ECHO2_LESSON,
    source=ECHO2_SOURCE,
    solution=ECHO2_SOLUTION,
    hints={
        "nlms_step": "error = mic_sample - weights @ recent. The new weights add step * error * recent divided by "
        "(recent @ recent + eps). Return both.",
        "nlms": "Put taps - 1 zeros before far with torch.cat. At sample n, padded[n : n + taps].flip(0) is recent. "
        "Store each error in a tensor of len(mic) zeros.",
        "misalignment_db": "F.pad(x, (0, extra)) adds extra zeros at the end. Take .norm() of the difference over "
        "response.norm(), as plain numbers, then 20 * math.log10.",
        "after_noise": "This line runs your NLMS on noise. If it crashed, fix nlms first.",
        "after_speech": "This line runs your NLMS on speech. If it crashed on the data, download it first.",
    },
    introduces=(),
    check=check_echo2,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# echo3
# ---------------------------------------------------------------------------

ECHO3_LESSON = prose(
    """
    echo1 found that on frames, a short room is a product per frequency and
    a long room spills into the next frames. That suggests the design, a
    filter per frequency that reaches back a few frames. Each frequency
    holds one complex weight per frame back, and the echo's guess adds them.

    ```text
    guess(f, t) = w(f, 0) far(f, t) + w(f, 1) far(f, t - 1) + ... + w(f, taps - 1) far(f, t - taps + 1)
    ```

    A complex weight scales a frequency's strength and shifts its timing,
    which is what a room does to it. Optimizers here train real numbers, so
    each weight is a real part and an imaginary part, joined by
    `torch.complex`. Adam fits them on the frames.

    The design follows 95_design_method's steps.

    - The toy task: one frequency, one weight, a known gain of 0.6 - 0.8i. After 300 steps the weight is 0.6000 - 0.8000i.
    - The simplest baseline: one frame per frequency, echo1's product alone.
    - The diagnosis: on a room of 1,024 the baseline misses even its training frames. So it fails to fit and needs more reach.
    - Gradient health: the weights start at 0, so the filter starts by doing nothing. Every weight gets a gradient at the first step.
    - The properties: an echo is a linear copy of the far end, and a live canceller can't read ahead. `is_linear` and `is_causal` test both.

    Each filter below fits on the first 4 seconds of 8 and is scored on the
    last 4, three seeds each. The miss is on the training frames, and ERLE,
    echo4's score, is how many decibels quieter the echo got.

    | Frames back | Room 256, miss | Room 256, ERLE | Room 1,024, miss | Room 1,024, ERLE |
    |---|---|---|---|---|
    | 1 | -11.7 to -9.4 dB | 10.5 to 13.9 dB | -3.2 to -2.4 dB | 1.5 to 3.5 dB |
    | 2 | -23.4 to -20.8 dB | 22.3 to 29.3 dB | -6.7 to -6.1 dB | 4.4 to 7.3 dB |
    | 4 | -30.2 to -26.7 dB | 38.0 to 42.0 dB | -13.0 to -12.2 dB | 11.8 to 13.3 dB |
    | 8 | -31.3 to -27.4 dB | 41.4 to 46.6 dB | -24.5 to -22.7 dB | 23.8 to 25.5 dB |

    Where the miss on training frames is high, the score on new frames is
    low too. That is a failure to fit, and reaching back further fixes it.

    The overlap matters as well. With a window every 128 samples, half
    overlapping as in stft1, 8 frames back did far worse. They missed the
    short room's training frames by only -15.4 to -14.2 dB and scored 13.9 to
    17.5 dB. Denser frames let each frequency's weights fit the room far
    better here.

    A fit takes seconds on the CPU, so no checkpoint is needed.
    """,
)

ECHO3_NAMES = '''
    generator = torch.Generator().manual_seed(0)
    far = speech_clip(["george", "jackson"], 4.0, generator)
    mic = microphone(echo_of(far, room_response(1024, 200.0, generator)), torch.zeros(len(far)), 0.01, generator)
    one_frame = SubbandFilter(N_FFT // 2 + 1, 1)
    eight_frames = SubbandFilter(N_FFT // 2 + 1, 8)
    one_frame_loss = fit_filter(one_frame, frames_of(far), frames_of(mic), 300, 0.05)
    eight_frames_loss = fit_filter(eight_frames, frames_of(far), frames_of(mic), 300, 0.05)
    linear = is_linear(eight_frames, frames_of(far[:4000]), frames_of(mic[:4000]))
    causal = is_causal(eight_frames, frames_of(far), 100)
'''

ECHO3_SOURCE = source_file(
    """
    # echo3.py
    """
    + FILE_HEADER
    + """
    pair = torch.complex(torch.tensor([1.0]), torch.tensor([2.0]))    # tensor([1.+2.j])
    shifted = F.pad(torch.tensor([[1.0, 2.0, 3.0]]), (2, 0))          # tensor([[0., 0., 1., 2., 3.]])
    same = torch.allclose(torch.tensor([1.0]), torch.tensor([1.0 + 1e-9]))    # True


    # The settings, the speech, and echo1's room and frames.
"""
    + indented(SETTINGS_PART)
    + "\n\n"
    + indented(SPEECH_PART)
    + "\n\n"
    + indented(ROOM_SOLUTION)
    + "\n\n"
    + indented(FRAMES_SOLUTION)
    + '''


    # TODO: Finish forward. far_frames has shape (frequencies, frames). Return
    #   the echo's estimate, of the same shape. Frame t of each frequency is the
    #   sum over k from 0 to taps - 1 of weights[:, k] times far frame t - k,
    #   with frames before the first counted as 0.
'''
    + indented(FILTER_INIT)
    + '''

        def forward(self, far_frames: Tensor) -> Tensor:
            weights = torch.complex(self.real, self.imag)
            ...


    # TODO: Fit model to the microphone's frames with Adam at rate lr for steps
    #   steps. The loss is the mean of .abs() squared of mic_frames minus
    #   model(far_frames). Zero the old gradients, backpropagate and step each
    #   time. Return the last loss as a plain number.
    def fit_filter(
        model: nn.Module,
        far_frames: Tensor,
        mic_frames: Tensor,
        steps: int,
        lr: float,
    ) -> float:
        ...


    # TODO: Return mic minus the echo model guesses from far. Inside
    #   torch.no_grad(), run model on frames_of(far). Turn the guess back into
    #   samples with torch.istft, with n_fft=N_FFT, hop_length=HOP, a Hann
    #   window of N_FFT in far's dtype and length=len(mic).
    def cancel(
        model: nn.Module,
        far: Tensor,
        mic: Tensor,
    ) -> Tensor:
        ...


    # TODO: Test that model is linear. Inside torch.no_grad(), compare
    #   model(2 * first + second) with 2 * model(first) + model(second), and
    #   return whether torch.allclose finds them equal with atol=1e-4.
    def is_linear(
        model: nn.Module,
        first: Tensor,
        second: Tensor,
    ) -> bool:
        ...


    # TODO: Test that model never reads ahead. Copy far_frames with .clone()
    #   and set every frame from cut on to 0. Inside torch.no_grad(), return
    #   whether the outputs' frames before cut match, by torch.allclose with
    #   atol=1e-6.
    def is_causal(
        model: nn.Module,
        far_frames: Tensor,
        cut: int,
    ) -> bool:
        ...


    # Your own design steps on four seconds of speech through a room of 128 milliseconds.
'''
    + ECHO3_NAMES,
)

ECHO3_SOLUTION = source_file(
    """
    # echo3.py, a solution for comparison
    """
    + FILE_HEADER
    + """
    pair = torch.complex(torch.tensor([1.0]), torch.tensor([2.0]))
    shifted = F.pad(torch.tensor([[1.0, 2.0, 3.0]]), (2, 0))
    same = torch.allclose(torch.tensor([1.0]), torch.tensor([1.0 + 1e-9]))


"""
    + indented(SETTINGS_PART)
    + "\n\n"
    + indented(SPEECH_PART)
    + "\n\n"
    + indented(ROOM_SOLUTION)
    + "\n\n"
    + indented(FRAMES_SOLUTION)
    + "\n\n"
    + indented(FILTER_INIT)
    + "\n\n"
    + textwrap.indent(FILTER_FORWARD.strip("\n"), "    ")
    + "\n\n\n"
    + indented(FIT_SOLUTION)
    + "\n\n\n"
    + indented(PROPERTY_SOLUTION)
    + "\n\n\n"
    + ECHO3_NAMES,
)


def random_filter(
    make: type,
    frequencies: int,
    taps: int,
    seed: int,
) -> nn.Module:
    model = make(frequencies, taps)
    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        model.real.copy_(torch.randn(frequencies, taps, generator=generator))
        model.imag.copy_(torch.randn(frequencies, taps, generator=generator))

    return model


def check_filter_forward(found: type) -> None:
    for frequencies, taps, count, seed in ((5, 3, 7, 1020), (3, 1, 4, 1022), (4, 4, 6, 1024)):
        model = expect_module(f"SubbandFilter({frequencies}, {taps})", random_filter(found, frequencies, taps, seed))
        reference = random_filter(REFERENCE["SubbandFilter"], frequencies, taps, seed)
        frames = complex_frames(seed + 1, frequencies, count)

        with torch.no_grad():
            expect_complex(
                call=f"SubbandFilter({frequencies}, {taps}) with random weights on frames of shape ({frequencies}, {count})",
                actual=model(frames),
                expected=reference(frames),
            )


def fitting_case(seed: int) -> tuple[Tensor, Tensor]:
    far = complex_frames(seed, 6, 20)
    gains = complex_frames(seed + 1, 6, 2)
    mic = gains[:, :1] * far + gains[:, 1:] * F.pad(far, (1, 0))[:, :20] + 0.1 * complex_frames(seed + 2, 6, 20)

    return far, mic


def check_fit_filter(found: object) -> None:
    for seed, steps in ((1030, 2), (1033, 25)):
        far, mic = fitting_case(seed)
        model = REFERENCE["SubbandFilter"](6, 2)
        expected_model = REFERENCE["SubbandFilter"](6, 2)
        call = f"fit_filter(SubbandFilter(6, 2), frames of shape (6, 20), their echo, {steps}, 0.05)"
        actual = found(model, far.clone(), mic.clone(), steps, 0.05)
        expected = REFERENCE["fit_filter"](expected_model, far.clone(), mic.clone(), steps, 0.05)
        expect_close(
            call=f"the last loss from {call}",
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        )
        expect_tensor(
            call=f"the real parts after {call}",
            actual=model.real.detach(),
            expected=expected_model.real.detach(),
        )


def check_cancel(found: object) -> None:
    for seed, length in ((1040, 900), (1043, 1300)):
        model = random_filter(REFERENCE["SubbandFilter"], N_FFT // 2 + 1, 3, seed).double()
        far = noise(seed + 1, length)
        mic = noise(seed + 2, length)
        call = f"cancel(a SubbandFilter with random weights, {length} samples of noise, {length} more)"
        actual = found(model, far.clone(), mic.clone())
        reject_missing_return(
            call=call,
            actual=actual,
        )

        if isinstance(actual, Tensor) and actual.requires_grad:
            raise Mismatch(f"{call} is still recorded for autograd. Run the model inside torch.no_grad().")

        expect_tensor(
            call=call,
            actual=actual,
            expected=REFERENCE["cancel"](model, far.clone(), mic.clone()),
        )


class FilterWithBias(nn.Module):
    """A filter that adds a fixed number to every output, so doubling the input doesn't double the output."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, far_frames: Tensor) -> Tensor:
        return self.inner(far_frames) + 0.5


class SquaringFilter(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, far_frames: Tensor) -> Tensor:
        return self.inner(far_frames * far_frames.abs())


class PeekingFilter(nn.Module):
    """A filter that adds a little of the next frame, which a canceller running live can't have yet."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, far_frames: Tensor) -> Tensor:
        ahead = F.pad(far_frames[:, 1:], (0, 1))

        return self.inner(far_frames) + 0.1 * ahead


def property_models(seed: int) -> dict[str, tuple[nn.Module, bool, bool]]:
    """Each model with whether it is linear and whether it is causal."""
    inner = random_filter(REFERENCE["SubbandFilter"], 4, 3, seed)

    return {
        "a SubbandFilter with random weights": (inner, True, True),
        "a filter that adds 0.5": (FilterWithBias(inner), False, True),
        "a filter that squares its input's size into it": (SquaringFilter(inner), False, True),
        "a filter that reads the next frame": (PeekingFilter(inner), True, False),
    }


def check_is_linear(found: object) -> None:
    for seed in (1050, 1060):
        first = complex_frames(seed + 1, 4, 6)
        second = complex_frames(seed + 2, 4, 6)

        for text, (model, linear, _) in property_models(seed).items():
            expect_value(
                call=f"is_linear({text}, two sets of frames)",
                actual=found(model, first.clone(), second.clone()),
                expected=linear,
            )


def check_is_causal(found: object) -> None:
    for seed, cut in ((1070, 3), (1080, 5)):
        frames = complex_frames(seed + 1, 4, 8)

        for text, (model, _, causal) in property_models(seed).items():
            expect_value(
                call=f"is_causal({text}, frames of shape (4, 8), {cut})",
                actual=found(model, frames.clone(), cut),
                expected=causal,
            )


@functools.cache
def reference_echo3_names() -> dict[str, object]:
    namespace = dict(REFERENCE)
    exec(textwrap.dedent(ECHO3_NAMES), namespace)

    return {name: namespace[name] for name in ("one_frame_loss", "eight_frames_loss", "linear", "causal")}


def check_echo3_loss(name: str) -> Callable[[object], None]:
    def run(value: object) -> None:
        expected = reference_echo3_names()[name]
        expect_close(
            call=name,
            actual=value,
            expected=expected,
            tolerance=1e-3 * abs(expected),
        )

    return run


def check_echo3_property(name: str) -> Callable[[object], None]:
    def run(value: object) -> None:
        expect_value(
            call=name,
            actual=value,
            expected=True,
        )

    return run


def check_echo3(checker: Checker) -> None:
    checker.test_method(
        class_name="SubbandFilter",
        method="forward",
        description="adds each frequency's weights times the last frames",
        run=check_filter_forward,
    )
    checker.test(
        function="fit_filter",
        description="fits the weights with Adam, zeroing the gradients each step",
        run=check_fit_filter,
    )
    checker.test(
        function="cancel",
        description="takes the guessed echo out of the microphone, without autograd",
        run=check_cancel,
    )
    checker.test(
        function="is_linear",
        description="tells a linear filter from three others",
        run=check_is_linear,
    )
    checker.test(
        function="is_causal",
        description="tells a filter that reads ahead from the others",
        run=check_is_causal,
    )

    for name in ("one_frame_loss", "eight_frames_loss"):
        checker.check_name(
            name=name,
            description="your own fit on four seconds of speech",
            run=check_echo3_loss(name),
        )

    for name in ("linear", "causal"):
        checker.check_name(
            name=name,
            description="your property test on your fitted filter",
            run=check_echo3_property(name),
        )


ECHO3 = Exercise(
    name="echo3",
    chapter=ECHO_CANCELLER,
    title="Project: a filter per frequency",
    lesson=ECHO3_LESSON,
    source=ECHO3_SOURCE,
    solution=ECHO3_SOLUTION,
    hints={
        "SubbandFilter.forward": "F.pad(far_frames, (self.taps - 1, 0)) puts taps - 1 zero frames first. For each k, "
        "weights[:, k : k + 1] multiplies count padded frames. They start at taps - 1 - k, and the product adds to "
        "the estimate.",
        "fit_filter": "Make the Adam optimizer once, before the loop. Each step: loss, zero_grad, backward, step. "
        "Return loss.item().",
        "cancel": "estimate = model(frames_of(far)) inside torch.no_grad(). Then torch.istft with the arguments the TODO "
        "names, and subtract it from mic.",
        "is_linear": "Compute both sides inside torch.no_grad(), then return torch.allclose(together, apart, atol=1e-4).",
        "is_causal": "changed = far_frames.clone(), then changed[:, cut:] = 0. Compare model(far_frames)[:, :cut] with "
        "model(changed)[:, :cut].",
        "one_frame_loss": "This line fits your filter with one frame. If it crashed, fix fit_filter first.",
        "eight_frames_loss": "This line fits your filter with eight frames. If it crashed, fix fit_filter first.",
        "linear": "Your fitted filter should pass your own test. If it doesn't, look at is_linear's tolerance.",
        "causal": "Your fitted filter should pass your own test. If it doesn't, look at the frames forward reads.",
    },
    introduces=(),
    check=check_echo3,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# echo4
# ---------------------------------------------------------------------------

ECHO4_LESSON = prose(
    """
    **ERLE**, echo return loss enhancement, is the score cancellers report.
    It is how many decibels quieter the canceller made the echo, 10 times
    the log of the echo's energy over the energy left of it. Wikipedia puts
    most cancellers at 18 to 35 dB. The rooms here are simulated, so the echo
    is known exactly, and ERLE counts the leftover echo alone, even while
    the near end talks.

    Both cancellers get the same 8 seconds. NLMS adapts along all of it. The
    filter fits on the first half and then stays fixed. Both are scored on
    the second half, over three seeds.

    | Case | NLMS | The filter per frequency |
    |---|---|---|
    | Single talk, a room of 256 | 28.6 to 30.6 dB | 44.2 to 45.0 dB |
    | Double talk, the near end speaking too | -5.0 to -1.9 dB | 18.9 to 28.3 dB |
    | The room changes at 4 seconds | 14.2 to 22.2 dB | -4.9 to -1.1 dB |
    | A long tail, a room of 1,024 | 18.3 to 19.5 dB | 24.4 to 25.2 dB |

    Each row has a reason.

    - Single talk: NLMS is still learning the room from speech, as echo2 showed. The filter saw 4 seconds at once.
    - Double talk: the near end's voice lands in NLMS's error and drags its steps to the wrong room. The fit averages that voice out.
    - The room changes: the fitted filter keeps the old room, and its guess is worse than none. NLMS follows the new room.
    - The long tail: 256 weights cover 32 milliseconds of a room that rings for 128. Eight frames back reach about 90.

    Neither wins everywhere, and real systems combine the ideas. Classic
    cancellers keep adapting, and pause the update while the near end talks,
    which takes a double talk detector of its own. Many learned systems keep
    a linear canceller and put a neural network after it, for the echo it
    leaves. Zhang and colleagues placed third overall in the ICASSP 2022
    challenge with such a pair.

    Your filter is linear, so least squares could fit it in one step.
    Training it with gradient descent leaves room for a network on top. The
    design came from the problem's own structure, a product per frequency
    that reaches back in time, and each choice was measured before the next.
    """,
)

ECHO4_NAMES = '''
    table = {name: score_both(*case(name, torch.Generator().manual_seed(0))) for name in CASES}
'''

ECHO4_SOURCE = source_file(
    """
    # echo4.py
    """
    + FILE_HEADER
    + """
    louder = 10 * math.log10(torch.tensor([3.0, 4.0]).pow(2).sum().item() / 1.0)    # 13.979..., 25 times the energy
    second_half = torch.arange(6.0)[3:]                                         # tensor([3., 4., 5.])


    # Everything from echo1 to echo3, and the four cases.
"""
    + indented(SETTINGS_PART)
    + "\n\n"
    + indented(SPEECH_PART)
    + "\n\n"
    + indented(ROOM_SOLUTION)
    + "\n\n"
    + indented(FRAMES_SOLUTION)
    + "\n\n"
    + indented(NLMS_SOLUTION)
    + "\n\n"
    + indented(FILTER_INIT)
    + "\n\n"
    + textwrap.indent(FILTER_FORWARD.strip("\n"), "    ")
    + "\n\n\n"
    + indented(FIT_SOLUTION)
    + "\n\n\n"
    + indented(CASES_PART)
    + '''


    # TODO: Return the echo return loss enhancement in decibels, as a plain
    #   number: 10 * log10 of the echo's energy over the residual echo's
    #   energy. The energy is the sum of the samples squared.
    def erle_db(
        echo: Tensor,
        residual_echo: Tensor,
    ) -> float:
        ...


    # TODO: Score both cancellers on the second half of one case, as
    #   (NLMS, filter), with half = len(far) // 2. Run nlms on all of far and
    #   mic, with 256 taps, a step of 0.5 and eps of 1e-3. Its echo guess is mic
    #   minus its residual. Fit a SubbandFilter(N_FFT // 2 + 1, 8) on
    #   frames_of(far[:half]) and frames_of(mic[:half]), 300 steps at 0.05. Its
    #   guess is mic minus cancel(model, far, mic). Each score is erle_db of
    #   echo[half:] against what is left of it, the echo minus that guess.
    def score_both(
        far: Tensor,
        echo: Tensor,
        mic: Tensor,
    ) -> tuple[float, float]:
        ...


    # Your own table, the four cases with seed 0. It takes a few seconds.
'''
    + ECHO4_NAMES,
)

ECHO4_SOLUTION = source_file(
    """
    # echo4.py, a solution for comparison
    """
    + FILE_HEADER
    + """
    louder = 10 * math.log10(torch.tensor([3.0, 4.0]).pow(2).sum().item() / 1.0)
    second_half = torch.arange(6.0)[3:]


"""
    + indented(SETTINGS_PART)
    + "\n\n"
    + indented(SPEECH_PART)
    + "\n\n"
    + indented(ROOM_SOLUTION)
    + "\n\n"
    + indented(FRAMES_SOLUTION)
    + "\n\n"
    + indented(NLMS_SOLUTION)
    + "\n\n"
    + indented(FILTER_INIT)
    + "\n\n"
    + textwrap.indent(FILTER_FORWARD.strip("\n"), "    ")
    + "\n\n\n"
    + indented(FIT_SOLUTION)
    + "\n\n\n"
    + indented(CASES_PART)
    + "\n\n\n"
    + indented(SCORE_SOLUTION)
    + "\n\n\n"
    + ECHO4_NAMES,
)


def score_case(seed: int) -> tuple[Described, Described, Described]:
    """A short case with no speech in it, so the check needs no download: colored noise through a room, plus noise."""
    far = colored_noise(seed, 2 * RATE // 4).float()
    response = REFERENCE["room_response"](64, 10.0, torch.Generator().manual_seed(seed + 1))
    echo = REFERENCE["echo_of"](far, response)
    mic = echo + 0.01 * noise(seed + 2, len(far)).float()

    return (
        Described(far, "half a second of colored noise"),
        Described(echo, "its echo through a room of 64 samples"),
        Described(mic, "the echo plus a little noise"),
    )


def check_score_both(found: object) -> None:
    for seed in (1090, 1095):
        far, echo, mic = score_case(seed)
        call = f"score_both({far.text}, {echo.text}, {mic.text})"
        expected = REFERENCE["score_both"](far.value.clone(), echo.value.clone(), mic.value.clone())
        close_scores(1e-3)(
            call=call,
            actual=found(far.value.clone(), echo.value.clone(), mic.value.clone()),
            expected=expected,
        )


@functools.cache
def reference_table() -> dict[str, tuple[float, float]]:
    namespace = dict(REFERENCE)
    exec(textwrap.dedent(ECHO4_NAMES), namespace)

    return namespace["table"]


def check_table(value: object) -> None:
    reject_missing_return(
        call="table",
        actual=value,
    )

    if not isinstance(value, dict):
        raise Mismatch(f"table is {value_text(value)}, but it should be a dict of the four cases.")

    expected = reference_table()

    for name, scores in expected.items():
        if name not in value:
            raise Mismatch(f"table has no row for {name!r}.")

        close_scores(1e-3)(
            call=f"table[{name!r}]",
            actual=value[name],
            expected=scores,
        )


def close_scores(tolerance: float) -> Callable[..., None]:
    """Two scores in decibels, which a sum taken in another order moves by far less than a thousandth."""

    def expect(
        call: str,
        actual: object,
        expected: tuple[float, float],
    ) -> None:
        reject_missing_return(
            call=call,
            actual=actual,
        )

        if not isinstance(actual, (tuple, list)) or len(actual) != 2:
            raise Mismatch(f"{call} should give 2 scores, the NLMS one and the filter's, but it gives {value_text(actual)}.")

        for text, part, wanted in zip(("the NLMS score", "the filter's score"), actual, expected):
            expect_close(
                call=f"{text} from {call}",
                actual=part,
                expected=wanted,
                tolerance=tolerance,
            )

    return expect


def check_echo4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="erle_db",
        description="gives how many decibels quieter the echo got",
        inputs=(
            (Described(noise(1100, 50), "an echo of 50 samples"), Described(0.1 * noise(1101, 50), "a residual of 50")),
            (Described(noise(1102, 80), "an echo of 80 samples"), Described(0.5 * noise(1103, 80), "a residual of 80")),
        ),
        reference=REFERENCE["erle_db"],
        expect=expect_close,
    )
    checker.test(
        function="score_both",
        description="scores NLMS and the filter on the second half",
        run=check_score_both,
    )
    checker.check_name(
        name="table",
        description="your own table of the four cases",
        run=check_table,
    )


ECHO4 = Exercise(
    name="echo4",
    chapter=ECHO_CANCELLER,
    title="Project: who wins where",
    lesson=ECHO4_LESSON,
    source=ECHO4_SOURCE,
    solution=ECHO4_SOLUTION,
    hints={
        "erle_db": "Sum echo.pow(2) and residual_echo.pow(2), divide the plain numbers, and take 10 * math.log10.",
        "score_both": "half = len(far) // 2. NLMS's guess is mic - residual. The filter fits on frames_of(far[:half]) and "
        "frames_of(mic[:half]), and its guess is mic - cancel(model, far, mic). Score echo[half:] against "
        "(echo - guess)[half:].",
        "table": "This line runs score_both on the four cases. If it crashed on the data, download it first.",
    },
    introduces=(),
    check=check_echo4,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_products(seeds: range = range(3)) -> dict[int, list[float]]:
    """echo1's table: how far a product per frequency misses each room's echo, in decibels, on 4 seconds of speech."""
    results: dict[int, list[float]] = {}

    with pinned_threads():
        for length, decay in ((64, 10.0), (256, 40.0), (1024, 200.0)):
            for seed in seeds:
                generator = torch.Generator().manual_seed(seed)
                far = REFERENCE["speech_clip"](FAR_SPEAKERS, 4.0, generator)
                response = REFERENCE["room_response"](length, decay, generator)
                echo = REFERENCE["echo_of"](far, response)
                error = REFERENCE["product_error"](
                    REFERENCE["frames_of"](echo),
                    REFERENCE["frames_of"](far),
                    REFERENCE["room_spectrum"](response),
                )
                results.setdefault(length, []).append(error)

    return results


def misalignment_track(
    far: Tensor,
    mic: Tensor,
    response: Tensor,
    every: int,
) -> list[float]:
    """NLMS run as echo2 runs it, with its misalignment noted every so many samples."""
    weights = torch.zeros(TAPS, dtype=far.dtype)
    padded = torch.cat([torch.zeros(TAPS - 1, dtype=far.dtype), far])
    track = []

    for n in range(len(mic)):
        weights, _ = REFERENCE["nlms_step"](weights, padded[n : n + TAPS].flip(0), mic[n], STEP, EPS)

        if (n + 1) % every == 0:
            track.append(REFERENCE["misalignment_db"](weights, response))

    return track


def measure_convergence(seeds: range = range(3)) -> dict[str, list[tuple[float | None, float | None, float, float]]]:
    """echo2's table: for noise and for speech through a room of 256, the seconds NLMS takes to get below -10 dB and
    below -20 dB, and its misalignment after 1 and after 4 seconds. The microphone adds noise at 0.01, as in echo4."""
    results: dict[str, list[tuple[float | None, float | None, float, float]]] = {}
    every = RATE // 20

    with pinned_threads():
        for kind in ("noise", "speech"):
            for seed in seeds:
                generator = torch.Generator().manual_seed(10 + seed)
                far = (
                    torch.randn(4 * RATE, generator=generator)
                    if kind == "noise"
                    else REFERENCE["speech_clip"](FAR_SPEAKERS, 4.0, generator)
                )
                response = REFERENCE["room_response"](256, 40.0, generator)
                mic = REFERENCE["microphone"](REFERENCE["echo_of"](far, response), torch.zeros(len(far)), 0.01, generator)
                track = misalignment_track(far, mic, response, every)
                below_10 = next(((index + 1) * every / RATE for index, value in enumerate(track) if value < -10), None)
                below_20 = next(((index + 1) * every / RATE for index, value in enumerate(track) if value < -20), None)
                results.setdefault(kind, []).append((below_10, below_20, track[19], track[-1]))

    return results


def measure_low_share(seeds: range = range(3)) -> dict[str, list[float]]:
    """echo2's claim about speech's spectrum: the share of 4 seconds' energy below 1,000 Hz, for speech and for noise."""
    results: dict[str, list[float]] = {}

    for seed in seeds:
        generator = torch.Generator().manual_seed(10 + seed)
        signals = {
            "noise": torch.randn(4 * RATE, generator=generator),
            "speech": REFERENCE["speech_clip"](FAR_SPEAKERS, 4.0, generator),
        }

        for kind, signal in signals.items():
            energy = torch.fft.rfft(signal).abs().pow(2)
            below = int(len(energy) * 1000 / (RATE / 2))
            results.setdefault(kind, []).append((energy[:below].sum() / energy.sum()).item())

    return results


def frames_with_hop(
    signal: Tensor,
    hop: int,
) -> Tensor:
    return torch.stft(
        signal,
        n_fft=N_FFT,
        hop_length=hop,
        window=torch.hann_window(N_FFT, dtype=signal.dtype),
        pad_mode="constant",
        return_complex=True,
    )


def filter_scores(
    far: Tensor,
    echo: Tensor,
    mic: Tensor,
    taps: int,
    hop: int,
) -> tuple[float, float]:
    """The filter's training miss in decibels on the first half, and its ERLE on the second half."""
    half = len(far) // 2
    far_frames = frames_with_hop(far[:half], hop)
    mic_frames = frames_with_hop(mic[:half], hop)
    model = REFERENCE["SubbandFilter"](N_FFT // 2 + 1, taps)
    loss = REFERENCE["fit_filter"](model, far_frames, mic_frames, FIT_STEPS, FIT_RATE)
    miss = 10 * math.log10(loss / mic_frames.abs().pow(2).mean().item())

    with torch.no_grad():
        estimate = model(frames_with_hop(far, hop))

    guess = torch.istft(estimate, n_fft=N_FFT, hop_length=hop, window=torch.hann_window(N_FFT), length=len(mic))

    return miss, REFERENCE["erle_db"](echo[half:], (echo - guess)[half:])


def measure_design(seeds: range = range(3)) -> dict[str, object]:
    """echo3's numbers: the toy task's recovered gain, the frames table for a room of 256 and one of 1024, the same
    filter with windows every 128 samples, and how many weights get a gradient at the start."""
    results: dict[str, object] = {}

    with pinned_threads():
        # The toy task: one frequency, a known complex gain, one weight.
        gain = torch.complex(torch.tensor(0.6), torch.tensor(-0.8))
        far = complex_frames(5, 1, 200)
        model = REFERENCE["SubbandFilter"](1, 1)
        REFERENCE["fit_filter"](model, far, gain * far, FIT_STEPS, FIT_RATE)
        results["toy gain"] = (model.real.item(), model.imag.item())

        for length, decay in ((256, 40.0), (1024, 200.0)):
            for taps in (1, 2, 4, 8):
                rows = []

                for seed in seeds:
                    generator = torch.Generator().manual_seed(20 + seed)
                    far = REFERENCE["speech_clip"](FAR_SPEAKERS, 8.0, generator)
                    echo = REFERENCE["echo_of"](far, REFERENCE["room_response"](length, decay, generator))
                    mic = REFERENCE["microphone"](echo, torch.zeros(len(far)), 0.01, generator)
                    rows.append(filter_scores(far, echo, mic, taps, HOP))

                results[f"room {length}, {taps} frames"] = rows

        rows = []

        for seed in seeds:
            generator = torch.Generator().manual_seed(20 + seed)
            far = REFERENCE["speech_clip"](FAR_SPEAKERS, 8.0, generator)
            echo = REFERENCE["echo_of"](far, REFERENCE["room_response"](256, 40.0, generator))
            mic = REFERENCE["microphone"](echo, torch.zeros(len(far)), 0.01, generator)
            rows.append(filter_scores(far, echo, mic, 8, 128))

        results["room 256, 8 frames, hop 128"] = rows

        generator = torch.Generator().manual_seed(20)
        far = REFERENCE["speech_clip"](FAR_SPEAKERS, 4.0, generator)
        mic = REFERENCE["microphone"](
            REFERENCE["echo_of"](far, REFERENCE["room_response"](256, 40.0, generator)),
            torch.zeros(len(far)),
            0.01,
            generator,
        )
        model = REFERENCE["SubbandFilter"](N_FFT // 2 + 1, 8)
        far_frames = REFERENCE["frames_of"](far)
        mic_frames = REFERENCE["frames_of"](mic)
        loss = (mic_frames - model(far_frames)).abs().pow(2).mean()
        loss.backward()
        moved = ((model.real.grad != 0) | (model.imag.grad != 0)).float().mean().item()
        results["start"] = (loss.item() / mic_frames.abs().pow(2).mean().item(), moved)

    return results


def measure_cases(seeds: range = range(3)) -> dict[str, list[tuple[float, float]]]:
    """echo4's table: both scores for each case, seeds 0 to 2."""
    results: dict[str, list[tuple[float, float]]] = {}

    with pinned_threads():
        for name in CASES:
            for seed in seeds:
                far, echo, mic = REFERENCE["case"](name, torch.Generator().manual_seed(seed))
                results.setdefault(name, []).append(REFERENCE["score_both"](far, echo, mic))

    return results
