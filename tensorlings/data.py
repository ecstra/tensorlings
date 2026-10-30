"""Datasets the course downloads once into data/, which git ignores, and the loaders exercises call."""

from __future__ import annotations

import functools
import gzip
import hashlib
import io
import shutil
import subprocess
import tarfile
import urllib.request
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .paths import DATA_DIR, START_COMMAND


@dataclass(frozen=True)
class RemoteFile:
    url: str
    md5: str

    @property
    def name(self) -> str:
        return self.url.rsplit("/", 1)[1]


# PyTorch's own mirror of MNIST, the one torchvision downloads from, with torchvision's checksums.
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

# Jakobovski's Free Spoken Digit Dataset, CC BY-SA 4.0, pinned to its last commit. Each file is one spoken digit,
# 8,000 samples a second, mono, 16 bits. These are three speakers' first two takes of every digit, fetched one by
# one, since an archive GitHub builds on the fly may not keep its checksum.
SPOKEN_DIGITS_MIRROR = (
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/26eb9aaf76e81b692f806f9140c2d2777410d7a1/recordings/"
)
SPOKEN_DIGITS_MD5: dict[str, str] = {
    "0_george_0.wav": "9ac95a34295c0abcf1ea39dfbd367cd6",
    "0_george_1.wav": "a4bb1fbe6acde92827b433d120552615",
    "1_george_0.wav": "a14e6562aca81b1dc3d7480c2b80dce6",
    "1_george_1.wav": "fc2c5acf5592dec333c28962d2128f74",
    "2_george_0.wav": "7c28ce59895d1eb786a62601c690ea8d",
    "2_george_1.wav": "faa58043abbaa1d08b0946a78b51e424",
    "3_george_0.wav": "79bce3f75e7467780008fd00031d3c7d",
    "3_george_1.wav": "b3635d4d099366349854613407fec232",
    "4_george_0.wav": "555e77dbb05f350383053c16b4929bdc",
    "4_george_1.wav": "b274f1c811c2f5912724b0c89d5fd9c3",
    "5_george_0.wav": "6228c14210486d2e9d273d0e7740e925",
    "5_george_1.wav": "49a4962a938b418571a143135f1860e0",
    "6_george_0.wav": "14b74a15b5b39dca504d0ac400133cf0",
    "6_george_1.wav": "407f9fef777ebbaab730b5372c2400b1",
    "7_george_0.wav": "4d12fe2a53a2a7e5bd6465d548f04e36",
    "7_george_1.wav": "31da1a817b058b0c3d68a366253c9eb8",
    "8_george_0.wav": "b42f8bae65981984d3ce080dbe0b2f99",
    "8_george_1.wav": "4d83bd673586b179335f5794923cb4b6",
    "9_george_0.wav": "b169b62dd6ec1f5c534995fcca1657e3",
    "9_george_1.wav": "a48c7a77d1ab663271750e3a88b9cf3e",
    "0_jackson_0.wav": "367ac1753da93ce4e65a7c9e0db0208d",
    "0_jackson_1.wav": "2d13a0fb75e1dd059aa2469afc8e30f9",
    "1_jackson_0.wav": "879c203087d8b0d4e6eb2645f4d71d77",
    "1_jackson_1.wav": "3fcd2f19a22eba0df1686dc47a7186c9",
    "2_jackson_0.wav": "561360bb203f3fa8aafe699284af0356",
    "2_jackson_1.wav": "d7bd1829dc5502630471315a7635f92b",
    "3_jackson_0.wav": "72f1fa50f205d08bd91a9bcacb33f02f",
    "3_jackson_1.wav": "59e62e96416acbe5e42279cf38bfb0e6",
    "4_jackson_0.wav": "afa011bec6078ae39841dce25403649a",
    "4_jackson_1.wav": "5f7582091decbf5b1721256a486ae69b",
    "5_jackson_0.wav": "a3b021e017de8fbd43e41e52880cdc03",
    "5_jackson_1.wav": "aabd1f53b2c918a5948da1a04ebac1d9",
    "6_jackson_0.wav": "61cb3053818ba915bad938f0073bbbe0",
    "6_jackson_1.wav": "1f9a78a7c9b82b475ec6503e66d2260d",
    "7_jackson_0.wav": "eeec6f6102c3e1b38966224804b0e44e",
    "7_jackson_1.wav": "20318b957d005ec13b08650f55c34a8e",
    "8_jackson_0.wav": "9a73ae09c0fd40cf9493128611fd2fab",
    "8_jackson_1.wav": "cab8584943ca6d7b412892d5a97c9d39",
    "9_jackson_0.wav": "a35c44102568ad39d3a6ca1229ca05e5",
    "9_jackson_1.wav": "c468dece5d1eb2f5c9de5ec75c76b057",
    "0_theo_0.wav": "7e1e8adb5dae0f9048f5b22e6a3928d1",
    "0_theo_1.wav": "8f81aa0af268190dd35f27d8b13917cc",
    "1_theo_0.wav": "7ce4b86dbc694a1924e18aca94f8fe3b",
    "1_theo_1.wav": "a5bcd1977883857f8fbba3e075c56f9b",
    "2_theo_0.wav": "2b1bbcf55eaeff12e2eeae887bde2a20",
    "2_theo_1.wav": "39e3f9fbd29d6fcea3beca05048758e3",
    "3_theo_0.wav": "cccdc7a74351821d54fe0ad736baa232",
    "3_theo_1.wav": "0fb59e8c1a2c708e4898980503484b80",
    "4_theo_0.wav": "8426fe8fb3dc488bc2d4fa412808e110",
    "4_theo_1.wav": "c393e1832a8725363eb880fd9b9c8646",
    "5_theo_0.wav": "41c0ddde83aa3229ce16bbdfca589166",
    "5_theo_1.wav": "767e248ea9016bb301b9bcf0dbe7135d",
    "6_theo_0.wav": "622c077775eb14e95e0d6a615ac7c26c",
    "6_theo_1.wav": "cfa7f4ca9271d1dfc822fc1cf9ce878f",
    "7_theo_0.wav": "46325b8df2e57c9e13fbd99de303bc40",
    "7_theo_1.wav": "4f87dfbe81e683db861f3119724efcce",
    "8_theo_0.wav": "669e360a78494fcd67b9b45e94c6df50",
    "8_theo_1.wav": "4b087425588591a884d1f31ffa7ca450",
    "9_theo_0.wav": "5d8cf056a617a68ac449c4f57f3ed69f",
    "9_theo_1.wav": "1f295e6ee7b4dd7f4787fbba9311a8ac",
}

DATASETS: dict[str, tuple[RemoteFile, ...]] = {
    "mnist": (
        RemoteFile(MNIST_MIRROR + "train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        RemoteFile(MNIST_MIRROR + "train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        RemoteFile(MNIST_MIRROR + "t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        RemoteFile(MNIST_MIRROR + "t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ),
    # The 32,033 names of Karpathy's makemore, one per line.
    "names": (
        RemoteFile("https://raw.githubusercontent.com/karpathy/makemore/master/names.txt", "401b3c4cac9e28a586ed15e81256680d"),
    ),
    # Karpathy's tiny Shakespeare, about a million characters of the plays in one file.
    "shakespeare": (
        RemoteFile("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "6fb458f1232090904fb40fe944165e91"),
    ),
    # The binary version from Krizhevsky's own page, read straight out of the archive, so no pickle is ever loaded.
    "cifar10": (
        RemoteFile("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", "c32a1d4ab5d03f1284b67883e8d87530"),
    ),
    # NLTK's 18 books from Project Gutenberg, pinned to the commit that last changed them, read out of the zip in memory.
    "gutenberg": (
        RemoteFile(
            "https://raw.githubusercontent.com/nltk/nltk_data/5f09e470fe339e1a119dd174ca284129a3c7a5bb/packages/corpora/gutenberg.zip",
            "48c9c8605cd70b0230687557ee543633",
        ),
    ),
    "spoken_digits": tuple(RemoteFile(SPOKEN_DIGITS_MIRROR + name, md5) for name, md5 in SPOKEN_DIGITS_MD5.items()),
}

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class MissingData(Exception):
    """A dataset an exercise needs is not downloaded yet. The message says how to get it."""


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch(
    url: str,
    path: Path,
) -> None:
    """Python from python.org on a Mac ships without certificates, so its own downloads fail to verify
    the site. curl uses the system's certificates, so it goes first, with Python as the fallback."""
    curl = shutil.which("curl")

    if curl is None:
        urllib.request.urlretrieve(url, path)
        return

    finished = subprocess.run(
        [curl, "--fail", "--location", "--silent", "--show-error", "--output", str(path), url],
        check=False,
    )

    if finished.returncode != 0:
        raise SystemExit(f"Downloading {url} failed. Check the internet connection and run the command again.")


def download(name: str) -> None:
    """Download every file of a dataset that isn't there yet, checking each against its checksum."""
    folder = DATA_DIR / name
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for remote in DATASETS[name]:
        path = folder / remote.name

        if path.exists() and md5_of(path) == remote.md5:
            print(f"{remote.name} is already here.")
            continue

        print(f"Downloading {remote.url}")
        # A download cut off halfway stays a .part file, so it never passes for a whole one.
        partial = path.with_name(path.name + ".part")
        fetch(
            url=remote.url,
            path=partial,
        )

        if md5_of(partial) != remote.md5:
            partial.unlink()
            raise SystemExit(f"{remote.name} came down damaged. Run the command again.")

        partial.rename(path)


def dataset_file(
    name: str,
    file_name: str,
) -> Path:
    path = DATA_DIR / name / file_name

    if not path.exists():
        raise MissingData(
            f"The {name} data isn't downloaded yet. Run this once from the project folder, then save the file "
            f"again.\n\n    {START_COMMAND} data {name}",
        )

    return path


def read_idx(path: Path) -> Tensor:
    """A gzipped IDX file, the format MNIST ships in. Its fourth byte is the number of dimensions, then
    comes each dimension's size as a 4 byte big endian number, then one byte per entry."""
    raw = gzip.decompress(path.read_bytes())
    dimensions = raw[3]
    sizes = [int.from_bytes(raw[4 + 4 * index : 8 + 4 * index], "big") for index in range(dimensions)]
    start = 4 + 4 * dimensions

    return torch.frombuffer(bytearray(raw[start:]), dtype=torch.uint8).reshape(sizes)


@functools.cache
def read_mnist(split: str) -> tuple[Tensor, Tensor]:
    prefix = "train" if split == "train" else "t10k"
    images = read_idx(dataset_file("mnist", f"{prefix}-images-idx3-ubyte.gz"))
    labels = read_idx(dataset_file("mnist", f"{prefix}-labels-idx1-ubyte.gz")).long()

    return images, labels


def load_mnist(split: str) -> tuple[Tensor, Tensor]:
    """MNIST's images, of shape (count, 28, 28) with pixels from 0 to 255, and their labels from 0 to 9.
    split is "train" for the 60,000 training images or "test" for the 10,000 test images."""
    images, labels = read_mnist(split)

    # Copies, so a learner's file that changes them in place can't change what the next check reads.
    return images.clone(), labels.clone()


def read_cifar_batch(raw: bytes) -> tuple[Tensor, Tensor]:
    """CIFAR-10's binary format, one record of 3,073 bytes per image. The first byte is the label,
    then come the red plane, the green and the blue, each 32 rows of 32 pixels."""
    records = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, 3073)

    return records[:, 1:].reshape(-1, 3, 32, 32), records[:, 0].long()


@functools.cache
def read_cifar10(split: str) -> tuple[Tensor, Tensor]:
    wanted = [f"data_batch_{number}.bin" for number in range(1, 6)] if split == "train" else ["test_batch.bin"]
    batches: dict[str, tuple[Tensor, Tensor]] = {}

    # Members are read in memory by their names and never extracted, so no path in the archive is ever written to.
    with tarfile.open(dataset_file("cifar10", "cifar-10-binary.tar.gz"), "r:gz") as archive:
        for member in archive:
            name = member.name.rsplit("/", 1)[-1]
            content = archive.extractfile(member) if member.isfile() and name in wanted else None

            if content is not None:
                batches[name] = read_cifar_batch(content.read())

    images, labels = zip(*(batches[name] for name in wanted))

    return torch.cat(images), torch.cat(labels)


def load_cifar10(split: str) -> tuple[Tensor, Tensor]:
    """CIFAR-10's images, of shape (count, 3, 32, 32) with pixels from 0 to 255, and their labels from 0 to 9,
    which CIFAR10_CLASSES names. split is "train" for the 50,000 training images or "test" for the 10,000 test images."""
    images, labels = read_cifar10(split)

    return images.clone(), labels.clone()


@functools.cache
def read_shakespeare() -> str:
    return dataset_file("shakespeare", "input.txt").read_text()


def load_shakespeare() -> str:
    """Tiny Shakespeare as one string, 1,115,394 characters of the plays."""
    return read_shakespeare()


@functools.cache
def read_persuasion() -> str:
    # Read by its name in memory and never extracted, so no path in the archive is ever written to.
    with zipfile.ZipFile(dataset_file("gutenberg", "gutenberg.zip")) as archive:
        return archive.read("gutenberg/austen-persuasion.txt").decode("ascii")


def load_persuasion() -> str:
    """Jane Austen's novel Persuasion as one string, 466,292 characters, from NLTK's copy of Project Gutenberg."""
    return read_persuasion()


@functools.cache
def read_spoken_digits() -> dict[str, tuple[Tensor, ...]]:
    recordings: dict[str, list[Tensor]] = {}

    for name in SPOKEN_DIGITS_MD5:
        speaker = name.split("_")[1]

        with wave.open(io.BytesIO(dataset_file("spoken_digits", name).read_bytes())) as recording:
            raw = recording.readframes(recording.getnframes())

        # 16 bit whole numbers, little endian as WAV stores them, scaled into -1 to 1.
        samples = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768
        recordings.setdefault(speaker, []).append(samples)

    return {speaker: tuple(pieces) for speaker, pieces in recordings.items()}


def load_spoken_digits() -> dict[str, list[Tensor]]:
    """Three speakers of the Free Spoken Digit Dataset, george, jackson and theo, each with two takes of every digit
    from 0 to 9 in that order. Each recording is a tensor of samples from -1 to 1, 8,000 a second."""
    return {speaker: [piece.clone() for piece in pieces] for speaker, pieces in read_spoken_digits().items()}


@functools.cache
def read_names() -> tuple[str, ...]:
    return tuple(dataset_file("names", "names.txt").read_text().split())


def load_names() -> list[str]:
    """The 32,033 names of makemore, in lowercase, in the file's order."""
    return list(read_names())
