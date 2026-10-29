"""Datasets the course downloads once into data/, which git ignores, and the loaders exercises call."""

from __future__ import annotations

import functools
import gzip
import hashlib
import shutil
import subprocess
import urllib.request
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
}


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


@functools.cache
def read_names() -> tuple[str, ...]:
    return tuple(dataset_file("names", "names.txt").read_text().split())


def load_names() -> list[str]:
    """The 32,033 names of makemore, in lowercase, in the file's order."""
    return list(read_names())
