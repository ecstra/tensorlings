"""Datasets the course downloads once into data/, which git ignores, and the loaders exercises call."""

from __future__ import annotations

import csv
import functools
import gzip
import hashlib
import io
import math
import shutil
import subprocess
import tarfile
import urllib.request
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
    # A URL that ends in a query, like the exoplanet archive's table service, needs a file name of its own.
    file_name: str = ""

    @property
    def name(self) -> str:
        return self.file_name or self.url.rsplit("/", 1)[1]


# PyTorch's own mirror of MNIST, the one torchvision downloads from, with torchvision's checksums.
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

# The measurements of the Kepler table the exoplanet project reads, in the order of its columns.
KEPLER_COLUMNS: tuple[str, ...] = (
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_impact",
    "koi_ror",
    "koi_srho",
    "koi_num_transits",
    "koi_model_snr",
    "koi_max_sngle_ev",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
    "koi_bin_oedp_sig",
    "koi_dikco_msky",
)

# NASA's Kepler DR25 table of objects of interest, a final delivery that no longer changes, asked of the exoplanet
# archive's table service in a fixed order so the same bytes come back each time. NASA's data is public domain.
KEPLER_QUERY = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+kepoi_name,koi_pdisposition,"
    + ",".join(KEPLER_COLUMNS)
    + "+from+q1_q17_dr25_koi+order+by+kepoi_name&format=csv"
)

# The two answers the Kepler team's vetting gave each signal, in the order of their numbers.
KEPLER_CLASSES: tuple[str, ...] = (
    "planet candidate",
    "false positive",
)

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
    "kepler": (
        RemoteFile(KEPLER_QUERY, "79699f62ff42cdcbb7b10d0d50f73511", "kepler_dr25.csv"),
    ),
    # Karpathy's tiny Shakespeare, about a million characters of the plays in one file.
    "shakespeare": (
        RemoteFile("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "6fb458f1232090904fb40fe944165e91"),
    ),
    # The binary version from Krizhevsky's own page, read straight out of the archive, so no pickle is ever loaded.
    "cifar10": (
        RemoteFile("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", "c32a1d4ab5d03f1284b67883e8d87530"),
    ),
    # The UCI splice junction set of Noordewier, Towell and Shavlik, 1991, under CC BY 4.0, read out of the zip in memory.
    "splice": (
        RemoteFile(
            "https://archive.ics.uci.edu/static/public/69/molecular+biology+splice+junction+gene+sequences.zip",
            "abb671f0730dabacd67bfb3e2f16dade",
            "splice.zip",
        ),
    ),
    # NLTK's 18 books from Project Gutenberg, pinned to the commit that last changed them, read out of the zip in memory.
    "gutenberg": (
        RemoteFile(
            "https://raw.githubusercontent.com/nltk/nltk_data/5f09e470fe339e1a119dd174ca284129a3c7a5bb/packages/corpora/gutenberg.zip",
            "48c9c8605cd70b0230687557ee543633",
        ),
    ),
    # Cora from the LINQS group's public copy, unchanged since 2020, read out of the archive in memory. LINQS states
    # no license for it. It is credited to Sen and colleagues, 2008, from Lu and Getoor's selection of McCallum's papers.
    "cora": (
        RemoteFile("https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz", "2fc040bee8ce3d920e4204effd1e9214"),
    ),
    # California housing from the 1990 census, Pace and Barry's table, in the copy of Géron's book repository
    # (Apache 2.0) that adds each block's distance to the ocean as a category, pinned to the commit that last changed it.
    "housing": (
        RemoteFile(
            "https://raw.githubusercontent.com/ageron/handson-ml2/aa555aa16b9d0af6c84e67c7757f3317d05acb12/datasets/housing/housing.csv",
            "d1c47305887e2252bf1ccbd74ff159a6",
        ),
    ),
    # The CDC's weekly counts of doctor visits for flu-like illness in the US, 2002 to 2020, the ILI table of the long
    # forecasting benchmarks. US government data is public domain. This copy is the one torch-timeseries downloads,
    # pinned to the commit that last changed it, read out of the zip in memory.
    "illness": (
        RemoteFile(
            "https://raw.githubusercontent.com/wwy155/multivariate_timeseries_datasets/87a83a1fbff3bb61c507341f2695d988192bb33b/illness.zip",
            "e18542fafd509ad52ffedde1b0f1018d",
        ),
    ),
}

# The housing table's number columns, in the order load_housing gives them.
HOUSING_COLUMNS: tuple[str, ...] = (
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
)

# The housing table's one category, each block's distance to the ocean, in the order of their numbers.
HOUSING_PLACES: tuple[str, ...] = (
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
)

# The flu table's seven weekly series. The last, which the benchmarks call OT, is the total of patients seen, since the
# flu-like visits over it give the unweighted share.
ILLNESS_COLUMNS: tuple[str, ...] = (
    "% WEIGHTED ILI",
    "%UNWEIGHTED ILI",
    "AGE 0-4",
    "AGE 5-24",
    "ILITOTAL",
    "NUM. OF PROVIDERS",
    "OT",
)

# What happens at the middle of each window of the splice set, in the order of their numbers.
SPLICE_CLASSES: tuple[str, ...] = (
    "exon to intron",
    "intron to exon",
    "neither",
)

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

# Cora's seven topics, in the order of their numbers.
CORA_TOPICS: tuple[str, ...] = (
    "case based",
    "genetic algorithms",
    "neural networks",
    "probabilistic methods",
    "reinforcement learning",
    "rule learning",
    "theory",
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


@functools.cache
def read_kepler() -> tuple[Tensor, Tensor]:
    rows = list(csv.DictReader(io.StringIO(dataset_file("kepler", "kepler_dr25.csv").read_text())))
    # The archive leaves a measurement it doesn't have empty, which becomes nan.
    features = torch.tensor([[float(row[column]) if row[column] else math.nan for column in KEPLER_COLUMNS] for row in rows])
    labels = torch.tensor([("CANDIDATE", "FALSE POSITIVE").index(row["koi_pdisposition"]) for row in rows])

    return features, labels


def load_kepler() -> tuple[Tensor, Tensor]:
    """The 8,054 signals of NASA's Kepler DR25 table of objects of interest. features has shape (8054, 18), one
    column per name in KEPLER_COLUMNS, with nan where the table has no value. labels holds 0 for a planet candidate
    and 1 for a false positive, the verdict of the Kepler team's vetting, which KEPLER_CLASSES names."""
    features, labels = read_kepler()

    return features.clone(), labels.clone()


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
def read_splice() -> tuple[tuple[str, ...], Tensor]:
    # Read by its name in memory and never extracted, so no path in the archive is ever written to.
    with zipfile.ZipFile(dataset_file("splice", "splice.zip")) as archive:
        text = archive.read("splice.data").decode("ascii")

    # Each line is the label, the window's GenBank name and its 60 bases, with spaces around the commas.
    rows = [[part.strip() for part in line.split(",")] for line in text.strip().splitlines()]
    sequences = tuple(row[2] for row in rows)
    labels = torch.tensor([("EI", "IE", "N").index(row[0]) for row in rows])

    return sequences, labels


def load_splice() -> tuple[list[str], Tensor]:
    """The 3,190 windows of primate DNA in the UCI splice junction set, each 60 bases from GenBank with the place
    in question between its 30th and 31st base. labels holds 0 where an exon ends and an intron begins, 1 where an
    intron ends and an exon begins, and 2 where neither happens, which SPLICE_CLASSES names. 15 windows also hold
    D, N, R or S, letters that mark a base the sequencing couldn't settle."""
    sequences, labels = read_splice()

    return list(sequences), labels.clone()


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
def read_names() -> tuple[str, ...]:
    return tuple(dataset_file("names", "names.txt").read_text().split())


def load_names() -> list[str]:
    """The 32,033 names of makemore, in lowercase, in the file's order."""
    return list(read_names())


@functools.cache
def read_cora() -> tuple[Tensor, Tensor, Tensor]:
    # Members are read in memory by their names and never extracted, so no path in the archive is ever written to.
    with tarfile.open(dataset_file("cora", "cora.tgz"), "r:gz") as archive:
        content = archive.extractfile("cora/cora.content").read().decode("ascii")
        cites = archive.extractfile("cora/cora.cites").read().decode("ascii")

    rows = [line.split("\t") for line in content.strip().split("\n")]
    names = sorted({row[-1] for row in rows})
    position = {row[0]: index for index, row in enumerate(rows)}
    words = torch.tensor([[int(value) for value in row[1:-1]] for row in rows], dtype=torch.float32)
    topics = torch.tensor([names.index(row[-1]) for row in rows])
    # 151 pairs of papers cite each other, so each link is kept once, the smaller number first.
    links = {tuple(sorted((position[cited], position[citing]))) for cited, citing in (line.split("\t") for line in cites.strip().split("\n"))}

    return words, topics, torch.tensor(sorted(links))


def load_cora() -> tuple[Tensor, Tensor, Tensor]:
    """The Cora citation graph of 2,708 machine learning papers. words has shape (2708, 1433), a 1 where a paper uses
    a word of the vocabulary. topics holds each paper's topic from 0 to 6, which CORA_TOPICS names. links has shape
    (5278, 2), each pair of papers joined by a citation listed once."""
    words, topics, links = read_cora()

    return words.clone(), topics.clone(), links.clone()


@functools.cache
def read_housing() -> tuple[Tensor, Tensor, Tensor]:
    with dataset_file("housing", "housing.csv").open(newline="") as table:
        rows = list(csv.DictReader(table))

    # 207 blocks have no bedroom count, so it reads as nan.
    numbers = torch.tensor([[float(row[column]) if row[column] else math.nan for column in HOUSING_COLUMNS] for row in rows])
    places = torch.tensor([HOUSING_PLACES.index(row["ocean_proximity"]) for row in rows])
    values = torch.tensor([float(row["median_house_value"]) for row in rows])

    return numbers, places, values


def load_housing() -> tuple[Tensor, Tensor, Tensor]:
    """California's 20,640 census blocks of 1990. numbers has shape (20640, 8), the columns HOUSING_COLUMNS names, with
    nan where a bedroom count is missing. places holds each block's distance to the ocean from 0 to 4, which
    HOUSING_PLACES names. values holds each block's median house value in dollars, capped at 500,001."""
    numbers, places, values = read_housing()

    return numbers.clone(), places.clone(), values.clone()


@functools.cache
def read_illness() -> Tensor:
    # Read by its name in memory and never extracted, so no path in the archive is ever written to.
    with zipfile.ZipFile(dataset_file("illness", "illness.zip")) as archive:
        text = archive.read("illness/national_illness.csv").decode("ascii")

    rows = list(csv.DictReader(io.StringIO(text)))

    return torch.tensor([[float(row[column]) for column in ILLNESS_COLUMNS] for row in rows])


def load_illness() -> Tensor:
    """The CDC's flu table, 966 weeks from 2002 to 2020, of shape (966, 7), the columns ILLNESS_COLUMNS names."""
    return read_illness().clone()
