"""MNIST's splits, which the chapters that train on handwritten digits share, from LeNet in Section 5 to the
autoencoders of Section 8. MNIST has no project of its own, since Section 2's project is the exoplanet hunter."""

from __future__ import annotations

import functools

import torch
from torch import Tensor

from ..data import load_mnist


@functools.cache
def mnist_sets() -> dict[str, Tensor]:
    """The splits the lessons quote numbers from, built once per run of the app."""
    images, labels = load_mnist("train")
    test_images, test_labels = load_mnist("test")
    x = images.reshape(len(images), -1).float() / 255
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))

    return {
        "train_x": x[order[:50000]],
        "train_y": labels[order[:50000]],
        "validation_x": x[order[50000:]],
        "validation_y": labels[order[50000:]],
        "small_x": x[order[:1000]],
        "small_y": labels[order[:1000]],
        "test_x": test_images.reshape(len(test_images), -1).float() / 255,
        "test_y": test_labels,
    }
