"""What exercises that train for minutes share, the best device and the checkpoint their check reads.

A long run happens outside the app, with `run.py train <exercise>`, which calls the exercise file's main().
It saves the model with save_checkpoint, and the exercise's check loads it with load_checkpoint on the CPU."""

from __future__ import annotations

import torch
import torch.nn as nn

from .paths import CHECKPOINTS_DIR, START_COMMAND


class MissingCheckpoint(Exception):
    """A check needs a model the learner hasn't trained yet. The message says how to train it."""


def best_device() -> torch.device:
    """The fastest device the machine has, CUDA, MPS or another accelerator, and the CPU when there is none, per DEC-016."""
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()

    return torch.device("cpu")


def save_checkpoint(
    model: nn.Module,
    name: str,
) -> None:
    CHECKPOINTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    # Moved to the CPU first, so a model trained on MPS or CUDA loads anywhere.
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, CHECKPOINTS_DIR / f"{name}.pt")


def load_checkpoint(
    model: nn.Module,
    name: str,
) -> nn.Module:
    path = CHECKPOINTS_DIR / f"{name}.pt"

    if not path.exists():
        raise MissingCheckpoint(
            f"There is no trained model for {name} yet. Train it once from the project folder, which takes a few "
            f"minutes, then save the file again.\n\n    {START_COMMAND} train {name}",
        )

    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    return model
