"""Shared helpers: device selection and seeding."""

import random

import numpy as np
import torch


def get_device(override: str | None = None) -> torch.device:
    """Return `override` if given, else MPS when available, else CPU."""
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
