"""Utility functions for dflash."""

import time
import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """Resolve the best available device or use the specified one.

    Args:
        device: Optional device string (e.g. 'cpu', 'cuda', 'cuda:0', 'mps').
                If None, auto-selects based on availability.

    Returns:
        A torch.device instance.
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: A PyTorch module.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def human_readable_size(num: int, suffix: str = "") -> str:
    """Convert a large integer to a human-readable string (K, M, B).

    Args:
        num: The number to format.
        suffix: Optional suffix to append (e.g. 'params').

    Returns:
        Formatted string.
    """
    for unit in ("", "K", "M", "B"):
        if abs(num) < 1000:
            label = f"{unit}{suffix}".strip()
            return f"{num:.1f} {label}" if label else f"{num:.1f}"
        num /= 1000  # type: ignore[assignment]
    label = f"T{suffix}".strip()
    return f"{num:.1f} {label}"


class Timer:
    """Simple context-manager timer.

    Usage::

        with Timer("forward pass") as t:
            output = model(input)
        print(t.elapsed_ms, "ms")
    """

    def __init__(self, name: str = "", verbose: bool = False) -> None:
        self.name = name
        self.verbose = verbose
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self.verbose:
            label = f"[{self.name}] " if self.name else ""
            logger.debug("%s%.2f ms", label, self.elapsed_ms)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy (if available), and PyTorch.

    Args:
        seed: Integer seed value.
    """
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to logits.

    Tokens with cumulative probability above *top_p* are masked out so they
    cannot be sampled.

    Args:
        logits: 1-D or 2-D float tensor of shape ``(vocab,)`` or
                ``(batch, vocab)``.
        top_p: Cumulative probability threshold in (0, 1].

    Returns:
        Filtered logits tensor with the same shape; masked positions are set
        to ``-inf``.
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens that push cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    # Scatter back to original ordering
    return logits.scatter(-1, sorted_indices, sorted_logits)
