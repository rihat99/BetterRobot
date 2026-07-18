"""The evaluation-context ``Residual`` protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch

from .._validation import check_tensor


def _configuration(ctx: Mapping[str, Any]) -> torch.Tensor:
    """Return the configuration tensor from a residual evaluation context."""
    return check_tensor("q", ctx["q"])


@runtime_checkable
class Residual(Protocol):
    """Callable residual over a declared evaluation context."""

    name: str
    dim: int
    reads: tuple[str, ...]

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        """Return the residual vector of shape ``(B..., dim)``."""
        ...
