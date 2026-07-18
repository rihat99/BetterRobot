"""The named-context ``Residual`` protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Residual(Protocol):
    """Callable residual over a declared named-block context."""

    name: str
    dim: int
    reads: tuple[str, ...]

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        """Return the residual vector of shape ``(B..., dim)``."""
        ...
