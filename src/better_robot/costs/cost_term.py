"""CostTerm: a single differentiable cost or constraint term."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch


@dataclass
class CostTerm:
    """A single differentiable cost or constraint term.

    Attributes:
        residual_fn: Pure function mapping variable tensor to residual vector.
            Must have signature ``(x: Tensor) -> Tensor``.
            Use ``functools.partial`` to bind extra arguments before passing.
        weight: Scalar weight applied to the residual.
        kind: ``'soft'`` terms are minimized via least squares.
            ``'constraint_leq_zero'`` terms are enforced (residual <= 0).
    """

    residual_fn: Callable[[torch.Tensor], torch.Tensor]
    weight: float = 1.0
    kind: Literal["soft", "constraint_leq_zero"] = "soft"
