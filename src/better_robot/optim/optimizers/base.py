"""``Optimizer`` protocol and ``OptimizationResult`` struct.

Every optimiser takes the same keyword components (``linear_solver``,
``kernel``, ``strategy``, ``scheduler``) — replacing one swaps one knob
without touching the optimisation loop.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §5, §9``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from ..problem import LeastSquaresProblem


@dataclass
class OptimizationResult:
    """Result of an optimiser run."""

    x: torch.Tensor
    residual: torch.Tensor
    iters: int
    converged: bool
    history: list[dict] = field(default_factory=list)


class Optimizer(Protocol):
    """Optimiser protocol. Concrete classes live under ``optim/optimizers/``."""

    def minimize(
        self,
        problem: LeastSquaresProblem,
        *,
        max_iter: int,
        linear_solver,
        kernel,
        strategy,
        scheduler=None,
    ) -> OptimizationResult:
        ...
