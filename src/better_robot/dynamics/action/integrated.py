"""``IntegratedActionModel`` — discrete-time wrapper around a differential model.

Crocoddyl 3-layer split, layer 2. Skeleton only.
See ``docs/06_DYNAMICS.md §6``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .action import ActionModel
from .differential import DifferentialActionModel


@dataclass
class IntegratedActionModelEuler(ActionModel):
    differential: DifferentialActionModel
    dt: float = 0.01
    with_cost_residual: bool = True

    def calc(self, data, x: torch.Tensor, u: torch.Tensor) -> None:
        """``x_{k+1} = state_manifold.integrate(x, dt * f(x, u))``.

        See docs/06_DYNAMICS.md §6.
        """
        raise NotImplementedError("see docs/06_DYNAMICS.md §6")

    def calc_diff(self, data, x: torch.Tensor, u: torch.Tensor) -> None:
        """Compose continuous-time Jacobians with manifold integrate Jacobians.

        See docs/06_DYNAMICS.md §6.
        """
        raise NotImplementedError("see docs/06_DYNAMICS.md §6")


@dataclass
class IntegratedActionModelRK4(ActionModel):
    differential: DifferentialActionModel
    dt: float = 0.01
    with_cost_residual: bool = True

    def calc(self, data, x: torch.Tensor, u: torch.Tensor) -> None:
        """RK4 step over the state manifold. See docs/06_DYNAMICS.md §6."""
        raise NotImplementedError("see docs/06_DYNAMICS.md §6")

    def calc_diff(self, data, x: torch.Tensor, u: torch.Tensor) -> None:
        raise NotImplementedError("see docs/06_DYNAMICS.md §6")
