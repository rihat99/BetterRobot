"""``DifferentialActionModel`` — continuous-time dynamics + per-knot cost.

Crocoddyl 3-layer split, layer 1. Skeleton only; see
``docs/06_DYNAMICS.md §6``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...data_model.model import Model
from ..state_manifold import StateMultibody


@dataclass
class DifferentialActionModel:
    """Continuous-time action model: defines ``ẋ = f(x, u)`` and a running cost."""

    model: Model
    state_manifold: StateMultibody
    cost_stack: object  # forward reference; avoid circular imports

    def calc(self, data, x: torch.Tensor, u: torch.Tensor) -> None:
        """Evaluate ``ẋ = f(x, u)`` and the running cost.

        See docs/06_DYNAMICS.md §6.
        """
        raise NotImplementedError("see docs/06_DYNAMICS.md §6")

    def calc_diff(self, data, x: torch.Tensor, u: torch.Tensor) -> None:
        """Evaluate ``Fx, Fu, Lx, Lu, Lxx, Lxu, Luu``.

        See docs/06_DYNAMICS.md §6.
        """
        raise NotImplementedError("see docs/06_DYNAMICS.md §6")


class DifferentialActionModelFreeFwd(DifferentialActionModel):
    """Forward dynamics with no contacts — will call ``aba`` + ``aba_derivatives``."""
