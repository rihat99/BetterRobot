"""``ActionModel`` — abstract base + paired ``ActionData`` for the OC solver.

The 3-layer Crocoddyl split:

* ``DifferentialActionModel`` — continuous-time dynamics ``ẋ = f(x, u)``
  and a running cost.
* ``IntegratedActionModel`` — discrete-time wrapper.
* ``ActionModel`` — the protocol the solver consumes; integrated models
  satisfy it by composition.

A DDP / iLQR solver is intentionally not yet wired in — the OC layer is
deferred to a follow-on release. The forward & gradient evaluation
machinery in this package is sufficient to plug into any external
solver that consumes the protocol below.

See ``docs/concepts/dynamics.md §6``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import torch

from ..state_manifold import StateMultibody


@dataclass
class ActionData:
    """Per-knot scratchpad written by ``calc`` / ``calc_diff``."""

    xnext: Optional[torch.Tensor] = None        # (..., nx) next state
    cost: Optional[torch.Tensor] = None         # (...,) running cost
    fx: Optional[torch.Tensor] = None           # (..., ndx, ndx)
    fu: Optional[torch.Tensor] = None           # (..., ndx, nu)
    lx: Optional[torch.Tensor] = None           # (..., ndx)
    lu: Optional[torch.Tensor] = None           # (..., nu)
    lxx: Optional[torch.Tensor] = None
    lxu: Optional[torch.Tensor] = None
    luu: Optional[torch.Tensor] = None
    extras: dict = field(default_factory=dict)


class ActionModel(Protocol):
    """Protocol for the action model consumed by DDP/iLQR."""

    state: StateMultibody
    nu: int

    def calc(self, data: ActionData, x: torch.Tensor, u: torch.Tensor) -> None: ...
    def calc_diff(self, data: ActionData, x: torch.Tensor, u: torch.Tensor) -> None: ...

    def create_data(self) -> ActionData:
        return ActionData()
