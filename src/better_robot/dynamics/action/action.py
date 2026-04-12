"""``ActionModel`` — the abstract base class the optimal-control solver sees.

Crocoddyl 3-layer split, layer 3. Skeleton only.
See ``docs/06_DYNAMICS.md §6``.
"""

from __future__ import annotations

from typing import Protocol

import torch

from ..state_manifold import StateMultibody


class ActionModel(Protocol):
    """Protocol for the action model consumed by DDP/iLQR."""

    state: StateMultibody
    nu: int

    def calc(self, data, x: torch.Tensor, u: torch.Tensor) -> None: ...
    def calc_diff(self, data, x: torch.Tensor, u: torch.Tensor) -> None: ...
