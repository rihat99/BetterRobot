"""Composite Rigid Body Algorithm — joint-space inertia matrix ``M(q)``.

**Skeleton only.** See ``docs/06_DYNAMICS.md §2``, milestone D3.
"""

from __future__ import annotations

import torch

from ..data_model.data import Data
from ..data_model.model import Model


def crba(
    model: Model,
    data: Data,
    q: torch.Tensor,
) -> torch.Tensor:
    """Joint-space inertia matrix ``M(q)``. Shape: ``(B..., nv, nv)``.

    Populates ``data.mass_matrix``.

    TODO(milestone D3). See docs/06_DYNAMICS.md §2.
    """
    raise NotImplementedError("TODO(milestone D3) — see docs/06_DYNAMICS.md §2")


def compute_minverse(
    model: Model,
    data: Data,
    q: torch.Tensor,
) -> torch.Tensor:
    """Direct ``M(q)^{-1}`` computation via the ABA factorisation.

    TODO(milestone D4). See docs/06_DYNAMICS.md §2.
    """
    raise NotImplementedError("TODO(milestone D4) — see docs/06_DYNAMICS.md §2")
