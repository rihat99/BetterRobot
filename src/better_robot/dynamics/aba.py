"""Forward dynamics (Featherstone's ABA).

**Skeleton only.** See ``docs/06_DYNAMICS.md §2``, milestone D4.
"""

from __future__ import annotations

import torch

from ..data_model.data import Data
from ..data_model.model import Model


def aba(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    *,
    fext: torch.Tensor | None = None,
) -> torch.Tensor:
    """Articulated Body Algorithm — ``ddq = M(q)^{-1} (τ - b(q, v) - g(q) + J^T fext)``.

    Populates ``data.ddq``.

    TODO(milestone D4). See docs/06_DYNAMICS.md §2.
    """
    raise NotImplementedError("TODO(milestone D4) — see docs/06_DYNAMICS.md §2")
