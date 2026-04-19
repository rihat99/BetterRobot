"""Centroidal dynamics — center of mass, centroidal momentum matrix, CCRBA.

``center_of_mass`` is the only function in ``dynamics/`` that lands in v1
(it needs only kinematic machinery). Everything else is skeleton.

See ``docs/06_DYNAMICS.md §3``.
"""

from __future__ import annotations

import torch

from ..data_model.data import Data
from ..data_model.model import Model


def center_of_mass(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
) -> torch.Tensor:
    """Whole-body center of mass. ``(B..., 3)``. Populates
    ``data.com_position`` (and ``data.com_velocity``/
    ``data.com_acceleration`` if ``v``/``a`` are given).

    Milestone D1 — implementable now (kinematic only). Skeleton for the
    moment; see docs/06_DYNAMICS.md §3.
    """
    raise NotImplementedError("TODO(milestone D1) — see docs/06_DYNAMICS.md §3")


def compute_centroidal_map(
    model: Model,
    data: Data,
    q: torch.Tensor,
) -> torch.Tensor:
    """Centroidal momentum matrix ``A_g(q)``. Shape: ``(B..., 6, nv)``.
    Populates ``data.centroidal_momentum_matrix``.

    TODO(milestone D5). See docs/06_DYNAMICS.md §3.
    """
    raise NotImplementedError("TODO(milestone D5) — see docs/06_DYNAMICS.md §3")


def compute_centroidal_momentum(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """``h_g = A_g(q) v``. Populates ``data.centroidal_momentum``.

    TODO(milestone D5). See docs/06_DYNAMICS.md §3.
    """
    raise NotImplementedError("TODO(milestone D5) — see docs/06_DYNAMICS.md §3")


def ccrba(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centroidal CRBA — returns ``(A_g, h_g)``.

    TODO(milestone D5). See docs/06_DYNAMICS.md §3.
    """
    raise NotImplementedError("TODO(milestone D5) — see docs/06_DYNAMICS.md §3")
