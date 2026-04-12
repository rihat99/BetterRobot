"""Analytic derivatives of RNEA / ABA / CRBA / centroidal dynamics.

These are what make DDP / iLQR cheap enough to use without autograd. All
signatures mirror Pinocchio's ``algorithm/derivatives/`` directory.

**Skeleton only.** See ``docs/06_DYNAMICS.md §4``, milestone D6.
"""

from __future__ import annotations

import torch

from ..data_model.data import Data
from ..data_model.model import Model


def compute_rnea_derivatives(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    fext: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(∂τ/∂q, ∂τ/∂v, ∂τ/∂a = M)`` each of shape ``(B..., nv, nv)``.

    TODO(milestone D6). See docs/06_DYNAMICS.md §4.
    """
    raise NotImplementedError("TODO(milestone D6) — see docs/06_DYNAMICS.md §4")


def compute_aba_derivatives(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    fext: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(∂a/∂q, ∂a/∂v, ∂a/∂τ = M^{-1})`` each of shape ``(B..., nv, nv)``.

    TODO(milestone D6). See docs/06_DYNAMICS.md §4.
    """
    raise NotImplementedError("TODO(milestone D6) — see docs/06_DYNAMICS.md §4")


def compute_crba_derivatives(
    model: Model,
    data: Data,
    q: torch.Tensor,
) -> torch.Tensor:
    """Return ``∂M/∂q`` of shape ``(B..., nv, nv, nv)``.

    TODO(milestone D6). See docs/06_DYNAMICS.md §4.
    """
    raise NotImplementedError("TODO(milestone D6) — see docs/06_DYNAMICS.md §4")


def compute_centroidal_dynamics_derivatives(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
):
    """Return centroidal dynamics derivatives.

    TODO(milestone D6). See docs/06_DYNAMICS.md §4.
    """
    raise NotImplementedError("TODO(milestone D6) — see docs/06_DYNAMICS.md §4")
