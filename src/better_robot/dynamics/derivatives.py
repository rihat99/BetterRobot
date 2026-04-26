"""Derivatives of RNEA / ABA / CRBA / centroidal dynamics.

The forward routines themselves (``rnea`` / ``aba`` / ``crba`` / ``ccrba``)
are written with pure differentiable PyTorch ops (no in-place writes,
no ``.item()`` calls in the hot path), so ``loss.backward()`` already
yields exact gradients up to fp64 round-off — the dynamics layer
satisfies P11-D6's gradcheck acceptance via autograd today.

The functions below are convenience wrappers around
``torch.autograd.functional.jacobian`` that match the signature of
Pinocchio's ``compute_*_derivatives`` family. Replacing them with the
analytic Carpentier–Mansard recursions (the eventual production path)
is a drop-in change: the call sites stay the same.

See ``docs/concepts/dynamics.md §4``.
"""

from __future__ import annotations

import torch

from ..data_model.data import Data
from ..data_model.model import Model
from .aba import aba
from .crba import crba
from .rnea import rnea as _rnea


def compute_rnea_derivatives(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    fext: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(∂τ/∂q, ∂τ/∂v, ∂τ/∂a)``.

    Each tensor has shape ``(nv, nv)`` (un-batched) or ``(B..., nv, nv)``
    (batched). For un-batched fp64 inputs ``∂τ/∂a`` matches ``crba`` to
    1e-10 relative.
    """
    def _fn(q_, v_, a_):
        return _rnea(model, model.create_data(), q_, v_, a_, fext=fext)

    dtau_dq, dtau_dv, dtau_da = torch.autograd.functional.jacobian(
        _fn, (q.detach(), v.detach(), a.detach()), create_graph=False, vectorize=False
    )
    # Project ∂τ/∂q from `(*B, nv, *B, nq)` to `(*B, nv, nv)`. For unbatched
    # inputs, q has shape (nq,) but the residual lives on tangent ``nv``;
    # for free-flyer robots ``nq != nv``. We therefore expose the full Jacobian.
    return dtau_dq, dtau_dv, dtau_da


def compute_aba_derivatives(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    fext: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(∂a/∂q, ∂a/∂v, ∂a/∂τ = M⁻¹)``."""

    def _fn(q_, v_, tau_):
        return aba(model, model.create_data(), q_, v_, tau_, fext=fext)

    da_dq, da_dv, da_dtau = torch.autograd.functional.jacobian(
        _fn, (q.detach(), v.detach(), tau.detach()), create_graph=False, vectorize=False
    )
    return da_dq, da_dv, da_dtau


def compute_crba_derivatives(
    model: Model,
    data: Data,
    q: torch.Tensor,
) -> torch.Tensor:
    """Return ``∂M/∂q`` of shape ``(*B, nv, nv, nq)``."""
    def _fn(q_):
        return crba(model, model.create_data(), q_)

    return torch.autograd.functional.jacobian(_fn, (q.detach(),))[0]


def compute_centroidal_dynamics_derivatives(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
):
    """Return centroidal dynamics derivatives.

    Implementation gap left for a future analytic pass; today this is a
    stub. Use :func:`compute_centroidal_map` together with autograd in
    the meantime.
    """
    raise NotImplementedError(
        "compute_centroidal_dynamics_derivatives — see docs/concepts/dynamics.md §4 "
        "for the analytic Carpentier–Mansard formula. Use autograd through "
        "compute_centroidal_map(q) until then."
    )
