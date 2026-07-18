"""Derivatives of RNEA / ABA / CRBA / centroidal dynamics.

The forward routines themselves (``rnea`` / ``aba`` / ``crba`` / ``ccrba``)
are written with pure differentiable PyTorch ops (no in-place writes,
no ``.item()`` calls in the hot path), so ordinary ``loss.backward()`` uses
PyTorch autograd through the implemented recursions.

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

    Unbatched shapes are ``(nv, nq)``, ``(nv, nv)``, and ``(nv, nv)``.
    With a shared batch prefix ``*B``, the full Jacobians retain output and
    input batch axes: ``(*B, nv, *B, nq)`` for ``q`` and
    ``(*B, nv, *B, nv)`` for ``v`` and ``a``.
    """

    def _fn(q_, v_, a_):
        return _rnea(model, q_, v_, a_, fext=fext)

    dtau_dq, dtau_dv, dtau_da = torch.autograd.functional.jacobian(
        _fn, (q.detach(), v.detach(), a.detach()), create_graph=False, vectorize=False
    )
    # Keep the full Jacobians. In particular, q has trailing size nq rather
    # than nv for free-flyer robots, and batched inputs retain both batch axes.
    return dtau_dq, dtau_dv, dtau_da


def compute_aba_derivatives(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    fext: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(∂a/∂q, ∂a/∂v, ∂a/∂τ = M⁻¹)`` as full Jacobians.

    Shapes follow :func:`compute_rnea_derivatives`, replacing output ``τ``
    with output acceleration ``a``.
    """

    def _fn(q_, v_, tau_):
        return aba(model, q_, v_, tau_, fext=fext)

    da_dq, da_dv, da_dtau = torch.autograd.functional.jacobian(
        _fn, (q.detach(), v.detach(), tau.detach()), create_graph=False, vectorize=False
    )
    return da_dq, da_dv, da_dtau


def compute_crba_derivatives(
    model: Model,
    data: Data,
    q: torch.Tensor,
) -> torch.Tensor:
    """Return the full ``∂M/∂q`` Jacobian.

    Its unbatched shape is ``(nv, nv, nq)``. With batch prefix ``*B`` it is
    ``(*B, nv, nv, *B, nq)``.
    """

    def _fn(q_):
        return crba(model, q_)

    return torch.autograd.functional.jacobian(_fn, (q.detach(),))[0]
