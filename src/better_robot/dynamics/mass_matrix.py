"""The joint-space inertia matrix of a `KinematicState`, and forward dynamics through it."""

from __future__ import annotations

import torch

from ..kinematics import KinematicState
from ..kinematics.configuration import reduced_mass_matrix
from ..lie._checks import check_tensor
from ..model._checks import expand_batch
from ..model.model import Model
from ._world import mass_matrix_full, world_kinematics
from .inverse_dynamics import _external_forces, _full_tangent, _torques


def mass_matrix(model: Model, state: KinematicState) -> torch.Tensor:
    """Joint-space inertia matrix ``M(q)``, ``(B..., nv, nv)``, symmetric positive definite.

    The motor armature ``model.armature (nv,)`` is added on the diagonal. Composite rigid bodies in the world frame
    (`_world`).
    """
    kin = world_kinematics(model, state)
    mass = reduced_mass_matrix(model, mass_matrix_full(kin))
    return mass.view(*kin.batch, model.nv, model.nv) + torch.diag(model.armature)


def forward_dynamics(
    model: Model,
    state: KinematicState,
    *,
    v: torch.Tensor,
    tau: torch.Tensor,
    external_forces: torch.Tensor | None = None,
) -> torch.Tensor:
    """Joint accelerations ``ddq = M(q)⁻¹ (τ - C(q, v) v - g(q) + Σ_i J_iᵀ f_i)``, ``(B..., nv)``.

    ``v, tau: (B..., nv)``; ``external_forces`` as in `inverse_dynamics`. The bias forces come from
    `inverse_dynamics` with zero acceleration, then a Cholesky solve with `mass_matrix`. Raises as `inverse_dynamics`.
    """
    kin = world_kinematics(model, state)
    v_full = _full_tangent(model, kin.batch, v, "v")
    bias = _torques(model, kin, v_full, torch.zeros_like(v_full), _external_forces(model, kin.batch, external_forces))
    check_tensor("tau", tau, shape=(model.nv,), dtype=model.dtype, device=model.device)
    tau = expand_batch("tau", tau, kin.batch, (model.nv,))
    factor = torch.linalg.cholesky(mass_matrix(model, state))
    return torch.cholesky_solve((tau - bias).unsqueeze(-1), factor).squeeze(-1)


__all__ = ["forward_dynamics", "mass_matrix"]
