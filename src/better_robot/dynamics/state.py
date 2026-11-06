"""Dynamics on a `KinematicState`.

``inverse_dynamics``, ``mass_matrix`` and ``forward_dynamics`` read the world transforms of a state computed by
`forward_kinematics`; the world-frame geometry they share (Jacobian columns) is built once per state and
memoised on it. See `_world` for the formulas.
"""

from __future__ import annotations

import math

import torch

from ..kinematics.configuration import full_tangent, reduced_columns, reduced_mass_matrix
from ..kinematics import KinematicState
from ..model._checks import check_tensor, expand_batch
from ..model.model import Model
from ._world import (
    DynamicsPlan,
    WorldKinematics,
    centroidal_map_full,
    dynamics_plan,
    inverse_dynamics_full,
    mass_matrix_full,
    world_geometry,
    world_kinematics,
)


def _plan(model: Model, state: KinematicState) -> DynamicsPlan:
    return dynamics_plan(model, dtype=state.q.dtype, device=state.q.device)


def _world(model: Model, state: KinematicState) -> tuple[WorldKinematics, tuple[int, ...]]:
    """World-frame kinematics of ``state`` and the state's batch shape.

    The model-independent geometry is memoised on the state; the inertias are read from ``model`` on every call.
    """
    batch = tuple(state.q.shape[:-1])
    if "world_geometry" not in state._memo:
        transform = state.joint_transforms.reshape(math.prod(batch), model.njoints, 3, 4)
        state._memo["world_geometry"] = world_geometry(model, transform, _plan(model, state))
    return world_kinematics(state._memo["world_geometry"], model, batch), batch


def _tangent(model: Model, batch: tuple[int, ...], value: torch.Tensor, name: str) -> torch.Tensor:
    """A checked ``(..., nv)`` tangent broadcast to the state's batch, flattened to ``(N, nv_full)``."""
    check_tensor(name, value, shape=(model.nv,), dtype=model.dtype, device=model.device)
    value = expand_batch(name, value, batch, (model.nv,))
    return full_tangent(model, value).reshape(math.prod(batch), model.nv_full)


def inverse_dynamics(
    model: Model, state: KinematicState, *, v: torch.Tensor, a: torch.Tensor, fext: torch.Tensor | None = None
) -> torch.Tensor:
    """``τ = M(q) a + C(q, v) v + g(q) - Jᵀ fext`` for ``v, a: (B..., nv)``. Returns ``(B..., nv)``.

    ``M`` includes the motor armature: ``model.armature * a`` (``armature: (nv,)``) is added per coordinate.
    ``fext`` is an optional ``(B..., njoints, 6)`` wrench per joint in the joint's own frame.
    Bias forces are ``inverse_dynamics(model, state, v=v, a=zeros)``; gravity torque uses zeros twice.
    """
    kin, batch = _world(model, state)
    count = math.prod(batch)
    v_full, a_full = _tangent(model, batch, v, "v"), _tangent(model, batch, a, "a")
    if fext is not None:
        check_tensor("fext", fext, shape=(model.njoints, 6), dtype=model.dtype, device=model.device)
        fext = expand_batch("fext", fext, batch, (model.njoints, 6)).reshape(count, model.njoints, 6)
    gravity = model.gravity.broadcast_to(*batch, 6).reshape(count, 6)
    tau_full, _, _, _ = inverse_dynamics_full(kin, _plan(model, state), gravity, v_full, a_full, fext)
    return reduced_columns(model, tau_full).view(*batch, model.nv) + model.armature * a


def mass_matrix(model: Model, state: KinematicState) -> torch.Tensor:
    """Joint-space inertia matrix ``M(q)``, ``(B..., nv, nv)``, symmetric positive definite.

    The motor armature ``model.armature (nv,)`` is added on the diagonal.
    """
    kin, batch = _world(model, state)
    mass = mass_matrix_full(kin, _plan(model, state))
    return reduced_mass_matrix(model, mass).view(*batch, model.nv, model.nv) + torch.diag(model.armature)


def forward_dynamics(
    model: Model, state: KinematicState, *, v: torch.Tensor, tau: torch.Tensor, fext: torch.Tensor | None = None
) -> torch.Tensor:
    """``ddq = M(q)⁻¹ (τ - C(q, v) v - g(q) + Jᵀ fext)`` for ``v, tau: (B..., nv)``. Returns ``(B..., nv)``."""
    bias = inverse_dynamics(model, state, v=v, a=torch.zeros_like(v), fext=fext)
    check_tensor("tau", tau, shape=(model.nv,), dtype=model.dtype, device=model.device)
    tau = expand_batch("tau", tau, tuple(bias.shape[:-1]), (model.nv,))
    factor = torch.linalg.cholesky(mass_matrix(model, state))
    return torch.cholesky_solve((tau - bias).unsqueeze(-1), factor).squeeze(-1)


def center_of_mass(model: Model, state: KinematicState) -> torch.Tensor:
    """Whole-body centre of mass in the world frame, ``(B..., 3)``."""
    inertias = model.body_inertias  # (..., njoints, 10): mass, com, inertia
    masses, com_local = inertias[..., 0], inertias[..., 1:4]
    com_world = (state.joint_rotations @ com_local.unsqueeze(-1)).squeeze(-1) + state.joint_translations
    return (masses.unsqueeze(-1) * com_world).sum(-2) / masses.sum(-1).unsqueeze(-1)


def centroidal_map(model: Model, state: KinematicState) -> torch.Tensor:
    """Centroidal momentum matrix ``A_g(q)``, ``(B..., 6, nv)``: ``h_g = A_g v`` in world axes at the COM."""
    kin, batch = _world(model, state)
    com = center_of_mass(model, state).reshape(math.prod(batch), 3)
    columns = centroidal_map_full(kin, _plan(model, state), com)
    return reduced_columns(model, columns).view(*batch, 6, model.nv)


def centroidal_momentum(model: Model, state: KinematicState, *, v: torch.Tensor) -> torch.Tensor:
    """Centroidal momentum ``h_g = A_g(q) v`` for ``v: (B..., nv)``, ``(B..., 6)``; rows as `centroidal_map`."""
    check_tensor("v", v, shape=(model.nv,), dtype=model.dtype, device=model.device)
    return (centroidal_map(model, state) @ v.unsqueeze(-1)).squeeze(-1)


__all__ = [
    "center_of_mass",
    "centroidal_map",
    "centroidal_momentum",
    "forward_dynamics",
    "inverse_dynamics",
    "mass_matrix",
]
