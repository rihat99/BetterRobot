"""Dynamics on a :class:`~better_robot.kinematics.state.KinematicState`.

``inverse_dynamics``, ``mass_matrix`` and ``forward_dynamics`` read the world transforms of a
state computed by :func:`~better_robot.kinematics.forward_kinematics_state`; the world-frame
quantities they share (Jacobian columns, world inertias) are built once per state and memoised
on it. See :mod:`._world` for the formulas.
"""

from __future__ import annotations

import math

import torch

from ..data_model.execution_batch import broadcast_to_execution_batch
from ..data_model.model import Model
from ..data_model.reduced_coordinates import (
    expand_tangent,
    reduce_generalized_force,
    reduce_jacobian,
    reduce_mass_matrix,
)
from ..kinematics.state import KinematicState
from ._world import (
    WorldKinematics,
    centroidal_map_full,
    get_dynamics_plan,
    inverse_dynamics_full,
    mass_matrix_full,
    world_geometry,
    world_kinematics,
)


def _world(model: Model, state: KinematicState) -> tuple[WorldKinematics, tuple[int, ...]]:
    """World-frame kinematics of ``state`` and the state's batch shape.

    The model-independent geometry is memoised on the state; the inertias are read from ``model``
    on every call (cached on its values, invalidated by in-place edits).
    """
    batch = tuple(state.q.shape[:-1])
    structure = model.structure
    if "world_geometry" not in state._memo:
        transform = state.joint_transform.reshape(math.prod(batch), structure.njoints, 3, 4)
        state._memo["world_geometry"] = world_geometry(structure, transform, get_dynamics_plan(structure))
    return world_kinematics(state._memo["world_geometry"], model.values, batch, structure), batch


def _tangent(model: Model, batch: tuple[int, ...], value: torch.Tensor, name: str) -> torch.Tensor:
    """Broadcast a ``(..., nv)`` tangent to the state's batch, flattened to ``(N, nv_full)``."""
    value = broadcast_to_execution_batch(value, batch, (model.nv,), name=name)
    return expand_tangent(model.structure, value).reshape(math.prod(batch), -1)


def inverse_dynamics(
    model: Model, state: KinematicState, v: torch.Tensor, a: torch.Tensor, *, fext: torch.Tensor | None = None
) -> torch.Tensor:
    """``τ = M(q) a + C(q, v) v + g(q) - Jᵀ fext`` for ``v, a: (B..., nv)``. Returns ``(B..., nv)``.

    ``fext`` is an optional ``(B..., njoints, 6)`` wrench per joint in the joint's own frame.
    Bias forces are ``inverse_dynamics(model, state, v, zeros)``; gravity torque uses zeros twice.
    """
    kin, batch = _world(model, state)
    count = math.prod(batch)
    structure, values = model.structure, model.values
    v_full, a_full = _tangent(model, batch, v, "v"), _tangent(model, batch, a, "a")
    if fext is not None:
        fext = broadcast_to_execution_batch(fext, batch, (structure.njoints, 6), name="fext").reshape(
            count, structure.njoints, 6
        )
    gravity = values.gravity.broadcast_to(*batch, 6).reshape(count, 6)
    tau_full, _, _, _ = inverse_dynamics_full(kin, get_dynamics_plan(structure), gravity, v_full, a_full, fext)
    return reduce_generalized_force(structure, tau_full).view(*batch, model.nv)


def mass_matrix(model: Model, state: KinematicState) -> torch.Tensor:
    """Joint-space inertia matrix ``M(q)``, ``(B..., nv, nv)``, symmetric positive definite."""
    kin, batch = _world(model, state)
    mass = mass_matrix_full(kin, get_dynamics_plan(model.structure))
    return reduce_mass_matrix(model.structure, mass).view(*batch, model.nv, model.nv)


def forward_dynamics(
    model: Model, state: KinematicState, v: torch.Tensor, tau: torch.Tensor, *, fext: torch.Tensor | None = None
) -> torch.Tensor:
    """``ddq = M(q)⁻¹ (τ - C(q, v) v - g(q) + Jᵀ fext)`` for ``v, tau: (B..., nv)``. Returns ``(B..., nv)``."""
    bias = inverse_dynamics(model, state, v, torch.zeros_like(v), fext=fext)
    tau = broadcast_to_execution_batch(tau, tuple(bias.shape[:-1]), (model.nv,), name="tau")
    factor = torch.linalg.cholesky(mass_matrix(model, state))
    return torch.cholesky_solve((tau - bias).unsqueeze(-1), factor).squeeze(-1)


def center_of_mass(model: Model, state: KinematicState) -> torch.Tensor:
    """Whole-body centre of mass in the world frame, ``(B..., 3)``."""
    inertias = model.values.body_inertias  # (..., njoints, 10): mass, com, inertia
    masses, com_local = inertias[..., 0], inertias[..., 1:4]
    com_world = (state.joint_rotation @ com_local.unsqueeze(-1)).squeeze(-1) + state.joint_translation
    return (masses.unsqueeze(-1) * com_world).sum(-2) / masses.sum(-1).unsqueeze(-1)


def centroidal_map(model: Model, state: KinematicState) -> torch.Tensor:
    """Centroidal momentum matrix ``A_g(q)``, ``(B..., 6, nv)``: ``h_g = A_g v`` in world axes at the COM."""
    kin, batch = _world(model, state)
    com = center_of_mass(model, state).reshape(math.prod(batch), 3)
    columns = centroidal_map_full(kin, get_dynamics_plan(model.structure), com)
    return reduce_jacobian(model.structure, columns).view(*batch, 6, model.nv)


__all__ = ["center_of_mass", "centroidal_map", "forward_dynamics", "inverse_dynamics", "mass_matrix"]
