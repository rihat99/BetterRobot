"""Joint-space dynamics of a `KinematicState`: the torques of an acceleration (`inverse_dynamics`), the mass
matrix, the accelerations of torques (`forward_dynamics`), and world forces at frames as the external wrenches both
take (`contact_wrenches`)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..kinematics import KinematicState
from ..kinematics.forward import check_state
from ..kinematics.configuration import full_tangent, reduced_columns, reduced_mass_matrix
from ..lie.checks import check_tensor
from ..model._checks import expand_batch
from ..model.model import Model
from ._world import WorldKinematics, inverse_dynamics_full, mass_matrix_full, world_kinematics


def _full_tangent(model: Model, batch: tuple[int, ...], value: torch.Tensor, name: str) -> torch.Tensor:
    """A checked ``(..., nv)`` tangent broadcast to the state's batch, flattened to ``(N, nv_full)``."""
    check_tensor(name, value, shape=(model.nv,), dtype=model.dtype, device=model.device)
    value = expand_batch(name, value, batch, (model.nv,))
    return full_tangent(model, value).reshape(math.prod(batch), model.nv_full)


def inverse_dynamics(
    model: Model,
    state: KinematicState,
    *,
    v: torch.Tensor,
    a: torch.Tensor,
    external_forces: torch.Tensor | None = None,
) -> torch.Tensor:
    """Joint torques ``τ = M(q) a + C(q, v) v + g(q) - Σ_i J_iᵀ f_i``, ``(B..., nv)``, for ``v, a: (B..., nv)``.

    ``M`` includes the motor armature: ``model.armature * a`` (``armature: (nv,)``) is added per coordinate.
    ``external_forces`` is an optional ``(B..., njoints, 6)`` wrench ``f_i = [force, torque]`` acting on the body
    of each joint ``i``, expressed in that joint's own frame. The bias forces are
    ``inverse_dynamics(model, state, v=v, a=zeros)``; the gravity torque also sets ``v`` to zeros. Recursive
    Newton-Euler in the world frame (`_world`). Raises ``TypeError``, ``ShapeError``, ``DtypeMismatchError`` or
    ``DeviceMismatchError`` for an input that is not a tensor of the model's dtype and device with the trailing shape
    above and a batch that broadcasts to the state's.
    """
    check_state(model, state)
    kin = world_kinematics(model, state)
    v_full, a_full = _full_tangent(model, kin.batch, v, "v"), _full_tangent(model, kin.batch, a, "a")
    forces = _external_forces(model, kin.batch, external_forces)
    return _torques(model, kin, v_full, a_full, forces) + model.armature * a


def _external_forces(model: Model, batch: tuple[int, ...], forces: torch.Tensor | None) -> torch.Tensor | None:
    """Checked ``external_forces (B..., njoints, 6)`` broadcast to the state's batch, flattened to ``(N, njoints, 6)``."""
    if forces is None:
        return None
    event = (model.njoints, 6)
    check_tensor("external_forces", forces, shape=event, dtype=model.dtype, device=model.device)
    return expand_batch("external_forces", forces, batch, event).reshape(math.prod(batch), *event)


def _torques(
    model: Model, kin: WorldKinematics, v_full: torch.Tensor, a_full: torch.Tensor, forces: torch.Tensor | None
) -> torch.Tensor:
    """RNEA torques ``(B..., nv)`` without the armature term, from checked, flattened ``(N, nv_full)`` inputs."""
    gravity = model.gravity.broadcast_to(*kin.batch, 6).reshape(math.prod(kin.batch), 6)
    tau_full = inverse_dynamics_full(kin, gravity, v_full, a_full, forces)
    return reduced_columns(model, tau_full).view(*kin.batch, model.nv)


def mass_matrix(model: Model, state: KinematicState) -> torch.Tensor:
    """Joint-space inertia matrix ``M(q)``, ``(B..., nv, nv)``, symmetric positive definite.

    The motor armature ``model.armature (nv,)`` is added on the diagonal. Composite rigid bodies in the world frame
    (`_world`).
    """
    check_state(model, state)
    return _mass_matrix(model, world_kinematics(model, state))


def _mass_matrix(model: Model, kin: WorldKinematics) -> torch.Tensor:
    """`mass_matrix` of the world kinematics ``kin`` of a checked state, ``(B..., nv, nv)``."""
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

    ``v, tau: (B..., nv)``; ``external_forces: (B..., njoints, 6)`` as in `inverse_dynamics`. The bias forces come
    from `inverse_dynamics` with zero acceleration, then a Cholesky solve with `mass_matrix`. A batch element whose
    mass matrix is not positive definite (a non-physical model, for example a massless moving body) gets NaN
    accelerations; the other elements keep theirs. Their gradients stay finite only for inputs batched per element:
    an input shared across the batch (for example ``q (nq,)`` against a model value batch) gets non-finite gradients,
    even from a loss that reads only the finite elements. Raises as `inverse_dynamics`.
    """
    check_state(model, state)
    kin = world_kinematics(model, state)
    v_full = _full_tangent(model, kin.batch, v, "v")
    bias = _torques(model, kin, v_full, torch.zeros_like(v_full), _external_forces(model, kin.batch, external_forces))
    check_tensor("tau", tau, shape=(model.nv,), dtype=model.dtype, device=model.device)
    tau = expand_batch("tau", tau, kin.batch, (model.nv,))
    # cholesky_ex and two triangular solves leave the check to us (torch.cholesky_solve reads its own back to the
    # host): no host sync, so the call can be captured in a CUDA graph.
    factor, info = torch.linalg.cholesky_ex(_mass_matrix(model, kin))
    half = torch.linalg.solve_triangular(factor, (tau - bias).unsqueeze(-1), upper=False)
    ddq = torch.linalg.solve_triangular(factor.mT, half, upper=True).squeeze(-1)
    return torch.where((info == 0).unsqueeze(-1), ddq, torch.nan)


def contact_wrenches(
    model: Model, state: KinematicState, *, forces: torch.Tensor, frames: Sequence[int | str]
) -> torch.Tensor:
    """``external_forces (B..., njoints, 6)`` of world ``forces (B..., C, 3)`` pushing at the origins of the ``C``
    frames ``frames`` (ids or names), for `inverse_dynamics` and `forward_dynamics`.

    Each force becomes a wrench ``[force, torque]`` on the body of its frame's parent joint, expressed in that joint's
    frame as ``external_forces`` reads it; forces on frames of one joint add up. ``forces`` broadcasts to the state's
    batch ``B...``. Raises ``UnknownNameError`` for a frame the model does not have, and as `inverse_dynamics` for
    ``forces`` that are not a tensor of the model's dtype and device ending in ``(C, 3)``.
    """
    check_state(model, state)
    batch, event = tuple(state.q.shape[:-1]), (len(frames), 3)
    check_tensor("forces", forces, shape=event, dtype=model.dtype, device=model.device)
    forces = expand_batch("forces", forces, batch, event)
    ids = [model.frame_id(frame) for frame in frames]
    frame_ids = torch.as_tensor(ids, dtype=torch.long, device=forces.device)
    parents = torch.as_tensor([model.frame_parent_joint_ids[i] for i in ids], dtype=torch.long, device=forces.device)
    joint = state.joint_transforms[..., parents, :, :]  # (B..., C, 3, 4)
    lever = state.frame_translations[..., frame_ids, :] - joint[..., 3]
    rotation = joint[..., :3]
    linear = (rotation.mT @ forces[..., None]).squeeze(-1)
    angular = (rotation.mT @ torch.linalg.cross(lever, forces)[..., None]).squeeze(-1)
    wrenches = forces.new_zeros((*batch, model.njoints, 6))
    return wrenches.index_add(-2, parents, torch.cat((linear, angular), dim=-1))


__all__ = ["contact_wrenches", "forward_dynamics", "inverse_dynamics", "mass_matrix"]
