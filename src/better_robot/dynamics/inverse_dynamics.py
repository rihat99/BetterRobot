"""Inverse dynamics on a `KinematicState`: the joint torques that produce a given acceleration."""

from __future__ import annotations

import math

import torch

from ..kinematics import KinematicState
from ..kinematics.configuration import full_tangent, reduced_columns
from ..lie._checks import check_tensor
from ..model._checks import expand_batch
from ..model.model import Model
from ._world import inverse_dynamics_full, world_kinematics


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
    kin = world_kinematics(model, state)
    count = math.prod(kin.batch)
    v_full, a_full = _full_tangent(model, kin.batch, v, "v"), _full_tangent(model, kin.batch, a, "a")
    if external_forces is not None:
        event = (model.njoints, 6)
        check_tensor("external_forces", external_forces, shape=event, dtype=model.dtype, device=model.device)
        external_forces = expand_batch("external_forces", external_forces, kin.batch, event).reshape(count, *event)
    gravity = model.gravity.broadcast_to(*kin.batch, 6).reshape(count, 6)
    tau_full = inverse_dynamics_full(kin, gravity, v_full, a_full, external_forces)
    return reduced_columns(model, tau_full).view(*kin.batch, model.nv) + model.armature * a


__all__ = ["inverse_dynamics"]
