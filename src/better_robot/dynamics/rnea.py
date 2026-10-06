"""Inverse dynamics — Recursive Newton–Euler in the world frame.

``τ = M(q) a + C(q, v) v + g(q) − Jᵀ fext``, evaluated by :mod:`._world` with ancestor and
subtree sums as support-matrix products instead of per-joint passes. Gravity enters as the base
acceleration ``−gravity``. All inputs and outputs carry a leading batch shape ``(B...,)``.

The per-joint bias acceleration ``Ṡ v`` is the derivative of the world Jacobian column, which
covers every joint whose motion subspace is constant in its own frame (all shipped kinds).

See ``docs/concepts/dynamics.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..data_model._kinematics_level import KinematicsLevel
from ..data_model.data import Data
from ..data_model.model import Model
from ..data_model.model_structure import ModelStructure
from ..data_model.model_values import ModelValues
from ..data_model.reduced_coordinates import (
    expand_tangent,
    reduce_generalized_force,
)
from ._execution import prepare_dynamics_inputs
from ._world import (
    get_dynamics_plan,
    inverse_dynamics_full,
    world_kinematics_from_q,
)


@dataclass(frozen=True)
class RNEAResult:
    """Fresh tensor outputs from :func:`rnea_raw`."""

    tau: torch.Tensor
    joint_pose_world: torch.Tensor
    joint_pose_local: torch.Tensor
    joint_velocity_local: torch.Tensor
    joint_acceleration_local: torch.Tensor
    joint_forces: torch.Tensor


def rnea_raw(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    *,
    fext: torch.Tensor | None = None,
) -> RNEAResult:
    """Inverse dynamics: ``τ = M(q)·a + b(q, v) + g(q) − Jᵀ fext``.

    Parameters
    ----------
    structure, values
        Pure compute-seam inputs.
    q : Tensor
        ``(B..., nq)`` configuration.
    v : Tensor
        ``(B..., nv)`` generalised velocity.
    a : Tensor
        ``(B..., nv)`` generalised acceleration.
    fext : Tensor, optional
        ``(B..., njoints, 6)`` external spatial wrench per joint in the
        joint's local frame ``[fx, fy, fz, τx, τy, τz]``. ``None`` ⇒ zero.

    Returns
    -------
    RNEAResult
        Torques plus all fresh pass intermediates needed by the public wrapper.
    """
    query_inputs: dict[str, tuple[torch.Tensor, tuple[int, ...]]] = {
        "v": (v, (structure.nv,)),
        "a": (a, (structure.nv,)),
    }
    if fext is not None:
        query_inputs["fext"] = (fext, (structure.njoints, 6))
    q, prepared, batch = prepare_dynamics_inputs(
        structure,
        values,
        q,
        query_inputs,
        validate=False,
    )
    v = prepared["v"]
    a = prepared["a"]
    fext = prepared.get("fext")
    count = math.prod(batch)
    njoints = structure.njoints
    kin, world_pose, local_pose = world_kinematics_from_q(structure, values, q)
    gravity = values.gravity.broadcast_to(*batch, 6).reshape(count, 6)
    tau_full, velocity, acceleration, wrench = inverse_dynamics_full(
        kin,
        get_dynamics_plan(structure),
        gravity,
        expand_tangent(structure, v).reshape(count, -1),
        expand_tangent(structure, a).reshape(count, -1),
        None if fext is None else fext.reshape(count, njoints, 6),
    )
    return RNEAResult(
        tau=reduce_generalized_force(structure, tau_full).view(*batch, structure.nv),
        joint_pose_world=world_pose,
        joint_pose_local=local_pose,
        joint_velocity_local=kin.to_joint_frames(velocity).view(*batch, njoints, 6),
        joint_acceleration_local=kin.to_joint_frames(acceleration).view(*batch, njoints, 6),
        joint_forces=kin.to_joint_frames_force(wrench).view(*batch, njoints, 6),
    )


def rnea(
    model: Model,
    q: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    *,
    fext: torch.Tensor | None = None,
    data: Data | None = None,
) -> torch.Tensor:
    """Return inverse dynamics, optionally populating ``data`` in place."""

    query_inputs: dict[str, tuple[torch.Tensor, tuple[int, ...]]] = {
        "v": (v, (model.nv,)),
        "a": (a, (model.nv,)),
    }
    if fext is not None:
        query_inputs["fext"] = (fext, (model.njoints, 6))
    q, prepared, batch = prepare_dynamics_inputs(
        model.structure,
        model.values,
        q,
        query_inputs,
    )
    v = prepared["v"]
    a = prepared["a"]
    fext = prepared.get("fext")
    if data is None:
        data = model.create_data(batch_shape=batch, device=q.device, dtype=q.dtype)
    data.q = q
    data.v = v
    data.a = a
    result = rnea_raw(model.structure, model.values, q, v, a, fext=fext)
    data.tau = result.tau
    data.joint_pose_world = result.joint_pose_world
    data.joint_pose_local = result.joint_pose_local
    data.joint_velocity_local = result.joint_velocity_local
    data.joint_acceleration_local = result.joint_acceleration_local
    data.joint_forces = result.joint_forces
    object.__setattr__(data, "_kinematics_level", KinematicsLevel.ACCELERATIONS)
    return result.tau


def bias_forces(
    model: Model,
    q: torch.Tensor,
    v: torch.Tensor,
    *,
    data: Data | None = None,
) -> torch.Tensor:
    """Bias forces ``b(q, v) = C(q, v)·v + g(q)`` — RNEA with ``a = 0``.

    Populates ``data.bias_forces`` in addition to the usual RNEA fields.
    """
    a_zero = torch.zeros_like(v)
    tau = rnea(model, q, v, a_zero, data=data)
    if data is not None:
        data.bias_forces = tau
    return tau


def compute_generalized_gravity(
    model: Model,
    q: torch.Tensor,
    *,
    data: Data | None = None,
) -> torch.Tensor:
    """Generalised gravity torque ``g(q)`` — RNEA with ``v = 0, a = 0``.

    Populates ``data.gravity_torque`` in addition to the usual RNEA fields.
    """
    zeros_v = torch.zeros(*q.shape[:-1], model.nv, device=q.device, dtype=q.dtype)
    tau = rnea(model, q, zeros_v, zeros_v, data=data)
    if data is not None:
        data.gravity_torque = tau
    return tau
