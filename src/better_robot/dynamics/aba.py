"""Forward dynamics — ``M(q) ddq = τ − b(q, v) + Jᵀ fext``.

Solved as a batched Cholesky system over the world-frame CRBA mass matrix and RNEA bias
(:mod:`._world`). The articulated-body recursion is sequential in the tree depth and has no
advantage in batched torch for ``nv`` up to a few hundred. Gravity is folded into the base
acceleration; external wrenches are given per joint in the joint frame. Inputs and outputs
carry a leading batch shape ``(B...,)``.

See ``docs/concepts/dynamics.md §2``.
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
from ..data_model.reduced_coordinates import expand_tangent, reduce_generalized_force, reduce_mass_matrix
from ._execution import prepare_dynamics_inputs
from ._world import (
    body_velocities,
    get_dynamics_plan,
    inverse_dynamics_full,
    mass_matrix_full,
    world_kinematics_from_q,
)


@dataclass(frozen=True)
class ABAResult:
    """Fresh tensor outputs from :func:`aba_raw`."""

    ddq: torch.Tensor
    joint_pose_world: torch.Tensor
    joint_pose_local: torch.Tensor
    joint_velocity_local: torch.Tensor
    joint_acceleration_local: torch.Tensor


def aba_raw(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    *,
    fext: torch.Tensor | None = None,
) -> ABAResult:
    """Pure articulated-body solve ``M(q) ddq = τ − b(q, v) + Jᵀ fext``.

    Parameters
    ----------
    structure, values
        Pure compute-seam inputs.
    q, v, tau
        ``q`` is ``(B..., nq)``; ``v`` and ``tau`` are ``(B..., nv)``.
    fext : Tensor, optional
        ``(B..., njoints, 6)`` external wrench per joint in the joint's
        local frame ``[fx, fy, fz, τx, τy, τz]``. ``None`` ⇒ zero.

    Returns
    -------
    ABAResult
        Fresh acceleration, pose, and local-motion tensors.
    """
    query_inputs: dict[str, tuple[torch.Tensor, tuple[int, ...]]] = {
        "v": (v, (structure.nv,)),
        "tau": (tau, (structure.nv,)),
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
    tau = prepared["tau"]
    fext = prepared.get("fext")

    # Solve the joint-space equations M ddq = tau - b. The articulated-body recursion is
    # sequential in the tree depth; in batched torch a Cholesky solve on M is cheaper and, with
    # mimic joints, the projected equations are the only correct form anyway.
    count = math.prod(batch)
    njoints = structure.njoints
    plan = get_dynamics_plan(structure)
    kin, world_pose, local_pose = world_kinematics_from_q(structure, values, q)
    gravity = values.gravity.broadcast_to(*batch, 6).reshape(count, 6)
    v_full = expand_tangent(structure, v).reshape(count, -1)
    bias_full, velocity, bias_acceleration, _ = inverse_dynamics_full(
        kin, plan, gravity, v_full, torch.zeros_like(v_full), None if fext is None else fext.reshape(count, njoints, 6)
    )
    bias = reduce_generalized_force(structure, bias_full).view(*batch, structure.nv)
    mass = reduce_mass_matrix(structure, mass_matrix_full(kin, plan)).view(*batch, structure.nv, structure.nv)
    ddq = torch.cholesky_solve((tau - bias).unsqueeze(-1), torch.linalg.cholesky(mass)).squeeze(-1)
    # The realised body acceleration adds the joint accelerations along each path to the bias term.
    acceleration = bias_acceleration + body_velocities(kin, plan, expand_tangent(structure, ddq).reshape(count, -1))
    return ABAResult(
        ddq=ddq,
        joint_pose_world=world_pose,
        joint_pose_local=local_pose,
        joint_velocity_local=kin.to_joint_frames(velocity).view(*batch, njoints, 6),
        joint_acceleration_local=kin.to_joint_frames(acceleration).view(*batch, njoints, 6),
    )


def aba(
    model: Model,
    q: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    *,
    fext: torch.Tensor | None = None,
    data: Data | None = None,
) -> torch.Tensor:
    """Return forward dynamics, optionally populating ``data`` in place."""

    query_inputs: dict[str, tuple[torch.Tensor, tuple[int, ...]]] = {
        "v": (v, (model.nv,)),
        "tau": (tau, (model.nv,)),
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
    tau = prepared["tau"]
    fext = prepared.get("fext")
    if data is None:
        data = model.create_data(batch_shape=batch, device=q.device, dtype=q.dtype)
    data.q = q
    data.v = v
    data.tau = tau
    result = aba_raw(model.structure, model.values, q, v, tau, fext=fext)
    data.ddq = result.ddq
    data.joint_pose_world = result.joint_pose_world
    data.joint_pose_local = result.joint_pose_local
    data.joint_velocity_local = result.joint_velocity_local
    data.joint_acceleration_local = result.joint_acceleration_local
    object.__setattr__(data, "_kinematics_level", KinematicsLevel.ACCELERATIONS)
    return result.ddq
