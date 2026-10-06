"""Composite Rigid Body Algorithm — joint-space inertia matrix ``M(q)``.

World-frame CRBA (:mod:`._world`): composite inertias are one support-matrix product over the
world spatial inertias, and ``M[r, c] = C_r · I^c C_c`` over the world Jacobian columns, so the
operator count does not depend on the joint count.

See ``docs/concepts/dynamics.md`` ("Canonical signatures").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data_model._kinematics_level import KinematicsLevel
from ..data_model.data import Data
from ..data_model.model import Model
from ..data_model.model_structure import ModelStructure
from ..data_model.model_values import ModelValues
from ..data_model.reduced_coordinates import reduce_mass_matrix
from ._execution import prepare_dynamics_inputs
from ._world import get_dynamics_plan, mass_matrix_full, world_kinematics_from_q


@dataclass(frozen=True)
class CRBAResult:
    """Fresh tensor outputs from :func:`crba_raw`."""

    mass_matrix: torch.Tensor
    joint_pose_world: torch.Tensor
    joint_pose_local: torch.Tensor


def crba_raw(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> CRBAResult:
    """Return a fresh joint-space inertia result over the pure seam."""
    q, _, batch = prepare_dynamics_inputs(
        structure,
        values,
        q,
        {},
        validate=False,
    )
    kin, world_pose, local_pose = world_kinematics_from_q(structure, values, q)
    mass = mass_matrix_full(kin, get_dynamics_plan(structure))
    return CRBAResult(
        reduce_mass_matrix(structure, mass).view(*batch, structure.nv, structure.nv), world_pose, local_pose
    )


def crba(
    model: Model,
    q: torch.Tensor,
    *,
    data: Data | None = None,
) -> torch.Tensor:
    """Return the mass matrix, optionally populating ``data`` in place."""

    q, _, batch = prepare_dynamics_inputs(model.structure, model.values, q, {})
    if data is None:
        data = model.create_data(batch_shape=batch, device=q.device, dtype=q.dtype)
    data.q = q
    result = crba_raw(model.structure, model.values, q)
    data.mass_matrix = result.mass_matrix
    data.joint_pose_world = result.joint_pose_world
    data.joint_pose_local = result.joint_pose_local
    object.__setattr__(data, "_kinematics_level", KinematicsLevel.PLACEMENTS)
    return result.mass_matrix
