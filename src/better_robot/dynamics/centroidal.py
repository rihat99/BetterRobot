"""Centroidal dynamics — center of mass, centroidal momentum matrix, CCRBA.

The centroidal frame is *world-axes at the whole-body COM*. The centroidal
momentum matrix ``A_g(q)`` is the linear map ``v ↦ h_g`` from generalised
velocity to spatial momentum expressed in that frame.

Implementation (:mod:`._world`): the world momentum per unit column velocity,
``I^c_{owner(c)} C_c`` with the composite inertia ``I^c`` of the subtree, is one
batched product; moving it to the centroidal frame is a translation of the
angular rows by the centre of mass.

See ``docs/concepts/dynamics.md`` ("Centroidal").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..data_model import KinematicsLevel
from ..data_model.data import Data
from ..data_model.model import Model
from ..data_model.model_structure import ModelStructure
from ..data_model.model_values import ModelValues
from ..data_model.reduced_coordinates import reduce_jacobian
from ..lie import se3
from ._execution import prepare_dynamics_inputs
from ._world import centroidal_map_full, get_dynamics_plan, world_kinematics_from_q


def _world_com(
    structure: ModelStructure,
    values: ModelValues,
    oMi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(total_mass, com_world)`` from ``oMi``.

    ``total_mass`` is a ``(*batch,)`` tensor; ``com_world`` is ``(*batch, 3)``.
    """
    inertias = values.body_inertias
    masses = inertias[..., 1 : structure.njoints, 0]
    com_local = inertias[..., 1 : structure.njoints, 1:4]
    com_each = se3.act(oMi[..., 1 : structure.njoints, :], com_local)
    total_mass = masses.sum(dim=-1)
    com_world = (masses[..., None] * com_each).sum(dim=-2)
    com_world = com_world / total_mass.unsqueeze(-1).clamp(min=1e-12)
    return total_mass, com_world


@dataclass(frozen=True)
class CentroidalResult:
    """Fresh tensors from the pure centroidal pass."""

    centroidal_map: torch.Tensor
    momentum: torch.Tensor | None
    total_mass: torch.Tensor
    com_position: torch.Tensor
    joint_pose_world: torch.Tensor
    joint_pose_local: torch.Tensor


@dataclass(frozen=True)
class CCRBAResult:
    """Named public return from :func:`ccrba`."""

    centroidal_map: torch.Tensor
    momentum: torch.Tensor


def ccrba_raw(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
    v: torch.Tensor | None = None,
) -> CentroidalResult:
    """Pure centroidal CRBA over ``(structure, values, q, v)``."""

    query_inputs = {} if v is None else {"v": (v, (structure.nv,))}
    q, prepared, batch = prepare_dynamics_inputs(
        structure,
        values,
        q,
        query_inputs,
        validate=False,
    )
    v = prepared.get("v")
    count = math.prod(batch)
    kin, oMi, liMi = world_kinematics_from_q(structure, values, q)
    total_mass, com_world = _world_com(structure, values, oMi)
    centroidal_map_full_ = centroidal_map_full(kin, get_dynamics_plan(structure), com_world.reshape(count, 3))
    centroidal_map = reduce_jacobian(structure, centroidal_map_full_).view(*batch, 6, structure.nv)
    momentum = None if v is None else (centroidal_map @ v.unsqueeze(-1)).squeeze(-1)
    return CentroidalResult(
        centroidal_map,
        momentum,
        total_mass,
        com_world,
        oMi,
        liMi,
    )


def center_of_mass(
    model: Model,
    q: torch.Tensor,
    v: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
    *,
    data: Data | None = None,
) -> torch.Tensor:
    """Whole-body center of mass. ``(B..., 3)``.

    Populates ``data.com_position`` and, when ``v`` is supplied,
    ``data.com_velocity``. Acceleration input is currently unsupported and
    raises :class:`NotImplementedError`; ``data.com_acceleration`` is not
    populated.

    The function runs its own FK pass; callers do not need to populate
    ``data`` beforehand.
    """
    if a is not None:
        raise NotImplementedError("center-of-mass acceleration is not implemented; omit a")
    query_inputs = {} if v is None else {"v": (v, (model.nv,))}
    q, prepared, batch = prepare_dynamics_inputs(
        model.structure,
        model.values,
        q,
        query_inputs,
    )
    v = prepared.get("v")
    if data is None:
        data = model.create_data(batch_shape=batch, device=q.device, dtype=q.dtype)
    data.q = q
    data.v = v
    result = ccrba_raw(model.structure, model.values, q, v)
    _populate_centroidal_data(data, result)
    if result.momentum is not None:
        data.com_velocity = result.momentum[..., :3] / result.total_mass.unsqueeze(-1).clamp(min=1e-12)
    return result.com_position


def compute_centroidal_map(
    model: Model,
    q: torch.Tensor,
    *,
    data: Data | None = None,
) -> torch.Tensor:
    """Centroidal momentum matrix ``A_g(q)`` — shape ``(B..., 6, nv)``.

    Populates ``data.centroidal_momentum_matrix`` and ``data.com_position``.
    """
    A_g, _ = _ccrba_impl(model, q, v=None, data=data)
    return A_g


def compute_centroidal_momentum(
    model: Model,
    q: torch.Tensor,
    v: torch.Tensor,
    *,
    data: Data | None = None,
) -> torch.Tensor:
    """Centroidal spatial momentum ``h_g = A_g(q) v`` — shape ``(B..., 6)``.

    Populates ``data.centroidal_momentum`` (and the matrix as a side
    effect).
    """
    _, h_g = _ccrba_impl(model, q, v=v, data=data)
    return h_g


def ccrba(
    model: Model,
    q: torch.Tensor,
    v: torch.Tensor,
    *,
    data: Data | None = None,
) -> CCRBAResult:
    """Return named ``centroidal_map`` and ``momentum`` tensors.

    The matching fields are populated on ``data`` when supplied.
    """
    centroidal_map, momentum = _ccrba_impl(model, q, v=v, data=data)
    assert momentum is not None
    return CCRBAResult(centroidal_map, momentum)


def _ccrba_impl(
    model: Model,
    q: torch.Tensor,
    *,
    v: torch.Tensor | None,
    data: Data | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    query_inputs = {} if v is None else {"v": (v, (model.nv,))}
    q, prepared, batch = prepare_dynamics_inputs(
        model.structure,
        model.values,
        q,
        query_inputs,
    )
    v = prepared.get("v")
    if data is None:
        data = model.create_data(batch_shape=batch, device=q.device, dtype=q.dtype)
    data.q = q
    data.v = v
    result = ccrba_raw(model.structure, model.values, q, v)
    _populate_centroidal_data(data, result)
    return result.centroidal_map, result.momentum


def _populate_centroidal_data(data: Data, result: CentroidalResult) -> None:
    data.joint_pose_world = result.joint_pose_world
    data.joint_pose_local = result.joint_pose_local
    data.com_position = result.com_position
    data.centroidal_momentum_matrix = result.centroidal_map
    data.centroidal_momentum = result.momentum
    object.__setattr__(data, "_kinematics_level", KinematicsLevel.PLACEMENTS)
