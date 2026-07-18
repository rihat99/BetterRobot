"""Centroidal dynamics — center of mass, centroidal momentum matrix, CCRBA.

The centroidal frame is *world-axes at the whole-body COM*. The centroidal
momentum matrix ``A_g(q)`` is the linear map ``v ↦ h_g`` from generalised
velocity to spatial momentum expressed in that frame.

Implementation strategy (Featherstone-style, no separate backward pass):

* Run CRBA's backward accumulation to obtain composite-rigid-body
  inertias ``Y_c[i]`` in each joint's local frame.
* For each joint ``i`` with ``nv_i > 0``, the unit-motion ``S_i`` produces
  spatial momentum ``F = Y_c[i] · S_i`` in joint ``i``'s frame.
* Transport ``F`` to the centroidal frame via
  ``A_g[:, iv_i:iv_i+nv_i] = Ad(T_{g, i})⁻ᵀ · F``.

See ``docs/concepts/dynamics.md`` ("Centroidal").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data_model import KinematicsLevel
from ..data_model.data import Data
from ..data_model.model import Model
from ..data_model.model_structure import ModelStructure
from ..data_model.model_values import ModelValues
from ..data_model.reduced_coordinates import reduce_jacobian
from ..kinematics.forward import forward_kinematics_raw
from ..lie import se3
from ._execution import prepare_dynamics_inputs


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
    device, dtype = q.device, q.dtype
    njoints = structure.njoints
    nv_full = structure.nv_full
    oMi, liMi = forward_kinematics_raw(structure, values, q)
    total_mass, com_world = _world_com(structure, values, oMi)
    spatial_inertias = values.spatial_inertias()
    motion_subspaces = structure.joint_motion_subspaces

    adjoint_inverse: list[torch.Tensor | None] = [None] * njoints
    for index in range(1, njoints):
        adjoint_inverse[index] = se3.adjoint_inv(liMi[..., index, :])

    composite = [spatial_inertias[..., index, :, :].expand(*batch, 6, 6) for index in range(njoints)]
    for index in reversed(structure.topo_order):
        if index == 0:
            continue
        parent = structure.parents[index]
        if parent < 0:
            continue
        transform = adjoint_inverse[index]
        composite[parent] = composite[parent] + transform.transpose(-1, -2) @ composite[index] @ transform

    centroidal_map_full = torch.zeros(*batch, 6, nv_full, device=device, dtype=dtype)
    for index in structure.topo_order:
        if index == 0:
            continue
        nv_i = structure.nvs_full[index]
        if nv_i == 0:
            continue
        iv = structure.idx_vs_full[index]
        subspace = motion_subspaces[index, :, :nv_i].expand(*batch, 6, nv_i)
        momentum_columns = composite[index] @ subspace
        shifted = torch.cat((oMi[..., index, :3] - com_world, oMi[..., index, 3:7]), dim=-1)
        to_centroidal = se3.adjoint_inv(shifted).transpose(-1, -2)
        centroidal_map_full[..., :, iv : iv + nv_i] = to_centroidal @ momentum_columns

    centroidal_map = reduce_jacobian(structure, centroidal_map_full)
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
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
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
    q, prepared, _ = prepare_dynamics_inputs(
        model.structure,
        model.values,
        q,
        query_inputs,
    )
    v = prepared.get("v")
    data.q = q
    data.v = v
    result = ccrba_raw(model.structure, model.values, q, v)
    _populate_centroidal_data(data, result)
    if result.momentum is not None:
        data.com_velocity = result.momentum[..., :3] / result.total_mass.unsqueeze(-1).clamp(min=1e-12)
    return result.com_position


def compute_centroidal_map(
    model: Model,
    data: Data,
    q: torch.Tensor,
) -> torch.Tensor:
    """Centroidal momentum matrix ``A_g(q)`` — shape ``(B..., 6, nv)``.

    Populates ``data.centroidal_momentum_matrix`` and ``data.com_position``.
    """
    A_g, _ = _ccrba_impl(model, data, q, v=None)
    return A_g


def compute_centroidal_momentum(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Centroidal spatial momentum ``h_g = A_g(q) v`` — shape ``(B..., 6)``.

    Populates ``data.centroidal_momentum`` (and the matrix as a side
    effect).
    """
    _, h_g = _ccrba_impl(model, data, q, v=v)
    return h_g


def ccrba(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centroidal CRBA — return ``(A_g, h_g)`` and populate the matching
    fields on ``data``.
    """
    return _ccrba_impl(model, data, q, v=v)


def _ccrba_impl(
    model: Model,
    data: Data,
    q: torch.Tensor,
    *,
    v: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    query_inputs = {} if v is None else {"v": (v, (model.nv,))}
    q, prepared, _ = prepare_dynamics_inputs(
        model.structure,
        model.values,
        q,
        query_inputs,
    )
    v = prepared.get("v")
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
