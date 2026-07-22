"""Composite Rigid Body Algorithm — joint-space inertia matrix ``M(q)``.

Body-frame CRBA following Featherstone *RBDA* Algorithm 6.2 / Pinocchio's
local-convention CRBA. Two passes:

* **Backward pass** (leaves → root): accumulate composite-rigid-body
  inertias via ``Y[λ[i]] += Ad(liMi[i])^{-T} · Y[i] · Ad(liMi[i])^{-1}``.
* **Forward pass**: for each joint ``i`` with ``nv_i > 0``, compute the
  composite force ``F = Y_c[i] · S_i`` (in joint ``i``'s local frame),
  fill ``M_{ii} = Sᵢᵀ F``, then walk the chain up to the root applying
  ``F ← Ad(liMi[k])^{-T} · F`` and writing ``M_{ji} = Sⱼᵀ F`` (and its
  transpose) for each ancestor ``j``.

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
from ..kinematics.forward import forward_kinematics_raw
from ..lie import se3
from ._execution import prepare_dynamics_inputs


@dataclass(frozen=True)
class CRBAResult:
    """Fresh tensor outputs from :func:`crba_raw`."""

    mass_matrix: torch.Tensor
    joint_pose_world: torch.Tensor
    joint_pose_local: torch.Tensor


def crba_raw(  # noqa: PLR0912, PLR0915 - composite-body passes are intentionally explicit
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
    device, dtype = q.device, q.dtype
    njoints = structure.njoints

    # ── FK pass: oMi (unused) and liMi (drives the adjoints) ─────────────
    fk_result = forward_kinematics_raw(structure, values, q)
    oMi = fk_result.joint_pose_world
    liMi = fk_result.joint_pose_local
    spatial_inertias = values.spatial_inertias()
    motion_subspaces = structure.joint_motion_subspaces

    # ── Pre-compute Ad(liMi[i])^{-1} for all joints in one batched build ──
    adjoint_inv_stacked = se3.adjoint_inv(liMi)  # (..., njoints, 6, 6)
    Ad_inv: list[torch.Tensor | None] = [None] + [adjoint_inv_stacked[..., i, :, :] for i in range(1, njoints)]

    # ── Initialise composite-inertia matrices Y_c[i] (one batched materialise) ─
    Y_c_init = spatial_inertias.expand(*batch, njoints, 6, 6).contiguous()
    Y_c = [Y_c_init[..., i, :, :] for i in range(njoints)]

    # ── Backward pass: accumulate Y_c up the kinematic tree ──────────────
    for i in reversed(structure.topo_order):
        if i == 0:
            continue
        p = structure.parents[i]
        if p < 0:
            continue
        A = Ad_inv[i]  # (..., 6, 6)
        Y_in_parent = A.transpose(-1, -2) @ Y_c[i] @ A
        Y_c[p] = Y_c[p] + Y_in_parent

    # ── Pre-compute motion-subspace S_i for every joint with nv_i > 0 ────
    S_cache: list[torch.Tensor | None] = [None] * njoints
    for i in structure.topo_order:
        if i == 0:
            continue
        nv_i = structure.nvs_full[i]
        if nv_i == 0:
            continue
        S_cache[i] = motion_subspaces[i, :, :nv_i].expand(*batch, 6, nv_i)

    # ── Forward pass: assemble M ─────────────────────────────────────────
    nv = structure.nv_full
    M = torch.zeros((*batch, nv, nv), device=device, dtype=dtype)

    for i in structure.topo_order:
        if i == 0:
            continue
        nv_i = structure.nvs_full[i]
        if nv_i == 0:
            continue
        iv_i = structure.idx_vs_full[i]
        S_i = S_cache[i]  # (..., 6, nv_i)
        F = Y_c[i] @ S_i  # (..., 6, nv_i)
        M_ii = S_i.transpose(-1, -2) @ F  # (..., nv_i, nv_i)
        M[..., iv_i : iv_i + nv_i, iv_i : iv_i + nv_i] = M_ii

        # Walk up the chain transporting F into each ancestor's frame.
        j = i
        while True:
            p = structure.parents[j]
            if p <= 0:
                break
            F = Ad_inv[j].transpose(-1, -2) @ F  # (..., 6, nv_i)
            nv_p = structure.nvs_full[p]
            if nv_p > 0:
                iv_p = structure.idx_vs_full[p]
                S_p = S_cache[p]
                M_pi = S_p.transpose(-1, -2) @ F  # (..., nv_p, nv_i)
                M[..., iv_p : iv_p + nv_p, iv_i : iv_i + nv_i] = M_pi
                M[..., iv_i : iv_i + nv_i, iv_p : iv_p + nv_p] = M_pi.transpose(-1, -2)
            j = p

    return CRBAResult(reduce_mass_matrix(structure, M), oMi, liMi)


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
