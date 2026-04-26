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

See ``docs/concepts/dynamics.md §2``.
"""

from __future__ import annotations

import torch

from ..data_model._kinematics_level import KinematicsLevel
from ..data_model.data import Data
from ..data_model.model import Model
from ..kinematics.forward import forward_kinematics_raw
from ..lie import se3
from ..spatial.inertia import Inertia


def crba(
    model: Model,
    data: Data,
    q: torch.Tensor,
) -> torch.Tensor:
    """Joint-space inertia matrix ``M(q)``. Shape: ``(B..., nv, nv)``.

    Populates ``data.mass_matrix``, ``data.joint_pose_local``, and
    ``data.joint_pose_world`` as a side effect.
    """
    *batch, _ = q.shape
    device, dtype = q.device, q.dtype
    njoints = model.njoints

    # ── FK pass: oMi (unused) and liMi (drives the adjoints) ─────────────
    oMi, liMi = forward_kinematics_raw(model, q)
    data.joint_pose_world = oMi
    data.joint_pose_local = liMi
    data._kinematics_level = KinematicsLevel.PLACEMENTS

    # ── Pre-compute Ad(liMi[i])^{-1} once per joint ──────────────────────
    Ad_inv: list[torch.Tensor | None] = [None] * njoints
    for i in range(1, njoints):
        Ad_inv[i] = se3.adjoint_inv(liMi[..., i, :])  # (..., 6, 6)

    # ── Initialise composite-inertia matrices Y_c[i] (broadcast to batch) ─
    Y_c: list[torch.Tensor] = [
        torch.zeros((*batch, 6, 6), device=device, dtype=dtype)
        for _ in range(njoints)
    ]
    for i in range(njoints):
        I_i = Inertia(model.body_inertias[i].to(device=device, dtype=dtype))._to_6x6()
        Y_c[i] = I_i.expand(*batch, 6, 6).contiguous()

    # ── Backward pass: accumulate Y_c up the kinematic tree ──────────────
    for i in reversed(model.topo_order):
        if i == 0:
            continue
        p = model.parents[i]
        if p < 0:
            continue
        A = Ad_inv[i]                                                # (..., 6, 6)
        Y_in_parent = A.transpose(-1, -2) @ Y_c[i] @ A
        Y_c[p] = Y_c[p] + Y_in_parent

    # ── Pre-compute motion-subspace S_i for every joint with nv_i > 0 ────
    S_cache: list[torch.Tensor | None] = [None] * njoints
    for i in model.topo_order:
        if i == 0:
            continue
        nv_i = model.nvs[i]
        if nv_i == 0:
            continue
        jm = model.joint_models[i]
        iq, nq_i = model.idx_qs[i], model.nqs[i]
        q_i = q[..., iq : iq + nq_i] if nq_i > 0 else q[..., :0]
        S_cache[i] = jm.joint_motion_subspace(q_i)                   # (..., 6, nv_i)

    # ── Forward pass: assemble M ─────────────────────────────────────────
    nv = model.nv
    M = torch.zeros((*batch, nv, nv), device=device, dtype=dtype)

    for i in model.topo_order:
        if i == 0:
            continue
        nv_i = model.nvs[i]
        if nv_i == 0:
            continue
        iv_i = model.idx_vs[i]
        S_i = S_cache[i]                                             # (..., 6, nv_i)
        F = Y_c[i] @ S_i                                             # (..., 6, nv_i)
        M_ii = S_i.transpose(-1, -2) @ F                             # (..., nv_i, nv_i)
        M[..., iv_i : iv_i + nv_i, iv_i : iv_i + nv_i] = M_ii

        # Walk up the chain transporting F into each ancestor's frame.
        j = i
        while True:
            p = model.parents[j]
            if p <= 0:
                break
            F = Ad_inv[j].transpose(-1, -2) @ F                      # (..., 6, nv_i)
            nv_p = model.nvs[p]
            if nv_p > 0:
                iv_p = model.idx_vs[p]
                S_p = S_cache[p]
                M_pi = S_p.transpose(-1, -2) @ F                     # (..., nv_p, nv_i)
                M[..., iv_p : iv_p + nv_p, iv_i : iv_i + nv_i] = M_pi
                M[..., iv_i : iv_i + nv_i, iv_p : iv_p + nv_p] = M_pi.transpose(-1, -2)
            j = p

    data.mass_matrix = M
    return M


def compute_minverse(
    model: Model,
    data: Data,
    q: torch.Tensor,
) -> torch.Tensor:
    """Direct ``M(q)^{-1}`` computation via the ABA factorisation.

    TODO(milestone D4). See docs/concepts/dynamics.md §2.
    """
    raise NotImplementedError("TODO(milestone D4) — see docs/concepts/dynamics.md §2")
