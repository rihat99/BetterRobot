"""Forward dynamics — Featherstone's Articulated Body Algorithm.

Three-pass body-frame ABA mirroring Pinocchio's local-convention
``aba.hxx`` and Featherstone *RBDA* Algorithm 7.1:

* **Pass 1** (forward): propagate body-frame spatial velocity and the
  articulated-body bias ``pA = v ×* (I · v)``; initialise the
  articulated-body inertia ``IA = I``.
* **Pass 2** (backward): factor each joint with ``U = IA · S``,
  ``D = Sᵀ U``, ``u = τ − Sᵀ pA``; subtract the joint contribution from
  ``IA`` / ``pA`` and transport into the parent's frame.
* **Pass 3** (forward): solve ``q̈ᵢ = D⁻¹ (u − Uᵀ aᵖ)`` and accumulate
  the body acceleration along the way.

Gravity is folded into the base bias (``a[0] = −gravity``); external
wrenches are subtracted from the per-body bias on the way down. Inputs
and outputs carry a leading batch shape ``(B...,)``.

See ``docs/concepts/dynamics.md §2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data_model._kinematics_level import KinematicsLevel
from ..data_model.data import Data
from ..data_model.model import Model
from ..data_model.model_structure import ModelStructure
from ..data_model.model_values import ModelValues
from ..kinematics.forward import forward_kinematics_raw
from ..lie import se3
from ._execution import prepare_dynamics_inputs
from .crba import crba_raw
from .rnea import _cross_motion, _cross_motion_force, rnea_raw


@dataclass(frozen=True)
class ABAResult:
    """Fresh tensor outputs from :func:`aba_raw`."""

    ddq: torch.Tensor
    joint_pose_world: torch.Tensor
    joint_pose_local: torch.Tensor
    joint_velocity_local: torch.Tensor
    joint_acceleration_local: torch.Tensor


def aba_raw(  # noqa: PLR0912, PLR0915 - articulated-body passes are intentionally explicit
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

    if structure.has_mimic:
        # Eliminating a full-space ABA acceleration after the solve does not
        # enforce the mimic constraint. Solve the projected equations instead:
        # Gvᵀ M Gv ddq = tau - Gvᵀ b. The final RNEA pass supplies the
        # same local motion caches as the articulated-body path.
        zero_acceleration = torch.zeros_like(v)
        bias = rnea_raw(
            structure,
            values,
            q,
            v,
            zero_acceleration,
            fext=fext,
        )
        mass = crba_raw(structure, values, q).mass_matrix
        ddq = torch.linalg.solve(
            mass,
            (tau - bias.tau).unsqueeze(-1),
        ).squeeze(-1)
        realized = rnea_raw(
            structure,
            values,
            q,
            v,
            ddq,
            fext=fext,
        )
        return ABAResult(
            ddq=ddq,
            joint_pose_world=realized.joint_pose_world,
            joint_pose_local=realized.joint_pose_local,
            joint_velocity_local=realized.joint_velocity_local,
            joint_acceleration_local=realized.joint_acceleration_local,
        )

    device, dtype = q.device, q.dtype
    njoints = structure.njoints
    nv = structure.nv_full

    # ── FK pass (drives the adjoint matrices) ────────────────────────────
    fk_result = forward_kinematics_raw(structure, values, q)
    oMi = fk_result.joint_pose_world
    liMi = fk_result.joint_pose_local

    Ad_inv: list[torch.Tensor | None] = [None] * njoints
    for i in range(1, njoints):
        Ad_inv[i] = se3.adjoint_inv(liMi[..., i, :])  # (..., 6, 6)

    zero6 = torch.zeros((*batch, 6), device=device, dtype=dtype)
    zero_motion_subspace = torch.empty((*batch, 6, 0), device=device, dtype=dtype)
    grav = values.gravity.expand(*batch, 6)
    spatial_inertias = values.spatial_inertias()
    motion_subspaces = structure.joint_motion_subspaces

    # ── Per-joint storage ────────────────────────────────────────────────
    v_body: list[torch.Tensor | None] = [None] * njoints
    c_body: list[torch.Tensor | None] = [None] * njoints
    IA: list[torch.Tensor | None] = [None] * njoints
    pA: list[torch.Tensor | None] = [None] * njoints
    S_cache: list[torch.Tensor | None] = [None] * njoints
    U_cache: list[torch.Tensor | None] = [None] * njoints
    D_inv_cache: list[torch.Tensor | None] = [None] * njoints
    u_cache: list[torch.Tensor | None] = [None] * njoints

    v_body[0] = zero6

    # ── Pass 1: forward — v, IA, pA ──────────────────────────────────────
    for i in structure.topo_order:
        if i == 0:
            continue
        p = structure.parents[i]
        iv, nv_i = structure.idx_vs_full[i], structure.nvs_full[i]

        if nv_i > 0:
            v_i_slice = v[..., iv : iv + nv_i]
            S_i = motion_subspaces[i, :, :nv_i].expand(*batch, 6, nv_i)
            vJ = (S_i @ v_i_slice.unsqueeze(-1)).squeeze(-1)
            cJ = zero6
        else:
            S_i = zero_motion_subspace
            vJ = zero6
            cJ = zero6
        S_cache[i] = S_i

        # Parent-frame motion expressed locally.
        v_parent_local = (Ad_inv[i] @ v_body[p].unsqueeze(-1)).squeeze(-1)
        v_i = v_parent_local + vJ
        v_body[i] = v_i
        # Velocity-product bias acceleration: c = v × vJ + cJ.
        c_body[i] = _cross_motion(v_i, vJ) + cJ

        # Articulated-body inertia / bias (init).
        I_i_6x6 = spatial_inertias[..., i, :, :]
        IA[i] = I_i_6x6.expand(*batch, 6, 6).contiguous()

        h_i = (IA[i] @ v_i.unsqueeze(-1)).squeeze(-1)
        pA[i] = _cross_motion_force(v_i, h_i)
        if fext is not None:
            pA[i] = pA[i] - fext[..., i, :]

    # ── Pass 2: backward — factorise + transport to parent ───────────────
    for i in reversed(structure.topo_order):
        if i == 0:
            continue
        nv_i = structure.nvs_full[i]
        iv = structure.idx_vs_full[i]
        S_i = S_cache[i]  # (..., 6, nv_i)
        IA_i = IA[i]
        pA_i = pA[i]

        if nv_i > 0:
            U = IA_i @ S_i  # (..., 6, nv_i)
            D = S_i.transpose(-1, -2) @ U  # (..., nv_i, nv_i)
            tau_i = tau[..., iv : iv + nv_i]
            u = tau_i - (S_i.transpose(-1, -2) @ pA_i.unsqueeze(-1)).squeeze(-1)  # (..., nv_i)
            D_inv = torch.linalg.inv(D)  # (..., nv_i, nv_i)
            U_cache[i] = U
            D_inv_cache[i] = D_inv
            u_cache[i] = u

            # Subtract the joint's contribution and transport.
            UDinvUT = U @ D_inv @ U.transpose(-1, -2)  # (..., 6, 6)
            Ia = IA_i - UDinvUT
            pa = pA_i + (Ia @ c_body[i].unsqueeze(-1)).squeeze(-1) + (U @ (D_inv @ u.unsqueeze(-1))).squeeze(-1)
        else:
            Ia = IA_i
            pa = pA_i + (IA_i @ c_body[i].unsqueeze(-1)).squeeze(-1)

        p = structure.parents[i]
        if p >= 0:
            A = Ad_inv[i]
            IA[p] = IA[p] + A.transpose(-1, -2) @ Ia @ A if IA[p] is not None else (A.transpose(-1, -2) @ Ia @ A)
            pA[p] = (
                pA[p] + (A.transpose(-1, -2) @ pa.unsqueeze(-1)).squeeze(-1)
                if pA[p] is not None
                else ((A.transpose(-1, -2) @ pa.unsqueeze(-1)).squeeze(-1))
            )

    # ── Pass 3: forward — solve for accelerations ────────────────────────
    a_body: list[torch.Tensor | None] = [None] * njoints
    a_body[0] = -grav
    ddq_slots: list[torch.Tensor | None] = [None] * nv

    for i in structure.topo_order:
        if i == 0:
            continue
        p = structure.parents[i]
        nv_i = structure.nvs_full[i]
        iv = structure.idx_vs_full[i]

        a_parent_local = (Ad_inv[i] @ a_body[p].unsqueeze(-1)).squeeze(-1)
        a_pre = a_parent_local + c_body[i]
        S_i = S_cache[i]

        if nv_i > 0:
            U = U_cache[i]
            D_inv = D_inv_cache[i]
            u = u_cache[i]
            UT_a = (U.transpose(-1, -2) @ a_pre.unsqueeze(-1)).squeeze(-1)  # (..., nv_i)
            ddq_i = (D_inv @ (u - UT_a).unsqueeze(-1)).squeeze(-1)  # (..., nv_i)
            for k in range(nv_i):
                ddq_slots[iv + k] = ddq_i[..., k]
            a_body[i] = a_pre + (S_i @ ddq_i.unsqueeze(-1)).squeeze(-1)
        else:
            a_body[i] = a_pre

    if nv > 0:
        ddq = torch.stack(ddq_slots, dim=-1)  # type: ignore[arg-type]
    else:
        ddq = torch.zeros((*batch, 0), device=device, dtype=dtype)

    return ABAResult(
        ddq=ddq,
        joint_pose_world=oMi,
        joint_pose_local=liMi,
        joint_velocity_local=torch.stack(v_body, dim=-2),  # type: ignore[arg-type]
        joint_acceleration_local=torch.stack(a_body, dim=-2),  # type: ignore[arg-type]
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
