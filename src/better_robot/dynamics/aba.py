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

See ``docs/design/06_DYNAMICS.md §2``.
"""

from __future__ import annotations

import torch

from ..data_model._kinematics_level import KinematicsLevel
from ..data_model.data import Data
from ..data_model.joint_models.base import joint_bias_acceleration
from ..data_model.model import Model
from ..kinematics.forward import forward_kinematics_raw
from ..lie import se3
from ..spatial.inertia import Inertia
from .rnea import _cross_motion, _cross_motion_force


def aba(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    *,
    fext: torch.Tensor | None = None,
) -> torch.Tensor:
    """Articulated Body Algorithm — solve ``M(q) ddq = τ − b(q, v) + Jᵀ fext``.

    Populates ``data.ddq`` in addition to the standard FK fields.

    Parameters
    ----------
    model, data, q, v, tau
        ``q`` is ``(B..., nq)``; ``v`` and ``tau`` are ``(B..., nv)``.
    fext : Tensor, optional
        ``(B..., njoints, 6)`` external wrench per joint in the joint's
        local frame ``[fx, fy, fz, τx, τy, τz]``. ``None`` ⇒ zero.

    Returns
    -------
    ddq : Tensor
        ``(B..., nv)`` joint-space accelerations.
    """
    *batch, _ = q.shape
    device, dtype = q.device, q.dtype
    njoints = model.njoints
    nv = model.nv

    # ── FK pass (drives the adjoint matrices) ────────────────────────────
    oMi, liMi = forward_kinematics_raw(model, q)
    data.joint_pose_world = oMi
    data.joint_pose_local = liMi
    data._kinematics_level = KinematicsLevel.PLACEMENTS

    Ad_inv: list[torch.Tensor | None] = [None] * njoints
    for i in range(1, njoints):
        Ad_inv[i] = se3.adjoint_inv(liMi[..., i, :])  # (..., 6, 6)

    zero6 = torch.zeros((*batch, 6), device=device, dtype=dtype)
    grav = model.gravity.to(device=device, dtype=dtype).expand(*batch, 6)

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
    for i in model.topo_order:
        if i == 0:
            continue
        jm = model.joint_models[i]
        p = model.parents[i]
        iq, nq_i = model.idx_qs[i], model.nqs[i]
        iv, nv_i = model.idx_vs[i], model.nvs[i]

        if nq_i > 0:
            q_i = q[..., iq : iq + nq_i]
            v_i_slice = v[..., iv : iv + nv_i]
            S_i = jm.joint_motion_subspace(q_i)                         # (..., 6, nv_i)
            vJ = jm.joint_velocity(q_i, v_i_slice)                      # (..., 6)
            cJ = joint_bias_acceleration(jm, q_i, v_i_slice)            # (..., 6)
        else:
            S_i = torch.zeros((*batch, 6, 0), device=device, dtype=dtype)  # bench-ok: zero-DoF placeholder
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
        I_i_6x6 = Inertia(model.body_inertias[i].to(device=device, dtype=dtype))._to_6x6()
        IA[i] = I_i_6x6.expand(*batch, 6, 6).contiguous()

        h_i = (IA[i] @ v_i.unsqueeze(-1)).squeeze(-1)
        pA[i] = _cross_motion_force(v_i, h_i)
        if fext is not None:
            pA[i] = pA[i] - fext[..., i, :]

    # ── Pass 2: backward — factorise + transport to parent ───────────────
    for i in reversed(model.topo_order):
        if i == 0:
            continue
        nv_i = model.nvs[i]
        iv = model.idx_vs[i]
        S_i = S_cache[i]                                                # (..., 6, nv_i)
        IA_i = IA[i]
        pA_i = pA[i]

        if nv_i > 0:
            U = IA_i @ S_i                                              # (..., 6, nv_i)
            D = S_i.transpose(-1, -2) @ U                               # (..., nv_i, nv_i)
            tau_i = tau[..., iv : iv + nv_i]
            u = tau_i - (S_i.transpose(-1, -2) @ pA_i.unsqueeze(-1)).squeeze(-1)  # (..., nv_i)
            D_inv = torch.linalg.inv(D)                                 # (..., nv_i, nv_i)
            U_cache[i] = U
            D_inv_cache[i] = D_inv
            u_cache[i] = u

            # Subtract the joint's contribution and transport.
            UDinvUT = U @ D_inv @ U.transpose(-1, -2)                   # (..., 6, 6)
            Ia = IA_i - UDinvUT
            pa = (
                pA_i
                + (Ia @ c_body[i].unsqueeze(-1)).squeeze(-1)
                + (U @ (D_inv @ u.unsqueeze(-1))).squeeze(-1)
            )
        else:
            Ia = IA_i
            pa = pA_i + (IA_i @ c_body[i].unsqueeze(-1)).squeeze(-1)

        p = model.parents[i]
        if p >= 0:
            A = Ad_inv[i]
            IA[p] = IA[p] + A.transpose(-1, -2) @ Ia @ A if IA[p] is not None else (
                A.transpose(-1, -2) @ Ia @ A
            )
            pA[p] = pA[p] + (A.transpose(-1, -2) @ pa.unsqueeze(-1)).squeeze(-1) if pA[p] is not None else (
                (A.transpose(-1, -2) @ pa.unsqueeze(-1)).squeeze(-1)
            )

    # ── Pass 3: forward — solve for accelerations ────────────────────────
    a_body: list[torch.Tensor | None] = [None] * njoints
    a_body[0] = -grav
    ddq_slots: list[torch.Tensor | None] = [None] * nv

    for i in model.topo_order:
        if i == 0:
            continue
        p = model.parents[i]
        nv_i = model.nvs[i]
        iv = model.idx_vs[i]

        a_parent_local = (Ad_inv[i] @ a_body[p].unsqueeze(-1)).squeeze(-1)
        a_pre = a_parent_local + c_body[i]
        S_i = S_cache[i]

        if nv_i > 0:
            U = U_cache[i]
            D_inv = D_inv_cache[i]
            u = u_cache[i]
            UT_a = (U.transpose(-1, -2) @ a_pre.unsqueeze(-1)).squeeze(-1)  # (..., nv_i)
            ddq_i = (D_inv @ (u - UT_a).unsqueeze(-1)).squeeze(-1)      # (..., nv_i)
            for k in range(nv_i):
                ddq_slots[iv + k] = ddq_i[..., k]
            a_body[i] = a_pre + (S_i @ ddq_i.unsqueeze(-1)).squeeze(-1)
        else:
            a_body[i] = a_pre

    if nv > 0:
        ddq = torch.stack(ddq_slots, dim=-1)  # type: ignore[arg-type]
    else:
        ddq = torch.zeros((*batch, 0), device=device, dtype=dtype)

    data.ddq = ddq
    data.joint_velocity_local = torch.stack(v_body, dim=-2)            # type: ignore[arg-type]
    data.joint_acceleration_local = torch.stack(a_body, dim=-2)        # type: ignore[arg-type]
    return ddq
