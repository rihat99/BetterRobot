"""Inverse dynamics — Featherstone's Recursive Newton–Euler Algorithm.

Two-pass body-frame RNEA mirroring Pinocchio's ``rnea.hxx``:

* **Forward pass** over ``model.topo_order``: propagate body-frame spatial
  velocity and acceleration from parent to child using ``Ad(liMi⁻¹)``,
  then compute the inertial wrench ``f_i = I_i·a_i + v_i ×* (I_i·v_i)``
  and subtract any external wrench.
* **Backward pass** in reverse: project the child wrench onto the joint
  subspace ``τ_i = Sᵢᵀ · f_i`` and transport the residual up the tree
  with ``Ad(liMi⁻¹)ᵀ``.

Gravity is folded into the base bias: ``a_body[0] = −model.gravity``.

All inputs and outputs carry a leading batch shape ``(B...,)``.

Known limitations:

* The per-joint bias acceleration ``c_J = Ṡ · v`` is dispatched through
  ``JointModel.joint_bias_acceleration`` (added in P11-pre). It is zero
  for every current joint type because their motion subspaces are body-
  frame constant; joints with ``q``-dependent subspaces only need to
  override the hook to plug in correctly.
* :func:`compute_coriolis_matrix` remains a stub — it requires a separate
  world-frame recursion (Pinocchio ``rnea.hxx`` §CoriolisMatrixForwardStep).

See ``docs/concepts/dynamics.md §2``.
"""
from __future__ import annotations

import torch

from ..data_model.data import Data
from ..data_model.joint_models.base import joint_bias_acceleration
from ..data_model.model import Model
from ..kinematics.forward import forward_kinematics_raw
from ..lie import se3
from ..spatial.inertia import Inertia


def _cross_motion(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``ad(v) · u`` — Motion × Motion spatial cross. ``(..., 6)``.

    Matches :meth:`Motion.cross_motion`::

        out_lin = ω_v × u_lin + v_lin × u_ang
        out_ang = ω_v × u_ang
    """
    v_lin, v_ang = v[..., :3], v[..., 3:]
    u_lin, u_ang = u[..., :3], u[..., 3:]
    out_lin = torch.linalg.cross(v_ang, u_lin) + torch.linalg.cross(v_lin, u_ang)
    out_ang = torch.linalg.cross(v_ang, u_ang)
    return torch.cat([out_lin, out_ang], dim=-1)


def _cross_motion_force(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """``ad*(v) · f`` — Motion × Force spatial cross. ``(..., 6)``.

    Matches :meth:`Motion.cross_force`::

        out_lin = ω_v × f_lin
        out_ang = ω_v × f_ang + v_lin × f_lin
    """
    v_lin, v_ang = v[..., :3], v[..., 3:]
    f_lin, f_ang = f[..., :3], f[..., 3:]
    out_lin = torch.linalg.cross(v_ang, f_lin)
    out_ang = torch.linalg.cross(v_ang, f_ang) + torch.linalg.cross(v_lin, f_lin)
    return torch.cat([out_lin, out_ang], dim=-1)


def rnea(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    *,
    fext: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inverse dynamics: ``τ = M(q)·a + b(q, v) + g(q) − Jᵀ fext``.

    Parameters
    ----------
    model : Model
    data : Data
        Mutable workspace. Populated fields on return:
        ``tau``, ``joint_pose_local``, ``joint_pose_world``,
        ``joint_velocity_local``, ``joint_acceleration_local``, ``joint_forces``.
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
    tau : Tensor
        ``(B..., nv)`` joint-space torques.
    """
    *batch, _ = q.shape
    device, dtype = q.device, q.dtype

    # ── Pass 0: forward kinematics (liMi needed for adjoints) ────────────
    oMi, liMi = forward_kinematics_raw(model, q)
    data.joint_pose_world = oMi
    data.joint_pose_local = liMi
    data._kinematics_level = 1

    # ── Base spatial velocity / acceleration ─────────────────────────────
    # a_gf[0] = −gravity: folds gravity into the inertial bias so the
    # forward recursion produces per-body spatial forces that include weight.
    zero6 = torch.zeros((*batch, 6), device=device, dtype=dtype)
    grav = model.gravity.to(device=device, dtype=dtype).expand(*batch, 6)

    # Per-joint storage (list-of-tensor + torch.stack for autograd safety).
    njoints = model.njoints
    v_body: list[torch.Tensor | None] = [None] * njoints
    a_body: list[torch.Tensor | None] = [None] * njoints
    f_body: list[torch.Tensor | None] = [None] * njoints
    S_cache: list[torch.Tensor | None] = [None] * njoints

    v_body[0] = zero6
    a_body[0] = -grav
    f_body[0] = zero6

    # ── Forward pass ─────────────────────────────────────────────────────
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
            a_i_slice = a[..., iv : iv + nv_i]
            S_i = jm.joint_motion_subspace(q_i)                     # (B..., 6, nv_i)
            vJ = jm.joint_velocity(q_i, v_i_slice)                  # (B..., 6)
            aJ = (S_i @ a_i_slice.unsqueeze(-1)).squeeze(-1)        # (B..., 6)
            cJ = joint_bias_acceleration(jm, q_i, v_i_slice)        # (B..., 6)
        else:
            # Fixed / zero-DoF joint: no velocity contribution, no subspace.
            S_i = torch.zeros((*batch, 6, 0), device=device, dtype=dtype)  # bench-ok: zero-DoF placeholder, not allocated per-iter
            vJ = zero6
            aJ = zero6
            cJ = zero6
        S_cache[i] = S_i

        # Parent motion transport: Ad(liMi⁻¹) expresses a parent-frame
        # motion 6-vector in the local joint frame.
        Ad_inv = se3.adjoint_inv(liMi[..., i, :])                   # (B..., 6, 6)
        v_parent_local = (Ad_inv @ v_body[p].unsqueeze(-1)).squeeze(-1)
        a_parent_local = (Ad_inv @ a_body[p].unsqueeze(-1)).squeeze(-1)

        v_i = v_parent_local + vJ
        # Featherstone: a_i = parent_transported + v_i × vJ + S·ddq + c_J.
        a_i = a_parent_local + _cross_motion(v_i, vJ) + aJ + cJ

        # Inertial wrench in body frame: f_i = I_i·a_i + v_i ×* (I_i·v_i).
        M_i = Inertia(model.body_inertias[i].to(device=device, dtype=dtype))._to_6x6()
        h_i = (M_i @ v_i.unsqueeze(-1)).squeeze(-1)
        f_i = (M_i @ a_i.unsqueeze(-1)).squeeze(-1) + _cross_motion_force(v_i, h_i)

        if fext is not None:
            f_i = f_i - fext[..., i, :]

        v_body[i] = v_i
        a_body[i] = a_i
        f_body[i] = f_i

    # ── Backward pass ────────────────────────────────────────────────────
    # Per-coordinate tau slots, filled in reverse topo order.
    tau_slots: list[torch.Tensor | None] = [None] * model.nv

    for i in reversed(model.topo_order):
        if i == 0:
            continue
        iv, nv_i = model.idx_vs[i], model.nvs[i]
        if nv_i > 0:
            S_i = S_cache[i]  # (B..., 6, nv_i)
            tau_i = (S_i.transpose(-1, -2) @ f_body[i].unsqueeze(-1)).squeeze(-1)
            for k in range(nv_i):
                tau_slots[iv + k] = tau_i[..., k]

        p = model.parents[i]
        if p >= 0:
            # Force transport from child to parent frame: Ad(liMi⁻¹)ᵀ · f.
            Ad_inv_T = se3.adjoint_inv(liMi[..., i, :]).transpose(-1, -2)
            f_transported = (Ad_inv_T @ f_body[i].unsqueeze(-1)).squeeze(-1)
            f_body[p] = f_body[p] + f_transported

    if model.nv > 0:
        tau = torch.stack(tau_slots, dim=-1)  # type: ignore[arg-type]
    else:
        tau = torch.zeros((*batch, 0), device=device, dtype=dtype)

    # ── Populate Data ────────────────────────────────────────────────────
    data.tau = tau
    data.joint_velocity_local = torch.stack(v_body, dim=-2)       # type: ignore[arg-type]
    data.joint_acceleration_local = torch.stack(a_body, dim=-2)   # type: ignore[arg-type]
    data.joint_forces = torch.stack(f_body, dim=-2)               # type: ignore[arg-type]
    return tau


def bias_forces(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Bias forces ``b(q, v) = C(q, v)·v + g(q)`` — RNEA with ``a = 0``.

    Populates ``data.bias_forces`` in addition to the usual RNEA fields.
    """
    a_zero = torch.zeros_like(v)
    tau = rnea(model, data, q, v, a_zero)
    data.bias_forces = tau
    return tau


# Deprecated alias — remove in v1.1. See docs/conventions/naming.md.
nle = bias_forces


def compute_generalized_gravity(
    model: Model,
    data: Data,
    q: torch.Tensor,
) -> torch.Tensor:
    """Generalised gravity torque ``g(q)`` — RNEA with ``v = 0, a = 0``.

    Populates ``data.gravity_torque`` in addition to the usual RNEA fields.
    """
    zeros_v = torch.zeros(*q.shape[:-1], model.nv, device=q.device, dtype=q.dtype)
    tau = rnea(model, data, q, zeros_v, zeros_v)
    data.gravity_torque = tau
    return tau


def compute_coriolis_matrix(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """``C(q, v)`` — Coriolis matrix. Populates ``data.coriolis_matrix``.

    Deferred: this requires a dedicated world-frame recursion with the
    composite-inertia variation ``B[i]`` (Pinocchio ``rnea.hxx``
    §CoriolisMatrixForwardStep). Not derivable as a slice of RNEA.
    """
    raise NotImplementedError(
        "compute_coriolis_matrix is a separate recursion; deferred to a later "
        "dynamics milestone — see docs/concepts/dynamics.md §2."
    )
