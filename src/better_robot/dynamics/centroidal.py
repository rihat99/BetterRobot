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

See ``docs/design/06_DYNAMICS.md §3``.
"""

from __future__ import annotations

import torch

from ..data_model import KinematicsLevel
from ..data_model.data import Data
from ..data_model.model import Model
from ..kinematics.forward import forward_kinematics_raw
from ..lie import se3
from ..spatial.inertia import Inertia


def _world_com(model: Model, q: torch.Tensor, oMi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(total_mass, com_world)`` from ``oMi``.

    ``total_mass`` is a ``(*batch,)`` tensor; ``com_world`` is ``(*batch, 3)``.
    """
    *batch, _ = q.shape
    device, dtype = q.device, q.dtype
    total_mass = torch.zeros(tuple(batch), device=device, dtype=dtype)
    com_world = torch.zeros((*batch, 3), device=device, dtype=dtype)
    for i in range(1, model.njoints):
        I_i = Inertia(model.body_inertias[i].to(device=device, dtype=dtype))
        m_i = I_i.mass
        c_local = I_i.com
        c_world_i = se3.act(oMi[..., i, :], c_local)
        total_mass = total_mass + m_i
        com_world = com_world + m_i * c_world_i
    com_world = com_world / total_mass.unsqueeze(-1).clamp(min=1e-12)
    return total_mass, com_world


def center_of_mass(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
) -> torch.Tensor:
    """Whole-body center of mass. ``(B..., 3)``.

    Populates ``data.com_position`` (and ``data.com_velocity`` /
    ``data.com_acceleration`` when ``v`` / ``a`` are given via
    :func:`compute_centroidal_momentum`).

    The function runs its own FK pass; callers do not need to populate
    ``data`` beforehand.
    """
    oMi, liMi = forward_kinematics_raw(model, q)
    data.joint_pose_world = oMi
    data.joint_pose_local = liMi
    data._kinematics_level = KinematicsLevel.PLACEMENTS

    total_mass, com_world = _world_com(model, q, oMi)
    data.com_position = com_world

    if v is not None:
        _, h_g = ccrba(model, data, q, v)
        com_vel = h_g[..., :3] / total_mass.unsqueeze(-1).clamp(min=1e-12)
        data.com_velocity = com_vel

    return com_world


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
    *batch, _ = q.shape
    device, dtype = q.device, q.dtype
    njoints = model.njoints
    nv = model.nv

    # ── FK ───────────────────────────────────────────────────────────────
    oMi, liMi = forward_kinematics_raw(model, q)
    data.joint_pose_world = oMi
    data.joint_pose_local = liMi
    data._kinematics_level = KinematicsLevel.PLACEMENTS

    # ── COM in world ─────────────────────────────────────────────────────
    total_mass, com_world = _world_com(model, q, oMi)
    data.com_position = com_world

    # ── CRBA backward accumulation: Y_c in each joint's local frame ──────
    Ad_inv: list[torch.Tensor | None] = [None] * njoints
    for i in range(1, njoints):
        Ad_inv[i] = se3.adjoint_inv(liMi[..., i, :])

    Y_c: list[torch.Tensor] = []
    for i in range(njoints):
        I_i = Inertia(model.body_inertias[i].to(device=device, dtype=dtype))._to_6x6()
        Y_c.append(I_i.expand(*batch, 6, 6).contiguous())

    for i in reversed(model.topo_order):
        if i == 0:
            continue
        p = model.parents[i]
        if p < 0:
            continue
        A = Ad_inv[i]
        Y_c[p] = Y_c[p] + A.transpose(-1, -2) @ Y_c[i] @ A

    # ── Per-joint columns of A_g ─────────────────────────────────────────
    A_g = torch.zeros(*batch, 6, nv, device=device, dtype=dtype)

    for i in model.topo_order:
        if i == 0:
            continue
        nv_i = model.nvs[i]
        if nv_i == 0:
            continue
        iv = model.idx_vs[i]
        jm = model.joint_models[i]
        iq, nq_i = model.idx_qs[i], model.nqs[i]
        q_i = q[..., iq : iq + nq_i] if nq_i > 0 else q[..., :0]
        S_i = jm.joint_motion_subspace(q_i)                                 # (..., 6, nv_i)
        F = Y_c[i] @ S_i                                                    # (..., 6, nv_i) — momentum in joint-i frame

        # T_{g, i}: world-axes-at-COM → body i. Translation is shifted by
        # −com_world; rotation is the body's world rotation (oMi[i].q).
        t_shifted = oMi[..., i, :3] - com_world
        q_oMi = oMi[..., i, 3:7]
        T_g_i = torch.cat([t_shifted, q_oMi], dim=-1)
        Phi_neg_T = se3.adjoint_inv(T_g_i).transpose(-1, -2)                # (..., 6, 6)
        F_g = Phi_neg_T @ F                                                 # (..., 6, nv_i)
        A_g[..., :, iv : iv + nv_i] = F_g

    data.centroidal_momentum_matrix = A_g

    if v is None:
        return A_g, None
    h_g = (A_g @ v.unsqueeze(-1)).squeeze(-1)
    data.centroidal_momentum = h_g
    return A_g, h_g
