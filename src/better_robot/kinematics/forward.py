"""Forward kinematics — one topological scan per ``Model``.

Replaces the legacy split between ``_fk_impl`` and the
``_solve_floating_*`` family. FK is a single function that walks
``model.topo_order`` and calls ``joint_models[j].joint_transform``. A
free-flyer root is not a special case — it's the joint at index 1.

See ``docs/05_KINEMATICS.md §2``.
"""

from __future__ import annotations

import torch

from ..data_model.data import Data
from ..data_model.model import Model
from ..lie import se3


def forward_kinematics_raw(
    model: Model,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor-only FK primitive — returns ``(oMi, liMi)`` without touching ``Data``.

    Autograd-safe: uses list accumulation + ``torch.stack`` instead of
    in-place writes so PyPose's backward can trace through the composition.

    Parameters
    ----------
    model : Model
    q : (B..., nq)

    Returns
    -------
    oMi : (B..., njoints, 7)   world-frame joint placements
    liMi : (B..., njoints, 7)  parent-frame joint placements

    See docs/05_KINEMATICS.md §6.
    """
    *batch, _ = q.shape
    device, dtype = q.device, q.dtype

    # Identity SE3 (no in-place writes to keep autograd happy)
    _id7 = torch.cat([
        torch.zeros(6, device=device, dtype=dtype),
        torch.ones(1, device=device, dtype=dtype),
    ])

    oMi_list: list[torch.Tensor] = [None] * model.njoints  # type: ignore[list-item]
    liMi_list: list[torch.Tensor] = [None] * model.njoints  # type: ignore[list-item]

    for j in model.topo_order:
        nq_j = model.nqs[j]
        T_placement = model.joint_placements[j].to(device=device, dtype=dtype)  # (7,)

        if nq_j > 0:
            q_j = q[..., model.idx_qs[j] : model.idx_qs[j] + nq_j]
            T_j = model.joint_models[j].joint_transform(q_j).to(dtype=dtype)  # (B..., 7)
        else:
            # nq=0 joints (fixed, universe) contribute identity transform
            T_j = _id7.expand(*batch, 7) if batch else _id7

        # liMi[j] = T_placement ∘ T_j  (parent-frame joint placement)
        liMi_j = se3.compose(T_placement, T_j)  # (B..., 7)
        liMi_list[j] = liMi_j

        parent = model.parents[j]
        if parent < 0:
            oMi_list[j] = liMi_j
        else:
            oMi_list[j] = se3.compose(oMi_list[parent], liMi_j)  # (B..., 7)

    # Stack along the joints dimension (len(batch) = q.ndim - 1)
    stack_dim = len(batch)
    oMi = torch.stack(oMi_list, dim=stack_dim)   # (B..., njoints, 7)
    liMi = torch.stack(liMi_list, dim=stack_dim)
    return oMi, liMi


def forward_kinematics(
    model: Model,
    q_or_data: torch.Tensor | Data,
    *,
    compute_frames: bool = False,
) -> Data:
    """Compute joint (and optionally frame) placements.

    Parameters
    ----------
    model : Model
        Immutable kinematic tree.
    q_or_data : torch.Tensor | Data
        Either a flat configuration tensor of shape ``(B..., nq)`` or a
        pre-allocated ``Data`` whose ``q`` field is populated.
    compute_frames : bool
        If true, also populate ``data.oMf``.

    Returns
    -------
    Data
        Data with ``liMi``, ``oMi`` (and optionally ``oMf``) filled.

    See docs/05_KINEMATICS.md §2.
    """
    if isinstance(q_or_data, Data):
        data = q_or_data
        q = data.q
    else:
        q = q_or_data
        data = model.create_data(
            batch_shape=tuple(q.shape[:-1]),
            device=q.device,
            dtype=q.dtype,
        )
        data.q = q

    oMi, liMi = forward_kinematics_raw(model, q)
    data.oMi = oMi
    data.liMi = liMi
    data._kinematics_level = 1

    if compute_frames:
        update_frame_placements(model, data)

    return data


def update_frame_placements(model: Model, data: Data) -> Data:
    """Populate ``data.oMf`` from ``data.oMi`` and the model's frame metadata.

    Requires ``data.oMi`` to be populated (call ``forward_kinematics`` first).

    See docs/05_KINEMATICS.md §2.
    """
    oMi = data.oMi
    assert oMi is not None, "call forward_kinematics before update_frame_placements"

    *batch, _ = data.q.shape
    device, dtype = data.q.device, data.q.dtype

    oMf = torch.zeros(*batch, model.nframes, 7, device=device, dtype=dtype)

    for f_id, frame in enumerate(model.frames):
        T_parent = oMi[..., frame.parent_joint, :]  # (B..., 7)
        T_local = frame.joint_placement.to(device=device, dtype=dtype)  # (7,)
        oMf[..., f_id, :] = se3.compose(T_parent, T_local)  # (B..., 7)

    data.oMf = oMf
    return data
