"""Forward kinematics — one topological scan per ``Model``.

Replaces the legacy split between ``_fk_impl`` and the
``_solve_floating_*`` family. FK is a single function that walks
``model.topo_order`` and calls ``joint_models[j].joint_transform``. A
free-flyer root is not a special case — it's the joint at index 1.

See ``docs/design/05_KINEMATICS.md §2``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..backends import default_backend
from ..data_model import KinematicsLevel
from ..data_model.data import Data
from ..data_model.joint_models import JointFreeFlyer
from ..data_model.model import Model
from ..exceptions import DeviceMismatchError, QuaternionNormError, ShapeError
from ..lie import se3

if TYPE_CHECKING:
    from ..backends.protocol import Backend


#: Tolerance on quaternion norm at a public entry point.
#: Anything within this band of 1.0 is re-normalised silently by downstream
#: code; anything outside is rejected (see docs/conventions/17_CONTRACTS.md §1.3).
_QUAT_NORM_TOL = 0.1


def _validate_q(model: Model, q: torch.Tensor) -> None:
    """Sanity-check a configuration tensor at a public entry point.

    Raises
    ------
    ShapeError
        If ``q.shape[-1] != model.nq``.
    DeviceMismatchError
        If ``q.device`` differs from the model's tensor device.
    QuaternionNormError
        If the model has a free-flyer root and ``q[..., 3:7]`` is outside
        ``[1 - TOL, 1 + TOL]``.

    See ``docs/conventions/17_CONTRACTS.md §1``.
    """
    if q.shape[-1] != model.nq:
        raise ShapeError(
            f"q has trailing size {q.shape[-1]}, expected model.nq={model.nq}"
        )
    model_device = model.joint_placements.device
    if q.device != model_device:
        raise DeviceMismatchError(
            f"q.device={q.device} != model.device={model_device}. "
            f"Call model.to(q.device) or q.to(model.device) first."
        )
    if model.njoints >= 2 and isinstance(model.joint_models[1], JointFreeFlyer):
        # Free-flyer q layout: [tx, ty, tz, qx, qy, qz, qw]
        quat = q[..., 3:7]
        norm = quat.norm(dim=-1)
        if bool(((norm - 1.0).abs() > _QUAT_NORM_TOL).any()):
            bad = float(norm.min()), float(norm.max())
            raise QuaternionNormError(
                f"free-flyer quaternion norm outside [{1 - _QUAT_NORM_TOL}, "
                f"{1 + _QUAT_NORM_TOL}] (observed range {bad}). "
                f"Normalise before passing."
            )


def forward_kinematics_raw(
    model: Model,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor-only FK primitive — returns world/local joint placements.

    Autograd-safe: uses list accumulation + ``torch.stack`` instead of
    in-place writes so the backward pass can trace through every SE3
    composition cleanly.

    Parameters
    ----------
    model : Model
    q : (B..., nq)

    Returns
    -------
    joint_pose_world : (B..., njoints, 7)
        World-frame joint placements (the quantity historically called ``oMi``).
    joint_pose_local : (B..., njoints, 7)
        Parent-frame joint placements (``liMi``).

    See docs/design/05_KINEMATICS.md §6 and docs/conventions/13_NAMING.md for the rename.
    """
    _validate_q(model, q)
    *batch, _ = q.shape
    device, dtype = q.device, q.dtype

    # Identity SE3 (no in-place writes to keep autograd happy)
    _id7 = torch.cat([
        torch.zeros(6, device=device, dtype=dtype),
        torch.ones(1, device=device, dtype=dtype),
    ])

    world_list: list[torch.Tensor] = [None] * model.njoints  # type: ignore[list-item]
    local_list: list[torch.Tensor] = [None] * model.njoints  # type: ignore[list-item]

    for j in model.topo_order:
        nq_j = model.nqs[j]
        T_placement = model.joint_placements[j].to(device=device, dtype=dtype)  # (7,)

        if nq_j > 0:
            q_j = q[..., model.idx_qs[j] : model.idx_qs[j] + nq_j]
            T_j = model.joint_models[j].joint_transform(q_j).to(dtype=dtype)  # (B..., 7)
        else:
            # nq=0 joints (fixed, universe) contribute identity transform
            T_j = _id7.expand(*batch, 7) if batch else _id7

        # joint_pose_local[j] = T_placement ∘ T_j  (parent-frame placement)
        local_j = se3.compose(T_placement, T_j)  # (B..., 7)
        local_list[j] = local_j

        parent = model.parents[j]
        if parent < 0:
            world_list[j] = local_j
        else:
            world_list[j] = se3.compose(world_list[parent], local_j)  # (B..., 7)

    # Stack along the joints dimension (len(batch) = q.ndim - 1)
    stack_dim = len(batch)
    joint_pose_world = torch.stack(world_list, dim=stack_dim)  # (B..., njoints, 7)
    joint_pose_local = torch.stack(local_list, dim=stack_dim)
    return joint_pose_world, joint_pose_local


def forward_kinematics(
    model: Model,
    q_or_data: torch.Tensor | Data,
    *,
    compute_frames: bool = False,
    backend: "Backend | None" = None,
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
        If true, also populate ``data.frame_pose_world``.
    backend : Backend, optional
        Backend instance to dispatch through. ``None`` uses the active
        default (see :func:`better_robot.backends.default_backend`).

    Returns
    -------
    Data
        ``Data`` with ``joint_pose_local`` and ``joint_pose_world`` populated
        (and ``frame_pose_world`` if ``compute_frames=True``).

    See docs/design/05_KINEMATICS.md §2.
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

    # Validate at the public boundary; the backend impl trusts inputs.
    _validate_q(model, q)
    backend = backend or default_backend()
    joint_pose_world, joint_pose_local = backend.kinematics.forward_kinematics(model, q)
    data.joint_pose_world = joint_pose_world
    data.joint_pose_local = joint_pose_local
    object.__setattr__(data, "_kinematics_level", KinematicsLevel.PLACEMENTS)

    if compute_frames:
        update_frame_placements(model, data)

    return data


def update_frame_placements(model: Model, data: Data) -> Data:
    """Populate ``data.frame_pose_world`` from ``data.joint_pose_world``
    and the model's frame metadata.

    Requires ``data.joint_pose_world`` to be populated (call
    :func:`forward_kinematics` first).

    See docs/design/05_KINEMATICS.md §2.
    """
    joint_pose_world = data.joint_pose_world
    assert joint_pose_world is not None, (
        "call forward_kinematics before update_frame_placements"
    )

    *batch, _ = data.q.shape
    device, dtype = data.q.device, data.q.dtype

    frame_pose_world = torch.zeros(*batch, model.nframes, 7, device=device, dtype=dtype)

    for f_id, frame in enumerate(model.frames):
        T_parent = joint_pose_world[..., frame.parent_joint, :]         # (B..., 7)
        T_local = frame.joint_placement.to(device=device, dtype=dtype)  # (7,)
        frame_pose_world[..., f_id, :] = se3.compose(T_parent, T_local) # (B..., 7)

    data.frame_pose_world = frame_pose_world
    return data
