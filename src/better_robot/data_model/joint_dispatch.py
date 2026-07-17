"""Static joint dispatch shared by whole-pass torch implementations."""

from __future__ import annotations

import torch

from ..lie import se3
from .joint_models.base import JointModel
from .model_structure import JOINT_KIND_CODES


_FIXED_CODES = {
    JOINT_KIND_CODES["universe"],
    JOINT_KIND_CODES["fixed"],
    JOINT_KIND_CODES["mimic"],
}
_REVOLUTE_CODES = {
    JOINT_KIND_CODES["revolute_rx"],
    JOINT_KIND_CODES["revolute_ry"],
    JOINT_KIND_CODES["revolute_rz"],
    JOINT_KIND_CODES["revolute_unaligned"],
}
_PRISMATIC_CODES = {
    JOINT_KIND_CODES["prismatic_px"],
    JOINT_KIND_CODES["prismatic_py"],
    JOINT_KIND_CODES["prismatic_pz"],
    JOINT_KIND_CODES["prismatic_unaligned"],
}


def joint_transform(
    joint: JointModel,
    kind_code: int,
    axis: torch.Tensor,
    pitch: torch.Tensor,
    q_slice: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one joint without per-call axis transfers.

    Non-axis joint kinds retain their statically-specialized implementation;
    axis-based kinds consume the already-device-resident packed tables.
    """

    if kind_code in _FIXED_CODES:
        return se3.identity(
            batch_shape=tuple(q_slice.shape[:-1]),
            device=q_slice.device,
            dtype=q_slice.dtype,
        )
    if kind_code in _REVOLUTE_CODES:
        return se3.from_axis_angle(axis, q_slice[..., 0])
    if kind_code == JOINT_KIND_CODES["revolute_unbounded"]:
        angle = torch.atan2(q_slice[..., 1], q_slice[..., 0])
        return se3.from_axis_angle(axis, angle)
    if kind_code in _PRISMATIC_CODES:
        return se3.from_translation(axis, q_slice[..., 0])
    if kind_code == JOINT_KIND_CODES["helical"]:
        angle = q_slice[..., 0]
        rotation = se3.from_axis_angle(axis, angle)
        translation = se3.from_translation(axis, pitch * angle)
        return se3.compose(translation, rotation)
    return joint.joint_transform(q_slice)


__all__ = ["joint_transform"]
