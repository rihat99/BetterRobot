"""SE3 Lie group operations (PyPose backend).

Convention
----------
SE3 pose:    [tx, ty, tz, qx, qy, qz, qw]  (PyPose native, scalar-last quaternion)
SE3 tangent: [tx, ty, tz, rx, ry, rz]       (PyPose native)
"""
from __future__ import annotations
import pypose as pp
import torch

__all__ = [
    "se3_identity",
    "se3_compose",
    "se3_inverse",
    "se3_log",
    "se3_exp",
    "se3_apply_base",
]


def se3_identity() -> torch.Tensor:
    """Identity SE3: zero translation, identity quaternion."""
    return torch.tensor([0., 0., 0., 0., 0., 0., 1.])


def se3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compose two SE3 transforms. a: (..., 7), b: (..., 7) -> (..., 7). Result = a @ b."""
    return (pp.SE3(a) @ pp.SE3(b)).tensor()


def se3_inverse(t: torch.Tensor) -> torch.Tensor:
    """Invert an SE3 transform. t: (..., 7) -> (..., 7)."""
    return pp.SE3(t).Inv().tensor()


def se3_log(t: torch.Tensor) -> torch.Tensor:
    """SE3 -> se3 tangent. t: (..., 7) -> (..., 6) [tx, ty, tz, rx, ry, rz]."""
    return pp.SE3(t).Log().tensor()


def se3_exp(tangent: torch.Tensor) -> torch.Tensor:
    """se3 tangent -> SE3. tangent: (..., 6) -> (..., 7)."""
    return pp.se3(tangent).Exp().tensor()


def se3_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """SE3 pose from pure rotation about an axis.

    Supports batched angles: angle can be (*batch,) and axis is (3,).

    Args:
        axis: (3,) unit rotation axis.
        angle: (*batch,) rotation angle(s) in radians.

    Returns:
        (*batch, 7) SE3 pose [tx, ty, tz, qx, qy, qz, qw] with zero translation.
    """
    half = angle / 2.0
    cos_h = torch.cos(half)
    sin_h = torch.sin(half)
    qxyz = sin_h.unsqueeze(-1) * axis
    zeros = torch.zeros(*angle.shape, 3, device=angle.device, dtype=angle.dtype)
    return torch.cat([zeros, qxyz, cos_h.unsqueeze(-1)], dim=-1)


def se3_from_translation(axis: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """SE3 pose from pure translation along an axis.

    Supports batched displacements: displacement can be (*batch,) and axis is (3,).

    Args:
        axis: (3,) translation direction.
        displacement: (*batch,) displacement magnitude(s).

    Returns:
        (*batch, 7) SE3 pose [tx, ty, tz, qx, qy, qz, qw] with identity rotation.
    """
    batch_shape = displacement.shape
    device, dtype = displacement.device, displacement.dtype
    trans = displacement.unsqueeze(-1) * axis.to(device=device, dtype=dtype)
    qxyz = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
    qw = torch.ones(*batch_shape, 1, device=device, dtype=dtype)
    return torch.cat([trans, qxyz, qw], dim=-1)


def se3_normalize(pose: torch.Tensor) -> torch.Tensor:
    """Normalize the quaternion part of an SE3 pose.

    Args:
        pose: (..., 7) SE3 pose tensor [tx, ty, tz, qx, qy, qz, qw].

    Returns:
        (..., 7) SE3 pose with normalized quaternion.
    """
    q = pose[..., 3:7]
    q_normed = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.cat([pose[..., :3], q_normed], dim=-1)


def se3_apply_base(base_pose: torch.Tensor, link_poses: torch.Tensor) -> torch.Tensor:
    """Apply a base SE3 transform to all link poses.

    Args:
        base_pose: (..., 7) SE3 base transform.
        link_poses: (..., num_links, 7) link poses in robot frame.

    Returns:
        (..., num_links, 7) link poses in world frame.
    """
    assert base_pose.shape[-1] == 7, f"base_pose last dim must be 7, got {base_pose.shape}"
    assert link_poses.shape[-1] == 7, f"link_poses last dim must be 7, got {link_poses.shape}"
    base = pp.SE3(base_pose.unsqueeze(-2))
    return (base @ pp.SE3(link_poses)).tensor()
