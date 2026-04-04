"""Lie group operations wrapping PyPose SE3/SO3.

All Lie group operations in BetterRobot go through this module.
To switch to a different backend, only this file needs changing.

Convention
----------
SE3 pose:    [tx, ty, tz, qx, qy, qz, qw]  (PyPose native)
SE3 tangent: [tx, ty, tz, rx, ry, rz]       (PyPose native)
"""
from __future__ import annotations
import pypose as pp
import torch


def se3_identity() -> torch.Tensor:
    """Identity SE3: zero translation, identity quaternion."""
    return torch.tensor([0., 0., 0., 0., 0., 0., 1.])


def se3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (..., 7), b: (..., 7) -> (..., 7). Result = a @ b."""
    return (pp.SE3(a) @ pp.SE3(b)).tensor()


def se3_inverse(t: torch.Tensor) -> torch.Tensor:
    """t: (..., 7) -> (..., 7)."""
    return pp.SE3(t).Inv().tensor()


def se3_log(t: torch.Tensor) -> torch.Tensor:
    """t: (..., 7) -> (..., 6) tangent [tx, ty, tz, rx, ry, rz]."""
    return pp.SE3(t).Log().tensor()


def se3_exp(tangent: torch.Tensor) -> torch.Tensor:
    """tangent: (..., 6) [tx, ty, tz, rx, ry, rz] -> (..., 7)."""
    return pp.se3(tangent).Exp().tensor()


def adjoint_se3(T: torch.Tensor) -> torch.Tensor:
    """6×6 Adjoint matrix of SE3 element T.

    For PyPose se3 tangent convention [tx, ty, tz, rx, ry, rz]:

        Ad(T) = [[R,          skew(p) @ R],
                  [zeros(3,3), R          ]]

    where p = T[:3], R = rotation_matrix(T[3:7]).

    Args:
        T: (7,) SE3 pose [tx, ty, tz, qx, qy, qz, qw].

    Returns:
        (6, 6) Adjoint matrix.
    """
    p = T[:3]
    R = pp.SO3(T[3:7]).matrix()   # (3, 3)

    skew = torch.zeros(3, 3, dtype=T.dtype, device=T.device)
    skew[0, 1] = -p[2];  skew[0, 2] =  p[1]
    skew[1, 0] =  p[2];  skew[1, 2] = -p[0]
    skew[2, 0] = -p[1];  skew[2, 1] =  p[0]

    Ad = torch.zeros(6, 6, dtype=T.dtype, device=T.device)
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew @ R
    Ad[3:, 3:] = R
    return Ad


def se3_apply_base(base_pose: torch.Tensor, link_poses: torch.Tensor) -> torch.Tensor:
    """Apply a base SE3 transform to all link poses.

    Args:
        base_pose: (..., 7) SE3 base transform [tx, ty, tz, qx, qy, qz, qw].
        link_poses: (..., num_links, 7) link poses in robot frame.

    Returns:
        (..., num_links, 7) link poses in world frame.
    """
    assert base_pose.shape[-1] == 7, f"base_pose last dim must be 7, got {base_pose.shape}"
    assert link_poses.shape[-1] == 7, f"link_poses last dim must be 7, got {link_poses.shape}"
    base = pp.SE3(base_pose.unsqueeze(-2))    # (..., 1) SE3 LieTensor
    return (base @ pp.SE3(link_poses)).tensor()  # broadcasts over num_links
