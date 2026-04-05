"""Spatial algebra: adjoint matrix and skew-symmetric operations."""
from __future__ import annotations
import pypose as pp
import torch


def adjoint_se3(T: torch.Tensor) -> torch.Tensor:
    """6x6 Adjoint matrix of SE3 element T.

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
