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
