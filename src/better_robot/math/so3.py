"""SO3 Lie group operations (PyPose backend)."""
from __future__ import annotations
import pypose as pp
import torch

__all__ = [
    "so3_rotation_matrix",
    "so3_act",
    "so3_from_matrix",
]

def so3_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Quaternion [qx, qy, qz, qw] -> (3, 3) rotation matrix."""
    return pp.SO3(q).matrix()

def so3_act(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q [qx, qy, qz, qw]."""
    return pp.SO3(q).Act(v)

def so3_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """(3, 3) rotation matrix -> [qx, qy, qz, qw] quaternion."""
    return pp.mat2SO3(R).tensor()
