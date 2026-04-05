"""JointInfo dataclass: joint metadata for the kinematic tree."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class JointInfo:
    """Stores joint metadata for the kinematic tree."""

    names: tuple[str, ...]
    """Ordered joint names (topological BFS order)."""

    num_joints: int
    """Total number of joints (including fixed)."""

    num_actuated_joints: int
    """Number of actuated (non-fixed) joints."""

    lower_limits: torch.Tensor
    """Shape: (num_actuated_joints,). Joint lower limits in radians/meters."""

    upper_limits: torch.Tensor
    """Shape: (num_actuated_joints,). Joint upper limits in radians/meters."""

    velocity_limits: torch.Tensor
    """Shape: (num_actuated_joints,). Max joint velocity limits."""

    parent_indices: tuple[int, ...]
    """Parent joint index for each joint (-1 for root joints)."""

    twists: torch.Tensor
    """Shape: (num_joints, 6). Screw axis (twist) for each joint in parent frame."""

    parent_transforms: torch.Tensor
    """Shape: (num_joints, 7). SE3 transform from parent to joint frame."""
