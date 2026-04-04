"""URDF parsing utilities. Converts yourdfpy.URDF into JointInfo and LinkInfo dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import yourdfpy


@dataclass
class JointInfo:
    """Stores joint metadata for the kinematic tree."""

    names: tuple[str, ...]
    """Ordered joint names (topological order)."""

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
    """Shape: (num_joints, 7). SE3 transform from parent to joint frame (wxyz+xyz)."""


@dataclass
class LinkInfo:
    """Stores link metadata."""

    names: tuple[str, ...]
    """Ordered link names."""

    num_links: int
    """Total number of links."""

    parent_joint_indices: tuple[int, ...]
    """Index of the parent joint for each link (-1 for base link)."""


class RobotURDFParser:
    """Parses a yourdfpy.URDF into JointInfo and LinkInfo."""

    @staticmethod
    def parse(urdf: yourdfpy.URDF) -> tuple[JointInfo, LinkInfo]:
        """Parse a yourdfpy.URDF object.

        Args:
            urdf: A loaded yourdfpy.URDF instance.

        Returns:
            Tuple of (JointInfo, LinkInfo).
        """
        raise NotImplementedError
