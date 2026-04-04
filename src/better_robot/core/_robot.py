"""Robot class: loads URDF and runs forward kinematics."""

from __future__ import annotations

import torch
import yourdfpy

from ._urdf_parser import JointInfo, LinkInfo, RobotURDFParser


class Robot:
    """Differentiable robot kinematic tree backed by PyTorch."""

    joints: JointInfo
    links: LinkInfo

    def __init__(self, joints: JointInfo, links: LinkInfo) -> None:
        self.joints = joints
        self.links = links

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        default_joint_cfg: torch.Tensor | None = None,
    ) -> "Robot":
        """Load a robot kinematic tree from a yourdfpy URDF.

        Args:
            urdf: Loaded yourdfpy.URDF instance.
            default_joint_cfg: Shape (num_actuated_joints,). Initial joint config.
                Defaults to midpoint of joint limits.

        Returns:
            Robot instance.
        """
        raise NotImplementedError

    def forward_kinematics(
        self,
        cfg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute world poses of all links.

        Args:
            cfg: Shape (*batch, num_actuated_joints). Joint configuration.

        Returns:
            Shape (*batch, num_links, 7). SE3 poses (wxyz+xyz) for each link.
        """
        raise NotImplementedError

    def get_link_index(self, link_name: str) -> int:
        """Return the index of a link by name.

        Args:
            link_name: Name of the link as it appears in the URDF.

        Returns:
            Integer index into the link dimension of forward_kinematics output.

        Raises:
            ValueError: If link_name not found.
        """
        raise NotImplementedError
