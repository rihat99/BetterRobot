"""RobotModel: immutable robot kinematic structure."""
from __future__ import annotations

import torch
import yourdfpy

from .joint_info import JointInfo
from .link_info import LinkInfo

__all__ = [
    "RobotModel",
]


class RobotModel:
    """Immutable robot kinematic tree backed by PyTorch.

    All robot structure data is set once at construction via from_urdf()
    and never modified. This makes RobotModel safe to share across threads
    and parallel optimization runs.

    Attributes:
        joints: Joint metadata (names, types, limits, axes, origins).
        links: Link metadata (names, parent joints).
    """

    joints: JointInfo
    links: LinkInfo
    _frozen: bool = False

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, '_frozen', False) and name != '_frozen':
            raise AttributeError(f"RobotModel is immutable. Cannot set '{name}'.")
        super().__setattr__(name, value)

    def __init__(self, joints: JointInfo, links: LinkInfo) -> None:
        self.joints = joints
        self.links = links

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        default_joint_cfg: torch.Tensor | None = None,
    ) -> "RobotModel":
        """Load a robot model from a yourdfpy URDF instance.

        Args:
            urdf: Loaded yourdfpy.URDF instance.
            default_joint_cfg: Optional (num_actuated_joints,) initial config.
                Defaults to midpoint of joint limits.

        Returns:
            RobotModel with all FK data structures precomputed.
        """
        from .parsers._urdf_impl import RobotURDFParser
        joints, links = RobotURDFParser.parse(urdf)
        model = RobotModel.__new__(RobotModel)
        model.joints = joints
        model.links = links

        joint_map = urdf.joint_map
        link_name_to_idx = {name: idx for idx, name in enumerate(links.names)}

        base_link_name = None
        for i, name in enumerate(links.names):
            if links.parent_joint_indices[i] == -1:
                base_link_name = name
                break
        model._root_link_idx = link_name_to_idx[base_link_name]

        actuated_types = {'revolute', 'continuous', 'prismatic'}
        fk_joint_parent_link: list[int] = []
        fk_joint_child_link: list[int] = []
        fk_joint_types: list[str] = []
        fk_cfg_indices: list[int] = []

        actuated_counter = 0
        for j_idx, jname in enumerate(joints.names):
            joint = joint_map[jname]
            fk_joint_parent_link.append(link_name_to_idx[joint.parent])
            fk_joint_child_link.append(link_name_to_idx[joint.child])
            fk_joint_types.append(joint.type)
            if joint.type in actuated_types:
                fk_cfg_indices.append(actuated_counter)
                actuated_counter += 1
            else:
                fk_cfg_indices.append(-1)

        model._fk_joint_parent_link = fk_joint_parent_link
        model._fk_joint_child_link = fk_joint_child_link
        model._fk_joint_origins = joints.parent_transforms.clone()
        model._fk_joint_types = fk_joint_types
        model._fk_cfg_indices = fk_cfg_indices
        model._fk_joint_order = list(range(joints.num_joints))

        axes = torch.zeros(joints.num_joints, 3)
        for j_idx in range(joints.num_joints):
            jtype = fk_joint_types[j_idx]
            if jtype in ('revolute', 'continuous'):
                axes[j_idx] = joints.twists[j_idx, :3]
            elif jtype == 'prismatic':
                axes[j_idx] = joints.twists[j_idx, 3:]
        model._fk_joint_axes = axes

        if default_joint_cfg is None:
            model._default_cfg = (joints.lower_limits + joints.upper_limits) / 2.0
        else:
            model._default_cfg = default_joint_cfg

        model._frozen = True
        return model

    def forward_kinematics(
        self,
        cfg: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute world poses of all links.

        Args:
            cfg: Shape (*batch, num_actuated_joints). Joint configuration.
            base_pose: Shape (*batch, 7). Optional SE3 base transform
                [tx, ty, tz, qx, qy, qz, qw]. Default None (robot at world origin).

        Returns:
            Shape (*batch, num_links, 7). SE3 poses for each link.
        """
        from ..algorithms.kinematics.forward import forward_kinematics
        return forward_kinematics(self, cfg, base_pose)

    def link_index(self, link_name: str) -> int:
        """Return the index of a link by name.

        Args:
            link_name: Name of the link as it appears in the URDF.

        Returns:
            Integer index into the link dimension of forward_kinematics output.

        Raises:
            ValueError: If link_name is not found.
        """
        try:
            return self.links.names.index(link_name)
        except ValueError:
            raise ValueError(
                f"Link '{link_name}' not found. Available: {list(self.links.names)}"
            )

    def get_chain(self, link_idx: int) -> list[int]:
        """Joint indices (actuated only) on the path from root to link_idx.

        Args:
            link_idx: Target link index.

        Returns:
            List of joint indices in root->EE order (actuated joints only).
            Empty list if link_idx is the root.
        """
        from ..algorithms.kinematics.chain import get_chain
        return get_chain(self, link_idx)
