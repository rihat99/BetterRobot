"""Robot class: loads URDF and runs forward kinematics."""

from __future__ import annotations

import torch
import yourdfpy

from ._lie_ops import se3_compose, se3_identity, se3_apply_base
from ._urdf_parser import JointInfo, LinkInfo, RobotURDFParser


def _revolute_transform(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Pure rotation SE3: [0, 0, 0, sin*ax, sin*ay, sin*az, cos(a/2)]

    Args:
        axis: (3,) unit rotation axis.
        angle: (*batch,) rotation angle in radians.

    Returns:
        (*batch, 7) SE3 transform.
    """
    half = angle / 2.0
    cos_h = torch.cos(half)                               # (*batch,)
    sin_h = torch.sin(half)                               # (*batch,)
    qxyz = sin_h.unsqueeze(-1) * axis                    # (*batch, 3)
    zeros = torch.zeros(*angle.shape, 3, device=angle.device, dtype=angle.dtype)
    return torch.cat([zeros, qxyz, cos_h.unsqueeze(-1)], dim=-1)   # (*batch, 7)


def _prismatic_transform(axis: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Pure translation SE3: [d*ax, d*ay, d*az, 0, 0, 0, 1]

    Args:
        axis: (3,) unit translation axis.
        displacement: (*batch,) translation magnitude.

    Returns:
        (*batch, 7) SE3 transform.
    """
    batch_shape = displacement.shape
    device, dtype = displacement.device, displacement.dtype
    trans = displacement.unsqueeze(-1) * axis.to(device=device, dtype=dtype)
    qxyz = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
    qw = torch.ones(*batch_shape, 1, device=device, dtype=dtype)
    return torch.cat([trans, qxyz, qw], dim=-1)   # (*batch, 7)


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
        joints, links = RobotURDFParser.parse(urdf)
        robot = Robot.__new__(Robot)
        robot.joints = joints
        robot.links = links

        # Build FK data structures
        joint_map = urdf.joint_map
        link_name_to_idx = {name: idx for idx, name in enumerate(links.names)}

        # Find base link (the one with parent_joint_indices == -1)
        base_link_name = None
        for i, name in enumerate(links.names):
            if links.parent_joint_indices[i] == -1:
                base_link_name = name
                break
        robot._root_link_idx = link_name_to_idx[base_link_name]

        # Build per-joint FK data
        actuated_types = {'revolute', 'continuous', 'prismatic'}
        fk_joint_parent_link: list[int] = []
        fk_joint_child_link: list[int] = []
        fk_joint_types: list[str] = []
        fk_cfg_indices: list[int] = []

        actuated_counter = 0
        for j_idx, jname in enumerate(joints.names):
            joint = joint_map[jname]
            parent_link = joint.parent
            child_link = joint.child
            fk_joint_parent_link.append(link_name_to_idx[parent_link])
            fk_joint_child_link.append(link_name_to_idx[child_link])
            fk_joint_types.append(joint.type)

            if joint.type in actuated_types:
                fk_cfg_indices.append(actuated_counter)
                actuated_counter += 1
            else:
                fk_cfg_indices.append(-1)

        robot._fk_joint_parent_link = fk_joint_parent_link
        robot._fk_joint_child_link = fk_joint_child_link
        robot._fk_joint_origins = joints.parent_transforms.clone()
        robot._fk_joint_types = fk_joint_types
        robot._fk_cfg_indices = fk_cfg_indices
        robot._fk_joint_order = list(range(joints.num_joints))

        # Derive axes from twists
        axes = torch.zeros(joints.num_joints, 3)
        for j_idx in range(joints.num_joints):
            jtype = fk_joint_types[j_idx]
            if jtype in ('revolute', 'continuous'):
                axes[j_idx] = joints.twists[j_idx, :3]
            elif jtype == 'prismatic':
                axes[j_idx] = joints.twists[j_idx, 3:]
        robot._fk_joint_axes = axes

        if default_joint_cfg is None:
            robot._default_cfg = (joints.lower_limits + joints.upper_limits) / 2.0
        else:
            robot._default_cfg = default_joint_cfg

        return robot

    def forward_kinematics(
        self,
        cfg: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute world poses of all links.

        Args:
            cfg: Shape (*batch, num_actuated_joints). Joint configuration.
            base_pose: Shape (*batch, 7). Optional SE3 base transform
                [tx, ty, tz, qx, qy, qz, qw]. When provided, all link poses
                are expressed in the base frame's world coordinates.
                Defaults to identity (robot root at world origin).

        Returns:
            Shape (*batch, num_links, 7). SE3 poses [tx, ty, tz, qx, qy, qz, qw] for each link.
        """
        batch_shape = cfg.shape[:-1]
        device, dtype = cfg.device, cfg.dtype

        # Initialize link world poses
        link_world: dict[int, torch.Tensor] = {}
        link_world[self._root_link_idx] = (
            se3_identity()
            .to(device=device, dtype=dtype)
            .expand(*batch_shape, 7)
            .clone()
        )

        for j_idx in self._fk_joint_order:
            parent_link = self._fk_joint_parent_link[j_idx]
            child_link = self._fk_joint_child_link[j_idx]

            T_parent = link_world[parent_link]  # (*batch, 7)
            T_origin = self._fk_joint_origins[j_idx].to(device=device, dtype=dtype)  # (7,)

            # Compose parent pose with fixed origin transform
            T = se3_compose(T_parent, T_origin.expand_as(T_parent))

            # Add joint motion if actuated
            cfg_idx = self._fk_cfg_indices[j_idx]
            if cfg_idx >= 0:
                q = cfg[..., cfg_idx]  # (*batch,)
                axis = self._fk_joint_axes[j_idx].to(device=device, dtype=dtype)  # (3,)
                jtype = self._fk_joint_types[j_idx]
                if jtype in ('revolute', 'continuous'):
                    T_motion = _revolute_transform(axis, q)
                else:  # prismatic
                    T_motion = _prismatic_transform(axis, q)
                T = se3_compose(T, T_motion)

            link_world[child_link] = T

        # Stack in link index order
        result = torch.stack(
            [link_world[i] for i in range(self.links.num_links)], dim=-2
        )
        if base_pose is not None:
            result = se3_apply_base(base_pose, result)
        return result

    def get_link_index(self, link_name: str) -> int:
        """Return the index of a link by name.

        Args:
            link_name: Name of the link as it appears in the URDF.

        Returns:
            Integer index into the link dimension of forward_kinematics output.

        Raises:
            ValueError: If link_name not found.
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
            List of joint indices in root→EE order (only actuated joints).
            Empty list if link_idx is the root.
        """
        # Map child_link → joint_index for fast lookup
        child_to_joint: dict[int, int] = {
            child: j for j, child in enumerate(self._fk_joint_child_link)
        }

        chain: list[int] = []
        current = link_idx
        while current != self._root_link_idx:
            if current not in child_to_joint:
                break
            j = child_to_joint[current]
            if self._fk_cfg_indices[j] >= 0:   # actuated joints only
                chain.append(j)
            current = self._fk_joint_parent_link[j]

        chain.reverse()   # was built EE→root, reverse to root→EE
        return chain
