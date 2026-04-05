"""RobotModel: immutable robot kinematic structure."""
from __future__ import annotations

import torch
import yourdfpy

from ..math.se3 import se3_compose, se3_identity, se3_apply_base
from .joint_info import JointInfo
from .link_info import LinkInfo


def _revolute_transform(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Pure rotation SE3: [0, 0, 0, sin*ax, sin*ay, sin*az, cos(a/2)]."""
    half = angle / 2.0
    cos_h = torch.cos(half)
    sin_h = torch.sin(half)
    qxyz = sin_h.unsqueeze(-1) * axis
    zeros = torch.zeros(*angle.shape, 3, device=angle.device, dtype=angle.dtype)
    return torch.cat([zeros, qxyz, cos_h.unsqueeze(-1)], dim=-1)


def _prismatic_transform(axis: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Pure translation SE3: [d*ax, d*ay, d*az, 0, 0, 0, 1]."""
    batch_shape = displacement.shape
    device, dtype = displacement.device, displacement.dtype
    trans = displacement.unsqueeze(-1) * axis.to(device=device, dtype=dtype)
    qxyz = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
    qw = torch.ones(*batch_shape, 1, device=device, dtype=dtype)
    return torch.cat([trans, qxyz, qw], dim=-1)


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
        batch_shape = cfg.shape[:-1]
        device, dtype = cfg.device, cfg.dtype

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

            T_parent = link_world[parent_link]
            T_origin = self._fk_joint_origins[j_idx].to(device=device, dtype=dtype)
            T = se3_compose(T_parent, T_origin.expand_as(T_parent))

            cfg_idx = self._fk_cfg_indices[j_idx]
            if cfg_idx >= 0:
                q = cfg[..., cfg_idx]
                axis = self._fk_joint_axes[j_idx].to(device=device, dtype=dtype)
                jtype = self._fk_joint_types[j_idx]
                if jtype in ('revolute', 'continuous'):
                    T_motion = _revolute_transform(axis, q)
                else:
                    T_motion = _prismatic_transform(axis, q)
                T = se3_compose(T, T_motion)

            link_world[child_link] = T

        result = torch.stack(
            [link_world[i] for i in range(self.links.num_links)], dim=-2
        )
        if base_pose is not None:
            result = se3_apply_base(base_pose, result)
        return result

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
        child_to_joint: dict[int, int] = {
            child: j for j, child in enumerate(self._fk_joint_child_link)
        }
        chain: list[int] = []
        current = link_idx
        while current != self._root_link_idx:
            if current not in child_to_joint:
                raise ValueError(
                    f"Link index {link_idx} is not reachable from root "
                    f"(stuck at link {current}). Possible orphaned link."
                )
            j = child_to_joint[current]
            if self._fk_cfg_indices[j] >= 0:
                chain.append(j)
            current = self._fk_joint_parent_link[j]
        chain.reverse()
        return chain
