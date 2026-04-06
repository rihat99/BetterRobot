"""Internal URDF parsing implementation."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import yourdfpy

from ..joint_info import JointInfo
from ..link_info import LinkInfo
from ...math.so3 import so3_from_matrix


def _transform_to_se3(T: np.ndarray) -> list[float]:
    """Convert 4x4 numpy transform to [tx, ty, tz, qx, qy, qz, qw]."""
    t = T[:3, 3]
    R = torch.from_numpy(T[:3, :3].astype(np.float32))
    q = so3_from_matrix(R)
    return [float(t[0]), float(t[1]), float(t[2]),
            float(q[0]), float(q[1]), float(q[2]), float(q[3])]


class RobotURDFParser:
    """Parses a yourdfpy.URDF into JointInfo and LinkInfo."""

    @staticmethod
    def parse(urdf: yourdfpy.URDF) -> tuple[JointInfo, LinkInfo]:
        """Parse a yourdfpy.URDF object into JointInfo and LinkInfo."""
        joint_map = urdf.joint_map
        link_map = urdf.link_map

        child_links: set[str] = set()
        parent_to_joints: dict[str, list[str]] = {}
        for jname, joint in joint_map.items():
            child_links.add(joint.child)
            parent_link = joint.parent
            if parent_link not in parent_to_joints:
                parent_to_joints[parent_link] = []
            parent_to_joints[parent_link].append(jname)

        base_link = None
        for lname in link_map:
            if lname not in child_links:
                base_link = lname
                break
        assert base_link is not None, "Could not find base link"

        link_order: list[str] = [base_link]
        joint_order: list[str] = []
        queue: deque[str] = deque([base_link])
        visited_links: set[str] = {base_link}

        while queue:
            current_link = queue.popleft()
            if current_link in parent_to_joints:
                for jname in parent_to_joints[current_link]:
                    joint = joint_map[jname]
                    child = joint.child
                    joint_order.append(jname)
                    if child not in visited_links:
                        visited_links.add(child)
                        link_order.append(child)
                        queue.append(child)

        joint_name_to_idx = {name: idx for idx, name in enumerate(joint_order)}
        num_joints = len(joint_order)
        actuated_types = {'revolute', 'continuous', 'prismatic'}

        parent_indices_list: list[int] = []
        twists_list: list[list[float]] = []
        transforms_list: list[list[float]] = []
        lower_limits: list[float] = []
        upper_limits: list[float] = []
        velocity_limits: list[float] = []

        for j_idx, jname in enumerate(joint_order):
            joint = joint_map[jname]
            jtype = joint.type

            parent_link_name = joint.parent
            parent_j_idx = -1
            if parent_link_name != base_link:
                for other_jname in joint_order[:j_idx]:
                    if joint_map[other_jname].child == parent_link_name:
                        parent_j_idx = joint_name_to_idx[other_jname]
                        break
            parent_indices_list.append(parent_j_idx)

            if joint.axis is not None:
                axis = [float(joint.axis[0]), float(joint.axis[1]), float(joint.axis[2])]
            else:
                axis = [1.0, 0.0, 0.0]
            norm = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
            if norm > 1e-12:
                axis = [a / norm for a in axis]

            if jtype in ('revolute', 'continuous'):
                twist = [axis[0], axis[1], axis[2], 0.0, 0.0, 0.0]
            elif jtype == 'prismatic':
                twist = [0.0, 0.0, 0.0, axis[0], axis[1], axis[2]]
            else:
                twist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            twists_list.append(twist)

            if joint.origin is not None:
                se3 = _transform_to_se3(joint.origin)
            else:
                se3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            transforms_list.append(se3)

            if jtype in actuated_types:
                if joint.limit is not None:
                    lo = float(joint.limit.lower) if joint.limit.lower is not None else 0.0
                    hi = float(joint.limit.upper) if joint.limit.upper is not None else 0.0
                    vel = float(joint.limit.velocity) if joint.limit.velocity is not None else 0.0
                else:
                    lo, hi, vel = 0.0, 0.0, 0.0
                lower_limits.append(lo)
                upper_limits.append(hi)
                velocity_limits.append(vel)

        num_actuated = len(lower_limits)

        joint_info = JointInfo(
            names=tuple(joint_order),
            num_joints=num_joints,
            num_actuated_joints=num_actuated,
            lower_limits=torch.tensor(lower_limits, dtype=torch.float32),
            upper_limits=torch.tensor(upper_limits, dtype=torch.float32),
            velocity_limits=torch.tensor(velocity_limits, dtype=torch.float32),
            parent_indices=tuple(parent_indices_list),
            twists=torch.tensor(twists_list, dtype=torch.float32),
            parent_transforms=torch.tensor(transforms_list, dtype=torch.float32),
        )

        link_parent_joint_indices: list[int] = []
        for lname in link_order:
            if lname == base_link:
                link_parent_joint_indices.append(-1)
            else:
                found = False
                for jname in joint_order:
                    if joint_map[jname].child == lname:
                        link_parent_joint_indices.append(joint_name_to_idx[jname])
                        found = True
                        break
                if not found:
                    link_parent_joint_indices.append(-1)

        link_info = LinkInfo(
            names=tuple(link_order),
            num_links=len(link_order),
            parent_joint_indices=tuple(link_parent_joint_indices),
        )

        return joint_info, link_info
