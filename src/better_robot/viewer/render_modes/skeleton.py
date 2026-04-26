"""``render_modes/skeleton.py`` — abstract kinematic skeleton render mode.

Draws one sphere per articulated joint and one cylinder per link.
Always available: only requires ``model.parents`` and
``data.joint_pose_world``.

See ``docs/concepts/viewer.md §4.3``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch

from .base import RenderContext
from ..helpers import quat_xyzw_to_wxyz

if TYPE_CHECKING:
    from ...data_model.data import Data
    from ...data_model.model import Model


def _align_z_to_vec(direction: torch.Tensor) -> torch.Tensor:
    """Return a unit quaternion [qx, qy, qz, qw] that rotates (0,0,1) to align
    with ``direction`` (a 3-vector, need not be unit length).
    """
    d = direction / (direction.norm() + 1e-10)
    z = torch.tensor([0.0, 0.0, 1.0], dtype=d.dtype, device=d.device)
    cross = torch.cross(z, d, dim=-1)
    cross_norm = cross.norm()
    if cross_norm < 1e-8:
        # Parallel or anti-parallel
        dot = z.dot(d)
        if dot > 0:
            return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=d.dtype, device=d.device)
        else:
            # 180° flip around X
            return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=d.dtype, device=d.device)
    axis = cross / cross_norm
    angle = math.acos(float(z.dot(d).clamp(-1.0, 1.0)))
    half = angle / 2.0
    s = math.sin(half)
    return torch.tensor([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)],
                         dtype=d.dtype, device=d.device)


class SkeletonMode:
    """Abstract kinematic skeleton — a sphere per joint, a cylinder per link.

    Always available: works off ``model.parents`` and
    ``data.joint_pose_world`` alone. This is the mode used for
    programmatically-built robots that never go through a URDF.
    """

    name = "Skeleton"
    description = "Spheres for joints, cylinders for links"

    def __init__(
        self,
        *,
        joint_radius: float = 0.03,
        link_radius: float = 0.015,
        show_root: bool = True,
        colour_by: Literal["uniform", "subtree", "depth"] = "uniform",
    ) -> None:
        self._joint_radius = joint_radius
        self._link_radius = link_radius
        self._show_root = show_root
        self._colour_by = colour_by

        self._ctx: RenderContext | None = None
        self._model: Model | None = None
        self._sphere_joints: list[int] = []   # joint indices that have a sphere
        self._cylinder_joints: list[int] = [] # joint indices that have a cylinder (j→parent)

    @classmethod
    def is_available(cls, model: "Model", data: "Data") -> bool:
        return True  # every Model has a topology

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self, context: RenderContext, model: "Model", data: "Data") -> None:
        self._ctx = context
        self._model = model
        backend = context.backend
        ns = context.namespace
        theme = context.theme

        joint_rgba = theme.joint_colour if theme else (0.2, 0.6, 1.0, 1.0)
        link_rgba = theme.link_colour if theme else (0.7, 0.7, 0.7, 1.0)

        # Pick the batch slice to use for initial attach geometry
        b = context.batch_index
        joint_pose_world = data.joint_pose_world  # (B..., njoints, 7) or (njoints, 7)

        for j in range(model.njoints):
            jm = model.joint_models[j]
            # Sphere: articulated joints (nv > 0) + optionally the universe joint
            if jm.nv > 0 or (j == 0 and self._show_root):
                name = f"{ns}/sphere_{j}"
                pose = self._get_joint_pose(joint_pose_world, j, b)
                backend.add_sphere(name, radius=self._joint_radius,
                                   rgba=self._colour_for_joint(j, model, joint_rgba))
                backend.set_transform(name, pose)
                self._sphere_joints.append(j)

            # Cylinder: every non-root joint
            if j > 0:
                p = model.parents[j]
                name = f"{ns}/link_{j}"
                p_j = self._get_joint_pos(joint_pose_world, j, b)
                p_p = self._get_joint_pos(joint_pose_world, p, b)
                length = float((p_j - p_p).norm())
                if length > 1e-4:
                    backend.add_cylinder(name, radius=self._link_radius,
                                         length=length, rgba=link_rgba)
                    mid_pose = self._link_pose(p_p, p_j)
                    backend.set_transform(name, mid_pose)
                    self._cylinder_joints.append(j)

    def update(self, data: "Data") -> None:
        if self._ctx is None or self._model is None:
            return
        backend = self._ctx.backend
        ns = self._ctx.namespace
        b = self._ctx.batch_index
        model = self._model
        joint_pose_world = data.joint_pose_world

        for j in self._sphere_joints:
            name = f"{ns}/sphere_{j}"
            pose = self._get_joint_pose(joint_pose_world, j, b)
            backend.set_transform(name, pose)

        for j in self._cylinder_joints:
            p = model.parents[j]
            name = f"{ns}/link_{j}"
            p_j = self._get_joint_pos(joint_pose_world, j, b)
            p_p = self._get_joint_pos(joint_pose_world, p, b)
            if float((p_j - p_p).norm()) > 1e-4:
                mid_pose = self._link_pose(p_p, p_j)
                backend.set_transform(name, mid_pose)

    def set_visible(self, visible: bool) -> None:
        if self._ctx is None:
            return
        backend = self._ctx.backend
        ns = self._ctx.namespace
        for j in self._sphere_joints:
            backend.set_visible(f"{ns}/sphere_{j}", visible)
        for j in self._cylinder_joints:
            backend.set_visible(f"{ns}/link_{j}", visible)

    def detach(self) -> None:
        if self._ctx is None:
            return
        backend = self._ctx.backend
        ns = self._ctx.namespace
        for j in self._sphere_joints:
            backend.remove(f"{ns}/sphere_{j}")
        for j in self._cylinder_joints:
            backend.remove(f"{ns}/link_{j}")
        self._sphere_joints = []
        self._cylinder_joints = []
        self._ctx = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_joint_pose(self, poses: torch.Tensor, j: int, b: int) -> torch.Tensor:
        """Extract the 7-vector pose for joint ``j`` from ``joint_pose_world``,
        handling batch dims."""
        if poses.dim() == 2:
            return poses[j]
        elif poses.dim() == 3:
            return poses[b, j]
        else:
            return poses[b, j]

    def _get_joint_pos(self, poses: torch.Tensor, j: int, b: int) -> torch.Tensor:
        return self._get_joint_pose(poses, j, b)[:3]

    def _link_pose(self, p_parent: torch.Tensor, p_child: torch.Tensor) -> torch.Tensor:
        """Compute a 7-vector pose for a cylinder between two joint positions."""
        mid = (p_parent + p_child) / 2.0
        direction = p_child - p_parent
        q = _align_z_to_vec(direction)
        return torch.cat([mid, q])

    def _colour_for_joint(
        self,
        j: int,
        model: "Model",
        default: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if self._colour_by == "uniform":
            return default
        elif self._colour_by == "depth":
            depth = len(model.supports[j])
            t = min(depth / 10.0, 1.0)
            r = 0.2 + 0.8 * t
            b = 1.0 - 0.8 * t
            return (r, 0.4, b, 1.0)
        else:
            return default
