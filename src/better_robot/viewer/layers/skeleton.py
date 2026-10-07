"""``SkeletonLayer``: spheres at joints and cylinders along bones, two batched nodes."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._math import RGB, as_frames, instance_mesh, quat_z_to, rgb_uint8, unit_cylinder, unit_sphere
from ..layer import Geometry, Layer

_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class SkeletonLayer(Layer):
    """Joint positions ``(T, J, 3)`` and ``parents (J,)`` (``-1`` for roots) drawn as a stick figure.

    ``radius`` is one float or ``(J,)`` per joint; ``colors`` is one RGB, ``(J, 3)`` per joint, or
    ``(T, J, 3)`` per step (for torque colouring). ``colors`` is kept as a uint8 array ``(J, 3)`` or
    ``(T, J, 3)``: edit it in place and call ``viewer.show()`` to recolour (per-step colours only).
    Per step only instance transforms are written: J sphere positions and B bone poses.
    """

    def __init__(
        self,
        name: str,
        joints: np.ndarray,
        parents: np.ndarray | list[int],
        *,
        radius: float | np.ndarray = 0.02,
        bone_radius: float | None = None,
        colors: np.ndarray | RGB = (0.20, 0.60, 1.00),
        bone_color: RGB = (0.72, 0.72, 0.72),
    ) -> None:
        self.joints = as_frames(joints, 2)
        super().__init__(name, self.joints.shape[0])
        self.parents = np.asarray(parents, dtype=np.int64)
        if self.parents.shape != (self.joints.shape[1],):
            raise ValueError("parents must have one entry per joint")
        self.radius = np.broadcast_to(np.asarray(radius, dtype=np.float32), (self.joints.shape[1],))
        self.bone_radius = float(self.radius.max()) * 0.6 if bone_radius is None else bone_radius
        self.bone_color = bone_color
        self._child = np.nonzero(self.parents >= 0)[0]
        self._parent = self.parents[self._child]
        colors = np.asarray(colors, dtype=np.float32)
        self.colors = rgb_uint8(colors, (self.joints.shape[1],)) if colors.ndim <= 2 else rgb_uint8(colors)
        self._spheres: Any = None
        self._bones: Any = None

    def build(self, scene: Any, root: str) -> None:
        vertices, faces = unit_sphere()
        n_joints = self.joints.shape[1]
        colors = self.colors if self.colors.ndim == 2 else self.colors[0]
        self._spheres = self._track(
            scene.add_batched_meshes_simple(
                f"{root}/joints",
                vertices,
                faces,
                batched_wxyzs=np.tile(_IDENTITY, (n_joints, 1)),
                batched_positions=self.joints[0],
                batched_scales=self.radius,
                batched_colors=colors,
                lod="off",
            )
        )
        if len(self._child):
            vertices, faces = unit_cylinder()
            positions, wxyzs, scales = self._bone_transforms(0)
            self._bones = self._track(
                scene.add_batched_meshes_simple(
                    f"{root}/bones",
                    vertices,
                    faces,
                    batched_wxyzs=wxyzs,
                    batched_positions=positions,
                    batched_scales=scales,
                    batched_colors=rgb_uint8(self.bone_color),
                    lod="off",
                )
            )

    def set_step(self, step: int) -> None:
        self._spheres.batched_positions = self.joints[step]
        if self.colors.ndim == 3:
            self._spheres.batched_colors = self.colors[step]
        if self._bones is not None:
            positions, wxyzs, scales = self._bone_transforms(step)
            self._bones.batched_positions = positions
            self._bones.batched_wxyzs = wxyzs
            self._bones.batched_scales = scales

    def geometry(self, step: int) -> list[Geometry]:
        sphere_v, sphere_f = unit_sphere(1)
        n_joints = self.joints.shape[1]
        vertices, faces = instance_mesh(
            sphere_v, sphere_f, self.joints[step], np.tile(_IDENTITY, (n_joints, 1)), self.radius
        )
        colors = (self.colors if self.colors.ndim == 2 else self.colors[step]).astype(np.float32) / 255.0
        out = [Geometry(vertices, faces, np.repeat(colors, len(sphere_f), axis=0))]
        if len(self._child):
            cylinder_v, cylinder_f = unit_cylinder(8)
            positions, wxyzs, scales = self._bone_transforms(step)
            vertices, faces = instance_mesh(cylinder_v, cylinder_f, positions, wxyzs, scales)
            out.append(Geometry(vertices, faces, np.asarray(self.bone_color, dtype=np.float32)))
        return out

    def _bone_transforms(self, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = self.joints[step, self._parent]
        end = self.joints[step, self._child]
        vector = end - start
        length = np.linalg.norm(vector, axis=-1)
        scales = np.stack(
            [np.full_like(length, self.bone_radius), np.full_like(length, self.bone_radius), length], axis=-1
        )
        return (start + end) * 0.5, quat_z_to(vector), scales
