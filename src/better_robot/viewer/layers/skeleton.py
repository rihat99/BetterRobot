"""``SkeletonLayer``: per-kind joint markers and cylinders along bones, one batched node per marker kind."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._math import (
    RGB,
    as_frames,
    axis_wxyz,
    instance_mesh,
    quat_z_to,
    rgb_uint8,
    unit_box,
    unit_cylinder,
    unit_sphere,
    xyzw_to_wxyz,
)
from ..layer import Geometry, Layer
from .frames import FramesLayer

_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_IDENTITY_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
# Joint kind (``JointModel.kind`` or its prefix before ``_``) -> marker; fixed joints draw nothing.
_MARKER = {
    "revolute": "cylinder", "helical": "cylinder", "mimic": "cylinder", "prismatic": "box",
    "spherical": "sphere", "free_flyer": "axes", "planar": "cube", "translation": "cube", "fixed": None,
}  # fmt: skip
# Marker -> (unit mesh, scale per unit joint radius); the mesh's +z is laid along the joint axis.
_MESHES = {
    "sphere": (unit_sphere, (1.0, 1.0, 1.0)),
    "cylinder": (unit_cylinder, (0.8, 0.8, 2.5)),
    "box": (unit_box, (0.6, 0.6, 3.0)),
    "cube": (unit_box, (1.4, 1.4, 1.4)),
}


class SkeletonLayer(Layer):
    """Joints ``(T, J, 3)`` positions or ``(T, J, 7)`` poses (xyzw) and ``parents (J,)`` (``-1`` for roots)
    drawn as a stick figure.

    Without ``kinds`` every joint is a sphere. With ``kinds`` (``JointModel.kind`` strings, see
    ``skeleton_spec``) and ``axes (J, 3)`` in the joint frame, revolute/helical joints are cylinders along
    the axis, prismatic thin boxes along the axis, spherical spheres, free-flyers small triads,
    planar/translation cubes, and fixed joints nothing. ``radius`` is one float or ``(J,)`` per joint;
    ``colors`` is one RGB, ``(J, 3)`` per joint, or ``(T, J, 3)`` per step (torque colouring), kept as uint8:
    edit it in place and call ``viewer.show()`` to recolour (per-step colours only). ``joints``
    ``(T, J, 3)`` and ``rotations`` ``(T, J, 4)`` are views of the input (kept by reference). Per step only instance
    transforms are written: one batched node per marker kind plus the bones.
    """

    def __init__(
        self,
        name: str,
        joints: np.ndarray,
        parents: np.ndarray | list[int],
        *,
        kinds: Sequence[str] | None = None,
        axes: np.ndarray | None = None,
        radius: float | np.ndarray = 0.02,
        bone_radius: float | None = None,
        colors: np.ndarray | RGB = (0.20, 0.60, 1.00),
        bone_color: RGB = (0.72, 0.72, 0.72),
    ) -> None:
        joints = as_frames(joints, 2)
        if joints.shape[-1] not in (3, 7):
            raise ValueError(f"joints must be (T, J, 3) positions or (T, J, 7) poses, got {joints.shape}")
        # Both are views into the caller's array (kept by reference); positions-only input has no rotation.
        self.joints = joints[..., :3]
        self.rotations = (
            joints[..., 3:] if joints.shape[-1] == 7 else np.broadcast_to(_IDENTITY_XYZW, (*joints.shape[:-1], 4))
        )
        super().__init__(name, self.joints.shape[0])
        n_joints = self.joints.shape[1]
        self.parents = np.asarray(parents, dtype=np.int64)
        if self.parents.shape != (n_joints,):
            raise ValueError("parents must have one entry per joint")
        kinds = ["spherical"] * n_joints if kinds is None else list(kinds)
        if len(kinds) != n_joints:
            raise ValueError("kinds must have one entry per joint")
        markers = [_MARKER.get(kind, _MARKER.get(kind.split("_")[0], "sphere")) for kind in kinds]
        self.groups = {m: np.array([j for j in range(n_joints) if markers[j] == m]) for m in set(markers) - {None}}
        axes = (0.0, 0.0, 1.0) if axes is None else axes
        self.axes = np.broadcast_to(np.asarray(axes, dtype=np.float32), (n_joints, 3))
        self.radius = np.broadcast_to(np.asarray(radius, dtype=np.float32), (n_joints,))
        self.bone_radius = float(self.radius.max()) * 0.6 if bone_radius is None else bone_radius
        self.bone_color = bone_color
        self._child = np.nonzero(self.parents >= 0)[0]
        self._parent = self.parents[self._child]
        colors = np.asarray(colors, dtype=np.float32)
        self.colors = rgb_uint8(colors, (n_joints,)) if colors.ndim <= 2 else rgb_uint8(colors)
        self._nodes: dict[str, Any] = {}
        self._bones: Any = None

    def build(self, scene: Any, root: str) -> None:
        colors = self.colors if self.colors.ndim == 2 else self.colors[0]
        for marker, index in self.groups.items():
            if marker == "axes":
                length = float(self.radius[index].max()) * 4.0
                self._nodes[marker] = self._track(
                    scene.add_batched_axes(
                        f"{root}/joints_axes",
                        batched_wxyzs=xyzw_to_wxyz(self.rotations[0, index]),
                        batched_positions=self.joints[0, index],
                        axes_length=length,
                        axes_radius=length * 0.05,
                    )
                )
                continue
            vertices, faces = _MESHES[marker][0]()
            positions, wxyzs, scales = self._marker_transforms(marker, 0)
            path = f"{root}/joints" if marker == "sphere" else f"{root}/joints_{marker}"
            self._nodes[marker] = self._track(
                scene.add_batched_meshes_simple(
                    path,
                    vertices,
                    faces,
                    batched_wxyzs=wxyzs,
                    batched_positions=positions,
                    batched_scales=scales,
                    batched_colors=colors[index],
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
        for marker, node in self._nodes.items():
            index = self.groups[marker]
            node.batched_positions = self.joints[step, index]
            if marker == "axes":
                node.batched_wxyzs = xyzw_to_wxyz(self.rotations[step, index])
                continue
            node.batched_wxyzs = self._marker_transforms(marker, step)[1]
            if self.colors.ndim == 3:
                node.batched_colors = self.colors[step, index]
        if self._bones is not None:
            positions, wxyzs, scales = self._bone_transforms(step)
            self._bones.batched_positions = positions
            self._bones.batched_wxyzs = wxyzs
            self._bones.batched_scales = scales

    def geometry(self, step: int) -> list[Geometry]:
        colors = (self.colors if self.colors.ndim == 2 else self.colors[step]).astype(np.float32) / 255.0
        out = []
        for marker, index in self.groups.items():
            if marker == "axes":
                length = float(self.radius[index].max()) * 4.0
                triads = FramesLayer(
                    self.name,
                    np.concatenate([self.joints[step, index], self.rotations[step, index]], axis=-1),
                    axes_length=length,
                    axes_radius=length * 0.05,
                )
                out += triads.geometry(0)
                continue
            vertices, faces = _MESHES[marker][0](1) if marker in ("sphere", "cylinder") else unit_box()
            vertices, faces = instance_mesh(vertices, faces, *self._marker_transforms(marker, step))
            out.append(Geometry(vertices, faces, np.repeat(colors[index], len(faces) // len(index), axis=0)))
        if len(self._child):
            cylinder_v, cylinder_f = unit_cylinder(8)
            positions, wxyzs, scales = self._bone_transforms(step)
            vertices, faces = instance_mesh(cylinder_v, cylinder_f, positions, wxyzs, scales)
            out.append(Geometry(vertices, faces, np.asarray(self.bone_color, dtype=np.float32)))
        return out

    def _marker_transforms(self, marker: str, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Positions ``(N, 3)``, wxyz ``(N, 4)`` and scales ``(N, 3)`` of one marker group at ``step``."""
        index = self.groups[marker]
        wxyzs = axis_wxyz(self.rotations[step, index], self.axes[index])
        scales = self.radius[index, None] * np.asarray(_MESHES[marker][1], dtype=np.float32)
        return self.joints[step, index], wxyzs, scales

    def _bone_transforms(self, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = self.joints[step, self._parent]
        end = self.joints[step, self._child]
        vector = end - start
        length = np.linalg.norm(vector, axis=-1)
        scales = np.stack(
            [np.full_like(length, self.bone_radius), np.full_like(length, self.bone_radius), length], axis=-1
        )
        return (start + end) * 0.5, quat_z_to(vector), scales
