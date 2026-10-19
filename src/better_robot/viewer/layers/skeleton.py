"""``SkeletonLayer``: per-kind joint markers and cylinders along bones, one batched node per marker kind."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._math import (
    RGB,
    as_frames,
    axis_wxyz,
    quat_z_to,
    rgb_uint8,
    rotate,
    unit_box,
    unit_cylinder,
    unit_sphere,
    xyzw_to_wxyz,
)
from ..layer import Layer

_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_IDENTITY_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
_CONJUGATE = np.array([-1.0, -1.0, -1.0, 1.0], dtype=np.float32)
_FOLD = 0.01  # m: a bone leg shorter than this folds into the bone's longest leg
_STRAIGHT = 0.15  # m: a longer bone from a joint with one child stays straight (a thigh)
_CORNER = 0.98  # corner sphere radius per bone radius: inside the bone's 16 flat sides, so no scalloped seam
# Joint kind (``JointModel.kind`` or its prefix before ``_``) -> marker; fixed joints draw nothing.
_MARKER = {
    "revolute": "cylinder", "helical": "cylinder", "mimic": "cylinder", "prismatic": "box",
    "spherical": "sphere", "free_flyer": "axes", "planar": "cube", "translation": "cube", "fixed": None,
}  # fmt: skip
# Marker -> (unit mesh, scale per unit joint radius); the mesh's +z is laid along the joint axis.
_MESHES = {
    "sphere": (unit_sphere, (1.0, 1.0, 1.0)),
    "cylinder": (unit_cylinder, (1.0, 1.0, 2.4)),
    "box": (unit_box, (0.6, 0.6, 3.0)),
    "cube": (unit_box, (1.4, 1.4, 1.4)),
}
# Marker -> (radius, half-length along the axis) per unit joint radius of the capsule that encloses it; the
# free-flyer triad counts as its hub only (its arms are thin lines).
_EXTENT = {
    "cylinder": (1.0, 1.2),
    "box": (0.3 * 2**0.5, 1.5),
    "sphere": (1.0, 0.0),
    "cube": (0.7 * 3**0.5, 0.0),
    "axes": (0.5, 0.0),
}


def marker_of(kind: str) -> str | None:
    """The marker a joint kind (``JointModel.kind`` or its prefix before ``_``) is drawn as: ``"cylinder"``,
    ``"box"``, ``"sphere"``, ``"axes"``, ``"cube"``, or None for fixed joints (nothing drawn)."""
    return _MARKER.get(kind, _MARKER.get(kind.split("_", 1)[0], "sphere"))


def marker_extent(marker: str) -> tuple[float, float]:
    """``(radius, half_length)`` per unit joint radius of the capsule enclosing ``marker``, the length along the
    joint axis (0 for markers without one)."""
    return _EXTENT[marker]


class SkeletonLayer(Layer):
    """Joints ``(T, J, 3)`` positions or ``(T, J, 7)`` poses (xyzw) and ``parents (J,)`` (``-1`` for roots)
    drawn as a stick figure.

    Without ``kinds`` every joint is a sphere. With ``kinds`` (``JointModel.kind`` strings, see
    ``skeleton_spec``) and ``axes (J, 3)`` in the joint frame, revolute/helical joints are cylinders along
    the axis, prismatic thin boxes along the axis, spherical spheres, free-flyers small triads,
    planar/translation cubes, and fixed joints nothing. ``radius`` is one float, ``(J,)`` per joint, or
    ``(T, J)`` per step; a radius of 0 hides the marker. ``thickness`` ``(J,)`` or ``(T, J)`` (a joint's
    load, per step) multiplies each marker's cross-section but not its length along the axis; markers
    without an axis grow whole. Marker scales are written per step only for a ``(T, J)`` radius or thickness.
    ``offsets (J, 3)``, in each joint's frame (``robot.spread_offsets``), move a marker off its joint
    origin, such as along its own axis; bones join the drawn marker centres, and a bone of zero length
    draws nothing. ``bone_radius`` is one float or ``(J,)`` per bone, entry ``j`` for the bone
    from ``parents[j]`` to ``j`` (roots' entries unused); 0 hides the bone. ``colors`` is one RGB, ``(J, 3)``
    per joint, or ``(T, J, 3)`` per step (torque colouring), kept as uint8: edit it in place and call
    ``viewer.show()`` to recolour (per-step colours only). ``joints`` ``(T, J, 3)`` and ``rotations``
    ``(T, J, 4)`` are views of the input (kept by reference). Per step only instance transforms are
    written: one batched node per marker kind plus the bones (and their corners).

    A bone with an axis marker (cylinder or box) at either end runs, textbook style, as up to three legs along
    the axes of its parent joint's frame, joined by spheres as thick as the bone: first along the parent's
    joint axis, last along the frame axis nearest the child's joint axis, the remaining axes between them,
    longest first (when both are the same frame axis, only the last leg runs along it). A leg shorter than
    1 cm folds into the bone's longest leg. The routing is chosen once, from step 0; each step the legs split
    the bone's current centre difference, in its parent's frame, over those axes, so they always end on the
    child's marker. Bones between joints without an axis marker (every bone without ``kinds``) and bones
    longer than 15 cm from a joint with one child (a thigh) stay straight.
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
        thickness: np.ndarray | None = None,
        offsets: np.ndarray | None = None,
        bone_radius: float | np.ndarray | None = None,
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
        markers = [marker_of(kind) for kind in kinds]
        self.groups = {m: np.array([j for j in range(n_joints) if markers[j] == m]) for m in set(markers) - {None}}
        axes = (0.0, 0.0, 1.0) if axes is None else axes
        self.axes = np.broadcast_to(np.asarray(axes, dtype=np.float32), (n_joints, 3))
        radius = np.asarray(radius, dtype=np.float32)
        self.radius = radius if radius.ndim == 2 else np.broadcast_to(radius, (n_joints,))
        if self.radius.shape[-1] != n_joints or (radius.ndim == 2 and len(radius) != self.nsteps):
            raise ValueError(f"radius must be one float, (J,) or (T, J) = {self.joints.shape[:2]}, got {radius.shape}")
        self.thickness = None if thickness is None else np.asarray(thickness, dtype=np.float32)
        if self.thickness is not None and self.thickness.shape not in ((n_joints,), (self.nsteps, n_joints)):
            raise ValueError(f"thickness must be (J,) or (T, J) = {self.joints.shape[:2]}, got {self.thickness.shape}")
        self.offsets = None if offsets is None else np.asarray(offsets, dtype=np.float32)
        if self.offsets is not None and self.offsets.shape != (n_joints, 3):
            raise ValueError(f"offsets must be (J, 3) = ({n_joints}, 3), got {self.offsets.shape}")
        # A float or (J,) per bone, entry j for the bone into joint j; read at every step.
        self.bone_radius = float(self.radius.max()) * 0.6 if bone_radius is None else bone_radius
        self.bone_color = bone_color
        self._child = np.nonzero(self.parents >= 0)[0]
        self._parent = self.parents[self._child]
        colors = np.asarray(colors, dtype=np.float32)
        self.colors = rgb_uint8(colors, (n_joints,)) if colors.ndim <= 2 else rgb_uint8(colors)
        self._legs = self._route(np.array([marker in ("cylinder", "box") for marker in markers]))
        self._nodes: dict[str, Any] = {}
        self._bones: Any = None
        self._corners: Any = None

    def build(self, scene: Any, root: str) -> None:
        colors = self.colors if self.colors.ndim == 2 else self.colors[0]
        centres = self._centres(0)
        for marker, index in self.groups.items():
            if marker == "axes":
                length = float(self.radius[..., index].max()) * 4.0
                self._nodes[marker] = self._track(
                    scene.add_batched_axes(
                        f"{root}/joints_axes",
                        batched_wxyzs=xyzw_to_wxyz(self.rotations[0, index]),
                        batched_positions=centres[index],
                        axes_length=length,
                        axes_radius=length * 0.05,
                    )
                )
                continue
            vertices, faces = _MESHES[marker][0]()
            wxyzs, scales = self._marker_transforms(marker, 0)
            path = f"{root}/joints" if marker == "sphere" else f"{root}/joints_{marker}"
            self._nodes[marker] = self._track(
                scene.add_batched_meshes_simple(
                    path,
                    vertices,
                    faces,
                    batched_wxyzs=wxyzs,
                    batched_positions=centres[index],
                    batched_scales=scales,
                    batched_colors=colors[index],
                    lod="off",
                )
            )
        if len(self._child):
            vertices, faces = unit_cylinder()
            positions, wxyzs, scales, corners, corner_scales = self._bone_transforms(centres, 0)
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
            if len(corners):
                vertices, faces = unit_sphere(3)  # finer than the bones' facets: a smooth elbow, no jagged seam
                self._corners = self._track(
                    scene.add_batched_meshes_simple(
                        f"{root}/bone_corners",
                        vertices,
                        faces,
                        batched_wxyzs=np.tile(_IDENTITY, (len(corners), 1)),
                        batched_positions=corners,
                        batched_scales=corner_scales,
                        batched_colors=rgb_uint8(self.bone_color),
                        lod="off",
                    )
                )

    def set_step(self, step: int) -> None:
        centres = self._centres(step)
        per_step_scale = self.radius.ndim == 2 or (self.thickness is not None and self.thickness.ndim == 2)
        for marker, node in self._nodes.items():
            index = self.groups[marker]
            node.batched_positions = centres[index]
            if marker == "axes":
                node.batched_wxyzs = xyzw_to_wxyz(self.rotations[step, index])
                continue
            wxyzs, scales = self._marker_transforms(marker, step)
            node.batched_wxyzs = wxyzs
            if per_step_scale:
                node.batched_scales = scales
            if self.colors.ndim == 3:
                node.batched_colors = self.colors[step, index]
        if self._bones is not None:
            positions, wxyzs, scales, corners, corner_scales = self._bone_transforms(centres, step)
            self._bones.batched_positions = positions
            self._bones.batched_wxyzs = wxyzs
            self._bones.batched_scales = scales
            if self._corners is not None:
                self._corners.batched_positions = corners
                self._corners.batched_scales = corner_scales

    def world_points(self, step: int) -> np.ndarray | None:
        """World positions ``(N, 3)`` drawn at ``step``, for camera fitting."""
        return self.joints[step]

    def _centres(self, step: int) -> np.ndarray:
        """Drawn marker centres ``(J, 3)`` at ``step``: the joints moved by their ``offsets`` (joint frame)."""
        if self.offsets is None:
            return self.joints[step]
        return self.joints[step] + rotate(self.rotations[step], self.offsets)

    def _marker_transforms(self, marker: str, step: int) -> tuple[np.ndarray, np.ndarray]:
        """wxyz ``(N, 4)`` and scales ``(N, 3)`` of one marker group at ``step``."""
        index = self.groups[marker]
        wxyzs = axis_wxyz(self.rotations[step, index], self.axes[index])
        radius = self.radius[step] if self.radius.ndim == 2 else self.radius
        scales = radius[index, None] * np.asarray(_MESHES[marker][1], dtype=np.float32)
        if self.thickness is not None:
            factor = (self.thickness[step] if self.thickness.ndim == 2 else self.thickness)[index, None]
            scales[:, : 3 if marker in ("sphere", "cube") else 2] *= factor  # markers without an axis grow whole
        return wxyzs, scales

    def _route(self, axial: np.ndarray) -> np.ndarray:
        """``(B, L, 3)`` 0/1 per bone ``B`` (``parents >= 0``): the components, in the parent joint's frame, of the
        bone's centre difference that leg ``k`` carries, from step 0 and ``axial (J,)`` (joints with an axis
        marker); ``L`` is the most legs any bone has (1 when every bone is straight)."""
        child, parent = self._child, self._parent
        inverse = self.rotations[0, parent] * _CONJUGATE
        centres = self._centres(0)
        lengths = np.abs(rotate(inverse, centres[child] - centres[parent]))  # (B, 3)
        child_axes = rotate(inverse, rotate(self.rotations[0, child], self.axes[child]))  # (B, 3) parent frame
        children = np.bincount(parent, minlength=len(self.parents))
        legs = np.zeros((len(child), 3, 3), dtype=np.float32)
        for bone, (size, p, c) in enumerate(zip(lengths, parent, child)):
            first = int(np.argmax(np.abs(self.axes[p]))) if axial[p] else None
            last = int(np.argmax(np.abs(child_axes[bone]))) if axial[c] else None
            if (first is None and last is None) or (np.linalg.norm(size) > _STRAIGHT and children[p] == 1):
                legs[bone, 0] = 1.0
                continue
            first = None if first == last else first
            middle = sorted({0, 1, 2} - {first, last}, key=lambda k: -size[k])
            order = [k for k in (first, *middle, last) if k is not None]
            kept = [k for k in order if size[k] >= _FOLD] or [max(order, key=lambda k: size[k])]
            longest = max(kept, key=lambda k: size[k])
            for k in order:
                legs[bone, kept.index(k if k in kept else longest), k] = 1.0
        return legs[:, : max(1, int(legs.any(axis=2).sum(axis=1).max(initial=1)))]

    def _bone_transforms(
        self, centres: np.ndarray, step: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Leg positions, wxyz and scales ``(B * L, 3 | 4 | 3)`` and corner positions ``(B * (L - 1), 3)`` and
        radii ``(B * (L - 1),)`` at ``step``; unused legs and corners have scale 0 (fixed instance counts)."""
        start = centres[self._parent]
        rotation = self.rotations[step, self._parent]
        local = rotate(rotation * _CONJUGATE, centres[self._child] - start)  # (B, 3) parent joint frame
        legs = rotate(rotation[:, None], self._legs * local[:, None])  # (B, L, 3) world
        ends = start[:, None] + np.cumsum(legs, axis=1)
        starts = np.concatenate([start[:, None], ends[:, :-1]], axis=1)
        length = np.linalg.norm(legs, axis=-1)  # (B, L)
        radius = np.broadcast_to(np.asarray(self.bone_radius, dtype=np.float32), self.parents.shape)[self._child]
        drawn = length > 1e-6  # a zero-length leg would draw a flat disc
        leg_radius = np.where(drawn, radius[:, None], 0.0)
        scales = np.stack([leg_radius, leg_radius, length], axis=-1).reshape(-1, 3)
        corner_radius = np.where(drawn[:, :-1] & drawn[:, 1:], radius[:, None] * _CORNER, 0.0).reshape(-1)
        positions = ((starts + ends) * 0.5).reshape(-1, 3)
        # A cylinder drawn either way round is the same: laid along each leg's largest component's positive side,
        # coaxial legs of two bones get the same facets, so where they overlap they do not stripe.
        directions = legs.reshape(-1, 3)
        directions = directions * np.sign(np.take_along_axis(directions, np.abs(directions).argmax(-1)[:, None], -1))
        return positions, quat_z_to(directions), scales, ends[:, :-1].reshape(-1, 3), corner_radius
