"""``SkinnedMeshLayer``: linear blend skinning on the client; per step only bone poses are sent."""

from __future__ import annotations

from typing import Any, final, override

import numpy as np

from ...exceptions import ShapeError
from .._lie import act, inverse, to_wxyz
from ..layer import as_frames, as_triangles
from ..theme import RGB, rgb_uint8
from ..layer import Layer
from .camera import CameraLayer

__all__ = ["CameraLayer", "SkinnedMeshLayer"]

_IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


@final
class SkinnedMeshLayer(Layer):
    """Bind-pose ``vertices (V, 3)``, ``faces (F, 3)``, ``weights (V, J)`` and ``bone_poses (T, J, 7)``.

    ``bind_poses (J, 7)`` are the world bone poses the vertices are given in; a vertex follows
    ``sum_j w_ij * T_j(t) * T_j(bind)^-1``. viser keeps the four largest weights per vertex, so
    the offline renderer does the same. Poses are xyzw (BetterRobot order).
    """

    def __init__(
        self,
        name: str,
        vertices: np.ndarray,
        *,
        faces: np.ndarray,
        weights: np.ndarray,
        bind_poses: np.ndarray,
        bone_poses: np.ndarray,
        color: RGB = (0.85, 0.72, 0.62),
        opacity: float = 1.0,
    ) -> None:
        self.bone_poses = as_frames(bone_poses, 2, name="bone_poses", width=7)
        super().__init__(name, self.bone_poses.shape[0])
        self.vertices = np.asarray(vertices, dtype=np.float32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ShapeError(f"vertices must be (V, 3) bind-pose positions, got {self.vertices.shape}")
        self.faces = as_triangles(faces)
        self.bind_poses = np.asarray(bind_poses, dtype=np.float32)
        if self.bind_poses.ndim != 2 or self.bind_poses.shape[1] != 7:
            raise ShapeError(f"bind_poses must be (J, 7) world bone poses (xyzw), got {self.bind_poses.shape}")
        n_bones = self.bind_poses.shape[0]
        if self.bone_poses.shape[1:] != (n_bones, 7) or np.shape(weights) != (len(self.vertices), n_bones):
            raise ShapeError("bone_poses must be (T, J, 7) and weights (V, J) with J bind poses")
        self.weights = top_k_weights(np.asarray(weights, dtype=np.float32), 4)
        self.color = color
        self._store_opacity(opacity)
        self._mesh: Any = None

    @override
    def build(self, scene: Any, root: str) -> None:
        # viser takes the four largest weights per vertex and so needs four bones: fewer get unweighted bones at the
        # origin, which never move.
        pad = max(0, 4 - len(self.bind_poses))
        bind = np.concatenate([self.bind_poses, np.tile(_IDENTITY_POSE, (pad, 1))])
        self._mesh = self._track(
            scene.add_mesh_skinned(
                f"{root}/mesh",
                self.vertices,
                self.faces,
                bone_wxyzs=to_wxyz(bind[:, 3:]),
                bone_positions=bind[:, :3],
                skin_weights=np.pad(self.weights, ((0, 0), (0, pad))),
                color=rgb_uint8(self.color),
                opacity=None if self.opacity >= 1.0 else self.opacity,
            )
        )

    @override
    def set_step(self, step: int) -> None:
        poses = self.bone_poses[step]
        wxyz = to_wxyz(poses[:, 3:])
        for handle, rotation, pose in zip(self._mesh.bones, wxyz, poses):  # padded bones stay at the origin
            handle.wxyz = rotation
            handle.position = pose[:3]

    def set_opacity(self, opacity: float) -> None:
        """Draw the mesh at ``opacity`` in ``(0, 1]`` (1 is opaque), live and after a rebuild."""
        self._fade(opacity)

    def world_points(self, step: int) -> np.ndarray | None:
        """World positions ``(N, 3)`` drawn at ``step``, for camera fitting."""
        return self.skin(step)

    def skin(self, step: int) -> np.ndarray:
        """Posed vertices ``(V, 3)`` of ``step`` by linear blend skinning (CPU, numpy)."""
        bind, posed = self.bind_poses, self.bone_poses[step]
        local = self.vertices[:, None] - bind[None, :, :3]  # (V, J, 3)
        local = act(inverse(bind[None, :, 3:]), local)
        world = act(posed[None, :, 3:], local) + posed[None, :, :3]
        return np.einsum("vj,vjk->vk", self.weights, world).astype(np.float32)


def top_k_weights(weights: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` largest weights per row and renormalise rows to sum to one."""
    if weights.shape[1] <= k:
        return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    top = np.argpartition(weights, -k, axis=1)[:, -k:]
    kept = np.zeros_like(weights)
    np.put_along_axis(kept, top, np.take_along_axis(weights, top, axis=1), axis=1)
    return kept / np.maximum(kept.sum(axis=1, keepdims=True), 1e-12)
