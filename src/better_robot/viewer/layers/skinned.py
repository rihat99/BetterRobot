"""``SkinnedMeshLayer``: linear blend skinning on the client; per frame only bone poses are sent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .._math import RGB, as_frames, rgb_uint8, rotate, xyzw_to_wxyz
from ..layer import Geometry, Layer


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
        faces: np.ndarray,
        weights: np.ndarray,
        bind_poses: np.ndarray,
        bone_poses: np.ndarray,
        *,
        color: RGB = (0.85, 0.72, 0.62),
        opacity: float = 1.0,
    ) -> None:
        self.bone_poses = as_frames(bone_poses, 2)
        super().__init__(name, self.bone_poses.shape[0])
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.uint32)
        self.bind_poses = np.asarray(bind_poses, dtype=np.float32)
        n_bones = self.bind_poses.shape[0]
        if self.bone_poses.shape[1:] != (n_bones, 7) or np.shape(weights) != (len(self.vertices), n_bones):
            raise ValueError("bone_poses must be (T, J, 7) and weights (V, J) with J bind poses")
        self.weights = top_k_weights(np.asarray(weights, dtype=np.float32), 4)
        self.color = color
        self.opacity = opacity
        self._mesh: Any = None

    def build(self, scene: Any, root: str) -> None:
        self._mesh = self._track(
            scene.add_mesh_skinned(
                f"{root}/mesh",
                self.vertices,
                self.faces,
                bone_wxyzs=xyzw_to_wxyz(self.bind_poses[:, 3:]),
                bone_positions=self.bind_poses[:, :3],
                skin_weights=self.weights,
                color=rgb_uint8(self.color),
                opacity=None if self.opacity >= 1.0 else self.opacity,
            )
        )

    def set_frame(self, frame: int) -> None:
        poses = self.bone_poses[frame]
        wxyz = xyzw_to_wxyz(poses[:, 3:])
        for bone, handle in enumerate(self._mesh.bones):
            handle.wxyz = wxyz[bone]
            handle.position = poses[bone, :3]

    def geometry(self, frame: int) -> list[Geometry]:
        return [Geometry(self.skin(frame), self.faces.astype(np.int64), np.asarray(self.color, dtype=np.float32))]

    def skin(self, frame: int) -> np.ndarray:
        """Posed vertices ``(V, 3)`` of ``frame`` by linear blend skinning (CPU, numpy)."""
        bind, posed = self.bind_poses, self.bone_poses[frame]
        local = self.vertices[:, None] - bind[None, :, :3]  # (V, J, 3)
        conj = bind[:, 3:] * np.array([-1.0, -1.0, -1.0, 1.0], dtype=np.float32)
        local = rotate(conj[None], local)
        world = rotate(posed[None, :, 3:], local) + posed[None, :, :3]
        return np.einsum("vj,vjk->vk", self.weights, world).astype(np.float32)


def top_k_weights(weights: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` largest weights per row and renormalise rows to sum to one."""
    if weights.shape[1] <= k:
        return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    cutoff = np.partition(weights, -k, axis=1)[:, -k][:, None]
    kept = np.where(weights >= cutoff, weights, 0.0)
    return kept / np.maximum(kept.sum(axis=1, keepdims=True), 1e-12)


class CameraLayer(Layer):
    """Camera frustums at ``poses (T, 7)`` (camera-to-world, OpenCV ``+z`` forward), optional images.

    ``image`` is ``(H, W, 3)`` uint8, ``(T, H, W, 3)``, or a callable ``frame -> image`` for lazy
    loading; the image is shown inside the frustum.
    """

    def __init__(
        self,
        name: str,
        poses: np.ndarray,
        *,
        fov: float,
        aspect: float,
        scale: float = 0.3,
        color: RGB = (0.15, 0.15, 0.15),
        image: np.ndarray | Callable[[int], np.ndarray] | None = None,
    ) -> None:
        self.poses = as_frames(poses, 1)
        super().__init__(name, self.poses.shape[0])
        self.fov, self.aspect, self.scale, self.color = fov, aspect, scale, color
        self._image = image
        self._frustum: Any = None

    def image_at(self, frame: int) -> np.ndarray | None:
        if self._image is None:
            return None
        if callable(self._image):
            return self._image(frame)
        return self._image if self._image.ndim == 3 else self._image[frame]

    def build(self, scene: Any, root: str) -> None:
        self._frustum = self._track(
            scene.add_camera_frustum(
                f"{root}/frustum",
                fov=self.fov,
                aspect=self.aspect,
                scale=self.scale,
                color=rgb_uint8(self.color),
                image=self.image_at(0),
                wxyz=xyzw_to_wxyz(self.poses[0, 3:]),
                position=self.poses[0, :3],
            )
        )

    def set_frame(self, frame: int) -> None:
        self._frustum.position = self.poses[frame, :3]
        self._frustum.wxyz = xyzw_to_wxyz(self.poses[frame, 3:])
        if self._image is not None and (callable(self._image) or self._image.ndim == 4):
            self._frustum.image = self.image_at(frame)
