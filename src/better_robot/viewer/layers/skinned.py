"""``SkinnedMeshLayer``: linear blend skinning on the client; per step only bone poses are sent."""

from __future__ import annotations

from typing import Any, final, override

import numpy as np
import torch

from ...exceptions import ShapeError
from ...lie import so3
from ..layer import Bounds, Layer, as_frames, as_triangles
from ..theme import RGB, rgb_uint8

_IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


@final
class SkinnedMeshLayer(Layer):
    """Bind-pose ``vertices (V, 3)``, ``faces (F, 3)``, ``weights (V, J)`` and ``bone_poses (T, J, 7)``.

    ``bind_poses (J, 7)`` are the world bone poses the vertices are given in; a vertex follows
    ``sum_j w_ij * T_j(t) * T_j(bind)^-1``. viser keeps the four largest weights per vertex, so
    the offline renderer does the same. Poses are xyzw (BetterRobot order). ``bone_poses`` is read at every push; the
    rest arrays ``vertices``, ``bind_poses`` and ``weights`` are sent once, when the layer is built.
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
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ShapeError(f"vertices must be (V, 3) bind-pose positions, got {self.vertices.shape}")
        self.faces = as_triangles(faces)
        self.bind_poses = np.ascontiguousarray(bind_poses, dtype=np.float32)
        if self.bind_poses.ndim != 2 or self.bind_poses.shape[1] != 7:
            raise ShapeError(f"bind_poses must be (J, 7) world bone poses (xyzw), got {self.bind_poses.shape}")
        n_bones = self.bind_poses.shape[0]
        if self.bone_poses.shape[1:] != (n_bones, 7):
            raise ShapeError(
                f"bone_poses must be (T, J, 7) with the J = {n_bones} bones of bind_poses, got {self.bone_poses.shape}"
            )
        if np.shape(weights) != (len(self.vertices), n_bones):
            raise ShapeError(
                f"weights must be (V, J) = ({len(self.vertices)}, {n_bones}), one column per bone of bind_poses (for a "
                f"RobotVisual's Skin, per model joint, the universe included), got {np.shape(weights)}"
            )
        self.weights = top_k_weights(np.asarray(weights, dtype=np.float32), 4)
        # For bounds without skinning: per bone, the box in its bind frame of the vertices it carries at least a quarter
        # of (each vertex's main bone always does), turned with the bone at each step. A vertex is a blend of points in
        # these boxes, plus what its bones under a quarter add.
        main = np.minimum(0.25, self.weights.max(axis=1, keepdims=True))
        rows, bones = np.nonzero((self.weights > 0.0) & (self.weights >= main))
        local = so3.act(
            so3.inverse(torch.from_numpy(self.bind_poses[bones, 3:])),
            torch.from_numpy(self.vertices[rows] - self.bind_poses[bones, :3]),
        ).numpy()
        low, high = np.full((n_bones, 3), np.inf, np.float32), np.full((n_bones, 3), -np.inf, np.float32)
        np.minimum.at(low, bones, local)
        np.maximum.at(high, bones, local)
        self._carriers = np.unique(bones)  # (B,) the bones that carry a vertex
        low, high = low[self._carriers], high[self._carriers]
        self._boxes = (low + high) / 2.0, (high - low) / 2.0  # centre and half extent (B, 3)
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
                bone_wxyzs=bind[:, [6, 3, 4, 5]],  # viser's scalar-first quaternions
                bone_positions=bind[:, :3],
                skin_weights=np.pad(self.weights, ((0, 0), (0, pad))),
                color=rgb_uint8(self.color),
                **self._surface(),
            )
        )

    @override
    def set_step(self, step: int) -> None:
        poses = self.bone_poses[step]
        wxyz = poses[:, [6, 3, 4, 5]]
        for handle, rotation, pose in zip(self._mesh.bones, wxyz, poses):  # padded bones stay at the origin
            handle.wxyz = rotation
            handle.position = pose[:3]

    def set_opacity(self, opacity: float) -> None:
        """Draw the mesh at ``opacity`` in ``(0, 1]`` (1 is opaque), live and after a rebuild."""
        self._fade(opacity)

    @override
    def bounds(self, step: int) -> Bounds | None:
        """A box per bone that carries vertices ``(lo (B, 3), hi (B, 3))``: the box of the vertices it carries at
        least a quarter of, turned with the bone; no vertex is skinned. It holds every vertex blended from such bones;
        on SMPL and SMPL-X under random poses it held every vertex and reached 2 to 5 cm beyond the skinned mesh's box
        in the median, 16 cm at most."""
        poses = self.bone_poses[step, self._carriers]
        turn = so3.to_matrix(torch.from_numpy(poses[:, 3:])).numpy()  # (B, 3, 3)
        centre, half = self._boxes
        middle = poses[:, :3] + np.einsum("bij,bj->bi", turn, centre)
        reach = np.einsum("bij,bj->bi", np.abs(turn), half)  # the half extent of the turned box along the world axes
        return middle - reach, middle + reach

    def skin(self, step: int) -> np.ndarray:
        """Posed vertices ``(V, 3)`` of ``step`` by linear blend skinning (CPU, numpy)."""
        bind, posed = torch.from_numpy(self.bind_poses), torch.from_numpy(self.bone_poses[step])
        local = torch.from_numpy(self.vertices)[:, None] - bind[None, :, :3]  # (V, J, 3)
        local = so3.act(so3.inverse(bind[None, :, 3:]), local)
        world = (so3.act(posed[None, :, 3:], local) + posed[None, :, :3]).numpy()
        return np.einsum("vj,vjk->vk", self.weights, world).astype(np.float32)


def top_k_weights(weights: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` largest weights per row and renormalise rows to sum to one."""
    if weights.shape[1] <= k:
        return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    top = np.argpartition(weights, -k, axis=1)[:, -k:]
    kept = np.zeros_like(weights)
    np.put_along_axis(kept, top, np.take_along_axis(weights, top, axis=1), axis=1)
    return kept / np.maximum(kept.sum(axis=1, keepdims=True), 1e-12)
