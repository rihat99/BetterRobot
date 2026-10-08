"""Mesh layers: one deforming/static mesh, rigid meshes following body poses, batched spheres."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._math import RGB, as_frames, compose_poses, rgb_uint8, rotate, unit_sphere, xyzw_to_wxyz
from ..layer import Layer

_IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class MeshLayer(Layer):
    """One triangle mesh: static ``vertices (V, 3)`` or deforming ``(T, V, 3)``, ``faces (F, 3)``."""

    def __init__(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        color: RGB = (0.80, 0.80, 0.80),
        opacity: float = 1.0,
        flat_shading: bool = False,
    ) -> None:
        self.vertices = as_frames(vertices, 2)
        super().__init__(name, self.vertices.shape[0])
        self.faces = np.ascontiguousarray(np.asarray(faces, dtype=np.uint32))
        self.color = color
        self.opacity = opacity
        self.flat_shading = flat_shading
        self._mesh: Any = None

    def build(self, scene: Any, root: str) -> None:
        self._mesh = self._track(
            scene.add_mesh_simple(
                f"{root}/mesh",
                self.vertices[0],
                self.faces,
                color=rgb_uint8(self.color),
                opacity=None if self.opacity >= 1.0 else self.opacity,
                flat_shading=self.flat_shading,
            )
        )

    def world_points(self, step: int) -> np.ndarray | None:
        """World positions ``(N, 3)`` drawn at ``step``, for camera fitting."""
        return self.vertices[step]

    def set_step(self, step: int) -> None:
        self._mesh.vertices = self.vertices[step]


@dataclass
class MeshSpec:
    """One rigid mesh attached to ``body`` at ``origin`` (``(7,)`` xyzw pose in the body step).

    ``trimesh`` keeps a textured/vertex-coloured source for ``add_mesh_trimesh``; otherwise the
    mesh is sent as ``vertices``/``faces`` with one ``color``.
    """

    vertices: np.ndarray
    faces: np.ndarray
    body: int
    origin: np.ndarray
    color: RGB = (0.80, 0.80, 0.80)
    trimesh: Any = None


class RigidMeshLayer(Layer):
    """Rigid meshes following ``body_poses (T, B, 7)``: one node per mesh, per step only poses move.

    Arrays are read at ``set_step`` time, so editing ``body_poses`` in place and calling
    ``viewer.show()`` updates the scene (live tools).
    """

    def __init__(self, name: str, meshes: list[MeshSpec], body_poses: np.ndarray, *, opacity: float = 1.0) -> None:
        self.body_poses = as_frames(body_poses, 2)
        super().__init__(name, self.body_poses.shape[0])
        self.meshes = meshes
        self.opacity = opacity
        self._bodies = np.array([spec.body for spec in meshes], dtype=np.int64)
        self._origins = (
            np.stack([spec.origin for spec in meshes]).astype(np.float32) if meshes else np.zeros((0, 7), np.float32)
        )
        self._nodes: list[Any] = []

    def build(self, scene: Any, root: str) -> None:
        for index, spec in enumerate(self.meshes):
            name = f"{root}/mesh_{index}"
            if spec.trimesh is not None:
                node = scene.add_mesh_trimesh(name, spec.trimesh)
            else:
                node = scene.add_mesh_simple(
                    name,
                    spec.vertices,
                    spec.faces,
                    color=rgb_uint8(spec.color),
                    opacity=None if self.opacity >= 1.0 else self.opacity,
                )
            self._nodes.append(self._track(node))

    def world_points(self, step: int) -> np.ndarray | None:
        """World positions ``(N, 3)`` drawn at ``step``, for camera fitting."""
        if not self.meshes:
            return None
        world = compose_poses(self.body_poses[step, self._bodies], self._origins)
        return np.concatenate(
            [rotate(world[i, 3:], spec.vertices) + world[i, :3] for i, spec in enumerate(self.meshes)]
        )

    def set_step(self, step: int) -> None:
        world = compose_poses(self.body_poses[step, self._bodies], self._origins)
        wxyz = xyzw_to_wxyz(world[:, 3:])
        for index, node in enumerate(self._nodes):
            node.position = world[index, :3]
            node.wxyz = wxyz[index]


class SpheresLayer(Layer):
    """Spheres with ``centers (T, S, 3)`` and ``radii (S,)`` as one batched node (collision models, contacts).

    ``colors`` is one RGB, ``(S, 3)``, or ``(T, S, 3)`` per step (a contact turning red or green).
    """

    def __init__(
        self,
        name: str,
        centers: np.ndarray,
        radii: np.ndarray,
        *,
        colors: np.ndarray | RGB = (0.95, 0.45, 0.20),
        opacity: float = 0.5,
    ) -> None:
        self.centers = as_frames(centers, 2)
        super().__init__(name, self.centers.shape[0])
        self.radii = np.asarray(radii, dtype=np.float32)
        if self.radii.shape != (self.centers.shape[1],):
            raise ValueError("radii must have one entry per sphere")
        colors = np.asarray(colors, dtype=np.float32)
        self._colors = rgb_uint8(colors, (self.centers.shape[1],)) if colors.ndim <= 2 else rgb_uint8(colors)
        if self._colors.ndim == 3 and self._colors.shape != self.centers.shape:
            raise ValueError(f"per-step colors must be (T, S, 3) = {self.centers.shape}, got {self._colors.shape}")
        self.opacity = opacity
        self._elements = self.centers.shape[1]
        self._spheres: Any = None

    def build(self, scene: Any, root: str) -> None:
        vertices, faces = unit_sphere()
        n = self.centers.shape[1]
        self._spheres = self._track(
            scene.add_batched_meshes_simple(
                f"{root}/spheres",
                vertices,
                faces,
                batched_wxyzs=np.tile(xyzw_to_wxyz(_IDENTITY_POSE[3:]), (n, 1)),
                batched_positions=self.centers[0],
                batched_scales=self.radii,
                batched_colors=self._colors if self._colors.ndim == 2 else self._colors[0],
                opacity=None if self.opacity >= 1.0 else self.opacity,
                lod="off",
            )
        )

    def world_points(self, step: int) -> np.ndarray | None:
        """World positions ``(N, 3)`` drawn at ``step``, for camera fitting."""
        centers, radii = self.centers[step], self.radii[:, None]
        if self.valid is not None and self.valid.ndim == 2:
            centers, radii = centers[self.valid[step]], radii[self.valid[step]]
        return np.concatenate([centers - radii, centers + radii])

    def set_step(self, step: int) -> None:
        self._spheres.batched_positions = self.centers[step]
        if self._colors.ndim == 3:
            self._spheres.batched_colors = self._colors[step]
        if self.valid is not None and self.valid.ndim == 2:
            self._spheres.batched_scales = self.radii * self.valid[step]  # radius 0 hides an invalid sphere
