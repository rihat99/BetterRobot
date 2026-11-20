"""Mesh layers: one deforming/static mesh, rigid meshes following body poses, batched spheres."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Any, final, override

import numpy as np
import torch

from ...exceptions import ArgumentError, ShapeError
from ...lie import se3, so3
from ..layer import Bounds, Layer, as_frames, as_triangles, box
from ..theme import RGB, linear_uint8, rgb_uint8

#: The most colours a deforming ``MeshLayer`` takes: one scene node per colour.
_MAX_COLOURS = 64


@final
class MeshLayer(Layer):
    """One triangle mesh: static ``vertices (V, 3)`` or deforming ``(T, V, 3)``, ``faces (F, 3)``.

    ``colors (V, 3)`` colours each vertex instead of the one ``color``, the same at every step. A static mesh sends
    them in one GLB node, blended across each face and always opaque (sent again by a push after its ``vertices``
    changed); a deforming one becomes one node per distinct colour, at most 64, each face taking the colour of its
    first vertex.
    """

    def __init__(
        self,
        name: str,
        vertices: np.ndarray,
        *,
        faces: np.ndarray,
        color: RGB = (0.80, 0.80, 0.80),
        colors: np.ndarray | None = None,
        opacity: float = 1.0,
        flat_shading: bool = False,
    ) -> None:
        self.vertices = as_frames(vertices, 2, name="vertices", width=3)
        super().__init__(name, self.vertices.shape[0])
        self.faces = as_triangles(faces, vertices=self.vertices.shape[1])
        self.color = color
        self.colors = None if colors is None else rgb_uint8(colors, (self.vertices.shape[1],))
        self._store_opacity(opacity)
        self.flat_shading = flat_shading
        # (colour, vertex rows, faces into those rows) of each node; none for a static coloured mesh (one GLB node).
        self._parts: list[tuple[np.ndarray, np.ndarray | slice, np.ndarray]] = []
        if self.colors is None:
            self._parts = [(rgb_uint8(color), slice(None), self.faces)]
        elif self.nsteps > 1:
            self._parts = _split_by_colour(self.faces, self.colors)
        self._nodes: list[tuple[Any, np.ndarray | slice]] = []
        self._glb: tuple[Any, np.ndarray] | None = None  # a static coloured mesh: its node and the vertices it holds

    @override
    def build(self, scene: Any, root: str) -> None:
        self._nodes = []
        if self.colors is not None and self.nsteps == 1:
            self._glb = self._track(scene.add_glb(f"{root}/mesh", self._export())), self.vertices[0].copy()
            return
        for index, (colour, rows, faces) in enumerate(self._parts):
            node = scene.add_mesh_simple(
                f"{root}/mesh" if self.colors is None else f"{root}/mesh_{index}",
                self.vertices[0][rows],
                faces,
                color=colour,
                **self._surface(),
                flat_shading=self.flat_shading,
            )
            self._nodes.append((self._track(node), rows))

    @override
    def bounds(self, step: int) -> Bounds | None:
        return box(self.vertices[step])

    @override
    def set_step(self, step: int) -> None:
        for node, rows in self._nodes:
            node.vertices = self.vertices[step][rows]
        # A static mesh is pushed only by redraw, show() and the like: the GLB goes again only after an edit.
        if self._glb is not None and not np.array_equal(self._glb[1], self.vertices[0]):
            self._glb[0].glb_data = self._export()
            self._glb = self._glb[0], self.vertices[0].copy()

    def set_opacity(self, opacity: float) -> None:
        """Draw the mesh at ``opacity`` in ``(0, 1]`` (1 is opaque), live and after a rebuild; a static mesh with
        ``colors`` stays opaque."""
        self._fade(opacity, [node for node, _ in self._nodes])

    def _export(self) -> bytes:
        """The static coloured mesh as a GLB."""
        from trimesh import Trimesh  # noqa: PLC0415 - viser's own dependency, loaded for a coloured static mesh
        from trimesh.visual import TextureVisuals  # noqa: PLC0415
        from trimesh.visual.material import PBRMaterial  # noqa: PLC0415

        # Without a material trimesh sends its empty textured one, drawn black, and glTF's default is a metal, drawn
        # darker than a plain mesh: give the matte one of the plain meshes. glTF's vertex colours are linear, the
        # plain meshes' sRGB; without normals the mesh is drawn faceted.
        assert self.colors is not None
        visual = TextureVisuals(material=PBRMaterial(metallicFactor=0.0, roughnessFactor=1.0))
        linear = linear_uint8(self.colors / 255.0)
        visual.vertex_attributes["color"] = np.concatenate([linear, np.full_like(linear[:, :1], 255)], 1)
        mesh = Trimesh(self.vertices[0], self.faces, visual=visual, process=False)
        glb = mesh.export(file_type="glb", include_normals=not self.flat_shading)
        assert isinstance(glb, bytes)
        return glb


def _split_by_colour(faces: np.ndarray, colors: np.ndarray) -> list[tuple[np.ndarray, np.ndarray | slice, np.ndarray]]:
    """``faces (F, 3)`` grouped by the colour of their first vertex (``colors (V, 3)`` uint8): per colour, the colour,
    the vertex rows its faces use and those faces indexing the rows."""
    palette, which = np.unique(colors[faces[:, 0]], axis=0, return_inverse=True)
    if len(palette) > _MAX_COLOURS:
        raise ArgumentError(
            f"a deforming MeshLayer takes at most {_MAX_COLOURS} distinct colors (one node each), got {len(palette)}; "
            "quantise the colours or show a static mesh"
        )
    parts: list[tuple[np.ndarray, np.ndarray | slice, np.ndarray]] = []
    for index, colour in enumerate(palette):
        rows, local = np.unique(faces[which.reshape(-1) == index], return_inverse=True)
        parts.append((colour, rows, local.reshape(-1, 3).astype(np.uint32)))
    return parts


@final
@dataclass(frozen=True)
class MeshSpec:
    """One rigid mesh attached to ``body`` at ``origin`` (``(7,)`` xyzw pose in the body frame).

    ``trimesh`` keeps a source with its own colours or texture, sent as a GLB; otherwise the mesh is sent as
    ``vertices``/``faces`` with one ``color``.
    """

    vertices: np.ndarray
    faces: np.ndarray
    _: KW_ONLY
    body: int
    origin: np.ndarray
    color: RGB = (0.80, 0.80, 0.80)
    trimesh: Any = None  # a trimesh.Trimesh; trimesh is imported in robot.py only

    @override
    def __repr__(self) -> str:
        return f"MeshSpec(V={len(self.vertices)}, F={len(self.faces)}, body={self.body})"


@final
class RigidMeshLayer(Layer):
    """Rigid meshes following ``body_poses (T, B, 7)``: one node per mesh, per step only poses move.

    Arrays are read at ``set_step`` time, so editing ``body_poses`` in place and calling
    ``viewer.show()`` updates the scene (live tools).
    """

    def __init__(self, name: str, meshes: list[MeshSpec], *, body_poses: np.ndarray, opacity: float = 1.0) -> None:
        self.body_poses = as_frames(body_poses, 2, name="body_poses", width=7)
        super().__init__(name, self.body_poses.shape[0])
        n_bodies = self.body_poses.shape[1]
        outside = [spec.body for spec in meshes if not 0 <= spec.body < n_bodies]
        if outside:
            raise ArgumentError(
                f"MeshSpec.body {outside} out of range for body_poses with {n_bodies} bodies; pass every body's pose "
                "(KinematicState.body_poses)"
            )
        for spec in meshes:  # checked here, sent to viser as given at build
            as_triangles(spec.faces, vertices=len(spec.vertices))
        self.meshes = meshes
        self._store_opacity(opacity)
        self._bodies = np.array([spec.body for spec in meshes], dtype=np.int64)
        self._origins = (
            np.stack([spec.origin for spec in meshes]).astype(np.float32) if meshes else np.zeros((0, 7), np.float32)
        )
        # (M, 8, 3): the corners of each mesh's box in its body frame, what ``bounds`` moves per step.
        lows = [np.asarray(spec.vertices, np.float32).reshape(-1, 3).min(axis=0) for spec in meshes]
        highs = [np.asarray(spec.vertices, np.float32).reshape(-1, 3).max(axis=0) for spec in meshes]
        self._corners = np.array(
            [np.stack(np.meshgrid(*zip(low, high)), -1).reshape(8, 3) for low, high in zip(lows, highs)], np.float32
        ).reshape(-1, 8, 3)
        self._nodes: list[Any] = []
        self._plain: list[Any] = []  # the nodes of meshes without their own colours, which take an opacity
        self._own: list[tuple[Any, Any]] = []  # (node, trimesh) of each mesh with its own colours: a GLB

    @override
    def build(self, scene: Any, root: str) -> None:
        self._nodes, self._plain, self._own = [], [], []
        for index, spec in enumerate(self.meshes):
            name = f"{root}/mesh_{index}"
            if spec.trimesh is not None:
                shadows = {key: value for key, value in self._surface().items() if key != "opacity"}
                node = scene.add_glb(name, glb_data=_glb(spec.trimesh, self.opacity), **shadows)
                self._own.append((node, spec.trimesh))
            else:
                node = scene.add_mesh_simple(
                    name,
                    spec.vertices,
                    spec.faces,
                    color=rgb_uint8(spec.color),
                    **self._surface(),
                )
                self._plain.append(node)
            self._nodes.append(self._track(node))

    @override
    def bounds(self, step: int) -> Bounds | None:
        if not self.meshes:
            return None
        # The corners of each mesh's own box stand in for its vertices: a clip of 300 k vertices frames in a millisecond.
        world = self._world(step)
        corners = so3.act(torch.from_numpy(world[:, None, 3:]), torch.from_numpy(self._corners)).numpy()
        return box(corners + world[:, None, :3])

    @override
    def set_step(self, step: int) -> None:
        world = self._world(step)
        wxyz = world[:, [6, 3, 4, 5]]  # viser's scalar-first quaternions
        for index, node in enumerate(self._nodes):
            node.position = world[index, :3]
            node.wxyz = wxyz[index]

    def _world(self, step: int) -> np.ndarray:
        """World poses ``(M, 7)`` of the meshes at ``step``: each body's pose composed with the mesh's origin."""
        return se3.compose(
            torch.from_numpy(self.body_poses[step, self._bodies]), torch.from_numpy(self._origins)
        ).numpy()

    def set_opacity(self, opacity: float) -> None:
        """Draw the meshes at ``opacity`` in ``(0, 1]`` (1 is opaque), live and after a rebuild; a mesh with its own
        colours or texture (``MeshSpec.trimesh``) keeps them, at that opacity (its GLB is sent again)."""
        self._fade(opacity, self._plain)
        for node, mesh in self._own:
            node.glb_data = _glb(mesh, opacity)
            node.cast_shadow = node.receive_shadow = opacity >= 1.0


def _glb(mesh: Any, opacity: float) -> bytes:
    """``mesh``, a ``trimesh.Trimesh`` with its own colours or texture, as a GLB drawn at ``opacity``.

    Below 1 its material's alpha is scaled and blended, so the colours and the texture stay; vertex or face colours
    take glTF's default material, the one they are drawn with at 1, made see-through.
    """
    if opacity < 1.0:
        from trimesh.visual import TextureVisuals  # noqa: PLC0415 - viser's own dependency, loaded for a ghost
        from trimesh.visual.material import PBRMaterial  # noqa: PLC0415

        mesh = mesh.copy()  # the source keeps its material for the next opacity
        if mesh.visual.kind == "texture":
            material = mesh.visual.material
            if not isinstance(material, PBRMaterial):
                material = mesh.visual.material = material.to_pbr()
        else:
            colors, material = mesh.visual.vertex_colors, PBRMaterial(metallicFactor=1.0, roughnessFactor=1.0)
            mesh.visual = TextureVisuals(material=material)
            mesh.visual.vertex_attributes["color"] = colors
        factor = material.baseColorFactor if material.baseColorFactor is not None else np.full(4, 255, np.uint8)
        material.baseColorFactor = [*factor[:3], round(float(factor[3]) * opacity)]
        material.alphaMode = "BLEND"
    glb = mesh.export(file_type="glb")
    assert isinstance(glb, bytes)
    return glb


@final
class SpheresLayer(Layer):
    """Spheres with ``centers (T, S, 3)`` and ``radii (S,)`` as one batched node (collision models, contacts).

    ``colors`` is one RGB, ``(S, 3)``, or ``(T, S, 3)`` per step (a contact turning red or green).
    """

    def __init__(
        self,
        name: str,
        centers: np.ndarray,
        *,
        radii: np.ndarray,
        colors: np.ndarray | RGB = (0.95, 0.45, 0.20),
        opacity: float = 0.5,
    ) -> None:
        self.centers = as_frames(centers, 2, name="centers", width=3)
        super().__init__(name, self.centers.shape[0])
        self.radii = np.ascontiguousarray(radii, dtype=np.float32)
        if self.radii.shape != (self.centers.shape[1],):
            raise ShapeError(f"radii must be (S,) = ({self.centers.shape[1]},), one per sphere, got {self.radii.shape}")
        colors = np.asarray(colors, dtype=np.float32)
        self._colors = linear_uint8(colors, (self.centers.shape[1],)) if colors.ndim <= 2 else linear_uint8(colors)
        if self._colors.ndim == 3 and self._colors.shape != self.centers.shape:
            raise ShapeError(f"per-step colors must be (T, S, 3) = {self.centers.shape}, got {self._colors.shape}")
        self._store_opacity(opacity)
        self._elements = self.centers.shape[1]
        self._spheres: Any = None
        self._scales = self.radii.copy()  # the radii the node holds: a push writes them only when they change

    @override
    def build(self, scene: Any, root: str) -> None:
        import trimesh  # noqa: PLC0415 - viser's own dependency, loaded where a unit sphere is needed

        sphere = trimesh.creation.icosphere(subdivisions=2)
        n = self.centers.shape[1]
        self._spheres = self._track(
            scene.add_batched_meshes_simple(
                f"{root}/spheres",
                sphere.vertices.astype(np.float32),
                sphere.faces.astype(np.uint32),
                batched_wxyzs=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
                batched_positions=self.centers[0],
                batched_scales=self._scales,
                batched_colors=self._colors if self._colors.ndim == 2 else self._colors[0],
                **self._surface(),
                lod="off",
            )
        )

    @override
    def bounds(self, step: int) -> Bounds | None:
        centers, radii = self.centers[step], self.radii[:, None]
        if self.valid is not None and self.valid.ndim == 2:
            centers, radii = centers[self.valid[step]], radii[self.valid[step]]
        return box(np.concatenate([centers - radii, centers + radii]))

    @override
    def set_step(self, step: int) -> None:
        self._spheres.batched_positions = self.centers[step]
        if self._colors.ndim == 3:
            self._spheres.batched_colors = self._colors[step]
        # Radius 0 hides an invalid sphere.
        scales = self.radii * self.valid[step] if self.valid is not None and self.valid.ndim == 2 else self.radii
        if not np.array_equal(scales, self._scales):  # an edit of radii, or another mask
            self._spheres.batched_scales = self._scales = scales.copy()

    def set_opacity(self, opacity: float) -> None:
        """Draw the spheres at ``opacity`` in ``(0, 1]`` (1 is opaque), live and after a rebuild."""
        self._fade(opacity)
