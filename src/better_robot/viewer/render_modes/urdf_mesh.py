"""``render_modes/urdf_mesh.py`` — URDF visual mesh render mode.

``trimesh`` is imported lazily here and ONLY here.

See ``docs/concepts/viewer.md §4.2`` and ``§17``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import RenderContext
from ..helpers import quat_xyzw_to_wxyz

if TYPE_CHECKING:
    from ...data_model.data import Data
    from ...data_model.model import Model
    from ...io.assets import AssetResolver
    from ...io.ir import IRGeom


def _resolve_mesh_path(path: str, resolver: "AssetResolver | None") -> str:
    """Resolve a URDF mesh URI/path through the active ``AssetResolver``.

    When ``resolver`` is ``None`` the raw path is returned unchanged
    (matches the legacy behaviour where ``yourdfpy._filename_handler``
    pre-resolves paths at parse time). When the resolver fails, the raw
    path is returned as a fallback so URDFs with already-absolute paths
    still load.
    """
    if resolver is None or not path:
        return path
    try:
        return str(resolver.resolve(path))
    except (FileNotFoundError, OSError):
        return path


def _load_geom(
    geom: "IRGeom",
    *,
    resolver: "AssetResolver | None" = None,
) -> tuple[Any | None, tuple]:
    """Return (trimesh_mesh_or_None, rgba) for a single IRGeom.

    For ``kind="mesh"`` the original trimesh object is returned so that
    the backend can render it with its embedded materials/colors.
    For analytical primitives the trimesh is created from scratch.

    ``resolver`` (optional) maps the parser-emitted mesh URI to a
    concrete filesystem path. When ``None``, the raw ``path`` from the
    IR is used directly — that is the path the URDF parser already
    resolved at load time via ``yourdfpy._filename_handler``. Passing a
    resolver lets the viewer find meshes that were referenced by URI
    (e.g. ``package://``) even when the URDF was parsed without a
    filesystem root.
    """
    import trimesh

    rgba = geom.rgba if geom.rgba is not None else (0.7, 0.7, 0.7, 1.0)
    kind = geom.kind

    if kind == "mesh":
        path = geom.params.get("path", "")
        scale = geom.params.get("scale", [1.0, 1.0, 1.0])
        if not path:
            return None, rgba
        resolved = _resolve_mesh_path(path, resolver)
        try:
            mesh = trimesh.load(resolved, force="mesh")
            mesh.apply_scale(scale)
            return mesh, rgba
        except Exception:
            return None, rgba

    elif kind == "box":
        sx, sy, sz = geom.params["size"]
        return trimesh.creation.box(extents=[sx, sy, sz]), rgba

    elif kind == "cylinder":
        r = geom.params["radius"]
        h = geom.params["length"]
        return trimesh.creation.cylinder(radius=r, height=h), rgba

    elif kind == "sphere":
        r = geom.params["radius"]
        return trimesh.creation.icosphere(subdivisions=2, radius=r), rgba

    elif kind == "capsule":
        r = geom.params["radius"]
        h = geom.params["length"]
        return trimesh.creation.capsule(radius=r, height=h), rgba

    return None, rgba


class URDFMeshMode:
    """Render the URDF visual meshes attached to each body.

    Pulls ``IRBody.visual_geoms`` off the ``io.IRModel`` stored at
    ``model.meta["ir"]`` (back-reference set by ``build_model``).

    Visual primitives (``<box>``, ``<cylinder>``, ``<sphere>``,
    ``<capsule>``) are tessellated via trimesh.  Mesh files (``.obj``,
    ``.dae``, ``.stl``) are loaded with trimesh too.  Non-loadable meshes
    are skipped with a warning.

    Asset resolution. Mesh URIs in ``IRGeom.params["path"]`` are routed
    through ``model.meta["asset_resolver"]`` when present (set by the
    URDF parser per ``docs/concepts/parsers_and_ir.md §6``). The constructor
    also accepts an explicit ``resolver=`` override — useful when the
    caller wants a different search root than the parse-time one. When
    no resolver is available the raw path is used directly, preserving
    the legacy yourdfpy ``_filename_handler`` flow.
    """

    name = "URDF mesh"
    description = "Visual meshes from the URDF / MJCF"

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        resolver: "AssetResolver | None" = None,
    ) -> None:
        self.alpha = alpha
        self._explicit_resolver = resolver
        self._ctx: RenderContext | None = None
        self._model: Model | None = None
        # Maps node name → (joint_index, geom_local_pose_7vec)
        self._nodes: dict[str, tuple[int, torch.Tensor]] = {}

    @classmethod
    def is_available(cls, model: "Model", data: "Data") -> bool:
        meta = getattr(model, "meta", {}) or {}
        ir = meta.get("ir")
        if ir is None:
            return False
        return any(
            len(getattr(b, "visual_geoms", [])) > 0
            for b in getattr(ir, "bodies", [])
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self, context: RenderContext, model: "Model", data: "Data") -> None:
        self._ctx = context
        self._model = model
        backend = context.backend
        ns = context.namespace
        b_idx = context.batch_index
        ir = model.meta["ir"]
        # Constructor override wins; otherwise pull the parse-time
        # resolver off ``model.meta`` (set by parse_urdf / parse_mjcf).
        resolver = self._explicit_resolver
        if resolver is None:
            resolver = (model.meta or {}).get("asset_resolver")

        # Build a map: body_name → model joint index (via body_names)
        body_to_mjidx: dict[str, int] = {
            name: i for i, name in enumerate(model.body_names)
        }

        has_trimesh_backend = hasattr(backend, "add_mesh_trimesh")

        for ir_body in ir.bodies:
            mjidx = body_to_mjidx.get(ir_body.name)
            if mjidx is None:
                continue
            for geom_idx, geom in enumerate(ir_body.visual_geoms):
                mesh, rgba = _load_geom(geom, resolver=resolver)
                if mesh is None:
                    continue

                node_name = f"{ns}/body_{mjidx}/geom_{geom_idx}"

                # Use add_mesh_trimesh when available — it preserves
                # embedded materials (DAE/OBJ). For meshes without
                # embedded colors (STL), apply the URDF material color
                # as vertex colors so viser picks it up.
                if has_trimesh_backend and geom.kind == "mesh":
                    # If the mesh has no meaningful color info (e.g. STL)
                    # and the URDF specifies a material, paint it.
                    if mesh.visual.kind is None or (
                        mesh.visual.kind == "vertex"
                        and len(set(map(tuple, mesh.visual.vertex_colors))) <= 1
                    ):
                        import numpy as np
                        c = [int(x * 255) for x in rgba[:3]]
                        a = int(rgba[3] * self.alpha * 255)
                        mesh.visual.vertex_colors = np.full(
                            (len(mesh.vertices), 4),
                            [c[0], c[1], c[2], a],
                            dtype=np.uint8,
                        )
                    backend.add_mesh_trimesh(node_name, mesh)
                else:
                    import numpy as np
                    verts = torch.from_numpy(
                        np.array(mesh.vertices, dtype=np.float32))
                    faces = torch.from_numpy(
                        np.array(mesh.faces, dtype=np.int32))
                    rgba = (rgba[0], rgba[1], rgba[2],
                            rgba[3] * self.alpha)
                    backend.add_mesh(node_name, verts, faces, rgba=rgba)

                self._nodes[node_name] = (mjidx, geom.origin)

        # Set initial transforms
        if data.joint_pose_world is not None:
            self._set_transforms(data, b_idx)

    def update(self, data: "Data") -> None:
        if self._ctx is None or data.joint_pose_world is None:
            return
        self._set_transforms(data, self._ctx.batch_index)

    def set_visible(self, visible: bool) -> None:
        if self._ctx is None:
            return
        backend = self._ctx.backend
        for name in self._nodes:
            backend.set_visible(name, visible)

    def detach(self) -> None:
        if self._ctx is None:
            return
        backend = self._ctx.backend
        for name in self._nodes:
            backend.remove(name)
        self._nodes.clear()
        self._ctx = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_transforms(self, data: "Data", b: int) -> None:
        backend = self._ctx.backend
        joint_pose_world = data.joint_pose_world  # (B..., njoints, 7) or (njoints, 7)
        for name, (mjidx, local_pose) in self._nodes.items():
            # Get body world pose
            if joint_pose_world.dim() == 2:
                body_world = joint_pose_world[mjidx]
            else:
                body_world = joint_pose_world[b, mjidx]
            # Compose with local geom offset
            from ...lie.se3 import compose
            world_pose = compose(body_world, local_pose.to(body_world.device))
            backend.set_transform(name, world_pose)
