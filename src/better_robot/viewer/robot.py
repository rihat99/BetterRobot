"""Robot adapter: a ``Model`` and a configuration clip ``q (T, nq)`` become viewer layers.

The only viewer file that knows ``Model``. Forward kinematics runs once for the whole clip; the
layers receive numpy arrays and never touch the model again.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ._math import colormap
from .layer import Layer
from .layers.frames import FramesLayer
from .layers.mesh import MeshSpec, RigidMeshLayer, SpheresLayer
from .layers.skeleton import SkeletonLayer
from .theme import LIGHT, Theme

if TYPE_CHECKING:
    from ..collision.robot_spheres import RobotSpheres
    from ..data_model.model import Model
    from ..io.ir import IRGeom


class RobotView:
    """Layers for one robot and a clip ``q (T, nq)``; ``update(q)`` rewrites them in place.

    ``visual``/``collision`` draw the URDF/MJCF geometry as ``RigidMeshLayer``s; ``skeleton``
    draws joints and links, coloured by ``|tau| / effort_limit`` when ``tau (T, nv)`` is given;
    ``frames`` lists frame names or ids for a ``FramesLayer``; ``spheres`` draws a
    ``RobotSpheres`` collision model. Forward kinematics runs once per ``update``.
    """

    def __init__(
        self,
        model: Model,
        q: torch.Tensor,
        *,
        name: str = "robot",
        visual: bool = True,
        collision: bool = False,
        skeleton: bool = False,
        frames: Sequence[str | int] = (),
        tau: torch.Tensor | None = None,
        spheres: RobotSpheres | None = None,
        theme: Theme = LIGHT,
        opacity: float = 1.0,
    ) -> None:
        self.model = model
        self.frame_ids = [model.frame_id(frame) if isinstance(frame, str) else int(frame) for frame in frames]
        self.spheres = spheres
        self.layers: list[Layer] = []
        joint_poses, frame_poses, centers = self._kinematics(q)
        if visual:
            meshes = robot_meshes(model, collision=False, default_color=theme.robot)
            if meshes:
                self.layers.append(RigidMeshLayer(name, meshes, joint_poses, opacity=opacity))
        if collision:
            meshes = robot_meshes(model, collision=True, default_color=theme.force)
            if meshes:
                self.layers.append(RigidMeshLayer(f"{name} collision", meshes, joint_poses, opacity=0.4))
        if skeleton:
            self.layers.append(robot_skeleton(model, joint_poses, name=f"{name} skeleton", tau=tau, theme=theme))
        if self.frame_ids:
            self.layers.append(FramesLayer(f"{name} frames", frame_poses))
        if spheres is not None:
            radii = spheres.radii.detach().cpu().numpy()
            self.layers.append(SpheresLayer(f"{name} spheres", centers, radii, colors=theme.torque))

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def update(self, q: torch.Tensor) -> None:
        """Recompute kinematics for ``q`` (same number of frames) into the existing layer arrays."""
        joint_poses, frame_poses, centers = self._kinematics(q)
        for layer in self.layers:
            if isinstance(layer, RigidMeshLayer):
                layer.body_poses[:] = joint_poses
            elif isinstance(layer, SkeletonLayer):
                layer.joints[:] = joint_poses[:, 1:, :3]
            elif isinstance(layer, FramesLayer):
                layer.poses[:] = frame_poses
            elif isinstance(layer, SpheresLayer):
                layer.centers[:] = centers

    def _kinematics(self, q: torch.Tensor) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        from ..kinematics.state import forward_kinematics_state  # noqa: PLC0415 - keep the core import-light

        if q.ndim == 1:
            q = q[None]
        if q.ndim != 2:
            raise ValueError("q must be (T, nq) or (nq,)")
        with torch.no_grad():
            state = forward_kinematics_state(self.model, q)
            joint_poses = state.joint_pose.cpu().numpy().astype(np.float32)
            frame_poses = state.frame_pose[:, self.frame_ids].cpu().numpy() if self.frame_ids else None
            centers = self.spheres.centers(state).cpu().numpy() if self.spheres is not None else None
        return joint_poses, frame_poses, centers


def robot_layers(model: Model, q: torch.Tensor, **options: Any) -> list[Layer]:
    """Layers for a clip ``q (T, nq)`` (or one pose ``(nq,)``); see ``RobotView`` for the options."""
    return RobotView(model, q, **options).layers


def robot_skeleton(
    model: Model,
    joint_poses: np.ndarray,
    *,
    name: str = "skeleton",
    tau: torch.Tensor | None = None,
    theme: Theme = LIGHT,
    radius: float = 0.02,
) -> SkeletonLayer:
    """Skeleton over joints 1.. of ``joint_poses (T, njoints, 7)``; fixed joints get small spheres."""
    moving = np.array([nv > 0 for nv in model.nvs[1:]])
    radii = np.where(moving, radius, radius * 0.4).astype(np.float32)
    parents = np.asarray(model.parents[1:], dtype=np.int64) - 1  # the universe becomes -1
    colors: Any = theme.joint
    if tau is not None:
        colors = _torque_colors(model, tau.detach().cpu().numpy(), theme, joint_poses.shape[0])
    return SkeletonLayer(name, joint_poses[:, 1:, :3], parents, radius=radii, colors=colors, bone_color=theme.link)


def _torque_colors(model: Model, tau: np.ndarray, theme: Theme, num_frames: int) -> np.ndarray:
    """``(T, njoints - 1, 3)`` colours from ``|tau_j| / effort_limit_j``; joints without a torque or a finite
    limit keep the base colour."""
    tau = tau.reshape(-1, model.nv)
    if tau.shape[0] not in (1, num_frames):
        raise ValueError(f"tau must have {num_frames} frames or one, got {tau.shape[0]}")
    limits = model.values.effort_limit.detach().cpu().numpy().reshape(-1)
    # Joints without a finite effort limit (free-flyer, unbounded URDF joints) keep the base colour.
    finite = np.isfinite(limits) & (limits > 0)
    limits = np.where(finite, limits, np.inf)
    ratio = np.zeros((tau.shape[0], model.njoints - 1), dtype=np.float32)
    for joint, (start, count) in enumerate(zip(model.idx_vs[1:], model.nvs[1:])):
        if count:
            block = np.abs(tau[:, start : start + count]) / limits[start : start + count]
            ratio[:, joint] = block.max(axis=1)
    colors = colormap(ratio, theme.joint, theme.force)
    return np.broadcast_to(colors, (num_frames, *colors.shape[1:])).copy()


def robot_meshes(model: Model, *, collision: bool = False, default_color: Any = (0.8, 0.8, 0.8)) -> list[MeshSpec]:
    """Mesh specs for the visual (or collision) geometry stored in ``model.meta["ir"]``.

    Mesh files resolve through ``model.meta["asset_resolver"]`` when present; unreadable files
    are skipped. Primitives are tessellated with trimesh.
    """
    ir = model.meta.get("ir")
    if ir is None:
        return []
    resolver = model.meta.get("asset_resolver")
    specs: list[MeshSpec] = []
    for ir_body in ir.bodies:
        body = model.body_name_to_id.get(ir_body.name)
        if body is None:
            continue
        for geom in ir_body.collision_geoms if collision else ir_body.visual_geoms:
            spec = _load_geom(geom, body, resolver, default_color)
            if spec is not None:
                specs.append(spec)
    return specs


def _load_geom(geom: IRGeom, body: int, resolver: Any, default_color: Any) -> MeshSpec | None:
    import trimesh  # noqa: PLC0415 - optional dependency, imported at the boundary

    origin = geom.origin.detach().cpu().numpy().astype(np.float32)
    color = tuple(geom.rgba[:3]) if geom.rgba is not None else tuple(default_color)
    kind, params = geom.kind, geom.params
    if kind == "mesh":
        path = params.get("path", "")
        if not path:
            return None
        if resolver is not None:
            try:
                path = str(resolver.resolve(path))
            except (FileNotFoundError, OSError):
                pass
        try:
            mesh = trimesh.load(path, force="mesh")
        except Exception:  # noqa: BLE001 - trimesh raises many loader-specific errors
            return None
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
            return None
        mesh.apply_scale(params.get("scale", [1.0, 1.0, 1.0]))
        source = mesh if _has_own_colors(mesh) else None
    elif kind == "box":
        mesh, source = trimesh.creation.box(extents=params["size"]), None
    elif kind == "cylinder":
        mesh, source = trimesh.creation.cylinder(radius=params["radius"], height=params["length"]), None
    elif kind == "sphere":
        mesh, source = trimesh.creation.icosphere(subdivisions=2, radius=params["radius"]), None
    elif kind == "capsule":
        mesh, source = trimesh.creation.capsule(radius=params["radius"], height=params["length"]), None
    else:
        return None
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    return MeshSpec(vertices, faces, body, origin, color=color, trimesh=source)


def _has_own_colors(mesh: Any) -> bool:
    """True for textured meshes or meshes with more than one vertex/face colour (DAE, OBJ)."""
    kind = mesh.visual.kind
    if kind == "texture":
        return True
    if kind == "vertex":
        return len(np.unique(np.asarray(mesh.visual.vertex_colors), axis=0)) > 1
    if kind == "face":
        return len(np.unique(np.asarray(mesh.visual.face_colors), axis=0)) > 1
    return False
