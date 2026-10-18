"""Robot adapter: a ``Model`` and a configuration clip ``q (T, nq)`` become viewer layers.

The only viewer file that knows ``Model``. Forward kinematics runs once for the whole clip; the
layers receive numpy arrays and never touch the model again.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ._math import RGB, colormap
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
        self.sphere_model = spheres
        # Typed layer slots (None when not drawn); ``update`` writes each one directly.
        self.mesh: RigidMeshLayer | None = None
        self.collision: RigidMeshLayer | None = None
        self.skeleton: SkeletonLayer | None = None
        self.frames: FramesLayer | None = None
        self.spheres: SpheresLayer | None = None
        joint_poses, frame_poses, centers = self._kinematics(q)
        if visual:
            meshes = robot_meshes(model, collision=False, default_color=theme.robot)
            if meshes:
                self.mesh = RigidMeshLayer(name, meshes, joint_poses, opacity=opacity)
        if collision:
            meshes = robot_meshes(model, collision=True, default_color=theme.force)
            if meshes:
                self.collision = RigidMeshLayer(f"{name} collision", meshes, joint_poses, opacity=0.4)
        if skeleton:
            self.skeleton = robot_skeleton(model, joint_poses, name=f"{name} skeleton", tau=tau, theme=theme)
        if self.frame_ids:
            self.frames = FramesLayer(f"{name} frames", frame_poses)
        if spheres is not None:
            radii = spheres.radii.detach().cpu().numpy()
            self.spheres = SpheresLayer(f"{name} spheres", centers, radii, colors=theme.torque)
        slots = (self.mesh, self.collision, self.skeleton, self.frames, self.spheres)
        self.layers: list[Layer] = [layer for layer in slots if layer is not None]

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def update(self, q: torch.Tensor) -> None:
        """Recompute kinematics for ``q`` (same number of frames) into the existing layer arrays."""
        joint_poses, frame_poses, centers = self._kinematics(q)
        for mesh in (self.mesh, self.collision):
            if mesh is not None:
                mesh.body_poses[:] = joint_poses
        if self.skeleton is not None:
            self.skeleton.joints[:] = joint_poses[:, 1:, :3]
            self.skeleton.rotations[:] = joint_poses[:, 1:, 3:]
        if self.frames is not None:
            self.frames.poses[:] = frame_poses
        if self.spheres is not None:
            self.spheres.centers[:] = centers

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
            centers = self.sphere_model.centers(state).cpu().numpy() if self.sphere_model is not None else None
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
    bone_radius: float | np.ndarray | None = None,
) -> SkeletonLayer:
    """Skeleton over joints 1.. of ``joint_poses (T, njoints, 7)``, one marker per joint kind (``skeleton_spec``).

    ``bone_radius`` is one float or ``(njoints - 1,)``, entry ``j`` for the bone into joint ``j + 1``; 0 hides a
    bone, such as one to a massless sensor mount (``model.body_inertias[1:, 0] == 0``). ``None`` keeps
    ``SkeletonLayer``'s default.
    """
    parents = np.asarray(model.parents[1:], dtype=np.int64) - 1  # the universe becomes -1
    kinds, axes = skeleton_spec(model)
    colors: Any = theme.joint
    if tau is not None:
        colors = _torque_colors(model, tau.detach().cpu().numpy(), theme, joint_poses.shape[0])
    return SkeletonLayer(
        name,
        joint_poses[:, 1:],
        parents,
        kinds=kinds,
        axes=axes,
        radius=radius,
        bone_radius=bone_radius,
        colors=colors,
        bone_color=theme.link,
    )


def torque_spheres(
    model: Model,
    joint_poses: np.ndarray,
    *,
    tau: torch.Tensor | np.ndarray,
    name: str = "torques",
    radius: tuple[float, float] = (0.015, 0.05),
    theme: Theme = LIGHT,
    over_limit: RGB = (1.0, 0.85, 0.0),
) -> SkeletonLayer:
    """Spheres at joints 1.. of ``joint_poses (T, njoints, 7)``, sized and coloured by ``|tau| / effort_limit``.

    ``tau`` is ``(T, nv)`` or one ``(nv,)``; a joint shows the largest ratio over its dofs. Up to the limit the
    radius grows from ``radius[0]`` to ``radius[1]`` and the colour blends from ``theme.joint`` to ``theme.force``;
    past it the sphere keeps ``radius[1]`` and turns ``over_limit``. Joints without a dof of finite positive limit
    (fixed joints, a free flyer) draw nothing.
    """
    ratio, limited = _torque_ratio(model, torch.as_tensor(tau).detach().cpu().numpy(), joint_poses.shape[0])
    fill = np.minimum(ratio, 1.0)
    radii = np.where(limited, radius[0] + (radius[1] - radius[0]) * fill, 0.0)
    over = np.asarray(over_limit, dtype=np.float32)
    colors = np.where((ratio > 1.0)[..., None], over, colormap(fill, theme.joint, theme.force))
    parents = np.full(model.njoints - 1, -1)  # spheres only, no bones
    return SkeletonLayer(name, joint_poses[:, 1:, :3], parents, radius=radii, colors=colors)


def skeleton_spec(model: Model) -> tuple[list[str], np.ndarray]:
    """Joint kinds and axes ``(njoints - 1, 3)`` (joint frame; ``+z`` for joints without one) of joints 1..,
    for ``SkeletonLayer(kinds=, axes=)``."""
    # The packed table, not ``joint.axis``: JointRX/RY/RZ and PX/PY/PZ keep their axis in a ClassVar (axis None).
    axes = model.structure.joint_axes[1:].detach().cpu().numpy().astype(np.float32)
    axes[~np.any(axes, axis=-1)] = (0.0, 0.0, 1.0)  # zero rows: spherical, free-flyer, planar, fixed, ...
    return [joint.kind for joint in model.joint_models[1:]], axes


def _torque_colors(model: Model, tau: np.ndarray, theme: Theme, nsteps: int) -> np.ndarray:
    """``(T, njoints - 1, 3)`` colours from ``|tau_j| / effort_limit_j``; joints without a torque or a finite
    limit keep the base colour."""
    return colormap(_torque_ratio(model, tau, nsteps)[0], theme.joint, theme.force)


def _torque_ratio(model: Model, tau: np.ndarray, nsteps: int) -> tuple[np.ndarray, np.ndarray]:
    """``|tau| / effort_limit`` ``(nsteps, njoints - 1)`` of joints 1.. for ``tau (nsteps or 1, nv)``, the largest
    over each joint's dofs, and ``(njoints - 1,)`` whether a joint has a dof with a finite positive limit."""
    tau = tau.reshape(-1, model.nv)
    if tau.shape[0] not in (1, nsteps):
        raise ValueError(f"tau must have {nsteps} frames or one, got {tau.shape[0]}")
    limits = model.values.effort_limit.detach().cpu().numpy().reshape(-1)
    # Dofs without a finite effort limit (free-flyer, unbounded URDF joints) contribute a ratio of 0.
    finite = np.isfinite(limits) & (limits > 0)
    limits = np.where(finite, limits, np.inf)
    ratio = np.zeros((tau.shape[0], model.njoints - 1), dtype=np.float32)
    limited = np.zeros(model.njoints - 1, dtype=bool)
    for joint, (start, count) in enumerate(zip(model.idx_vs[1:], model.nvs[1:])):
        if count:
            block = np.abs(tau[:, start : start + count]) / limits[start : start + count]
            ratio[:, joint] = block.max(axis=1)
            limited[joint] = finite[start : start + count].any()
    return np.broadcast_to(ratio, (nsteps, ratio.shape[1])).copy(), limited


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
