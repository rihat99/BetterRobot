"""MJCF parser: MuJoCo compiles the file and ``parse_mjcf`` reads the compiled ``MjModel`` (record 0036).

MuJoCo resolves defaults, units (``compiler angle="degree"``), ``limited="auto"``, ``euler`` and ``axisangle``
orientations, ``fromto`` geoms and geom-derived inertias, so nothing here re-implements them. Oracle:
``tests/parity/mujoco/test_mjcf_kinematics.py`` compares body poses, site frames and limits with ``mj_kinematics``.

MuJoCo moves a body about ``jnt_pos``, a point of the body frame; BetterRobot moves a body about its joint frame's
origin. A body's joint frame is therefore its MuJoCo body frame translated by ``jnt_pos``, and every quantity
MuJoCo gives in the body frame (centre of mass, sites, geoms, child body positions) is shifted by ``-jnt_pos``.
MuJoCo moves a hinge or slide by ``qpos - ref``; the joint's motion at ``-ref`` is folded into its placement, so a
model's ``q`` is MuJoCo's ``qpos``, the limits stay in ``qpos`` units and ``q_neutral`` is ``qpos0`` (a free body's
``pos``/``quat`` included) clamped into the limits.
A body carries at most one joint (a chain of joints in one body needs a body per joint); a body without a joint is
attached by a fixed joint named after the body; geoms of the world body are scenery, not robot geometry, and are
not read. ``mujoco`` is imported here only (extra ``mjcf``).
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from ..exceptions import ModelError, OptionalDependencyError
from ..lie import se3, so3
from ..model.geometry import Box, Capsule, Cylinder, Mesh, Shape, Sphere
from ..model.joints import JointFixed, JointFreeFlyer, JointModel, JointPrismatic, JointRevolute, JointSpherical
from .ir import IRBody, IRFrame, IRJoint, IRModel, IRShape


def _pose(position: np.ndarray, quaternion_wxyz: np.ndarray) -> torch.Tensor:
    """MuJoCo ``pos (3,)`` and ``quat (w, x, y, z)`` as one pose ``(7,)`` ``[t, qx, qy, qz, qw]`` in float64."""
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    return torch.tensor([*position.tolist(), x, y, z, w], dtype=torch.float64)


def _named(element: Any, kind: str, index: int, path: Path) -> str:
    """The name of ``element``, the ``index``-th ``kind``; an unnamed body, joint or site is an error (names are the API)."""
    if not element.name:
        raise ModelError(f"{path}: {kind} {index} has no name; name every body, joint and site")
    return element.name


def _joint(mujoco: ModuleType, compiled: Any, body: int, path: Path) -> IRJoint | None:
    """The one joint moving ``body`` relative to its parent, or ``None`` for a fixed attachment."""
    count = int(compiled.body_jntnum[body])
    name = _named(compiled.body(body), "body", body, path)
    if count > 1:
        raise ModelError(f"{path}: body {name!r} has {count} joints; BetterRobot takes one joint per body")
    if count == 0:
        return None
    joint = int(compiled.body_jntadr[body])
    joint_type = compiled.jnt_type[joint]
    x, y, z = (float(value) for value in compiled.jnt_axis[joint])  # unit: MuJoCo's compiler normalises it
    kind: JointModel
    if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        kind = JointRevolute(axis=(x, y, z))
    elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
        kind = JointPrismatic(axis=(x, y, z))
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        kind = JointSpherical()
    else:
        kind = JointFreeFlyer()
    limited = isinstance(kind, (JointRevolute, JointPrismatic)) and bool(compiled.jnt_limited[joint])
    dof = int(compiled.jnt_dofadr[joint])
    start = int(compiled.jnt_qposadr[joint])
    qpos0 = compiled.qpos0[start : start + kind.nq]
    if kind.nq in (4, 7):  # MuJoCo writes a quaternion w first
        qpos0 = np.concatenate([qpos0[:-4], qpos0[-3:], qpos0[-4:-3]])
    return IRJoint(
        name=_named(compiled.joint(joint), "joint", joint, path),
        parent_body="",  # filled by the caller, which knows the body names
        child_body=name,
        joint=kind,
        lower=float(compiled.jnt_range[joint, 0]) if limited else None,
        upper=float(compiled.jnt_range[joint, 1]) if limited else None,
        damping=float(compiled.dof_damping[dof]),
        armature=float(compiled.dof_armature[dof]),
        neutral=torch.tensor(qpos0, dtype=torch.float64),
    )


def _shapes(mujoco: ModuleType, compiled: Any, spec: Any, body: int, offset: np.ndarray, path: Path) -> list[IRShape]:
    """The geoms of ``body`` as shapes placed in the body's joint frame; one that collides with nothing is display only."""
    meshes = {mesh.name: mesh for mesh in spec.meshes}
    name = compiled.body(body).name
    shapes: list[IRShape] = []
    for geom in range(compiled.ngeom):
        if int(compiled.geom_bodyid[geom]) != body:
            continue
        geom_type = compiled.geom_type[geom]
        size = compiled.geom_size[geom]
        pose = _pose(compiled.geom_pos[geom] - offset, compiled.geom_quat[geom])
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            # MuJoCo recentres a mesh on its inertial frame (mesh_pos, mesh_quat) and folds that frame into the
            # geom pose; the shape keeps the file's vertices, so the recentring is undone here.
            data = int(compiled.geom_dataid[geom])
            pose = se3.compose_raw(pose, se3.inverse_raw(_pose(compiled.mesh_pos[data], compiled.mesh_quat[data])))
        tx, ty, tz, qx, qy, qz, qw = pose.tolist()
        placement = (tx, ty, tz, qx, qy, qz, qw)
        red, green, blue, alpha = (float(value) for value in compiled.geom_rgba[geom])
        color = (red, green, blue, alpha)
        shape: Shape
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            shape = Sphere(radius=float(size[0]), placement=placement, color=color)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            shape = Capsule(radius=float(size[0]), length=2.0 * float(size[1]), placement=placement, color=color)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shape = Cylinder(radius=float(size[0]), length=2.0 * float(size[1]), placement=placement, color=color)
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            x, y, z = (2.0 * float(value) for value in size)
            shape = Box(size=(x, y, z), placement=placement, color=color)
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = meshes[compiled.mesh(int(compiled.geom_dataid[geom])).name]
            uri = str(Path(spec.meshdir) / mesh.file) if spec.meshdir else mesh.file
            x, y, z = (float(value) for value in mesh.scale)
            shape = Mesh(uri=uri, scale=(x, y, z), placement=placement, color=color)
        else:
            raise ModelError(f"{path}: geom {geom} of body {name!r} has type {geom_type!r}, not a shape")
        # MuJoCo's collision filter: a geom with contype and conaffinity both 0 never collides.
        collision = int(compiled.geom_contype[geom]) != 0 or int(compiled.geom_conaffinity[geom]) != 0
        shapes.append(IRShape(body=name, shape=shape, collision=collision))
    return shapes


def parse_mjcf(path: str | Path) -> IRModel:
    """Parse an MJCF file into an ``IRModel`` through MuJoCo's compiler; raises ``ModelError`` naming the file."""
    try:
        import mujoco  # noqa: PLC0415 - the mjcf extra, imported at its boundary
    except ImportError as exc:
        raise OptionalDependencyError("mujoco", install="uv sync --extra mjcf") from exc
    path = Path(path)
    if not path.is_file():
        raise ModelError(f"{path}: no such file")
    try:
        spec = mujoco.MjSpec.from_file(str(path))
        compiled = spec.compile()
    except ValueError as exc:
        raise ModelError(f"{path}: {exc}") from exc

    joints = [None] + [_joint(mujoco, compiled, body, path) for body in range(1, compiled.nbody)]
    offsets = np.zeros((compiled.nbody, 3))
    for body, joint in enumerate(joints):
        if joint is not None and not isinstance(joint.joint, JointFreeFlyer):
            offsets[body] = compiled.jnt_pos[int(compiled.body_jntadr[body])]

    bodies: list[IRBody] = []
    ir_joints: list[IRJoint] = []
    frames: list[IRFrame] = []
    shapes: list[IRShape] = []
    for body in range(1, compiled.nbody):
        name = compiled.body(body).name
        parent = int(compiled.body_parentid[body])
        parent_name = "world" if parent == 0 else compiled.body(parent).name
        # The joint's placement in the parent joint frame: the parent's offset undone, the body pose, this
        # body's offset applied.
        placement = _pose(compiled.body_pos[body] - offsets[parent], compiled.body_quat[body])
        placement = se3.compose_raw(
            placement, se3.from_translation_raw(torch.tensor(offsets[body], dtype=torch.float64))
        )
        joint = joints[body]
        if joint is None:
            joint = IRJoint(name=name, parent_body=parent_name, child_body=name, joint=JointFixed())
        elif isinstance(joint.joint, JointFreeFlyer):  # a free joint places the body in the world by ``qpos`` alone
            placement = se3.identity(dtype=torch.float64)
        elif isinstance(joint.joint, (JointRevolute, JointPrismatic)):
            # MuJoCo moves a hinge or slide by ``qpos - ref``: the joint's own motion at ``-ref`` folded into the
            # placement keeps q equal to qpos, so the limits stay in qpos units.
            placement = se3.compose_raw(placement, joint.joint.placement(-joint.neutral))
        joint.parent_body = parent_name
        joint.origin = placement
        ir_joints.append(joint)
        inertial_rotation = so3.to_matrix_raw(_pose(np.zeros(3), compiled.body_iquat[body])[3:])
        bodies.append(
            IRBody(
                name=name,
                mass=float(compiled.body_mass[body]),
                com=torch.tensor(compiled.body_ipos[body] - offsets[body], dtype=torch.float64),
                inertia=inertial_rotation @ torch.diag(torch.tensor(compiled.body_inertia[body])) @ inertial_rotation.T,
            )
        )
        shapes.extend(_shapes(mujoco, compiled, spec, body, offsets[body], path))
    for site in range(compiled.nsite):
        body = int(compiled.site_bodyid[site])
        frames.append(
            IRFrame(
                name=_named(compiled.site(site), "site", site, path),
                parent_body=compiled.body(body).name,
                placement=_pose(compiled.site_pos[site] - offsets[body], compiled.site_quat[site]),
            )
        )
    gravity = torch.cat([torch.tensor(compiled.opt.gravity, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)])
    return IRModel(name=spec.modelname, bodies=bodies, joints=ir_joints, frames=frames, shapes=shapes, gravity=gravity)
