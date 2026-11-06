"""MJCF parser: MuJoCo compiles the file and ``parse_mjcf`` reads the compiled ``MjModel`` (record 0036).

MuJoCo resolves defaults, units (``compiler angle="degree"``), ``limited="auto"``, ``euler`` and ``axisangle``
orientations, ``fromto`` geoms and geom-derived inertias, so nothing here re-implements them. Oracle:
``tests/parity/mujoco/test_mjcf_kinematics.py`` compares body poses, site frames and limits with ``mj_kinematics``.

MuJoCo moves a body about ``jnt_pos``, a point of the body frame; BetterRobot moves a body about its joint frame's
origin. A body's joint frame is therefore its MuJoCo body frame translated by ``jnt_pos``, and every quantity
MuJoCo gives in the body frame (centre of mass, sites, geoms, child body positions) is shifted by ``-jnt_pos``.
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
from .ir import IRBody, IRFrame, IRGeom, IRJoint, IRModel


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
    kinds = {
        mujoco.mjtJoint.mjJNT_HINGE: "revolute",
        mujoco.mjtJoint.mjJNT_SLIDE: "prismatic",
        mujoco.mjtJoint.mjJNT_BALL: "spherical",
        mujoco.mjtJoint.mjJNT_FREE: "free_flyer",
    }
    scalar = joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    limited = scalar and bool(compiled.jnt_limited[joint])
    dof = int(compiled.jnt_dofadr[joint])
    return IRJoint(
        name=_named(compiled.joint(joint), "joint", joint, path),
        parent_body="",  # filled by the caller, which knows the body names
        child_body=name,
        kind=kinds[joint_type],
        axis=torch.tensor(compiled.jnt_axis[joint], dtype=torch.float64) if scalar else None,
        lower=float(compiled.jnt_range[joint, 0]) if limited else None,
        upper=float(compiled.jnt_range[joint, 1]) if limited else None,
        damping=float(compiled.dof_damping[dof]),
        armature=float(compiled.dof_armature[dof]),
    )


def _geoms(mujoco: ModuleType, compiled: Any, spec: Any, body: int, offset: np.ndarray, path: Path) -> list[IRGeom]:
    """The geoms of ``body`` as IR primitives placed in the body's joint frame."""
    meshes = {mesh.name: mesh for mesh in spec.meshes}
    geoms: list[IRGeom] = []
    for geom in range(compiled.ngeom):
        if int(compiled.geom_bodyid[geom]) != body:
            continue
        geom_type = compiled.geom_type[geom]
        size = compiled.geom_size[geom]
        placement = _pose(compiled.geom_pos[geom] - offset, compiled.geom_quat[geom])
        rgba = tuple(float(value) for value in compiled.geom_rgba[geom])
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            kind, params = "sphere", {"radius": float(size[0])}
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            kind, params = "capsule", {"radius": float(size[0]), "length": 2.0 * float(size[1])}
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            kind, params = "cylinder", {"radius": float(size[0]), "length": 2.0 * float(size[1])}
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            kind, params = "box", {"size": [2.0 * float(value) for value in size]}
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = meshes[compiled.mesh(int(compiled.geom_dataid[geom])).name]
            uri = str(Path(spec.meshdir) / mesh.file) if spec.meshdir else mesh.file
            kind, params = "mesh", {"path": uri, "scale": [float(value) for value in mesh.scale]}
        else:
            body_name = compiled.body(body).name
            raise ModelError(f"{path}: geom {geom} of body {body_name!r} has type {geom_type!r}, not a shape")
        geoms.append(IRGeom(kind=kind, params=params, origin=placement, rgba=rgba))
    return geoms


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
        if joint is not None and joint.kind != "free_flyer":
            offsets[body] = compiled.jnt_pos[int(compiled.body_jntadr[body])]

    bodies: list[IRBody] = []
    ir_joints: list[IRJoint] = []
    frames: list[IRFrame] = []
    for body in range(1, compiled.nbody):
        name = compiled.body(body).name
        parent = int(compiled.body_parentid[body])
        parent_name = "world" if parent == 0 else compiled.body(parent).name
        # The joint's placement in the parent joint frame: the parent's offset undone, the body pose, this
        # body's offset applied.
        placement = _pose(compiled.body_pos[body] - offsets[parent], compiled.body_quat[body])
        placement = se3.compose(placement, se3.from_translation(torch.tensor(offsets[body], dtype=torch.float64)))
        joint = joints[body]
        if joint is None:
            joint = IRJoint(name=name, parent_body=parent_name, child_body=name, kind="fixed")
        elif joint.kind == "free_flyer":  # a free joint places the body in the world by ``qpos`` alone
            placement = se3.identity(dtype=torch.float64)
        joint.parent_body = parent_name
        joint.origin = placement
        ir_joints.append(joint)
        inertial_rotation = so3.to_matrix(_pose(np.zeros(3), compiled.body_iquat[body])[3:])
        bodies.append(
            IRBody(
                name=name,
                mass=float(compiled.body_mass[body]),
                com=torch.tensor(compiled.body_ipos[body] - offsets[body], dtype=torch.float64),
                inertia=inertial_rotation @ torch.diag(torch.tensor(compiled.body_inertia[body])) @ inertial_rotation.T,
                collision_geoms=_geoms(mujoco, compiled, spec, body, offsets[body], path),
            )
        )
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
    return IRModel(name=spec.modelname, bodies=bodies, joints=ir_joints, frames=frames, gravity=gravity)
