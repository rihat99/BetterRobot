"""MJCF parser — ``mujoco.MjSpec``-backed, emits an ``IRModel``.

MJCF has strictly richer joint syntax than URDF — ball joints, free joints,
sites — so the MJCF parser fills more of the IR than the URDF parser.
``mujoco`` is imported lazily so ``import better_robot`` does not require
it.

Covered: bodies, joints (hinge/slide/ball/free), sites (→ frames).
Not covered (deferred to future work): mesh loading, tendons, actuators.

See ``docs/design/04_PARSERS.md §5``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..ir import IRBody, IRFrame, IRGeom, IRJoint, IRModel

if TYPE_CHECKING:
    from ..assets import AssetResolver


def _pos_quat_to_se3(pos: np.ndarray | None, quat: np.ndarray | None) -> torch.Tensor:
    """Convert MuJoCo pos (3,) + quat (w,x,y,z) to [tx,ty,tz,qx,qy,qz,qw]."""
    if pos is None:
        t = torch.zeros(3)
    else:
        t = torch.tensor(pos[:3].astype(np.float32))

    if quat is None:
        q = torch.tensor([0., 0., 0., 1.])
    else:
        # MuJoCo quat convention is (w, x, y, z) — convert to (x,y,z,w)
        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        q = torch.tensor([x, y, z, w])

    return torch.cat([t, q])


def parse_mjcf(
    source: str | Path,
    *,
    resolver: "AssetResolver | None" = None,
) -> IRModel:
    """Parse an MJCF file into an ``IRModel``.

    Requires ``mujoco`` to be installed. Imports lazily so the rest of the
    library does not depend on it.

    Parameters
    ----------
    source : str | Path
    resolver : AssetResolver, optional
        Resolves mesh / texture references encountered inside the MJCF.
        Defaults to a :class:`~better_robot.io.assets.FilesystemResolver`
        rooted at the MJCF file's parent directory.

    See docs/design/04_PARSERS.md §5.
    """
    if resolver is None:
        from ..assets import FilesystemResolver
        resolver = FilesystemResolver(base_path=Path(source).parent)
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "mujoco is required to parse MJCF files. "
            "Install with: pip install mujoco"
        ) from exc

    source = str(source)
    spec = mujoco.MjSpec.from_file(source)

    ir_bodies: list[IRBody] = []
    ir_joints: list[IRJoint] = []
    ir_frames: list[IRFrame] = []

    # ── traverse body tree ────────────────────────────────────────────────────
    def _visit(body: mujoco.MjsBody, parent_name: str | None) -> None:
        bname = body.name if body.name else f"_body_{len(ir_bodies)}"

        # Inertia
        mass = float(body.mass) if hasattr(body, "mass") else 0.0
        pos_inertial = np.array(body.ipos, dtype=np.float32) if hasattr(body, "ipos") else np.zeros(3, np.float32)
        inertia_diag = np.array(body.inertia, dtype=np.float32) if hasattr(body, "inertia") else np.zeros(3, np.float32)
        inertia_mat = np.diag(inertia_diag) if inertia_diag.shape == (3,) else np.zeros((3, 3), np.float32)

        # Collision geoms
        collision_geoms: list[IRGeom] = []
        for g in body.geoms:
            gname = g.name or ""
            gtype = int(g.type) if hasattr(g, "type") else -1
            gpos = np.array(g.pos, dtype=np.float32) if hasattr(g, "pos") else np.zeros(3, np.float32)
            gquat = np.array(g.quat, dtype=np.float32) if hasattr(g, "quat") else np.array([1., 0., 0., 0.], np.float32)
            placement = _pos_quat_to_se3(gpos, gquat)
            gsize = np.array(g.size, dtype=np.float32) if hasattr(g, "size") else np.zeros(3, np.float32)

            # mujoco geom types: 0=plane,1=hfield,2=sphere,3=capsule,4=ellipsoid,5=cylinder,6=box,7=mesh
            if gtype == 2:  # sphere
                collision_geoms.append(IRGeom("sphere", {"radius": float(gsize[0])}, placement))
            elif gtype == 3:  # capsule
                collision_geoms.append(IRGeom("capsule", {"radius": float(gsize[0]), "length": float(gsize[1]) * 2}, placement))
            elif gtype == 5:  # cylinder
                collision_geoms.append(IRGeom("cylinder", {"radius": float(gsize[0]), "length": float(gsize[1]) * 2}, placement))
            elif gtype == 6:  # box
                collision_geoms.append(IRGeom("box", {"size": [float(gsize[0]) * 2, float(gsize[1]) * 2, float(gsize[2]) * 2]}, placement))

        ir_bodies.append(IRBody(
            name=bname,
            mass=mass,
            com=torch.tensor(pos_inertial),
            inertia=torch.tensor(inertia_mat),
            collision_geoms=collision_geoms,
        ))

        # ── joints on this body ───────────────────────────────────────────────
        for jnt in body.joints:
            jname = jnt.name if jnt.name else f"_joint_{len(ir_joints)}"
            jtype = int(jnt.type) if hasattr(jnt, "type") else 0
            jpos = np.array(jnt.pos, dtype=np.float32) if hasattr(jnt, "pos") else np.zeros(3, np.float32)
            jaxis = np.array(jnt.axis, dtype=np.float32) if hasattr(jnt, "axis") else np.array([0., 0., 1.], np.float32)
            norm = np.linalg.norm(jaxis)
            if norm > 1e-12:
                jaxis = jaxis / norm

            # Joint placement: pos in parent body frame, identity rotation
            origin = _pos_quat_to_se3(jpos, None)

            # Limits
            jrange = np.array(jnt.range, dtype=np.float32) if hasattr(jnt, "range") else None
            limited = bool(jnt.limited) if hasattr(jnt, "limited") else False
            lo: float | None = None
            hi: float | None = None
            if jrange is not None and limited:
                lo = float(jrange[0])
                hi = float(jrange[1])

            # mujoco joint types: 0=free,1=ball,2=slide,3=hinge
            if jtype == 0:  # free
                kind = "free_flyer"
                axis_t = None
            elif jtype == 1:  # ball
                kind = "spherical"
                axis_t = None
            elif jtype == 2:  # slide (prismatic)
                kind = "prismatic"
                axis_t = torch.tensor(jaxis)
            elif jtype == 3:  # hinge (revolute)
                kind = "revolute"
                axis_t = torch.tensor(jaxis)
            else:
                kind = "fixed"
                axis_t = None

            parent_name_safe = parent_name if parent_name is not None else "world"
            ir_joints.append(IRJoint(
                name=jname,
                parent_body=parent_name_safe,
                child_body=bname,
                kind=kind,
                axis=axis_t,
                origin=origin,
                lower=lo,
                upper=hi,
            ))

        if parent_name is not None and not body.joints:
            # Body with no joints → implicit fixed joint
            bpos = np.array(body.pos, dtype=np.float32) if hasattr(body, "pos") else np.zeros(3, np.float32)
            bquat = np.array(body.quat, dtype=np.float32) if hasattr(body, "quat") else np.array([1., 0., 0., 0.], np.float32)
            origin = _pos_quat_to_se3(bpos, bquat)
            jname = f"_fixed_{bname}"
            ir_joints.append(IRJoint(
                name=jname,
                parent_body=parent_name,
                child_body=bname,
                kind="fixed",
                origin=origin,
            ))

        # Sites → operational frames
        for site in getattr(body, "sites", []):
            sname = site.name if site.name else f"_site_{len(ir_frames)}"
            spos = np.array(site.pos, dtype=np.float32) if hasattr(site, "pos") else np.zeros(3, np.float32)
            squat = np.array(site.quat, dtype=np.float32) if hasattr(site, "quat") else np.array([1., 0., 0., 0.], np.float32)
            placement = _pos_quat_to_se3(spos, squat)
            ir_frames.append(IRFrame(name=sname, parent_body=bname, placement=placement, frame_type="op"))

        # Recurse into child bodies
        for child_body in body.bodies:
            _visit(child_body, bname)

    # MjSpec worldbody
    worldbody = spec.worldbody
    # Add a "world" body (no inertia)
    ir_bodies.append(IRBody(name="world", mass=0.0))
    for child in worldbody.bodies:
        _visit(child, "world")

    # Root body = first non-world body
    child_set = {j.child_body for j in ir_joints}
    root_body = ""
    for b in ir_bodies:
        if b.name != "world" and b.name not in child_set:
            root_body = b.name
            break
    if not root_body:
        root_body = "world"

    name = getattr(spec, "modelname", "") or Path(source).stem

    meta: dict = {}
    if resolver is not None:
        meta["asset_resolver"] = resolver

    return IRModel(
        name=name,
        bodies=ir_bodies,
        joints=ir_joints,
        frames=ir_frames,
        root_body=root_body,
        meta=meta,
    )
