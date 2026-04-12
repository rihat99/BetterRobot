"""URDF parser — ``yourdfpy``-backed, emits an ``IRModel``.

The ``yourdfpy`` import is confined to this file. Mimic joints, continuous
joints, and multi-child links are handled at IR level. Free-flyer is not
added here — ``build_model(root_joint=JointFreeFlyer())`` adds it.

See ``docs/04_PARSERS.md §4``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..ir import IRBody, IRFrame, IRGeom, IRJoint, IRModel


def _mat4_to_se3(T: np.ndarray) -> torch.Tensor:
    """Convert a 4×4 homogeneous numpy matrix to ``[tx,ty,tz,qx,qy,qz,qw]``."""
    from ...lie.so3 import from_matrix as _so3_from_matrix

    t = torch.tensor(T[:3, 3].astype(np.float32))
    R = torch.from_numpy(T[:3, :3].astype(np.float32))
    q = _so3_from_matrix(R)  # [qx, qy, qz, qw]
    return torch.cat([t, q])


def _extract_geoms(
    geom_list: list | None,
    origin_fn=None,
    filename_handler=None,
    material_map: dict | None = None,
) -> list[IRGeom]:
    """Extract visual / collision geometries from a list of yourdfpy geometry objects."""
    result: list[IRGeom] = []
    if geom_list is None:
        return result
    for geom in geom_list:
        origin: np.ndarray | None = getattr(geom, "origin", None)
        if origin is not None:
            placement = _mat4_to_se3(origin)
        else:
            placement = torch.tensor([0., 0., 0., 0., 0., 0., 1.])

        rgba = None
        if hasattr(geom, "material") and geom.material is not None:
            mat = geom.material
            # Try inline color first
            c = getattr(mat, "color", None)
            if c is not None:
                rgba_arr = getattr(c, "rgba", None)
                if rgba_arr is not None:
                    rgba = tuple(float(x) for x in rgba_arr[:4])
            # Fall back to material name reference
            if rgba is None and material_map is not None:
                mat_name = getattr(mat, "name", None)
                if mat_name and mat_name in material_map:
                    ref = material_map[mat_name]
                    c = getattr(ref, "color", None)
                    if c is not None:
                        rgba_arr = getattr(c, "rgba", None)
                        if rgba_arr is not None:
                            rgba = tuple(float(x) for x in rgba_arr[:4])

        g = getattr(geom, "geometry", None)
        if g is None:
            continue

        sphere = getattr(g, "sphere", None)
        box = getattr(g, "box", None)
        cylinder = getattr(g, "cylinder", None)
        capsule = getattr(g, "capsule", None)
        mesh = getattr(g, "mesh", None)

        if sphere is not None:
            result.append(IRGeom(
                kind="sphere",
                params={"radius": float(sphere.radius)},
                origin=placement,
                rgba=rgba,
            ))
        elif box is not None:
            size = np.array(box.size, dtype=np.float32)
            result.append(IRGeom(
                kind="box",
                params={"size": size.tolist()},
                origin=placement,
                rgba=rgba,
            ))
        elif cylinder is not None:
            result.append(IRGeom(
                kind="cylinder",
                params={"radius": float(cylinder.radius), "length": float(cylinder.length)},
                origin=placement,
                rgba=rgba,
            ))
        elif capsule is not None:
            result.append(IRGeom(
                kind="capsule",
                params={"radius": float(capsule.radius), "length": float(capsule.length)},
                origin=placement,
                rgba=rgba,
            ))
        elif mesh is not None:
            scale = getattr(mesh, "scale", None)
            scale_list = [float(s) for s in scale] if scale is not None else [1., 1., 1.]
            raw_path = getattr(mesh, "filename", "") or ""
            if filename_handler is not None and raw_path:
                try:
                    raw_path = filename_handler(raw_path)
                except Exception:
                    pass
            result.append(IRGeom(
                kind="mesh",
                params={"path": raw_path, "scale": scale_list},
                origin=placement,
                rgba=rgba,
            ))
    return result


def parse_urdf(source: str | Path | Any) -> IRModel:
    """Parse a URDF file, path, or ``yourdfpy.URDF`` object into an ``IRModel``.

    See docs/04_PARSERS.md §4.
    """
    import yourdfpy

    if isinstance(source, (str, Path)):
        urdf = yourdfpy.URDF.load(str(source))
    else:
        urdf = source

    # ── bodies (one per URDF link) ────────────────────────────────────
    ir_bodies: list[IRBody] = []
    for link_name, link in urdf.link_map.items():
        mass = 0.0
        com = torch.zeros(3)
        inertia = torch.zeros(3, 3)

        if link.inertial is not None:
            if link.inertial.mass is not None:
                mass = float(link.inertial.mass)
            if link.inertial.origin is not None:
                com = torch.tensor(
                    link.inertial.origin[:3, 3].astype(np.float32)
                )
            if link.inertial.inertia is not None:
                ii = link.inertial.inertia
                ixx = float(getattr(ii, "ixx", 0.0) or 0.0)
                iyy = float(getattr(ii, "iyy", 0.0) or 0.0)
                izz = float(getattr(ii, "izz", 0.0) or 0.0)
                ixy = float(getattr(ii, "ixy", 0.0) or 0.0)
                ixz = float(getattr(ii, "ixz", 0.0) or 0.0)
                iyz = float(getattr(ii, "iyz", 0.0) or 0.0)
                inertia = torch.tensor(
                    [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
                    dtype=torch.float32,
                )

        fn_handler = getattr(urdf, "_filename_handler", None)
        mat_map = getattr(urdf, "_material_map", None)
        visual_geoms = _extract_geoms(getattr(link, "visuals", None), filename_handler=fn_handler, material_map=mat_map)
        collision_geoms = _extract_geoms(getattr(link, "collisions", None), filename_handler=fn_handler)

        ir_bodies.append(IRBody(
            name=link_name,
            mass=mass,
            com=com,
            inertia=inertia,
            visual_geoms=visual_geoms,
            collision_geoms=collision_geoms,
        ))

    # ── joints (one per URDF joint) ───────────────────────────────────
    ir_joints: list[IRJoint] = []
    for jname, joint in urdf.joint_map.items():
        # axis
        if joint.axis is not None:
            axis_np = np.array(joint.axis, dtype=np.float32)
            norm = float(np.linalg.norm(axis_np))
            if norm > 1e-12:
                axis_np = axis_np / norm
            axis_t: torch.Tensor | None = torch.from_numpy(axis_np)
        else:
            axis_t = None

        # origin
        if joint.origin is not None:
            origin_t = _mat4_to_se3(joint.origin)
        else:
            origin_t = torch.tensor([0., 0., 0., 0., 0., 0., 1.])

        # limits
        lower: float | None = None
        upper: float | None = None
        vel_limit: float | None = None
        effort_limit: float | None = None
        if joint.limit is not None:
            lo = getattr(joint.limit, "lower", None)
            hi = getattr(joint.limit, "upper", None)
            vel = getattr(joint.limit, "velocity", None)
            eff = getattr(joint.limit, "effort", None)
            lower = float(lo) if lo is not None else None
            upper = float(hi) if hi is not None else None
            vel_limit = float(vel) if vel is not None else None
            effort_limit = float(eff) if eff is not None else None

        # mimic
        mimic_source: str | None = None
        mimic_mult = 1.0
        mimic_off = 0.0
        if getattr(joint, "mimic", None) is not None:
            mimic_source = getattr(joint.mimic, "joint", None)
            m = getattr(joint.mimic, "multiplier", None)
            o = getattr(joint.mimic, "offset", None)
            if m is not None:
                mimic_mult = float(m)
            if o is not None:
                mimic_off = float(o)

        ir_joints.append(IRJoint(
            name=jname,
            parent_body=joint.parent,
            child_body=joint.child,
            kind=joint.type,
            axis=axis_t,
            origin=origin_t,
            lower=lower,
            upper=upper,
            velocity_limit=vel_limit,
            effort_limit=effort_limit,
            mimic_source=mimic_source,
            mimic_multiplier=mimic_mult,
            mimic_offset=mimic_off,
        ))

    # ── root body ─────────────────────────────────────────────────────
    child_bodies = {j.child_body for j in ir_joints}
    root_body = ""
    for b in ir_bodies:
        if b.name not in child_bodies:
            root_body = b.name
            break

    # ── model name ────────────────────────────────────────────────────
    name = getattr(urdf, "name", "") or ""

    return IRModel(
        name=name,
        bodies=ir_bodies,
        joints=ir_joints,
        root_body=root_body,
    )
