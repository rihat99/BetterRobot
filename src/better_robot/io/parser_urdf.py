"""URDF parser: a typed adapter over the locked ``yourdfpy`` (record 0047).

``yourdfpy`` reads the XML into plain records (``Robot``, ``Link``, ``Joint``, ``Visual``, ``Collision``) and this
module reads those records by direct attribute access, so a yourdfpy upgrade that renames an attribute fails here
with an ``AttributeError`` and never silently with a default. Every failure a file can cause is a ``ModelError``
naming the file and the element. Mesh files are never opened: a mesh geometry keeps the ``filename`` the file
wrote (record 0048) and the viewer resolves it. A joint axis is normalised here (a file may carry one of any
length); the joint kinds accept unit axes only. The free-flyer root is not added here:
``build_model(root_joint=JointFreeFlyer())`` adds it. ``yourdfpy`` is imported here only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..exceptions import ModelError
from ..lie import so3
from .assets import FilesystemResolver
from .ir import IRBody, IRGeom, IRJoint, IRModel

if TYPE_CHECKING:
    import yourdfpy

#: URDF joint ``type`` to the IR kind.
_KINDS = {
    "revolute": "revolute",
    "continuous": "continuous",
    "prismatic": "prismatic",
    "fixed": "fixed",
    "floating": "free_flyer",
    "planar": "planar",
}


def _pose(origin: np.ndarray | None) -> torch.Tensor:
    """A URDF ``<origin>`` as a ``(4, 4)`` matrix, or ``None`` for identity, as one pose ``(7,)`` ``[t, qx, qy, qz, qw]``."""
    if origin is None:
        return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    matrix = torch.tensor(origin, dtype=torch.float64)
    return torch.cat([matrix[:3, 3], so3.from_matrix(matrix[:3, :3])])


def _color(
    material: yourdfpy.Material | None, materials: dict[str, yourdfpy.Material]
) -> tuple[float, float, float, float] | None:
    """The RGBA of a visual's material: inline, else the robot-level ``<material>`` of that name, else ``None``."""
    if material is None:
        return None
    if material.color is None:
        material = materials.get(material.name)
        if material is None or material.color is None:
            return None
    red, green, blue, alpha = (float(value) for value in material.color.rgba)
    return red, green, blue, alpha


def _geom(
    geometry: yourdfpy.Geometry, origin: np.ndarray | None, rgba: tuple[float, float, float, float] | None
) -> IRGeom:
    """One ``<visual>`` or ``<collision>`` as an IR primitive placed in the link frame (yourdfpy rejects any other shape)."""
    if geometry.sphere is not None:
        kind, params = "sphere", {"radius": float(geometry.sphere.radius)}
    elif geometry.box is not None:
        kind, params = "box", {"size": [float(value) for value in geometry.box.size]}
    elif geometry.cylinder is not None:
        kind, params = (
            "cylinder",
            {"radius": float(geometry.cylinder.radius), "length": float(geometry.cylinder.length)},
        )
    else:
        scale = geometry.mesh.scale  # None, one float for all axes, or (3,)
        scale = np.ones(3) if scale is None else np.broadcast_to(np.asarray(scale, dtype=np.float64), (3,))
        kind, params = "mesh", {"path": geometry.mesh.filename, "scale": scale.tolist()}
    return IRGeom(kind=kind, params=params, origin=_pose(origin), rgba=rgba)


def _body(link: yourdfpy.Link, materials: dict[str, yourdfpy.Material]) -> IRBody:
    """A ``<link>`` as an IR body: mass, centre of mass and inertia in the link frame, visual and collision geoms."""
    inertial = link.inertial
    mass = 0.0 if inertial is None or inertial.mass is None else float(inertial.mass)
    origin = np.eye(4) if inertial is None or inertial.origin is None else inertial.origin
    inertia = (
        np.zeros((3, 3))
        if inertial is None or inertial.inertia is None
        else np.asarray(inertial.inertia, dtype=np.float64)
    )
    # The file gives the inertia in the inertial frame; the IR wants it in the link frame about the centre of mass.
    rotation = origin[:3, :3]
    return IRBody(
        name=link.name,
        mass=mass,
        com=torch.tensor(origin[:3, 3], dtype=torch.float64),
        inertia=torch.tensor(rotation @ inertia @ rotation.T, dtype=torch.float64),
        visual_geoms=[_geom(v.geometry, v.origin, _color(v.material, materials)) for v in link.visuals],
        collision_geoms=[_geom(c.geometry, c.origin, None) for c in link.collisions],
    )


def _joint(joint: yourdfpy.Joint, links: set[str], path: Path) -> IRJoint:
    """A ``<joint>`` as an IR joint; its type, its parent and child links and its axis are checked here."""
    element = f"{path}: <joint name={joint.name!r}>"
    if joint.type not in _KINDS:
        raise ModelError(f"{element}: type {joint.type!r} is not one of {sorted(_KINDS)}")
    for role, link in (("parent", joint.parent), ("child", joint.child)):
        if link not in links:
            raise ModelError(f"{element}: {role} link {link!r} is not a <link> of the file")
    axis = np.asarray(joint.axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ModelError(f"{element}: <axis> is zero")
    limit, dynamics, mimic = joint.limit, joint.dynamics, joint.mimic
    lower, upper, velocity, effort = (
        (None, None, None, None) if limit is None else (limit.lower, limit.upper, limit.velocity, limit.effort)
    )
    # yourdfpy 0.0.60 keeps the <dynamics> attributes as the strings it read (its other numbers are floats).
    damping, friction = (
        (0.0, 0.0) if dynamics is None else (float(dynamics.damping or 0.0), float(dynamics.friction or 0.0))
    )
    source, multiplier, offset = (
        (None, 1.0, 0.0)
        if mimic is None
        else (
            mimic.joint,
            1.0 if mimic.multiplier is None else mimic.multiplier,
            0.0 if mimic.offset is None else mimic.offset,
        )
    )
    return IRJoint(
        name=joint.name,
        parent_body=joint.parent,
        child_body=joint.child,
        kind=_KINDS[joint.type],
        axis=torch.tensor(axis / norm),
        origin=_pose(joint.origin),
        lower=lower,
        upper=upper,
        velocity_limit=velocity,
        effort_limit=effort,
        damping=damping,
        friction=friction,
        mimic_source=source,
        mimic_multiplier=multiplier,
        mimic_offset=offset,
    )


def parse_urdf(path: str | Path) -> IRModel:
    """Parse a URDF file into an ``IRModel``; raises ``ModelError`` naming the file and the element at fault."""
    import yourdfpy  # noqa: PLC0415 - 0.5 s of import (it pulls trimesh), paid by the loader only

    path = Path(path)
    if not path.is_file():
        raise ModelError(f"{path}: no such file")
    try:
        urdf = yourdfpy.URDF.load(
            str(path),
            build_scene_graph=False,
            build_collision_scene_graph=False,
            load_meshes=False,
            load_collision_meshes=False,
        )
    except (yourdfpy.URDFError, SyntaxError, TypeError, ValueError, IndexError) as exc:
        # yourdfpy 0.0.60 fails on a broken file with whatever builtin its parse stumbles on: lxml's XMLSyntaxError
        # for an empty file, TypeError for no root element or a <visual> without <geometry>, ValueError for an
        # unknown tag such as <capsule>, IndexError for an empty <geometry>.
        raise ModelError(f"{path}: {exc}") from exc
    robot = urdf.robot
    materials = {material.name: material for material in robot.materials}
    bodies = [_body(link, materials) for link in robot.links]
    links = {link.name for link in robot.links}
    joints = [_joint(joint, links, path) for joint in robot.joints]
    children = {joint.child_body for joint in joints}
    root_body = next((link.name for link in robot.links if link.name not in children), "")
    return IRModel(
        name=robot.name,
        bodies=bodies,
        joints=joints,
        root_body=root_body,
        meta={"asset_resolver": FilesystemResolver(base_path=path.parent)},
    )
