"""Spatial algebra on tensors: cross products of twists, and frame changes of twists, wrenches and inertias.

A twist (motion) and a wrench (force) are ``(..., 6)`` with the linear part first, the order of the ``lie``
tangents; an inertia is packed ``(..., 10)`` as in ``spatial.inertia``. A pose ``(..., 7)`` ``[tx, ty, tz, qx, qy,
qz, qw]`` places a frame B in a frame A, and each ``transform_*`` re-expresses a quantity given in B in A. Leading
axes broadcast, dtype and device are preserved, and no input is checked: dynamics calls these per joint.
"""

from __future__ import annotations

import torch

from ..lie import so3
from .inertia import _pack, _rotational_inertia


def cross_motion(twist: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    """Motion cross product ``twist ×m motion = ad(twist) @ motion``. ``(..., 6), (..., 6) → (..., 6)``.

    For ``twist = [v, w]`` and ``motion = [u, z]`` it is ``[w × u + v × z, w × z]``: the rate of change of a motion
    fixed in a frame that moves with ``twist``.
    """
    twist, motion = torch.broadcast_tensors(twist, motion)
    linear, angular = twist[..., :3], twist[..., 3:]
    return torch.cat(
        (
            torch.linalg.cross(angular, motion[..., :3]) + torch.linalg.cross(linear, motion[..., 3:]),
            torch.linalg.cross(angular, motion[..., 3:]),
        ),
        dim=-1,
    )


def cross_force(twist: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
    """Force cross product ``twist ×f force = -ad(twist)^T @ force``. ``(..., 6), (..., 6) → (..., 6)``.

    For ``twist = [v, w]`` and ``force = [f, n]`` it is ``[w × f, w × n + v × f]``, the dual of `cross_motion`:
    ``cross_force(twist, force) · motion == -force · cross_motion(twist, motion)``.
    """
    twist, force = torch.broadcast_tensors(twist, force)
    linear, angular = twist[..., :3], twist[..., 3:]
    return torch.cat(
        (
            torch.linalg.cross(angular, force[..., :3]),
            torch.linalg.cross(angular, force[..., 3:]) + torch.linalg.cross(linear, force[..., :3]),
        ),
        dim=-1,
    )


def transform_motion(pose: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    """Twist ``motion (..., 6)`` given in frame B, expressed in frame A: ``Ad(pose) @ motion``, ``(..., 6)``.

    ``pose (..., 7)`` is B's pose in A, with rotation ``R`` and translation ``p``; the result is
    ``[R v + p × (R w), R w]`` for ``motion = [v, w]``.
    """
    batch = torch.broadcast_shapes(pose.shape[:-1], motion.shape[:-1])
    pose, motion = pose.expand(*batch, 7), motion.expand(*batch, 6)
    rotation = pose[..., 3:7]
    angular = so3.act(rotation, motion[..., 3:])
    linear = so3.act(rotation, motion[..., :3]) + torch.linalg.cross(pose[..., :3], angular)
    return torch.cat((linear, angular), dim=-1)


def transform_force(pose: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
    """Wrench ``force (..., 6)`` given in frame B, expressed in frame A: ``Ad(pose)^-T @ force``, ``(..., 6)``.

    ``pose (..., 7)`` is B's pose in A, with rotation ``R`` and translation ``p``; the result is
    ``[R f, R n + p × (R f)]`` for ``force = [f, n]``, so the power ``force · twist`` is the same in both frames.
    """
    batch = torch.broadcast_shapes(pose.shape[:-1], force.shape[:-1])
    pose, force = pose.expand(*batch, 7), force.expand(*batch, 6)
    rotation = pose[..., 3:7]
    linear = so3.act(rotation, force[..., :3])
    angular = so3.act(rotation, force[..., 3:]) + torch.linalg.cross(pose[..., :3], linear)
    return torch.cat((linear, angular), dim=-1)


def transform_inertia(pose: torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
    """Packed inertia ``inertia (..., 10)`` given in frame B, expressed in frame A, ``(..., 10)``.

    ``pose (..., 7)`` is B's pose in A, with rotation ``R`` and translation ``p``. The mass stays, the centre of
    mass maps to ``R c + p`` and the rotational inertia to ``R I_c R^T``, so ``inertia_matrix`` of the result is
    ``Ad(pose)^-T @ inertia_matrix(inertia) @ Ad(pose)^-1``.
    """
    batch = torch.broadcast_shapes(pose.shape[:-1], inertia.shape[:-1])
    pose, inertia = pose.expand(*batch, 7), inertia.expand(*batch, 10)
    rotation = so3.to_matrix(pose[..., 3:7])
    com = (rotation @ inertia[..., 1:4, None]).squeeze(-1) + pose[..., :3]
    return _pack(inertia[..., 0], com, rotation @ _rotational_inertia(inertia) @ rotation.mT)
