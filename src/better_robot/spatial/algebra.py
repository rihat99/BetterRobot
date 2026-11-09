"""Spatial algebra on tensors: cross products of twists, and frame changes of twists, wrenches and inertias.

A twist (motion) and a wrench (force) are ``(..., 6)`` with the linear part first, the order of the ``lie``
tangents; an inertia is packed ``(..., 10)`` as in ``spatial.inertia``. A pose ``(..., 7)`` ``[tx, ty, tz, qx, qy,
qz, qw]`` places a frame B in a frame A, and each ``transform_*`` re-expresses a quantity given in B in A. Leading
axes broadcast and dtype and device are preserved. Each public function checks its inputs' types, trailing shapes,
dtypes and devices (never their values) and calls its unchecked ``*_raw`` twin, which dynamics calls per joint
after checking its own inputs once.
"""

from __future__ import annotations

import torch

from ..lie import so3
from ..lie._checks import check_tensors
from .inertia import inertia_from_mass_com_raw, rotational_inertia_raw


def cross_motion(twist: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Motion cross product of two twists, ``twist ×m other = ad(twist) @ other``. ``(..., 6), (..., 6) → (..., 6)``.

    For ``twist = [v, w]`` and ``other = [u, z]`` it is ``[w × u + v × z, w × z]``: the rate of change of a twist
    fixed in a frame that moves with ``twist``. Raises ``TypeError`` for a non-tensor, ``ShapeError`` for a
    trailing size other than 6, ``DtypeMismatchError`` for a non-floating or mismatched dtype and
    ``DeviceMismatchError`` for mismatched devices.
    """
    check_tensors(twist=(twist, (6,)), other=(other, (6,)))
    return cross_motion_raw(twist, other)


def cross_motion_raw(twist: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """`cross_motion` without input checks, for checked inputs: ``(..., 6), (..., 6) → (..., 6)``."""
    twist, other = torch.broadcast_tensors(twist, other)
    linear, angular = twist[..., :3], twist[..., 3:]
    return torch.cat(
        (
            torch.linalg.cross(angular, other[..., :3]) + torch.linalg.cross(linear, other[..., 3:]),
            torch.linalg.cross(angular, other[..., 3:]),
        ),
        dim=-1,
    )


def cross_force(twist: torch.Tensor, wrench: torch.Tensor) -> torch.Tensor:
    """Force cross product ``twist ×f wrench = -ad(twist)^T @ wrench``. ``(..., 6), (..., 6) → (..., 6)``.

    For ``twist = [v, w]`` and ``wrench = [f, n]`` it is ``[w × f, w × n + v × f]``, the dual of `cross_motion`:
    ``cross_force(twist, wrench) · other == -wrench · cross_motion(twist, other)``. Raises as `cross_motion`.
    """
    check_tensors(twist=(twist, (6,)), wrench=(wrench, (6,)))
    return cross_force_raw(twist, wrench)


def cross_force_raw(twist: torch.Tensor, wrench: torch.Tensor) -> torch.Tensor:
    """`cross_force` without input checks, for checked inputs: ``(..., 6), (..., 6) → (..., 6)``."""
    twist, wrench = torch.broadcast_tensors(twist, wrench)
    linear, angular = twist[..., :3], twist[..., 3:]
    return torch.cat(
        (
            torch.linalg.cross(angular, wrench[..., :3]),
            torch.linalg.cross(angular, wrench[..., 3:]) + torch.linalg.cross(linear, wrench[..., :3]),
        ),
        dim=-1,
    )


def transform_motion(pose: torch.Tensor, twist: torch.Tensor) -> torch.Tensor:
    """Twist ``twist (..., 6)`` given in frame B, expressed in frame A: ``Ad(pose) @ twist``, ``(..., 6)``.

    ``pose (..., 7)`` is B's pose in A, with rotation ``R`` and translation ``p``; the result is
    ``[R v + p × (R w), R w]`` for ``twist = [v, w]``. Raises ``TypeError`` for a non-tensor, ``ShapeError`` for
    a trailing size other than 7 and 6, ``DtypeMismatchError`` for a non-floating or mismatched dtype and
    ``DeviceMismatchError`` for mismatched devices.
    """
    check_tensors(pose=(pose, (7,)), twist=(twist, (6,)))
    return transform_motion_raw(pose, twist)


def transform_motion_raw(pose: torch.Tensor, twist: torch.Tensor) -> torch.Tensor:
    """`transform_motion` without input checks, for checked inputs: ``(..., 7), (..., 6) → (..., 6)``."""
    batch = torch.broadcast_shapes(pose.shape[:-1], twist.shape[:-1])
    pose, twist = pose.expand(*batch, 7), twist.expand(*batch, 6)
    rotation = pose[..., 3:7]
    angular = so3.act(rotation, twist[..., 3:])
    linear = so3.act(rotation, twist[..., :3]) + torch.linalg.cross(pose[..., :3], angular)
    return torch.cat((linear, angular), dim=-1)


def transform_force(pose: torch.Tensor, wrench: torch.Tensor) -> torch.Tensor:
    """Wrench ``wrench (..., 6)`` given in frame B, expressed in frame A: ``Ad(pose)^-T @ wrench``, ``(..., 6)``.

    ``pose (..., 7)`` is B's pose in A, with rotation ``R`` and translation ``p``; the result is
    ``[R f, R n + p × (R f)]`` for ``wrench = [f, n]``, so the power ``wrench · twist`` is the same in both frames.
    Raises as `transform_motion`.
    """
    check_tensors(pose=(pose, (7,)), wrench=(wrench, (6,)))
    return transform_force_raw(pose, wrench)


def transform_force_raw(pose: torch.Tensor, wrench: torch.Tensor) -> torch.Tensor:
    """`transform_force` without input checks, for checked inputs: ``(..., 7), (..., 6) → (..., 6)``."""
    batch = torch.broadcast_shapes(pose.shape[:-1], wrench.shape[:-1])
    pose, wrench = pose.expand(*batch, 7), wrench.expand(*batch, 6)
    rotation = pose[..., 3:7]
    linear = so3.act(rotation, wrench[..., :3])
    angular = so3.act(rotation, wrench[..., 3:]) + torch.linalg.cross(pose[..., :3], linear)
    return torch.cat((linear, angular), dim=-1)


def transform_inertia(pose: torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
    """Packed inertia ``inertia (..., 10)`` given in frame B, expressed in frame A, ``(..., 10)``.

    ``pose (..., 7)`` is B's pose in A, with rotation ``R`` and translation ``p``. The mass stays, the centre of
    mass maps to ``R c + p`` and the rotational inertia to ``R I_c R^T``, so ``inertia_matrix`` of the result is
    ``Ad(pose)^-T @ inertia_matrix(inertia) @ Ad(pose)^-1``. Raises ``TypeError`` for a non-tensor,
    ``ShapeError`` for a trailing size other than 7 and 10, ``DtypeMismatchError`` for a non-floating or mismatched
    dtype and ``DeviceMismatchError`` for mismatched devices.
    """
    check_tensors(pose=(pose, (7,)), inertia=(inertia, (10,)))
    return transform_inertia_raw(pose, inertia)


def transform_inertia_raw(pose: torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
    """`transform_inertia` without input checks, for checked inputs: ``(..., 7), (..., 10) → (..., 10)``."""
    batch = torch.broadcast_shapes(pose.shape[:-1], inertia.shape[:-1])
    pose, inertia = pose.expand(*batch, 7), inertia.expand(*batch, 10)
    rotation = so3.to_matrix(pose[..., 3:7])
    com = (rotation @ inertia[..., 1:4, None]).squeeze(-1) + pose[..., :3]
    return inertia_from_mass_com_raw(
        inertia[..., 0], com, rotational_inertia=rotation @ rotational_inertia_raw(inertia) @ rotation.mT
    )
