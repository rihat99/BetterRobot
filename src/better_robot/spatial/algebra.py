"""Spatial algebra on tensors: cross products of twists, and frame changes of twists, wrenches and inertias.

A twist (motion) and a wrench (force) are ``(..., 6)`` with the linear part first, the order of the ``lie``
tangents; an inertia is packed ``(..., 10)`` as in ``spatial.inertia``. A pose ``(..., 7)`` ``[tx, ty, tz, qx, qy,
qz, qw]`` places a frame B in a frame A, and each ``transform_*`` re-expresses a quantity given in B in A. Leading
axes broadcast and dtype and device are preserved. Each public function checks its inputs' types, trailing shapes,
dtypes and devices (never their values). The cross products call unchecked ``*_raw`` twins, which dynamics calls
per joint after checking its own inputs once; dynamics changes frames with rotation matrices instead of the
``transform_*`` (``src/better_robot/dynamics/_world.py``), so those have no twin.
"""

from __future__ import annotations

import torch

from ..lie import so3
from ..lie._checks import broadcast_batch, check_tensors
from .inertia import inertia_from_mass_com_raw, rotational_inertia_raw


def cross_motion(twist: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Motion cross product of two twists, ``twist ×m other = ad(twist) @ other``. ``(..., 6), (..., 6) → (..., 6)``.

    For ``twist = [v, w]`` and ``other = [u, z]`` it is ``[w × u + v × z, w × z]``: the rate of change of a twist
    fixed in a frame that moves with ``twist``. Raises ``TypeError`` for a non-tensor, ``ShapeError`` for a
    trailing size other than 6, ``DtypeMismatchError`` for a non-floating or mismatched dtype and
    ``DeviceMismatchError`` for mismatched devices.
    """
    check_tensors(twist=(twist, (6,)), other=(other, (6,)))
    return cross_motion_raw(*torch.broadcast_tensors(twist, other))


def cross_motion_raw(twist: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """`cross_motion` without input checks, for checked inputs with as many axes: ``(..., 6), (..., 6) → (..., 6)``.

    ``torch.linalg.cross`` broadcasts sizes but not a missing axis; `cross_motion` aligns the axes first.
    """
    linear, angular = twist[..., :3], twist[..., 3:]
    other_linear, other_angular = other[..., :3], other[..., 3:]
    return torch.cat(
        (
            torch.linalg.cross(angular, other_linear) + torch.linalg.cross(linear, other_angular),
            torch.linalg.cross(angular, other_angular),
        ),
        dim=-1,
    )


def cross_force(twist: torch.Tensor, wrench: torch.Tensor) -> torch.Tensor:
    """Force cross product ``twist ×f wrench = -ad(twist)^T @ wrench``. ``(..., 6), (..., 6) → (..., 6)``.

    For ``twist = [v, w]`` and ``wrench = [f, n]`` it is ``[w × f, w × n + v × f]``, the dual of `cross_motion`:
    ``cross_force(twist, wrench) · other == -wrench · cross_motion(twist, other)``. Raises as `cross_motion`.
    """
    check_tensors(twist=(twist, (6,)), wrench=(wrench, (6,)))
    return cross_force_raw(*torch.broadcast_tensors(twist, wrench))


def cross_force_raw(twist: torch.Tensor, wrench: torch.Tensor) -> torch.Tensor:
    """`cross_force` without input checks, for checked inputs with as many axes: ``(..., 6), (..., 6) → (..., 6)``."""
    linear, angular = twist[..., :3], twist[..., 3:]
    force, torque = wrench[..., :3], wrench[..., 3:]
    return torch.cat(
        (
            torch.linalg.cross(angular, force),
            torch.linalg.cross(angular, torque) + torch.linalg.cross(linear, force),
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
    batch = broadcast_batch(pose.shape[:-1], twist.shape[:-1])
    pose, twist = pose.expand(*batch, 7), twist.expand(*batch, 6)
    rotation = pose[..., 3:7]
    angular = so3.act_raw(rotation, twist[..., 3:])
    linear = so3.act_raw(rotation, twist[..., :3]) + torch.linalg.cross(pose[..., :3], angular)
    return torch.cat((linear, angular), dim=-1)


def transform_force(pose: torch.Tensor, wrench: torch.Tensor) -> torch.Tensor:
    """Wrench ``wrench (..., 6)`` given in frame B, expressed in frame A: ``Ad(pose)^-T @ wrench``, ``(..., 6)``.

    ``pose (..., 7)`` is B's pose in A, with rotation ``R`` and translation ``p``; the result is
    ``[R f, R n + p × (R f)]`` for ``wrench = [f, n]``, so the power ``wrench · twist`` is the same in both frames.
    Raises as `transform_motion`.
    """
    check_tensors(pose=(pose, (7,)), wrench=(wrench, (6,)))
    batch = broadcast_batch(pose.shape[:-1], wrench.shape[:-1])
    pose, wrench = pose.expand(*batch, 7), wrench.expand(*batch, 6)
    rotation = pose[..., 3:7]
    linear = so3.act_raw(rotation, wrench[..., :3])
    angular = so3.act_raw(rotation, wrench[..., 3:]) + torch.linalg.cross(pose[..., :3], linear)
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
    batch = broadcast_batch(pose.shape[:-1], inertia.shape[:-1])
    pose, inertia = pose.expand(*batch, 7), inertia.expand(*batch, 10)
    rotation = so3.to_matrix_raw(pose[..., 3:7])
    com = (rotation @ inertia[..., 1:4, None]).squeeze(-1) + pose[..., :3]
    return inertia_from_mass_com_raw(
        inertia[..., 0], com, rotational_inertia=rotation @ rotational_inertia_raw(inertia) @ rotation.mT
    )
