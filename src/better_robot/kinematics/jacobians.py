"""Jacobians of joints, frames and points, and their time variation, read from a `KinematicState`.

Every Jacobian masks the state's shared world column table (`KinematicState.jacobian_columns`) by the support of
the joint that owns the frame, reduces it to the public ``nv`` (mimic joints) and re-expresses the world Jacobian
at the frame's origin and axes. Three references: ``"world"`` (velocity at the world origin), ``"local_world_aligned"``
(velocity of the frame origin in world axes, the default) and ``"local"`` (the body-frame twist).

The time variation ``J̇`` pairs with ``a = J v̇ + J̇ v``: the spatial acceleration for ``"world"`` and ``"local"``, the
classical (point) acceleration for ``"local_world_aligned"``, whose basis point moves with the frame. It is
transport-only, since every shipped joint has a configuration-constant motion subspace: each full-width column
``c`` owned by joint ``a`` drifts as ``dJ[:, c] = ad(ov_a) J[:, c]`` with ``ov_a = J_a v`` the world spatial
velocity of ``a``, evaluated column-wise on the same table, masked per joint and mimic-reduced last. The closed
forms for the other two references are checked against Pinocchio (``tests/parity/pinocchio``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from ..exceptions import ArgumentError
from ..model._checks import check_tensor, expand_batch
from ..model.model import Model
from ._plans import jacobian_plan
from .configuration import reduced_columns
from .forward_kinematics import KinematicState

_Reference = Literal["world", "local", "local_world_aligned"]


def frame_jacobian(
    model: Model, state: KinematicState, *, frame: int | str, reference: _Reference = "local_world_aligned"
) -> torch.Tensor:
    """Jacobian of one frame (id or name), ``(B..., 6, nv)`` with rows ``[linear, angular]``."""
    frame_id = model.frame_id(frame)
    world = _world_jacobian(model, state, model.frame_parent_joint_ids[frame_id])
    return _express(
        world, state.frame_translations[..., frame_id, :], state.frame_rotations[..., frame_id, :, :], reference
    )


def joint_jacobian(
    model: Model, state: KinematicState, *, joint: int | str, reference: _Reference = "local_world_aligned"
) -> torch.Tensor:
    """Jacobian of one joint (id or name), ``(B..., 6, nv)`` with rows ``[linear, angular]``; references as `frame_jacobian`."""
    joint = model.joint_id(joint)
    world = _world_jacobian(model, state, joint)
    return _express(world, state.joint_translations[..., joint, :], state.joint_rotations[..., joint, :, :], reference)


def point_jacobian(
    model: Model, state: KinematicState, *, frames: Sequence[int | str], points: torch.Tensor
) -> torch.Tensor:
    """Jacobian of points fixed to frames, ``(B..., K, 3, nv)``: the world velocity of each point.

    Point ``k`` sits at ``points[..., k, :]`` in the coordinates of ``frames[k]`` (id or name); ``points`` is
    ``(K, 3)`` or ``(B..., K, 3)``. The rows are the linear rows of a ``local_world_aligned`` frame Jacobian moved to
    the point. ``J.mT @ f`` maps world forces ``f: (B..., K, 3, 1)`` at the points to one generalized force per point,
    ``(B..., K, nv, 1)``; sum over ``K`` (``.sum(-3)``) for the generalized force of them all.
    """
    if not frames:
        raise ArgumentError("frames is empty; point_jacobian needs at least one frame")
    check_tensor("points", points, shape=(len(frames), 3), dtype=model.dtype, device=model.device)
    frame_ids = [model.frame_id(frame) for frame in frames]
    joints = [model.frame_parent_joint_ids[frame_id] for frame_id in frame_ids]
    owners = sorted(set(joints))
    world = torch.stack([_world_jacobian(model, state, joint) for joint in owners], dim=-3)  # (B..., U, 6, nv)
    world = world[..., [owners.index(joint) for joint in joints], :, :]
    transform = state.frame_transforms[..., frame_ids, :, :]  # (B..., K, 3, 4)
    position = (transform[..., :3] @ points.unsqueeze(-1)).squeeze(-1) + transform[..., 3]
    return world[..., :3, :] - torch.linalg.cross(position.unsqueeze(-1), world[..., 3:, :], dim=-2)


def frame_jacobian_time_variation(
    model: Model,
    state: KinematicState,
    *,
    v: torch.Tensor,
    frame: int | str,
    reference: _Reference = "local_world_aligned",
) -> torch.Tensor:
    """Time derivative ``dJ/dt`` of `frame_jacobian` along ``v: (B..., nv)``, ``(B..., 6, nv)``.

    ``a = J v̇ + J̇ v`` is the spatial acceleration for ``"world"`` and ``"local"``, the classical acceleration of the
    frame origin for ``"local_world_aligned"``.
    """
    v = _tangent(model, state, v)
    frame_id = model.frame_id(frame)
    jacobian, variation = _world_time_variation(model, state, v, model.frame_parent_joint_ids[frame_id])
    translation, rotation = state.frame_translations[..., frame_id, :], state.frame_rotations[..., frame_id, :, :]
    return _express_time_variation(jacobian, variation, translation, rotation, v, reference)


def joint_jacobian_time_variation(
    model: Model,
    state: KinematicState,
    *,
    v: torch.Tensor,
    joint: int | str,
    reference: _Reference = "local_world_aligned",
) -> torch.Tensor:
    """Time derivative ``dJ/dt`` of `joint_jacobian` along ``v: (B..., nv)``, ``(B..., 6, nv)``; pairing as `frame_jacobian_time_variation`."""
    v = _tangent(model, state, v)
    joint = model.joint_id(joint)
    jacobian, variation = _world_time_variation(model, state, v, joint)
    translation, rotation = state.joint_translations[..., joint, :], state.joint_rotations[..., joint, :, :]
    return _express_time_variation(jacobian, variation, translation, rotation, v, reference)


def _tangent(model: Model, state: KinematicState, v: torch.Tensor) -> torch.Tensor:
    """``v`` checked and broadcast to the state's batch, ``(B..., nv)``."""
    check_tensor("v", v, shape=(model.nv,), dtype=model.dtype, device=model.device)
    return expand_batch("v", v, tuple(state.q.shape[:-1]), (model.nv,))


def _world_jacobian(model: Model, state: KinematicState, joint: int) -> torch.Tensor:
    """World Jacobian of ``joint``, ``(B..., 6, nv)``: the shared column table masked by its support and reduced."""
    plan = jacobian_plan(model, dtype=state.q.dtype, device=state.q.device)
    return reduced_columns(model, state.jacobian_columns * plan.support_mask[joint])


def _world_time_variation(
    model: Model, state: KinematicState, v: torch.Tensor, joint: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """World ``J`` and ``J̇`` of ``joint``, ``(B..., 6, nv)`` each, by transport of the shared column table."""
    plan = jacobian_plan(model, dtype=state.q.dtype, device=state.q.device)
    columns = state.jacobian_columns  # (B..., 6, nv_full)
    jacobians = reduced_columns(model, columns.unsqueeze(-3) * plan.support_mask)  # (B..., njoints, 6, nv)
    velocities = torch.einsum("...jck,...k->...jc", jacobians, v)  # world spatial velocity of every joint
    # Each full-width column drifts with its owner's twist: ad([u; w]) = [[ŵ, û], [0, ŵ]], one cross per block.
    owner = velocities.index_select(-2, plan.column_owner)  # (B..., nv_full, 6)
    column = columns.transpose(-1, -2)  # (B..., nv_full, 6)
    linear = torch.linalg.cross(owner[..., 3:], column[..., :3]) + torch.linalg.cross(owner[..., :3], column[..., 3:])
    angular = torch.linalg.cross(owner[..., 3:], column[..., 3:])
    variation = torch.cat((linear, angular), dim=-1).transpose(-1, -2)  # (B..., 6, nv_full)
    return jacobians[..., joint, :, :], reduced_columns(model, variation * plan.support_mask[joint])


def _express(world: torch.Tensor, translation: torch.Tensor, rotation: torch.Tensor, reference: str) -> torch.Tensor:
    """Re-express a world Jacobian ``(B..., 6, nv)`` at a point ``translation (B..., 3)`` with axes ``rotation``."""
    if reference == "world":
        return world
    angular = world[..., 3:, :]
    aligned = torch.cat(
        (world[..., :3, :] - torch.linalg.cross(translation.unsqueeze(-1), angular, dim=-2), angular), dim=-2
    )
    if reference == "local_world_aligned":
        return aligned
    if reference == "local":
        rotation = rotation.mT
        return torch.cat((rotation @ aligned[..., :3, :], rotation @ aligned[..., 3:, :]), dim=-2)
    raise ArgumentError(f"Unsupported reference frame: {reference!r}")


def _express_time_variation(
    jacobian: torch.Tensor,
    variation: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
    v: torch.Tensor,
    reference: str,
) -> torch.Tensor:
    """``J̇`` in ``reference`` from the world ``J`` and ``J̇`` ``(B..., 6, nv)`` of a frame at ``translation`` with axes ``rotation``."""
    if reference == "world":
        return variation
    velocity = (jacobian @ v.unsqueeze(-1)).squeeze(-1)  # (B..., 6) world spatial velocity of the frame
    if reference == "local_world_aligned":
        # The basis point moves with the frame origin, so its world velocity enters the linear rows; without this
        # term (a static translation of J̇) the error is order one, not roundoff.
        point_velocity = velocity[..., :3] + torch.linalg.cross(velocity[..., 3:], translation)
        linear = (
            variation[..., :3, :]
            - torch.linalg.cross(translation.unsqueeze(-1), variation[..., 3:, :], dim=-2)
            - torch.linalg.cross(point_velocity.unsqueeze(-1), jacobian[..., 3:, :], dim=-2)
        )
        return torch.cat((linear, variation[..., 3:, :]), dim=-2)
    if reference == "local":
        # J̇_local = Ad⁻¹ J̇ − ad(v_local) (Ad⁻¹ J), with v_local = Ad⁻¹ ov the frame's own twist.
        local = _express(jacobian, translation, rotation, "local")
        twist = _express(velocity.unsqueeze(-1), translation, rotation, "local")  # (B..., 6, 1)
        linear_twist, angular_twist = twist[..., :3, :], twist[..., 3:, :]
        correction = torch.cat(
            (
                torch.linalg.cross(angular_twist, local[..., :3, :], dim=-2)
                + torch.linalg.cross(linear_twist, local[..., 3:, :], dim=-2),
                torch.linalg.cross(angular_twist, local[..., 3:, :], dim=-2),
            ),
            dim=-2,
        )
        return _express(variation, translation, rotation, "local") - correction
    raise ArgumentError(f"Unsupported reference frame: {reference!r}")


__all__ = [
    "frame_jacobian",
    "frame_jacobian_time_variation",
    "joint_jacobian",
    "joint_jacobian_time_variation",
    "point_jacobian",
]
