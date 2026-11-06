"""Immutable forward-kinematics result and the functions that consume it.

``forward_kinematics_state`` returns a frozen `KinematicState` holding world transforms as ``[R | p]`` matrices;
``frame_jacobian`` reads it. Poses in ``[t, q]`` layout are derived on request.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch

from ..exceptions import ArgumentError, QuaternionNormError
from ..lie import so3
from ..model._checks import batch_shape, check_tensor, expand_batch
from ..model.joints import JointFreeFlyer
from ..model.model import Model
from ._fk import fk_plan, lean_core, placement_matrices
from ._jacobian import jacobian_plan, world_column_table_from_matrices
from .configuration import full_configuration, reduced_columns
from .jacobian import (
    _time_variation_in_reference,
    frame_jacobian_time_variation_raw,
    joint_jacobians_time_variation_raw,
)

__all__ = [
    "KinematicState",
    "forward_kinematics_state",
    "frame_jacobian",
    "frame_jacobian_time_variation",
    "joint_jacobian",
    "joint_jacobian_time_variation",
    "point_jacobian",
]

#: Tolerance used by the opt-in free-flyer quaternion debug check.
_QUAT_NORM_TOL = 0.1


def _validate_free_flyer_quaternion_norm(model: Model, q: torch.Tensor) -> None:
    """Debug-check the free-flyer quaternion norm.

    This check converts a tensor predicate to a Python ``bool`` and therefore synchronizes accelerators. It must
    only be called from an explicitly requested public-boundary check, never from tensor-only primitives.
    """
    if model.njoints >= 2 and isinstance(model.joints[1], JointFreeFlyer):
        # Free-flyer q layout: [tx, ty, tz, qx, qy, qz, qw]
        norm = q[..., 3:7].norm(dim=-1)
        if bool(((norm - 1.0).abs() > _QUAT_NORM_TOL).any()):  # bench-ok: opt-in public-boundary debug validation
            bad = float(norm.min()), float(norm.max())  # bench-ok: opt-in public-boundary debug validation
            raise QuaternionNormError(
                f"free-flyer quaternion norm outside [{1 - _QUAT_NORM_TOL}, {1 + _QUAT_NORM_TOL}] "
                f"(observed range {bad}). Normalise before passing."
            )


def _to_pose(transform: torch.Tensor) -> torch.Tensor:
    """``[R | p]: (..., 3, 4)`` to ``[x y z qx qy qz qw]: (..., 7)``."""
    return torch.cat([transform[..., 3], so3.from_matrix(transform[..., :3])], dim=-1)


@dataclass(frozen=True, eq=False)
class KinematicState:
    """World placements of every joint and frame at one configuration batch."""

    model: Model  #: The model the state was computed from.
    q: torch.Tensor  #: ``(B..., nq)`` configuration the state was computed at.
    joint_transform: torch.Tensor  #: ``(B..., njoints, 3, 4)`` world ``[R | p]`` of every joint.
    frame_transform: torch.Tensor  #: ``(B..., nframes, 3, 4)`` world ``[R | p]`` of every frame.
    # Memo of values derived from the fields above (poses, joint Jacobians), filled on first use; never copied.
    _memo: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def joint_translation(self) -> torch.Tensor:
        """``(B..., njoints, 3)`` world joint origins."""
        return self.joint_transform[..., 3]

    @property
    def joint_rotation(self) -> torch.Tensor:
        """``(B..., njoints, 3, 3)`` world joint rotation matrices."""
        return self.joint_transform[..., :3]

    @property
    def frame_translation(self) -> torch.Tensor:
        """``(B..., nframes, 3)`` world frame origins."""
        return self.frame_transform[..., 3]

    @property
    def frame_rotation(self) -> torch.Tensor:
        """``(B..., nframes, 3, 3)`` world frame rotation matrices."""
        return self.frame_transform[..., :3]

    @property
    def joint_pose(self) -> torch.Tensor:
        """``(B..., njoints, 7)`` world poses ``[x y z qx qy qz qw]``, converted on first use."""
        if "joint_pose" not in self._memo:
            self._memo["joint_pose"] = _to_pose(self.joint_transform)
        return self._memo["joint_pose"]

    @property
    def frame_pose(self) -> torch.Tensor:
        """``(B..., nframes, 7)`` world poses ``[x y z qx qy qz qw]``, converted on first use."""
        if "frame_pose" not in self._memo:
            self._memo["frame_pose"] = _to_pose(self.frame_transform)
        return self._memo["frame_pose"]

    def frame_pose_of(self, frame: int | str) -> torch.Tensor:
        """``(B..., 7)`` world pose of one frame (id or name), without converting the others."""
        return _to_pose(self.frame_transform[..., self.model.frame_id(frame), :, :])


def forward_kinematics_state(model: Model, q: torch.Tensor, *, check_quaternion_norm: bool = False) -> KinematicState:
    """Forward kinematics. ``q: (B..., nq)`` → `KinematicState` of joints and frames.

    ``check_quaternion_norm=True`` raises `QuaternionNormError` when a free-flyer quaternion is far from unit
    norm. It syncs accelerators, so it is off by default.
    """
    check_tensor("q", q, shape=(model.nq,), dtype=model.dtype, device=model.device)
    batch = batch_shape(model, "q", q, (model.nq,))
    q = expand_batch("q", q, batch, (model.nq,))
    if check_quaternion_norm:
        _validate_free_flyer_quaternion_norm(model, q)
    q_full = full_configuration(model, q)
    placements, frames = placement_matrices(model)
    count = math.prod(batch)  # explicit: a model without coordinates has zero-width q
    nj, nf = model.njoints, model.nframes
    # Unbatched model values broadcast inside the matmuls; batched ones are flattened with q.
    if placements.ndim > 3:
        placements = placements.broadcast_to(*batch, nj, 4, 4).reshape(count, nj, 4, 4)
    if frames.ndim > 3:
        frames = frames.broadcast_to(*batch, nf, 4, 4).reshape(count, nf, 4, 4)
    world, _ = lean_core(model, q_full.reshape(count, model.nq_full), placements)
    parents = world.index_select(1, fk_plan(model, dtype=placements.dtype, device=placements.device).frame_parents)
    # einsum is several times faster than a broadcast matmul on CPU when one side has no batch axis.
    frame_world = torch.einsum("nfab,fbc->nfac", parents, frames) if frames.ndim == 3 else parents @ frames
    return KinematicState(
        model=model,
        q=q,
        joint_transform=world.view(*batch, nj, 3, 4),
        frame_transform=frame_world.view(*batch, nf, 3, 4),
    )


def frame_jacobian(
    model: Model,
    state: KinematicState,
    *,
    frame: int | str,
    reference: Literal["world", "local", "local_world_aligned"] = "local_world_aligned",
) -> torch.Tensor:
    """Jacobian of one frame. Returns ``(B..., 6, nv)``, rows ``[linear, angular]``.

    The world column table shared by all frames is built once per state; each call masks it with the columns
    that support the frame's parent joint.
    """
    frame_id = model.frame_id(frame)
    return _express(
        _world_jacobian(model, state, model.frame_parent_joint_ids[frame_id]),
        state.frame_translation[..., frame_id, :],
        state.frame_rotation[..., frame_id, :, :],
        reference,
    )


def point_jacobian(
    model: Model,
    state: KinematicState,
    *,
    frames: Sequence[int | str],
    points: torch.Tensor,
) -> torch.Tensor:
    """Jacobian of points fixed to frames. Returns ``(B..., K, 3, nv)``: the world velocity of each point.

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
    transform = state.frame_transform[..., frame_ids, :, :]  # (B..., K, 3, 4)
    position = (transform[..., :3] @ points.unsqueeze(-1)).squeeze(-1) + transform[..., 3]
    return world[..., :3, :] - torch.linalg.cross(position.unsqueeze(-1), world[..., 3:, :], dim=-2)


def joint_jacobian(
    model: Model,
    state: KinematicState,
    *,
    joint: int | str,
    reference: Literal["world", "local", "local_world_aligned"] = "local_world_aligned",
) -> torch.Tensor:
    """Jacobian of one joint (id or name). Returns ``(B..., 6, nv)``, rows ``[linear, angular]``; references as `frame_jacobian`."""
    joint = model.joint_id(joint)
    return _express(
        _world_jacobian(model, state, joint),
        state.joint_translation[..., joint, :],
        state.joint_rotation[..., joint, :, :],
        reference,
    )


def frame_jacobian_time_variation(
    model: Model,
    state: KinematicState,
    *,
    v: torch.Tensor,
    frame: int | str,
    reference: Literal["world", "local", "local_world_aligned"] = "local_world_aligned",
) -> torch.Tensor:
    """Time derivative ``dJ/dt`` of `frame_jacobian` along ``v: (B..., nv)``. Returns ``(B..., 6, nv)``.

    Acceleration pairing ``a = J dv + dJ v``: spatial for ``"world"``/``"local"``, classical for
    ``"local_world_aligned"``.
    """
    check_tensor("v", v, shape=(model.nv,), dtype=model.dtype, device=model.device)
    frame_id = model.frame_id(frame)
    return frame_jacobian_time_variation_raw(model, state.q, v, state.joint_pose, frame_id, reference=reference)


def joint_jacobian_time_variation(
    model: Model,
    state: KinematicState,
    *,
    v: torch.Tensor,
    joint: int | str,
    reference: Literal["world", "local", "local_world_aligned"] = "local_world_aligned",
) -> torch.Tensor:
    """Time derivative ``dJ/dt`` of `joint_jacobian` along ``v: (B..., nv)``. Returns ``(B..., 6, nv)``."""
    check_tensor("v", v, shape=(model.nv,), dtype=model.dtype, device=model.device)
    joint = model.joint_id(joint)
    result = joint_jacobians_time_variation_raw(model, state.q, v, state.joint_pose)
    return _time_variation_in_reference(
        result.joint_jacobians[..., joint, :, :],
        result.joint_jacobians_dot[..., joint, :, :],
        state.joint_pose[..., joint, :],
        v,
        reference,
    )


def _world_jacobian(model: Model, state: KinematicState, joint: int) -> torch.Tensor:
    """WORLD Jacobian of ``joint``, ``(B..., 6, nv)``; the column table is built once per state."""
    plan = jacobian_plan(model, dtype=state.q.dtype, device=state.q.device)
    if "jacobian_columns" not in state._memo:
        state._memo["jacobian_columns"] = world_column_table_from_matrices(
            state.joint_rotation, state.joint_translation, plan, device=state.q.device, dtype=state.q.dtype
        )
    return reduced_columns(model, state._memo["jacobian_columns"] * plan.support_mask[joint])


def _express(world: torch.Tensor, position: torch.Tensor, rotation: torch.Tensor, reference: str) -> torch.Tensor:
    """Re-express a WORLD Jacobian ``(B..., 6, nv)`` at a point ``position (B..., 3)`` with axes ``rotation``."""
    if reference == "world":
        return world
    angular = world[..., 3:, :]
    aligned = torch.cat(
        (world[..., :3, :] - torch.linalg.cross(position.unsqueeze(-1), angular, dim=-2), angular), dim=-2
    )
    if reference == "local_world_aligned":
        return aligned
    if reference == "local":
        rotation = rotation.mT
        return torch.cat((rotation @ aligned[..., :3, :], rotation @ aligned[..., 3:, :]), dim=-2)
    raise ArgumentError(f"Unsupported reference frame: {reference!r}")
