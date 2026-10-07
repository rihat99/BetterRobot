"""Immutable forward-kinematics result and the functions that consume it.

``forward_kinematics_state`` returns a frozen :class:`KinematicState` holding world
transforms as ``[R | p]`` matrices; ``frame_jacobian`` reads it. This is the interface new
code is written against. Poses in ``[t, q]`` layout are derived on request.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch

from ..data_model.execution_batch import broadcast_to_execution_batch
from ..data_model.model import Model
from ..data_model.reduced_coordinates import expand_configuration, reduce_jacobian
from ..lie import so3
from ._fk_lean import lean_core, placement_matrices
from .forward import _validate_q
from ._jacobian_columns import get_plan, world_column_table_from_matrices
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
]


def _to_pose(transform: torch.Tensor) -> torch.Tensor:
    """``[R | p]: (..., 3, 4)`` to ``[x y z qx qy qz qw]: (..., 7)``."""
    return torch.cat([transform[..., 3], so3.from_matrix(transform[..., :3])], dim=-1)


@dataclass(frozen=True)
class KinematicState:
    """World placements of every joint and frame at one configuration batch."""

    q: torch.Tensor  #: ``(B..., nq)`` configuration the state was computed at.
    joint_transform: torch.Tensor  #: ``(B..., njoints, 3, 4)`` world ``[R | p]`` of every joint.
    frame_transform: torch.Tensor  #: ``(B..., nframes, 3, 4)`` world ``[R | p]`` of every frame.
    # Memo of values derived from the fields above (poses, joint Jacobians), filled on first use.
    _memo: dict[str, torch.Tensor] = field(default_factory=dict, repr=False, compare=False)

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

    def frame_pose_of(self, frame_id: int) -> torch.Tensor:
        """``(B..., 7)`` world pose of one frame, without converting the others."""
        return _to_pose(self.frame_transform[..., frame_id, :, :])


def forward_kinematics_state(model: Model, q: torch.Tensor) -> KinematicState:
    """Forward kinematics. ``q: (B..., nq)`` → :class:`KinematicState` of joints and frames."""
    structure, values = model.structure, model.values
    _validate_q(structure, values, q)
    batch_shape = values._execution_batch_shape(q)
    nj, nf = structure.njoints, structure.nframes
    q = broadcast_to_execution_batch(q, batch_shape, (structure.nq,), name="q")
    q_full = expand_configuration(structure, q)
    placements, frames = placement_matrices(values)
    count = math.prod(batch_shape)  # explicit: a model without coordinates has zero-width q
    # Unbatched model values broadcast inside the matmuls; batched ones are flattened with q.
    if placements.ndim > 3:
        placements = placements.broadcast_to(*batch_shape, nj, 4, 4).reshape(count, nj, 4, 4)
    if frames.ndim > 3:
        frames = frames.broadcast_to(*batch_shape, nf, 4, 4).reshape(count, nf, 4, 4)
    world, _ = lean_core(structure, q_full.reshape(count, -1), placements)
    parents = world.index_select(1, structure.frame_parent_joints.to(torch.int64))
    # einsum is several times faster than a broadcast matmul on CPU when one side has no batch axis.
    frame_world = torch.einsum("nfab,fbc->nfac", parents, frames) if frames.ndim == 3 else parents @ frames
    return KinematicState(
        q=q,
        joint_transform=world.view(*batch_shape, nj, 3, 4),
        frame_transform=frame_world.view(*batch_shape, nf, 3, 4),
    )


def frame_jacobian(
    model: Model,
    state: KinematicState,
    frame: int | str,
    *,
    reference: Literal["world", "local", "local_world_aligned"] = "local_world_aligned",
) -> torch.Tensor:
    """Jacobian of one frame. Returns ``(B..., 6, nv)``, rows ``[linear, angular]``.

    The world column table shared by all frames is built once per state; each call masks it
    with the columns that support the frame's parent joint.
    """
    structure = model.structure
    frame_id = model.frame_id(frame) if isinstance(frame, str) else frame
    parent = structure.frame_parent_joint_ids[frame_id]
    return _express(
        _world_jacobian(model, state, parent),
        state.frame_translation[..., frame_id, :],
        state.frame_rotation[..., frame_id, :, :],
        reference,
    )


def joint_jacobian(
    model: Model,
    state: KinematicState,
    joint: int,
    *,
    reference: Literal["world", "local", "local_world_aligned"] = "local_world_aligned",
) -> torch.Tensor:
    """Jacobian of one joint. Returns ``(B..., 6, nv)``, rows ``[linear, angular]``; references as :func:`frame_jacobian`."""
    return _express(
        _world_jacobian(model, state, joint),
        state.joint_translation[..., joint, :],
        state.joint_rotation[..., joint, :, :],
        reference,
    )


def frame_jacobian_time_variation(
    model: Model,
    state: KinematicState,
    v: torch.Tensor,
    frame: int | str,
    *,
    reference: Literal["world", "local", "local_world_aligned"] = "local_world_aligned",
) -> torch.Tensor:
    """Time derivative ``dJ/dt`` of :func:`frame_jacobian` along ``v: (B..., nv)``. Returns ``(B..., 6, nv)``.

    Acceleration pairing ``a = J dv + dJ v``: spatial for ``"world"``/``"local"``, classical for
    ``"local_world_aligned"``.
    """
    frame_id = model.frame_id(frame) if isinstance(frame, str) else frame
    return frame_jacobian_time_variation_raw(
        model.structure, model.values, state.q, v, state.joint_pose, frame_id, reference=reference
    )


def joint_jacobian_time_variation(
    model: Model,
    state: KinematicState,
    v: torch.Tensor,
    joint: int,
    *,
    reference: Literal["world", "local", "local_world_aligned"] = "local_world_aligned",
) -> torch.Tensor:
    """Time derivative ``dJ/dt`` of :func:`joint_jacobian` along ``v: (B..., nv)``. Returns ``(B..., 6, nv)``."""
    result = joint_jacobians_time_variation_raw(model.structure, state.q, v, state.joint_pose)
    return _time_variation_in_reference(
        result.joint_jacobians[..., joint, :, :],
        result.joint_jacobians_dot[..., joint, :, :],
        state.joint_pose[..., joint, :],
        v,
        reference,
    )


def _world_jacobian(model: Model, state: KinematicState, joint: int) -> torch.Tensor:
    """WORLD Jacobian of ``joint``, ``(B..., 6, nv)``; the column table is built once per state."""
    structure = model.structure
    plan = get_plan(structure)
    if "jacobian_columns" not in state._memo:
        state._memo["jacobian_columns"] = world_column_table_from_matrices(
            structure, state.joint_rotation, state.joint_translation, plan, device=state.q.device, dtype=state.q.dtype
        )
    return reduce_jacobian(structure, state._memo["jacobian_columns"] * plan.support_mask[joint])


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
    raise ValueError(f"Unsupported reference frame: {reference!r}")
