"""Immutable forward-kinematics result and the functions that consume it.

``forward_kinematics_state`` returns a frozen :class:`KinematicState`; ``frame_jacobian`` reads it.
This is the interface new code is written against. It currently wraps the
``Data``-based passes; consumers should use the rotation / translation
accessors, which stay valid when the storage changes from poses to matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from ..data_model.model import Model
from ..lie import so3
from .forward import forward_kinematics
from .jacobian import frame_jacobian_raw, joint_jacobians_raw

__all__ = ["KinematicState", "forward_kinematics_state", "frame_jacobian"]


@dataclass(frozen=True)
class KinematicState:
    """World placements of every joint and frame at one configuration batch."""

    q: torch.Tensor  #: ``(B..., nq)`` configuration the state was computed at.
    joint_pose: torch.Tensor  #: ``(B..., njoints, 7)`` world poses ``[x y z qx qy qz qw]``.
    frame_pose: torch.Tensor  #: ``(B..., nframes, 7)`` world poses ``[x y z qx qy qz qw]``.
    # Memo of values derived from the fields above (joint Jacobians), filled on first use.
    _memo: dict[str, torch.Tensor] = field(default_factory=dict, repr=False, compare=False)

    @property
    def joint_translation(self) -> torch.Tensor:
        """``(B..., njoints, 3)`` world joint origins."""
        return self.joint_pose[..., :3]

    @property
    def joint_rotation(self) -> torch.Tensor:
        """``(B..., njoints, 3, 3)`` world joint rotation matrices."""
        return so3.to_matrix(self.joint_pose[..., 3:])

    @property
    def frame_translation(self) -> torch.Tensor:
        """``(B..., nframes, 3)`` world frame origins."""
        return self.frame_pose[..., :3]

    @property
    def frame_rotation(self) -> torch.Tensor:
        """``(B..., nframes, 3, 3)`` world frame rotation matrices."""
        return so3.to_matrix(self.frame_pose[..., 3:])


def forward_kinematics_state(model: Model, q: torch.Tensor) -> KinematicState:
    """Forward kinematics. ``q: (B..., nq)`` → :class:`KinematicState` of joints and frames."""
    data = forward_kinematics(model, q, compute_frames=True)
    return KinematicState(q=data.q, joint_pose=data.joint_pose_world, frame_pose=data.frame_pose_world)


def frame_jacobian(
    model: Model,
    state: KinematicState,
    frame: int | str,
    *,
    reference: Literal["world", "local", "local_world_aligned"] = "local_world_aligned",
) -> torch.Tensor:
    """Jacobian of one frame. Returns ``(B..., 6, nv)``, rows ``[linear, angular]``."""
    frame_id = model.frame_id(frame) if isinstance(frame, str) else frame
    if "joint_jacobians" not in state._memo:
        state._memo["joint_jacobians"] = joint_jacobians_raw(model.structure, state.q, state.joint_pose).joint_jacobians
    return frame_jacobian_raw(
        model.structure,
        model.values,
        state.q,
        state.joint_pose,
        frame_id,
        reference=reference,
        joint_jacobians=state._memo["joint_jacobians"],
    )
