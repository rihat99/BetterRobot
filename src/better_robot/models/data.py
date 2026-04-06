"""RobotData: mutable robot state and cached computation results."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["RobotData"]


@dataclass
class RobotData:
    """Mutable robot state and cached computation results.

    Created via model.create_data(). Stores the robot's current
    configuration and any quantities computed from it.

    Attributes:
        q: (n_actuated,) joint positions.
        v: (n_actuated,) joint velocities. None if not set.
        a: (n_actuated,) joint accelerations. None if not set.
        tau: (n_actuated,) joint torques/forces. None if not set.
        base_pose: (7,) SE3 base pose [tx, ty, tz, qx, qy, qz, qw]. None for fixed base.
        fk_poses: (n_links, 7) FK link poses. None until computed.
        _model_id: id(model) set at creation for safety checks. Do not modify.
    """

    q: torch.Tensor
    v: torch.Tensor | None = None
    a: torch.Tensor | None = None
    tau: torch.Tensor | None = None
    base_pose: torch.Tensor | None = None
    fk_poses: torch.Tensor | None = None
    _model_id: int = -1

    def clone(self) -> "RobotData":
        """Deep copy of all tensors."""
        return RobotData(
            q=self.q.clone(),
            v=self.v.clone() if self.v is not None else None,
            a=self.a.clone() if self.a is not None else None,
            tau=self.tau.clone() if self.tau is not None else None,
            base_pose=self.base_pose.clone() if self.base_pose is not None else None,
            fk_poses=self.fk_poses.clone() if self.fk_poses is not None else None,
            _model_id=self._model_id,
        )

    def invalidate_cache(self) -> None:
        """Clear cached quantities. Call after modifying q, v, or base_pose."""
        self.fk_poses = None
