"""Torch-native ``KinematicsOps`` — forwards to ``kinematics/forward.py`` and
``kinematics/jacobian.py``. Imports are lazy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from ...data_model.data import Data
    from ...data_model.model import Model


class TorchNativeKinematicsOps:
    """Concrete :class:`~better_robot.backends.protocol.KinematicsOps`.

    Today's implementation forwards to the topological-walk in
    ``kinematics.forward.forward_kinematics_raw`` and the propagation
    Jacobian in ``kinematics.jacobian._compute_joint_jacobians_raw``.
    """

    def forward_kinematics(
        self, model: "Model", q: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        from ...kinematics.forward import forward_kinematics_raw
        return forward_kinematics_raw(model, q)

    def compute_joint_jacobians(
        self, model: "Model", data: "Data",
    ) -> "torch.Tensor":
        from ...kinematics.jacobian import _compute_joint_jacobians_raw
        return _compute_joint_jacobians_raw(model, data)


__all__ = ["TorchNativeKinematicsOps"]
