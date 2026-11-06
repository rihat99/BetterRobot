"""Static per-joint tables read by the kinematics plans and the viewer."""

from __future__ import annotations

import torch

from .joint_models.base import JointModel

_ALIGNED_AXES = {
    "revolute_rx": (1.0, 0.0, 0.0),
    "revolute_ry": (0.0, 1.0, 0.0),
    "revolute_rz": (0.0, 0.0, 1.0),
    "prismatic_px": (1.0, 0.0, 0.0),
    "prismatic_py": (0.0, 1.0, 0.0),
    "prismatic_pz": (0.0, 0.0, 1.0),
}


def joint_axes(joints: tuple[JointModel, ...]) -> torch.Tensor:
    """Axis of every joint, ``(njoints, 3)`` float64 on the CPU; zero for joints without one."""
    rows = []
    for joint in joints:
        if joint.kind in _ALIGNED_AXES:
            rows.append(_ALIGNED_AXES[joint.kind])
        elif joint.axis is None:
            rows.append((0.0, 0.0, 0.0))
        else:
            rows.append(tuple(joint.axis.detach().to(device="cpu", dtype=torch.float64).tolist()))
    return torch.tensor(rows, dtype=torch.float64).reshape(len(joints), 3)


def joint_pitches(joints: tuple[JointModel, ...]) -> torch.Tensor:
    """Helical pitch of every joint, ``(njoints,)`` float64 on the CPU; zero for other kinds."""
    return torch.tensor([float(getattr(joint, "pitch", 0.0)) for joint in joints], dtype=torch.float64)


__all__ = ["joint_axes", "joint_pitches"]
