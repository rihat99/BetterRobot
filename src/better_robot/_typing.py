"""Shape-annotated type aliases (jaxtyping-style).

These are hints that document the shapes expected by functions; they do
**not** enforce anything at runtime. The intent is that IDEs and readers
can tell at a glance whether a tensor is a configuration ``(B..., nq)``
or a joint-space Jacobian ``(B..., njoints, 6, nv)`` without digging into
a docstring.

See ``docs/10_BATCHING_AND_BACKENDS.md §1``.
"""

from __future__ import annotations

import torch

# Scalar element tensor types
Float = torch.Tensor
Int = torch.Tensor
Bool = torch.Tensor

# Configuration / tangent
Q = torch.Tensor  # (B..., nq)
V = torch.Tensor  # (B..., nv)
Tau = torch.Tensor  # (B..., nv)

# Lie groups
SE3 = torch.Tensor  # (B..., 7)  [tx, ty, tz, qx, qy, qz, qw]
SO3 = torch.Tensor  # (B..., 4)  [qx, qy, qz, qw]
se3 = torch.Tensor  # (B..., 6)
so3 = torch.Tensor  # (B..., 3)

# Spatial algebra
Twist = torch.Tensor  # (B..., 6)
Wrench = torch.Tensor  # (B..., 6)
SpatialInertia = torch.Tensor  # (B..., 10)

# Bulk kinematics
JointsSE3 = torch.Tensor  # (B..., njoints, 7)
FramesSE3 = torch.Tensor  # (B..., nframes, 7)
JointJac = torch.Tensor  # (B..., 6, nv)
AllJointJacs = torch.Tensor  # (B..., njoints, 6, nv)

# Dynamics
MassMatrix = torch.Tensor  # (B..., nv, nv)
CoriolisMatrix = torch.Tensor  # (B..., nv, nv)

__all__ = [
    "Float",
    "Int",
    "Bool",
    "Q",
    "V",
    "Tau",
    "SE3",
    "SO3",
    "se3",
    "so3",
    "Twist",
    "Wrench",
    "SpatialInertia",
    "JointsSE3",
    "FramesSE3",
    "JointJac",
    "AllJointJacs",
    "MassMatrix",
    "CoriolisMatrix",
]
