"""Shape-annotated type aliases (jaxtyping-style).

These are hints that document the shapes expected by functions; they do
**not** enforce anything at runtime. The intent is that IDEs and readers
can tell at a glance whether a tensor is a configuration ``(B..., nq)``
or a joint-space Jacobian ``(B..., 6, nv)`` without digging into a
docstring.

The jaxtyping aliases live under ``if TYPE_CHECKING:`` so the runtime
``import better_robot._typing`` does **not** pull ``jaxtyping``. Plain
``torch.Tensor`` aliases stay in scope at runtime so existing call sites
keep type-checking with the lighter ``Tensor`` alias too.

See ``docs/conventions/13_NAMING.md §2.9``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

# ──────────────────────────────────────────────────────────────────────
# Runtime-friendly aliases (always available — no extra deps).
# ──────────────────────────────────────────────────────────────────────

Float = torch.Tensor
Int = torch.Tensor
Bool = torch.Tensor

if TYPE_CHECKING:
    from jaxtyping import Float as _Float  # type: ignore[import-not-found]
    from torch import Tensor as _Tensor

    # Single SE3 / SO3 storage tensors
    SE3Tensor = _Float[_Tensor, "*B 7"]
    SO3Tensor = _Float[_Tensor, "*B 4"]
    Quaternion = _Float[_Tensor, "*B 4"]
    TangentSE3 = _Float[_Tensor, "*B 6"]
    TangentSO3 = _Float[_Tensor, "*B 3"]

    # Per-joint and per-frame stacks
    JointPoseStack = _Float[_Tensor, "*B njoints 7"]
    FramePoseStack = _Float[_Tensor, "*B nframes 7"]

    # Configurations and tangents
    ConfigTensor = _Float[_Tensor, "*B nq"]
    VelocityTensor = _Float[_Tensor, "*B nv"]

    # Jacobians
    JointJacobian = _Float[_Tensor, "*B 6 nv"]
    JointJacobianStack = _Float[_Tensor, "*B njoints 6 nv"]
else:
    SE3Tensor = torch.Tensor
    SO3Tensor = torch.Tensor
    Quaternion = torch.Tensor
    TangentSE3 = torch.Tensor
    TangentSO3 = torch.Tensor
    JointPoseStack = torch.Tensor
    FramePoseStack = torch.Tensor
    ConfigTensor = torch.Tensor
    VelocityTensor = torch.Tensor
    JointJacobian = torch.Tensor
    JointJacobianStack = torch.Tensor


__all__ = [
    "Float",
    "Int",
    "Bool",
    "SE3Tensor",
    "SO3Tensor",
    "Quaternion",
    "TangentSE3",
    "TangentSO3",
    "JointPoseStack",
    "FramePoseStack",
    "ConfigTensor",
    "VelocityTensor",
    "JointJacobian",
    "JointJacobianStack",
]
