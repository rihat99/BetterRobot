"""``JointSpherical``: a ball joint on a unit quaternion. ``nq = 4``, ``nv = 3``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, final, override

import torch

from ...lie import so3
from ..manifolds import ROTATION, JointManifold
from .base import JointModel


@final
@dataclass(frozen=True)
class JointSpherical(JointModel):
    """Rotation ``q = [qx, qy, qz, qw]`` with the SO(3) tangent ``v``. ``nq = 4``, ``nv = 3``."""

    nq: ClassVar[int] = 4
    nv: ClassVar[int] = 3
    manifold: ClassVar[JointManifold] = ROTATION
    axis: ClassVar[None] = None

    @override
    def placement(self, q: torch.Tensor) -> torch.Tensor:
        return torch.cat((q.new_zeros(*q.shape[:-1], 3), so3.normalize_raw(q)), dim=-1)

    @override
    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        angular = torch.eye(3, dtype=q.dtype, device=q.device)
        return torch.cat((torch.zeros_like(angular), angular)).expand(*q.shape[:-1], 6, 3)

    @override
    def neutral(self) -> torch.Tensor:
        return torch.tensor([0.0, 0.0, 0.0, 1.0])

    @override
    def random(self, generator: torch.Generator, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        quaternion = torch.randn(4, generator=generator, dtype=lower.dtype)
        return quaternion / quaternion.norm().clamp(min=1e-8)
