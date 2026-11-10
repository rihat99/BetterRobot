"""``JointFixed``: a rigid link to the parent. ``nq = nv = 0``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, final, override

import torch

from ..manifolds import FIXED, JointManifold
from .base import JointModel


@final
@dataclass(frozen=True)
class JointFixed(JointModel):
    """Rigid link to the parent: the identity placement (the fixed offset is the model's joint placement)."""

    nq: ClassVar[int] = 0
    nv: ClassVar[int] = 0
    manifold: ClassVar[JointManifold] = FIXED
    axis: ClassVar[None] = None

    @override
    def placement(self, q: torch.Tensor) -> torch.Tensor:
        return q.new_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expand(*q.shape[:-1], 7)

    @override
    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        return q.new_zeros(*q.shape[:-1], 6, 0)

    @override
    def neutral(self) -> torch.Tensor:
        return torch.zeros(0)

    @override
    def random(self, generator: torch.Generator, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return torch.zeros(0, dtype=lower.dtype)
