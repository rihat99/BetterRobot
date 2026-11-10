"""``JointTranslation``: a free translation. ``nq = nv = 3``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, final, override

import torch

from ...lie import se3
from ..manifolds import EUCLIDEAN, JointManifold
from .base import JointModel


@final
@dataclass(frozen=True)
class JointTranslation(JointModel):
    """Translation ``q = [x, y, z]`` with ``v = [vx, vy, vz]``. ``nq = nv = 3``."""

    nq: ClassVar[int] = 3
    nv: ClassVar[int] = 3
    manifold: ClassVar[JointManifold] = EUCLIDEAN
    axis: ClassVar[None] = None

    @override
    def placement(self, q: torch.Tensor) -> torch.Tensor:
        return se3.from_translation_raw(q)

    @override
    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        linear = torch.eye(3, dtype=q.dtype, device=q.device)
        return torch.cat((linear, torch.zeros_like(linear))).expand(*q.shape[:-1], 6, 3)

    @override
    def neutral(self) -> torch.Tensor:
        return torch.zeros(3)

    @override
    def random(self, generator: torch.Generator, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return lower + (upper - lower) * torch.rand(3, generator=generator, dtype=lower.dtype, device=lower.device)
