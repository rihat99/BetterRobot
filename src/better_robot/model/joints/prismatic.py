"""``JointPrismatic``: a translation along ``axis``. ``nq = nv = 1``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, final

import torch

from ...lie import se3
from ..manifolds import EUCLIDEAN, Manifold
from .base import JointModel, _axis_column


@final
@dataclass(frozen=True)
class JointPrismatic(JointModel):
    """Translation along ``axis`` by the displacement ``q``. ``nq = nv = 1``."""

    axis: tuple[float, float, float]
    nq: ClassVar[int] = 1
    nv: ClassVar[int] = 1
    manifold: ClassVar[Manifold] = EUCLIDEAN

    def placement(self, q: torch.Tensor) -> torch.Tensor:
        return se3.from_translation(q.new_tensor(self.axis) * q)

    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        return _axis_column(q.new_tensor(self.axis), q.shape[:-1], row=0)

    def neutral(self) -> torch.Tensor:
        return torch.zeros(1)

    def random(self, generator: torch.Generator | None, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return lower + (upper - lower) * torch.rand(1, generator=generator)
