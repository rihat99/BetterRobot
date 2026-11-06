"""``JointHelical``: a rotation about ``axis`` coupled to a translation along it. ``nq = nv = 1``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, final

import torch

from ...lie import se3
from ..manifolds import EUCLIDEAN, Manifold
from .base import JointModel, check_axis


@final
@dataclass(frozen=True)
class JointHelical(JointModel):
    """Screw motion: rotation by the angle ``q`` about ``axis`` and translation ``pitch * q`` along it."""

    axis: tuple[float, float, float]
    pitch: float  #: translation per radian
    nq: ClassVar[int] = 1
    nv: ClassVar[int] = 1
    manifold: ClassVar[Manifold] = EUCLIDEAN

    def __post_init__(self) -> None:
        check_axis(self.axis)

    def placement(self, q: torch.Tensor) -> torch.Tensor:
        axis, angle = q.new_tensor(self.axis), q[..., 0]
        return se3.compose(se3.from_translation(axis * (self.pitch * q)), se3.from_axis_angle(axis, angle))

    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        axis = q.new_tensor(self.axis)
        return torch.cat((self.pitch * axis, axis)).expand(*q.shape[:-1], 6).unsqueeze(-1)

    def neutral(self) -> torch.Tensor:
        return torch.zeros(1)

    def random(self, generator: torch.Generator, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return lower + (upper - lower) * torch.rand(1, generator=generator, dtype=lower.dtype)
