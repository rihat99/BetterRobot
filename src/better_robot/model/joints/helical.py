"""``JointHelical``: a rotation about ``axis`` coupled to a translation along it. ``nq = nv = 1``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, final, override

import torch

from ...lie import se3
from ..manifolds import EUCLIDEAN, JointManifold
from .base import JointModel, check_axis


@final
@dataclass(frozen=True)
class JointHelical(JointModel):
    """Screw motion: rotation by the angle ``q`` about ``axis`` and translation ``pitch * q`` along it."""

    axis: tuple[float, float, float]
    pitch: float  #: translation per radian
    nq: ClassVar[int] = 1
    nv: ClassVar[int] = 1
    manifold: ClassVar[JointManifold] = EUCLIDEAN

    def __post_init__(self) -> None:
        """Raise ``ArgumentError`` unless ``axis`` is a tuple of three floats with unit norm."""
        check_axis(self.axis)

    @override
    def placement(self, q: torch.Tensor) -> torch.Tensor:
        axis, angle = q.new_tensor(self.axis), q[..., 0]
        return se3.compose_raw(se3.from_translation_raw(axis * (self.pitch * q)), se3.from_axis_angle_raw(axis, angle))

    @override
    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        axis = q.new_tensor(self.axis)
        return torch.cat((self.pitch * axis, axis)).expand(*q.shape[:-1], 6).unsqueeze(-1)

    @override
    def neutral(self) -> torch.Tensor:
        return torch.zeros(1)

    @override
    def random(self, generator: torch.Generator, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return lower + (upper - lower) * torch.rand(1, generator=generator, dtype=lower.dtype)
