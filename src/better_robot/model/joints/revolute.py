"""``JointRevolute`` (an angle about ``axis``) and ``JointRevoluteUnbounded`` (the same on ``[cos θ, sin θ]``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, final

import torch

from ...lie import se3
from ..manifolds import CIRCLE, EUCLIDEAN, Manifold
from .base import JointModel, _axis_column, check_axis


def _random_angle(generator: torch.Generator, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Uniform angle ``(1,)`` in ``[lower, upper]``; a missing limit is replaced by ``-pi`` or ``pi``."""
    lower = torch.where(torch.isfinite(lower), lower, torch.full_like(lower, -torch.pi))
    upper = torch.where(torch.isfinite(upper), upper, torch.full_like(upper, torch.pi))
    return lower + (upper - lower) * torch.rand(1, generator=generator, dtype=lower.dtype)


@final
@dataclass(frozen=True)
class JointRevolute(JointModel):
    """Rotation about ``axis`` by the angle ``q``. ``nq = nv = 1``."""

    axis: tuple[float, float, float]
    nq: ClassVar[int] = 1
    nv: ClassVar[int] = 1
    manifold: ClassVar[Manifold] = EUCLIDEAN

    def __post_init__(self) -> None:
        check_axis(self.axis)

    def placement(self, q: torch.Tensor) -> torch.Tensor:
        return se3.from_axis_angle(q.new_tensor(self.axis), q[..., 0])

    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        return _axis_column(q.new_tensor(self.axis), q.shape[:-1], row=3)

    def neutral(self) -> torch.Tensor:
        return torch.zeros(1)

    def random(self, generator: torch.Generator, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return _random_angle(generator, lower, upper)


@final
@dataclass(frozen=True)
class JointRevoluteUnbounded(JointModel):
    """Rotation about ``axis`` without limits, ``q = [cos θ, sin θ]``. ``nq = 2``, ``nv = 1``."""

    axis: tuple[float, float, float]
    nq: ClassVar[int] = 2
    nv: ClassVar[int] = 1
    manifold: ClassVar[Manifold] = CIRCLE

    def __post_init__(self) -> None:
        check_axis(self.axis)

    def placement(self, q: torch.Tensor) -> torch.Tensor:
        return se3.from_axis_angle(q.new_tensor(self.axis), torch.atan2(q[..., 1], q[..., 0]))

    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        return _axis_column(q.new_tensor(self.axis), q.shape[:-1], row=3)

    def neutral(self) -> torch.Tensor:
        return torch.tensor([1.0, 0.0])

    def random(self, generator: torch.Generator, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        # Drawn in the limits' dtype: a float32 (cos, sin) pair cast to float64 is not unit.
        angle = (torch.rand(1, generator=generator, dtype=lower.dtype) * 2.0 - 1.0) * torch.pi
        return torch.cat((torch.cos(angle), torch.sin(angle)))
