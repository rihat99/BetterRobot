"""``JointPlanar``: a rigid motion in the joint's ``xy`` plane. ``nq = 4``, ``nv = 3``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, final

import torch

from ..manifolds import PLANE, Manifold
from .base import JointModel


@final
@dataclass(frozen=True)
class JointPlanar(JointModel):
    """Translation ``[x, y]`` in the plane and rotation ``θ`` about ``z``, ``q = [x, y, cos θ, sin θ]``; the tangent
    ``v = [vx, vy, dθ]`` is the SE(2) twist in the child frame. ``nq = 4``, ``nv = 3``."""

    nq: ClassVar[int] = 4
    nv: ClassVar[int] = 3
    manifold: ClassVar[Manifold] = PLANE
    axis: ClassVar[None] = None

    def placement(self, q: torch.Tensor) -> torch.Tensor:
        x, y, cos_t, sin_t = q.unbind(-1)
        # The half angle through atan2: the half-angle identities have an infinite derivative at θ = 0.
        half = torch.atan2(sin_t, cos_t) * 0.5
        zero = torch.zeros_like(x)
        return torch.stack((x, y, zero, zero, zero, torch.sin(half), torch.cos(half)), dim=-1)

    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        columns = torch.zeros(6, 3, dtype=q.dtype, device=q.device)
        columns[0, 0] = columns[1, 1] = columns[5, 2] = 1.0
        return columns.expand(*q.shape[:-1], 6, 3)

    def neutral(self) -> torch.Tensor:
        return torch.tensor([0.0, 0.0, 1.0, 0.0])

    def random(self, generator: torch.Generator | None, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """``x`` and ``y`` uniform inside their limits, ``θ`` uniform on the circle."""
        x = lower[0] + (upper[0] - lower[0]) * torch.rand(1, generator=generator)
        y = lower[1] + (upper[1] - lower[1]) * torch.rand(1, generator=generator)
        # Drawn in the limits' dtype: a float32 (cos, sin) pair cast to float64 is not unit.
        angle = (torch.rand(1, generator=generator, dtype=lower.dtype) * 2.0 - 1.0) * torch.pi
        return torch.cat((x, y, torch.cos(angle), torch.sin(angle)))
