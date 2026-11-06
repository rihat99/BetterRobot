"""``JointFreeFlyer``: a floating base on a pose. ``nq = 7``, ``nv = 6``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, final

import torch

from ...lie import se3
from ..manifolds import RIGID, Manifold
from .base import JointModel


@final
@dataclass(frozen=True)
class JointFreeFlyer(JointModel):
    """Pose ``q = [tx, ty, tz, qx, qy, qz, qw]`` with the SE(3) body-frame twist ``v``. ``nq = 7``, ``nv = 6``.

    A floating base is this joint at index 1 of the model; the rest of the library never treats it as a special
    case.
    """

    nq: ClassVar[int] = 7
    nv: ClassVar[int] = 6
    manifold: ClassVar[Manifold] = RIGID
    axis: ClassVar[None] = None

    def placement(self, q: torch.Tensor) -> torch.Tensor:
        return se3.normalize(q)

    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        return torch.eye(6, dtype=q.dtype, device=q.device).expand(*q.shape[:-1], 6, 6)

    def neutral(self) -> torch.Tensor:
        return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def random(self, generator: torch.Generator | None, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Unit Gaussian translation and a uniformly random rotation; the limits are not read."""
        translation = torch.randn(3, generator=generator, dtype=lower.dtype)
        quaternion = torch.randn(4, generator=generator, dtype=lower.dtype)
        return torch.cat((translation, quaternion / quaternion.norm().clamp(min=1e-8)))
