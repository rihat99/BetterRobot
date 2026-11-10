"""``JointFreeFlyer``: a floating base on a pose. ``nq = 7``, ``nv = 6``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, final, override

import torch

from ...lie import se3
from ..manifolds import RIGID, JointManifold
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
    manifold: ClassVar[JointManifold] = RIGID
    axis: ClassVar[None] = None

    @override
    def placement(self, q: torch.Tensor) -> torch.Tensor:
        return se3.normalize_raw(q)

    @override
    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        return torch.eye(6, dtype=q.dtype, device=q.device).expand(*q.shape[:-1], 6, 6)

    @override
    def neutral(self) -> torch.Tensor:
        return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def random(self, generator: torch.Generator, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """A configuration ``(7,)``: unit Gaussian translation, uniformly random rotation; ``lower, upper: (7,)`` unread."""
        translation = torch.randn(3, generator=generator, dtype=lower.dtype, device=lower.device)
        quaternion = torch.randn(4, generator=generator, dtype=lower.dtype, device=lower.device)
        return torch.cat((translation, quaternion / quaternion.norm().clamp(min=1e-8)))
