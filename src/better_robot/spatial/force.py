"""``Force`` — 6D spatial wrench value type (dual of ``Motion``).

Stored as ``(..., 6)`` tensor ``[fx, fy, fz, tx, ty, tz]``.

See ``docs/03_LIE_AND_SPATIAL.md §7``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..lie import tangents as _tan


@dataclass(frozen=True)
class Force:
    """6D spatial force (linear + torque)."""

    data: torch.Tensor

    @classmethod
    def zero(
        cls,
        *,
        batch_shape: tuple[int, ...] = (),
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Force":
        return cls(torch.zeros((*batch_shape, 6), device=device, dtype=dtype))

    @property
    def linear(self) -> torch.Tensor:
        return self.data[..., :3]

    @property
    def angular(self) -> torch.Tensor:
        return self.data[..., 3:]

    def cross_motion(self, other) -> "Motion":  # other: Motion
        """Force cross Motion — dual of Motion.cross_force.

        Defined so that <f, m.cross_motion(v)> = <m.cross_force(v), f>.
        """
        from .motion import Motion
        # This is ad^*(f) applied via the dual structure
        # f × m: not a standard operation; raise for now
        raise NotImplementedError(
            "Force.cross_motion is not a standard spatial operation. "
            "Use Motion.cross_force instead."
        )

    def se3_action(self, T: torch.Tensor) -> "Force":
        """Apply an SE3 transform to the wrench (dual adjoint action).

        F_new = Ad(T)^{-T} * F = Ad^*(T) * F
        For spatial forces: Ad^*(T) = Ad(T^{-1})^T
        """
        from ..lie import se3 as _se3
        # Dual action: Ad^{-T}(T) * f = Ad(T^{-1})^T * f
        Ad_inv = _se3.adjoint_inv(T)    # (..., 6, 6)  = Ad(T^{-1})
        Ad_inv_T = Ad_inv.transpose(-1, -2)  # (..., 6, 6)
        return Force((Ad_inv_T @ self.data.unsqueeze(-1)).squeeze(-1))

    def __neg__(self) -> "Force":
        return Force(-self.data)

    def __add__(self, other: "Force") -> "Force":
        return Force(self.data + other.data)

    def __sub__(self, other: "Force") -> "Force":
        return Force(self.data - other.data)
