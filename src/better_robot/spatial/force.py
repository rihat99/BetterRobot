"""``Force`` — 6D spatial wrench value type (dual of ``Motion``).

Stored as ``(..., 6)`` tensor ``[fx, fy, fz, tx, ty, tz]``.

See ``docs/design/03_LIE_AND_SPATIAL.md §7``.
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
        """Not a standard spatial operation — use :meth:`Motion.cross_force`.

        ``Force × Motion`` is not part of the spatial-algebra primitives.
        The dual of :meth:`Motion.cross_force` is what callers usually want.
        """
        raise NotImplementedError(
            "Force.cross_motion is not a standard spatial operation. "
            "Use Motion.cross_force instead — see "
            "docs/design/03_LIE_AND_SPATIAL.md §7."
        )

    def se3_action(self, T) -> "Force":
        """Apply an SE3 transform to the wrench (dual adjoint action).

        ``T`` may be a raw ``(..., 7)`` tensor or an
        :class:`~better_robot.lie.types.SE3` value (its ``.tensor`` is
        unwrapped).

        ``F_new = Ad(T)^{-T} * F = Ad^*(T) * F``;
        for spatial forces ``Ad^*(T) = Ad(T^{-1})^T``.
        """
        from ..lie import se3 as _se3
        from ..lie.types import SE3 as _SE3
        if isinstance(T, _SE3):
            T = T.tensor
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
