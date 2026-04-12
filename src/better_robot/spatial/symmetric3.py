"""3x3 symmetric matrix value type.

Stored as a packed ``(..., 6)`` tensor ``[xx, yy, zz, xy, xz, yz]``.

See ``docs/03_LIE_AND_SPATIAL.md §7``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Symmetric3:
    """Packed 3x3 symmetric matrix.

    ``data``: ``(..., 6)`` laid out as ``[xx, yy, zz, xy, xz, yz]``.
    """

    data: torch.Tensor

    @classmethod
    def zero(
        cls,
        *,
        batch_shape: tuple[int, ...] = (),
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Symmetric3":
        return cls(torch.zeros((*batch_shape, 6), device=device, dtype=dtype))

    @classmethod
    def from_matrix(cls, M: torch.Tensor) -> "Symmetric3":
        """Extract the symmetric part of a ``(..., 3, 3)`` matrix.

        Packing: [xx, yy, zz, xy, xz, yz] where xx = M[0,0] etc.
        """
        S = (M + M.transpose(-1, -2)) * 0.5
        data = torch.stack([
            S[..., 0, 0],
            S[..., 1, 1],
            S[..., 2, 2],
            S[..., 0, 1],
            S[..., 0, 2],
            S[..., 1, 2],
        ], dim=-1)
        return cls(data)

    def to_matrix(self) -> torch.Tensor:
        """Expand to ``(..., 3, 3)``."""
        xx = self.data[..., 0]
        yy = self.data[..., 1]
        zz = self.data[..., 2]
        xy = self.data[..., 3]
        xz = self.data[..., 4]
        yz = self.data[..., 5]
        return torch.stack([
            torch.stack([xx, xy, xz], dim=-1),
            torch.stack([xy, yy, yz], dim=-1),
            torch.stack([xz, yz, zz], dim=-1),
        ], dim=-2)

    def add(self, other: "Symmetric3") -> "Symmetric3":
        return Symmetric3(self.data + other.data)
