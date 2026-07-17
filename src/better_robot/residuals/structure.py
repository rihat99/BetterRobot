"""Dependency-light temporal sparsity declarations for residual authors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalPattern:
    """Static row-to-knot incidence for one residual/trajectory-variable pair.

    Row group ``r`` touches knots ``r + row_origin + offset`` for every entry
    in ``offsets``. Numeric Jacobian blocks remain an optional residual hook;
    this value describes support only and deliberately has no optimizer import.
    """

    rows: int
    row_width: int
    row_origin: int
    offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("rows", "row_width", "row_origin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"TemporalPattern {name} must be an int")
        if self.rows <= 0:
            raise ValueError("TemporalPattern rows must be positive")
        if self.row_width <= 0:
            raise ValueError("TemporalPattern row_width must be positive")
        if not isinstance(self.offsets, tuple):
            raise TypeError("TemporalPattern offsets must be a tuple[int, ...]")
        if not self.offsets:
            raise ValueError("TemporalPattern offsets must be non-empty")
        if any(isinstance(offset, bool) or not isinstance(offset, int) for offset in self.offsets):
            raise TypeError("TemporalPattern offsets must contain only ints")
        if self.offsets != tuple(sorted(set(self.offsets))):
            raise ValueError("TemporalPattern offsets must be sorted and unique")

    @property
    def hessian_offsets(self) -> tuple[int, ...]:
        """Sorted normal-matrix offsets induced by this Jacobian stencil."""
        return tuple(sorted({left - right for left in self.offsets for right in self.offsets}))


__all__ = ["TemporalPattern"]
