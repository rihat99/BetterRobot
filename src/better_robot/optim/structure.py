"""Pure structural values shared by block assembly and linear solvers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, TypeAlias

import torch

if TYPE_CHECKING:
    from ..residuals.structure import TemporalPattern


LinearizationMode: TypeAlias = Literal["auto", "dense", "structured", "matrix_free"]
LinearSystemKind: TypeAlias = Literal["dense", "banded", "operator"]
MatVec: TypeAlias = Callable[[torch.Tensor], torch.Tensor]


class LinearizationReason(str, Enum):
    """Stable, machine-readable reason for a linearization route."""

    FORCED_DENSE = "forced_dense"
    ELIGIBLE_BANDED = "eligible_banded"
    EXPLICIT_MATRIX_FREE = "explicit_matrix_free"
    NO_TIME_VARIABLE = "no_time_variable"
    MULTIPLE_OPTIMIZED_VARIABLES = "multiple_optimized_variables"
    NONSEPARABLE_MASK = "nonseparable_mask"
    UNDECLARED_TEMPORAL_RESIDUAL = "undeclared_temporal_residual"
    MISSING_TEMPORAL_BLOCKS = "missing_temporal_blocks"
    MIXED_OPTIMIZED_DEPENDENCY = "mixed_optimized_dependency"
    EXPLICIT_DENSE_SOLVER = "explicit_dense_solver"
    INCOMPATIBLE_SOLVER = "incompatible_solver"


@dataclass(frozen=True)
class LinearizationDecision:
    """Resolved static linearization representation and its explanation."""

    requested: LinearizationMode
    used: Literal["dense", "banded", "matrix_free"]
    reason: LinearizationReason
    detail: str


@dataclass(frozen=True)
class TemporalAnalysis:
    """Problem-construction-time analysis of one temporal optimization block."""

    operator_eligible: bool
    direct_eligible: bool
    reason: LinearizationReason
    detail: str
    variable_name: str | None = None
    time_length: int | None = None
    tangent_width: int | None = None
    reduced_width: int | None = None
    bandwidth: int = 0
    patterns: tuple[tuple[str, "TemporalPattern"], ...] = ()


def _check_flat_operand(name: str, value: torch.Tensor, matrix: "BlockBandedMatrix") -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < 1 or value.shape[-1] != matrix.size:
        raise ValueError(f"{name} must end in flattened size {matrix.size}, got {tuple(value.shape)}")
    if value.dtype != matrix.bands.dtype or value.device != matrix.bands.device:
        raise ValueError(
            f"{name} must share band dtype/device {matrix.bands.dtype}/{matrix.bands.device}, "
            f"got {value.dtype}/{value.device}"
        )


@dataclass(frozen=True)
class BlockBandedMatrix:
    """Symmetric block matrix in padded lower-band storage.

    ``bands[..., t, k]`` stores ``H[t, t-k]``. Slots with ``t < k`` are
    padding. Production operations never densify.
    """

    bands: torch.Tensor
    bandwidth: int

    def __post_init__(self) -> None:
        if not isinstance(self.bands, torch.Tensor):
            raise TypeError("BlockBandedMatrix bands must be a torch.Tensor")
        if self.bands.ndim < 4:
            raise ValueError("BlockBandedMatrix bands must have shape (B..., T, w+1, d, d)")
        if isinstance(self.bandwidth, bool) or not isinstance(self.bandwidth, int):
            raise TypeError("BlockBandedMatrix bandwidth must be an int")
        if self.bandwidth < 0:
            raise ValueError("BlockBandedMatrix bandwidth must be non-negative")
        if self.bands.shape[-1] != self.bands.shape[-2] or self.bands.shape[-1] <= 0:
            raise ValueError("BlockBandedMatrix blocks must be non-empty and square")
        if self.bands.shape[-3] != self.bandwidth + 1:
            raise ValueError(
                "BlockBandedMatrix band axis must have bandwidth + 1 entries; "
                f"got {self.bands.shape[-3]} and bandwidth={self.bandwidth}"
            )
        if self.bands.shape[-4] <= 0:
            raise ValueError("BlockBandedMatrix time length must be positive")
        if self.bandwidth >= self.bands.shape[-4]:
            raise ValueError("BlockBandedMatrix bandwidth must be smaller than time length")
        if not self.bands.is_floating_point():
            raise TypeError("BlockBandedMatrix bands must use a floating dtype")

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.bands.shape[:-4])

    @property
    def time_length(self) -> int:
        return self.bands.shape[-4]

    @property
    def block_size(self) -> int:
        return self.bands.shape[-1]

    @property
    def size(self) -> int:
        return self.time_length * self.block_size

    @property
    def finite(self) -> torch.Tensor:
        return torch.isfinite(self.bands).all(dim=(-4, -3, -2, -1))

    def diagonal(self) -> torch.Tensor:
        """Return the scalar diagonal with shape ``(B..., T*d)``."""
        diagonal = self.bands[..., :, 0, :, :].diagonal(dim1=-2, dim2=-1)
        return diagonal.reshape(*self.batch_shape, self.size)

    def densify(self) -> torch.Tensor:
        """Materialize the symmetric dense matrix for tests and small oracles."""
        dense = self.bands.new_zeros(*self.batch_shape, self.size, self.size)
        d = self.block_size
        for time in range(self.time_length):
            row = slice(time * d, (time + 1) * d)
            for offset in range(min(self.bandwidth, time) + 1):
                earlier = time - offset
                column = slice(earlier * d, (earlier + 1) * d)
                block = self.bands[..., time, offset, :, :]
                dense[..., row, column] = block
                if offset:
                    dense[..., column, row] = block.mT
        return dense

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        """Multiply by a flat ``(B..., T*d)`` vector without densifying."""
        _check_flat_operand("vector", vector, self)
        batch_shape = torch.broadcast_shapes(self.batch_shape, tuple(vector.shape[:-1]))
        bands = self.bands.expand(*batch_shape, *self.bands.shape[-4:])
        shaped = vector.expand(*batch_shape, self.size).reshape(
            *batch_shape,
            self.time_length,
            self.block_size,
        )
        result = torch.matmul(bands[..., :, 0, :, :], shaped.unsqueeze(-1)).squeeze(-1)
        for offset in range(1, self.bandwidth + 1):
            lower = bands[..., offset:, offset, :, :]
            later_product = torch.matmul(lower, shaped[..., :-offset, :].unsqueeze(-1)).squeeze(-1)
            earlier_product = torch.matmul(lower.mT, shaped[..., offset:, :].unsqueeze(-1)).squeeze(-1)
            result[..., offset:, :] = result[..., offset:, :] + later_product
            result[..., :-offset, :] = result[..., :-offset, :] + earlier_product
        return result.reshape(*batch_shape, self.size)

    def add_diagonal(self, diagonal: torch.Tensor) -> "BlockBandedMatrix":
        """Return a copy with a flat diagonal added to band zero."""
        _check_flat_operand("diagonal", diagonal, self)
        batch_shape = torch.broadcast_shapes(self.batch_shape, tuple(diagonal.shape[:-1]))
        bands = self.bands.expand(*batch_shape, *self.bands.shape[-4:]).clone()
        shaped = diagonal.expand(*batch_shape, self.size).reshape(
            *batch_shape,
            self.time_length,
            self.block_size,
        )
        bands[..., :, 0, :, :].diagonal(dim1=-2, dim2=-1).add_(shaped)
        return BlockBandedMatrix(bands, self.bandwidth)

    def scaled_restricted(
        self,
        scale: torch.Tensor,
        movable: torch.Tensor,
        diagonal: torch.Tensor,
    ) -> "BlockBandedMatrix":
        """Return ``M S H S M + diag(diagonal)`` in the same storage."""
        for name, operand in (("scale", scale), ("movable", movable), ("diagonal", diagonal)):
            _check_flat_operand(name, operand, self)
        batch_shape = torch.broadcast_shapes(
            self.batch_shape,
            tuple(scale.shape[:-1]),
            tuple(movable.shape[:-1]),
            tuple(diagonal.shape[:-1]),
        )
        bands = self.bands.expand(*batch_shape, *self.bands.shape[-4:])
        factor = (
            (scale * movable)
            .expand(*batch_shape, self.size)
            .reshape(
                *batch_shape,
                self.time_length,
                self.block_size,
            )
        )
        transformed = torch.zeros_like(bands)
        for offset in range(self.bandwidth + 1):
            lower = bands[..., offset:, offset, :, :]
            left = factor[..., offset:, :].unsqueeze(-1)
            right = factor[..., : self.time_length - offset, :].unsqueeze(-2)
            transformed[..., offset:, offset, :, :] = lower * left * right
        result = BlockBandedMatrix(transformed, self.bandwidth)
        return result.add_diagonal(diagonal)


@dataclass(frozen=True)
class NormalOperator:
    """Sized normal-equation operator with explicit inverse-preconditioner."""

    size: int
    matvec: MatVec
    preconditioner: MatVec | None = None
    block_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("NormalOperator size must be a positive int")
        if not callable(self.matvec):
            raise TypeError("NormalOperator matvec must be callable")
        if self.preconditioner is not None and not callable(self.preconditioner):
            raise TypeError("NormalOperator preconditioner must be callable or None")
        if self.block_shape is not None:
            if (
                not isinstance(self.block_shape, tuple)
                or len(self.block_shape) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.block_shape
                )
            ):
                raise ValueError("NormalOperator block_shape must be a pair of positive ints")
            if self.block_shape[0] * self.block_shape[1] != self.size:
                raise ValueError("NormalOperator block_shape product must equal size")

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        return self.matvec(vector)


__all__ = [
    "BlockBandedMatrix",
    "LinearizationDecision",
    "LinearizationMode",
    "LinearizationReason",
    "LinearSystemKind",
    "MatVec",
    "NormalOperator",
    "TemporalAnalysis",
]
