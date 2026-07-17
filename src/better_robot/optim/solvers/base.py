"""Linear-system representations and the pluggable solver protocols.

The stable public seam remains ``solve(A, b, ridge=None) -> Tensor``. M5 adds
structured and matrix-free system representations without making diagnostics
or warm starts mandatory for existing third-party solvers. Implementations
that can report per-element health may additionally satisfy
``InformativeLinearSolver``.

See ``docs/concepts/solver_stack.md §5`` and ``docs/conventions/extension.md §5``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple, Protocol, TypeAlias, runtime_checkable

import torch

from ..structure import BlockBandedMatrix, NormalOperator


LinearSystem: TypeAlias = torch.Tensor | BlockBandedMatrix | NormalOperator


class LinearSolveStatus(IntEnum):
    """Per-element status emitted by an informative linear solver."""

    SUCCESS = 0
    MAX_ITER = 1
    NONFINITE = 2
    BREAKDOWN = 3
    NOT_SPD = 4


class LinearSolveResult(NamedTuple):
    """Fixed-structure tensor diagnostics for one batched linear solve."""

    solution: torch.Tensor
    converged: torch.Tensor
    finite: torch.Tensor
    ok: torch.Tensor
    iterations: torch.Tensor
    residual_norm: torch.Tensor
    relative_residual: torch.Tensor
    status: torch.Tensor


@runtime_checkable
class LinearSolver(Protocol):
    """Linear solver for ``(A + ridge I) x = b``.

    ``b`` has shape ``(B..., n)`` and a tensor ``ridge`` is scalar or
    broadcastable to ``B...``. Individual implementations advertise which
    system representations they accept; the two original implementations
    intentionally remain dense-only.
    """

    def solve(
        self,
        A: LinearSystem,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Return ``x`` such that ``(A + ridge I) x ≈ b``."""
        ...


@runtime_checkable
class InformativeLinearSolver(Protocol):
    """Optional diagnostic and warm-start capability for a linear solver."""

    def solve_with_info(
        self,
        A: LinearSystem,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
        *,
        initial: torch.Tensor | None = None,
    ) -> LinearSolveResult:
        """Solve a system and return fixed-shape per-element diagnostics."""
        ...


def _broadcast_ridge(
    ridge: torch.Tensor | float | None,
    *,
    batch_shape: tuple[int, ...],
    exemplar: torch.Tensor,
) -> torch.Tensor | None:
    """Validate and broadcast a scalar ridge to an independent batch shape.

    The error wording deliberately retains the original dense ``A`` contract;
    callers and tests already rely on it, and every structured representation
    follows the same working dtype/device rule.
    """
    if ridge is None:
        return None
    if isinstance(ridge, torch.Tensor):
        if ridge.dtype != exemplar.dtype or ridge.device != exemplar.device:
            raise ValueError(
                "Tensor ridge must share A's dtype and device; "
                f"got ridge ({ridge.dtype}, {ridge.device}) and "
                f"A ({exemplar.dtype}, {exemplar.device})"
            )
        ridge_tensor = ridge
    else:
        ridge_tensor = exemplar.new_tensor(ridge)

    try:
        return torch.broadcast_to(ridge_tensor, batch_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"ridge shape {tuple(ridge_tensor.shape)} is not broadcastable to A batch shape {batch_shape}"
        ) from exc


def _regularized_matrix(
    A: torch.Tensor,
    ridge: torch.Tensor | float | None,
) -> torch.Tensor:
    """Return ``A + ridge I`` without modifying ``A``.

    Tensor ridges must already share the matrix dtype and device. Python
    numbers are materialized in the matrix's working dtype and device.
    """
    if ridge is None:
        return A

    if A.ndim < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError(f"A must have shape (B..., n, n); received {tuple(A.shape)}")

    batch_shape = A.shape[:-2]
    ridge_tensor = _broadcast_ridge(ridge, batch_shape=batch_shape, exemplar=A)
    assert ridge_tensor is not None

    regularized = A.clone()
    regularized.diagonal(dim1=-2, dim2=-1).add_(ridge_tensor[..., None])
    return regularized
