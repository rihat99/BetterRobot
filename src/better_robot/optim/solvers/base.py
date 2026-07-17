"""``LinearSolver`` protocol — solves dense systems inside an optimiser.

The shipped implementations are ``Cholesky`` and ``LSTSQ``. A concrete class
is a ``LinearSolver`` if it provides a
``solve(A, b, ridge=None) -> Tensor`` method with the right signature;
``@runtime_checkable`` lets callers verify that with ``isinstance``.

See ``docs/concepts/solver_stack.md §5`` and ``docs/conventions/extension.md §5``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LinearSolver(Protocol):
    """Dense linear solver for ``(A + ridge I) x = b``.

    ``A`` has shape ``(B..., n, n)``, ``b`` has shape ``(B..., n)``, and a
    tensor ``ridge`` is scalar or broadcastable to ``B...``. The future
    matrix-free solver contract accepts a matvec in place of ``A``; the two
    implementations shipped here intentionally support dense tensors only.
    """

    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Return ``x`` such that ``(A + ridge I) x ≈ b``."""
        ...


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

    if isinstance(ridge, torch.Tensor):
        if ridge.dtype != A.dtype or ridge.device != A.device:
            raise ValueError(
                "Tensor ridge must share A's dtype and device; "
                f"got ridge ({ridge.dtype}, {ridge.device}) and "
                f"A ({A.dtype}, {A.device})"
            )
        ridge_tensor = ridge
    else:
        ridge_tensor = A.new_tensor(ridge)

    batch_shape = A.shape[:-2]
    try:
        ridge_tensor = torch.broadcast_to(ridge_tensor, batch_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"ridge shape {tuple(ridge_tensor.shape)} is not broadcastable to A batch shape {tuple(batch_shape)}"
        ) from exc

    regularized = A.clone()
    regularized.diagonal(dim1=-2, dim2=-1).add_(ridge_tensor[..., None])
    return regularized
