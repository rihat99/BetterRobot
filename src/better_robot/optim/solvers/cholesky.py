"""Cholesky linear solver — dense SPD.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

import torch

from .base import LinearSolveResult, LinearSolveStatus, _regularized_matrix


class Cholesky:
    """Dense batched Cholesky solver for ``(A + ridge I) x = b``.

    Failed batch elements use a least-squares fallback while successful
    elements retain their Cholesky solutions. Capture-ready LM handles its
    stricter zero-step/info-mask policy in the solver update itself.
    """

    supported_systems = frozenset(("dense",))

    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Solve the dense system, preserving the right-hand-side dtype."""
        matrix = _regularized_matrix(A, ridge)
        rhs = b.to(A.dtype)
        factor, info = torch.linalg.cholesky_ex(matrix)
        chol_solution = torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)
        lstsq_solution = torch.linalg.lstsq(matrix, rhs).solution
        ok = info == 0
        solution = torch.where(
            ok[..., None],
            torch.nan_to_num(chol_solution),
            lstsq_solution,
        )
        return solution.to(b.dtype)

    def solve_with_info(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
        *,
        initial: torch.Tensor | None = None,
    ) -> LinearSolveResult:
        """Solve without the public least-squares fallback and report health.

        Failed elements receive a safe zero solution. The existing
        :meth:`solve` method deliberately keeps its rank-deficient LSTSQ
        fallback for backward compatibility with direct callers.
        """
        if initial is not None:
            raise ValueError("Cholesky does not accept an initial solution")
        if not isinstance(A, torch.Tensor):
            raise TypeError("Cholesky requires a dense torch.Tensor system")
        if not isinstance(b, torch.Tensor):
            raise TypeError("b must be a torch.Tensor")
        if A.ndim < 2 or A.shape[-2] != A.shape[-1]:
            raise ValueError(f"A must have shape (B..., n, n); received {tuple(A.shape)}")
        expected = (*A.shape[:-2], A.shape[-1])
        if tuple(b.shape) != expected:
            raise ValueError(f"b must have shape {expected}, got {tuple(b.shape)}")
        if b.device != A.device:
            raise ValueError(f"b must share A's device {A.device}, got {b.device}")

        matrix = _regularized_matrix(A, ridge)
        rhs = b.to(dtype=A.dtype)
        factor, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
        factor_ok = info == 0
        identity = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        safe_factor = torch.where(factor_ok[..., None, None], factor, identity)
        raw_solution = torch.cholesky_solve(rhs.unsqueeze(-1), safe_factor).squeeze(-1)
        solution_work = torch.where(factor_ok[..., None], raw_solution, torch.zeros_like(raw_solution))

        residual = (matrix @ solution_work.unsqueeze(-1)).squeeze(-1) - rhs
        raw_residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        rhs_norm = torch.linalg.vector_norm(rhs, dim=-1)
        raw_relative_residual = raw_residual_norm / rhs_norm.clamp(min=torch.finfo(A.dtype).tiny)
        finite = (
            torch.isfinite(matrix).all(dim=(-2, -1))
            & torch.isfinite(rhs).all(dim=-1)
            & torch.isfinite(raw_solution).all(dim=-1)
            & torch.isfinite(raw_residual_norm)
            & torch.isfinite(raw_relative_residual)
        )
        infinity = torch.full_like(raw_residual_norm, torch.inf)
        residual_norm = torch.where(finite, raw_residual_norm, infinity)
        relative_residual = torch.where(finite, raw_relative_residual, infinity)
        converged = factor_ok & finite
        ok = converged

        success = torch.full_like(info, LinearSolveStatus.SUCCESS, dtype=torch.int8)
        not_spd = torch.full_like(success, LinearSolveStatus.NOT_SPD)
        nonfinite = torch.full_like(success, LinearSolveStatus.NONFINITE)
        status = torch.where(~finite, nonfinite, torch.where(factor_ok, success, not_spd))
        iterations = torch.zeros_like(info, dtype=torch.int64)
        solution = torch.where(ok[..., None], solution_work, torch.zeros_like(solution_work)).to(dtype=b.dtype)
        return LinearSolveResult(
            solution=solution,
            converged=converged,
            finite=finite,
            ok=ok,
            iterations=iterations,
            residual_norm=residual_norm,
            relative_residual=relative_residual,
            status=status,
        )
