"""Fixed-work batched conjugate gradients for sized normal operators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import torch

from ..structure import NormalOperator
from .base import LinearSolveResult, LinearSolveStatus, _broadcast_ridge


@dataclass(frozen=True)
class NormalCG:
    """Preconditioned CG with tensor-masked per-element convergence."""

    max_iter: int = 100
    rtol: float = 1e-5
    atol: float = 0.0
    supported_systems: ClassVar[frozenset[str]] = frozenset(("operator",))
    supports_initial: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("NormalCG max_iter must be a non-negative int")
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if isinstance(value, (bool, torch.Tensor)) or not isinstance(value, (int, float)):
                raise TypeError(f"NormalCG {name} must be a non-negative Python number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"NormalCG {name} must be a finite non-negative Python number")

    @staticmethod
    def _validate_operand(name: str, value: torch.Tensor, reference: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"NormalOperator {name} must return a torch.Tensor")
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"NormalOperator {name} must return shape {tuple(reference.shape)}, got {tuple(value.shape)}"
            )
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(
                f"NormalOperator {name} must preserve dtype/device "
                f"{reference.dtype}/{reference.device}, got {value.dtype}/{value.device}"
            )

    def solve(
        self,
        A: NormalOperator,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Return the solution from a zero initial value."""
        return self.solve_with_info(A, b, ridge).solution

    def solve_with_info(  # noqa: PLR0915 - fixed-work recurrence keeps per-element status auditable
        self,
        A: NormalOperator,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
        *,
        initial: torch.Tensor | None = None,
    ) -> LinearSolveResult:
        """Run a fixed iteration cap and return independent batch diagnostics."""
        if not isinstance(A, NormalOperator):
            raise TypeError("NormalCG requires a NormalOperator")
        if not isinstance(b, torch.Tensor):
            raise TypeError("b must be a torch.Tensor")
        if not b.is_floating_point():
            raise TypeError("b must use a floating dtype")
        if b.ndim < 1 or b.shape[-1] != A.size:
            raise ValueError(f"b must end in operator size {A.size}, got {tuple(b.shape)}")
        if initial is not None:
            if not isinstance(initial, torch.Tensor):
                raise TypeError("NormalCG initial must be a torch.Tensor or None")
            if tuple(initial.shape) != tuple(b.shape):
                raise ValueError(f"NormalCG initial must have shape {tuple(b.shape)}, got {tuple(initial.shape)}")
            if initial.dtype != b.dtype or initial.device != b.device:
                raise ValueError(
                    "NormalCG initial must share b's dtype and device; "
                    f"got {initial.dtype}/{initial.device} and {b.dtype}/{b.device}"
                )

        batch_shape = tuple(b.shape[:-1])
        ridge_tensor = _broadcast_ridge(ridge, batch_shape=batch_shape, exemplar=b)

        def matvec(vector: torch.Tensor) -> torch.Tensor:
            output = A(vector)
            self._validate_operand("matvec", output, vector)
            if ridge_tensor is not None:
                output = output + ridge_tensor[..., None] * vector
            return output

        def precondition(vector: torch.Tensor) -> torch.Tensor:
            if A.preconditioner is None:
                return vector
            output = A.preconditioner(vector)
            self._validate_operand("preconditioner", output, vector)
            return output

        x = torch.zeros_like(b) if initial is None else initial.clone()
        zero_vector = torch.zeros_like(b)
        Ax = matvec(x)
        residual = b - Ax
        z = precondition(residual)
        rz = (residual * z).sum(dim=-1)
        rhs_norm = torch.linalg.vector_norm(b, dim=-1)
        residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        relative_residual = residual_norm / rhs_norm.clamp(min=torch.finfo(b.dtype).tiny)
        threshold = self.atol + self.rtol * rhs_norm

        finite = (
            torch.isfinite(b).all(dim=-1)
            & torch.isfinite(x).all(dim=-1)
            & torch.isfinite(Ax).all(dim=-1)
            & torch.isfinite(residual).all(dim=-1)
            & torch.isfinite(z).all(dim=-1)
            & torch.isfinite(rz)
            & torch.isfinite(residual_norm)
            & torch.isfinite(relative_residual)
        )
        if ridge_tensor is not None:
            finite = finite & torch.isfinite(ridge_tensor)
        ever_nonfinite = ~finite
        converged = finite & (residual_norm <= threshold)
        initial_breakdown = finite & ~converged & (rz <= 0.0)

        success_value = torch.full_like(converged, LinearSolveStatus.SUCCESS, dtype=torch.int8)
        max_iter_value = torch.full_like(success_value, LinearSolveStatus.MAX_ITER)
        nonfinite_value = torch.full_like(success_value, LinearSolveStatus.NONFINITE)
        breakdown_value = torch.full_like(success_value, LinearSolveStatus.BREAKDOWN)
        status = torch.where(
            ever_nonfinite,
            nonfinite_value,
            torch.where(converged, success_value, torch.where(initial_breakdown, breakdown_value, max_iter_value)),
        )
        iterations = torch.zeros_like(converged, dtype=torch.int64)
        p = torch.where((status == LinearSolveStatus.MAX_ITER)[..., None], z, zero_vector)

        for _iteration in range(self.max_iter):
            live = status == LinearSolveStatus.MAX_ITER
            Ap = matvec(p)
            denominator = (p * Ap).sum(dim=-1)
            recurrence_finite = (
                torch.isfinite(Ap).all(dim=-1)
                & torch.isfinite(denominator)
                & torch.isfinite(rz)
            )
            recurrence_nonfinite = live & ~recurrence_finite
            recurrence_breakdown = live & recurrence_finite & ((denominator <= 0.0) | (rz <= 0.0))
            can_update = live & recurrence_finite & ~recurrence_breakdown
            safe_denominator = torch.where(can_update, denominator, torch.ones_like(denominator))
            alpha = torch.where(can_update, rz / safe_denominator, torch.zeros_like(rz))
            candidate_x = x + alpha[..., None] * p
            candidate_residual = residual - alpha[..., None] * Ap
            preconditioner_input = torch.where(can_update[..., None], candidate_residual, zero_vector)
            candidate_z = precondition(preconditioner_input)
            candidate_rz = (candidate_residual * candidate_z).sum(dim=-1)
            candidate_residual_norm = torch.linalg.vector_norm(candidate_residual, dim=-1)
            candidate_relative_residual = candidate_residual_norm / rhs_norm.clamp(min=torch.finfo(b.dtype).tiny)
            candidate_finite = (
                torch.isfinite(candidate_x).all(dim=-1)
                & torch.isfinite(candidate_residual).all(dim=-1)
                & torch.isfinite(candidate_z).all(dim=-1)
                & torch.isfinite(candidate_rz)
                & torch.isfinite(candidate_residual_norm)
                & torch.isfinite(candidate_relative_residual)
            )
            candidate_nonfinite = can_update & ~candidate_finite
            accepted_recurrence = can_update & candidate_finite
            x = torch.where(accepted_recurrence[..., None], candidate_x, x)
            residual = torch.where(accepted_recurrence[..., None], candidate_residual, residual)
            z = torch.where(accepted_recurrence[..., None], candidate_z, z)
            residual_norm = torch.where(accepted_recurrence, candidate_residual_norm, residual_norm)
            relative_residual = torch.where(accepted_recurrence, candidate_relative_residual, relative_residual)
            iterations = iterations + can_update.to(dtype=iterations.dtype)

            just_converged = accepted_recurrence & (candidate_residual_norm <= threshold)
            continuing = accepted_recurrence & ~just_converged
            beta_breakdown = continuing & (candidate_rz <= 0.0)
            safe_rz = torch.where(continuing & ~beta_breakdown, rz, torch.ones_like(rz))
            beta = torch.where(continuing & ~beta_breakdown, candidate_rz / safe_rz, torch.zeros_like(rz))
            candidate_p = candidate_z + beta[..., None] * p
            direction_nonfinite = continuing & ~beta_breakdown & ~torch.isfinite(candidate_p).all(dim=-1)

            ever_nonfinite = ever_nonfinite | recurrence_nonfinite | candidate_nonfinite | direction_nonfinite
            status = torch.where(recurrence_nonfinite, nonfinite_value, status)
            status = torch.where(recurrence_breakdown, breakdown_value, status)
            status = torch.where(candidate_nonfinite | direction_nonfinite, nonfinite_value, status)
            status = torch.where(beta_breakdown, breakdown_value, status)
            status = torch.where(just_converged, success_value, status)
            converged = converged | just_converged
            rz = torch.where(accepted_recurrence, candidate_rz, rz)
            keep_direction = continuing & ~beta_breakdown & ~direction_nonfinite
            p = torch.where(keep_direction[..., None], candidate_p, zero_vector)

        metrics_finite = torch.isfinite(residual_norm) & torch.isfinite(relative_residual)
        finite = ~ever_nonfinite & metrics_finite & torch.isfinite(x).all(dim=-1)
        converged = converged & (status == LinearSolveStatus.SUCCESS)
        ok = converged & finite
        infinity = torch.full_like(residual_norm, torch.inf)
        residual_norm = torch.where(metrics_finite, residual_norm, infinity)
        relative_residual = torch.where(metrics_finite, relative_residual, infinity)
        solution = torch.where(ok[..., None], x, zero_vector)
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


__all__ = ["NormalCG"]
