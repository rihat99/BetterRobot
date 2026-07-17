"""Batched Cholesky solve for symmetric block-banded SPD systems."""

from __future__ import annotations

import torch

from ..structure import BlockBandedMatrix
from .base import (
    LinearSolveResult,
    LinearSolveStatus,
    _broadcast_ridge,
)


class BandedCholesky:
    """Solve padded lower-band systems in linear storage for fixed bandwidth."""

    supported_systems = frozenset(("banded",))
    supports_initial = False

    @staticmethod
    def _validate(
        A: BlockBandedMatrix,
        b: torch.Tensor,
        initial: torch.Tensor | None,
    ) -> None:
        if not isinstance(A, BlockBandedMatrix):
            raise TypeError("BandedCholesky requires a BlockBandedMatrix")
        if not isinstance(b, torch.Tensor):
            raise TypeError("b must be a torch.Tensor")
        expected = (*A.batch_shape, A.size)
        if tuple(b.shape) != expected:
            raise ValueError(f"b must have shape {expected}, got {tuple(b.shape)}")
        if b.device != A.bands.device:
            raise ValueError(f"b must share A's device {A.bands.device}, got {b.device}")
        if initial is not None:
            raise ValueError("BandedCholesky does not accept an initial solution")

    def solve(
        self,
        A: BlockBandedMatrix,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Return the solution, using safe zeros for failed batch elements."""
        return self.solve_with_info(A, b, ridge).solution

    def solve_with_info(  # noqa: PLR0915 - factor and two triangular sweeps share one status program
        self,
        A: BlockBandedMatrix,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
        *,
        initial: torch.Tensor | None = None,
    ) -> LinearSolveResult:
        """Factor and solve with independent status for every batch element."""
        self._validate(A, b, initial)
        bands = A.bands
        rhs = b.to(dtype=bands.dtype).reshape(*A.batch_shape, A.time_length, A.block_size)
        ridge_tensor = _broadcast_ridge(ridge, batch_shape=A.batch_shape, exemplar=bands)

        eye = torch.eye(A.block_size, dtype=bands.dtype, device=bands.device)
        zero_block = torch.zeros_like(bands[..., 0, 0, :, :])
        zero_vector = torch.zeros_like(rhs[..., 0, :])
        input_finite = A.finite & torch.isfinite(rhs).all(dim=(-2, -1))
        if ridge_tensor is not None:
            input_finite = input_finite & torch.isfinite(ridge_tensor)
        running_ok = input_finite
        saw_nonfinite = ~input_finite
        saw_not_spd = torch.zeros_like(input_finite)

        # ``factor_rows[t][k]`` stores L[t, t-k]. Python lists avoid
        # version-counter hazards from modifying a shared tensor after views of
        # earlier blocks have participated in autograd operations.
        factor_rows: list[list[torch.Tensor | None]] = [
            [None] * (min(A.bandwidth, time) + 1) for time in range(A.time_length)
        ]

        for time in range(A.time_length):
            schur = bands[..., time, 0, :, :]
            if ridge_tensor is not None:
                schur = schur + ridge_tensor[..., None, None] * eye
            for offset in range(1, min(A.bandwidth, time) + 1):
                block = factor_rows[time][offset]
                assert block is not None
                schur = schur - block @ block.mT

            schur_finite = torch.isfinite(schur).all(dim=(-2, -1))
            safe_schur = torch.where((running_ok & schur_finite)[..., None, None], schur, eye)
            diagonal_factor, info = torch.linalg.cholesky_ex(safe_schur, check_errors=False)
            pivot_not_spd = running_ok & schur_finite & (info != 0)
            pivot_nonfinite = running_ok & ~schur_finite
            saw_not_spd = saw_not_spd | pivot_not_spd
            saw_nonfinite = saw_nonfinite | pivot_nonfinite
            running_ok = running_ok & schur_finite & (info == 0)
            diagonal_factor = torch.where(running_ok[..., None, None], diagonal_factor, eye)
            factor_rows[time][0] = diagonal_factor

            for row in range(time + 1, min(A.time_length, time + A.bandwidth + 1)):
                offset = row - time
                off_diagonal = bands[..., row, offset, :, :]
                first_shared = max(0, row - A.bandwidth, time - A.bandwidth)
                for shared in range(first_shared, time):
                    row_block = factor_rows[row][row - shared]
                    time_block = factor_rows[time][time - shared]
                    assert row_block is not None and time_block is not None
                    off_diagonal = off_diagonal - row_block @ time_block.mT

                off_finite = torch.isfinite(off_diagonal).all(dim=(-2, -1))
                safe_off_diagonal = torch.where(
                    (running_ok & off_finite)[..., None, None],
                    off_diagonal,
                    zero_block,
                )
                solved = torch.linalg.solve_triangular(
                    diagonal_factor,
                    safe_off_diagonal.mT,
                    upper=False,
                ).mT
                solved_finite = torch.isfinite(solved).all(dim=(-2, -1))
                new_nonfinite = running_ok & (~off_finite | ~solved_finite)
                saw_nonfinite = saw_nonfinite | new_nonfinite
                running_ok = running_ok & off_finite & solved_finite
                factor_rows[row][offset] = torch.where(
                    running_ok[..., None, None],
                    solved,
                    zero_block,
                )

        factor_ok = running_ok & ~saw_nonfinite & ~saw_not_spd

        forward_values: list[torch.Tensor] = []
        for time in range(A.time_length):
            value = torch.where(factor_ok[..., None], rhs[..., time, :], zero_vector)
            for offset in range(1, min(A.bandwidth, time) + 1):
                block = factor_rows[time][offset]
                assert block is not None
                value = value - (block @ forward_values[time - offset].unsqueeze(-1)).squeeze(-1)
            diagonal_factor = factor_rows[time][0]
            assert diagonal_factor is not None
            solved = torch.linalg.solve_triangular(
                diagonal_factor,
                value.unsqueeze(-1),
                upper=False,
            ).squeeze(-1)
            forward_values.append(torch.where(factor_ok[..., None], solved, zero_vector))

        reverse_values: list[torch.Tensor | None] = [None] * A.time_length
        for time in range(A.time_length - 1, -1, -1):
            value = forward_values[time]
            for row in range(time + 1, min(A.time_length, time + A.bandwidth + 1)):
                block = factor_rows[row][row - time]
                later_value = reverse_values[row]
                assert block is not None and later_value is not None
                value = value - (block.mT @ later_value.unsqueeze(-1)).squeeze(-1)
            diagonal_factor = factor_rows[time][0]
            assert diagonal_factor is not None
            solved = torch.linalg.solve_triangular(
                diagonal_factor.mT,
                value.unsqueeze(-1),
                upper=True,
            ).squeeze(-1)
            reverse_values[time] = torch.where(factor_ok[..., None], solved, zero_vector)

        solution_work = torch.stack([value for value in reverse_values if value is not None], dim=-2).reshape(
            *A.batch_shape,
            A.size,
        )
        solution_finite = torch.isfinite(solution_work).all(dim=-1)
        saw_nonfinite = saw_nonfinite | ~solution_finite
        converged = factor_ok & solution_finite
        solution_work = torch.where(converged[..., None], solution_work, torch.zeros_like(solution_work))

        residual = A.matvec(solution_work) - b.to(dtype=bands.dtype)
        if ridge_tensor is not None:
            residual = residual + ridge_tensor[..., None] * solution_work
        raw_residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        rhs_norm = torch.linalg.vector_norm(b.to(dtype=bands.dtype), dim=-1)
        raw_relative_residual = raw_residual_norm / rhs_norm.clamp(min=torch.finfo(bands.dtype).tiny)
        metrics_finite = torch.isfinite(raw_residual_norm) & torch.isfinite(raw_relative_residual)
        finite = ~saw_nonfinite & metrics_finite
        ok = converged & finite
        infinity = torch.full_like(raw_residual_norm, torch.inf)
        residual_norm = torch.where(metrics_finite, raw_residual_norm, infinity)
        relative_residual = torch.where(metrics_finite, raw_relative_residual, infinity)

        success = torch.full_like(ok, LinearSolveStatus.SUCCESS, dtype=torch.int8)
        not_spd = torch.full_like(success, LinearSolveStatus.NOT_SPD)
        nonfinite = torch.full_like(success, LinearSolveStatus.NONFINITE)
        status = torch.where(saw_nonfinite | ~metrics_finite, nonfinite, torch.where(saw_not_spd, not_spd, success))
        iterations = torch.zeros_like(ok, dtype=torch.int64)
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


__all__ = ["BandedCholesky"]
