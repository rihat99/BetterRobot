"""Dense and block-banded direct linear solvers."""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple, Protocol, TypeAlias, runtime_checkable

import torch

from .temporal import BlockBandedMatrix

LinearSystem: TypeAlias = torch.Tensor | BlockBandedMatrix


class LinearSolveStatus(IntEnum):
    SUCCESS = 0
    MAX_ITER = 1
    NONFINITE = 2
    BREAKDOWN = 3
    NOT_SPD = 4


class LinearSolveResult(NamedTuple):
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
    def solve(
        self,
        A: LinearSystem,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor: ...


@runtime_checkable
class InformativeLinearSolver(Protocol):
    def solve_with_info(
        self,
        A: LinearSystem,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
        *,
        initial: torch.Tensor | None = None,
    ) -> LinearSolveResult: ...


def _broadcast_ridge(
    ridge: torch.Tensor | float | None,
    batch_shape: tuple[int, ...],
    exemplar: torch.Tensor,
) -> torch.Tensor | None:
    if ridge is None:
        return None
    if isinstance(ridge, torch.Tensor):
        if ridge.dtype != exemplar.dtype or ridge.device != exemplar.device:
            raise ValueError(
                "Tensor ridge must share A's dtype and device; "
                f"got ridge ({ridge.dtype}, {ridge.device}) and A ({exemplar.dtype}, {exemplar.device})"
            )
        value = ridge
    else:
        value = exemplar.new_tensor(ridge)
    try:
        return torch.broadcast_to(value, batch_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"ridge shape {tuple(value.shape)} is not broadcastable to A batch shape {batch_shape}"
        ) from exc


def _regularized(A: torch.Tensor, ridge: torch.Tensor | float | None) -> torch.Tensor:
    if ridge is None:
        return A
    if A.ndim < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError(f"A must have shape (B..., n, n); received {tuple(A.shape)}")
    value = _broadcast_ridge(ridge, tuple(A.shape[:-2]), A)
    assert value is not None
    result = A.clone()
    result.diagonal(dim1=-2, dim2=-1).add_(value[..., None])
    return result


def _direct_result(
    solution: torch.Tensor,
    residual: torch.Tensor,
    rhs: torch.Tensor,
    succeeded: torch.Tensor,
    nonfinite: torch.Tensor,
    factorization_failed: torch.Tensor,
    *,
    failure_status: LinearSolveStatus = LinearSolveStatus.NOT_SPD,
) -> LinearSolveResult:
    solution_finite = torch.isfinite(solution).all(dim=-1)
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=-1)
    relative = residual_norm / rhs_norm.clamp_min(torch.finfo(rhs.dtype).tiny)
    metrics_finite = torch.isfinite(residual_norm) & torch.isfinite(relative)
    finite = ~nonfinite & solution_finite & metrics_finite
    ok = succeeded & finite
    infinity = torch.full_like(residual_norm, torch.inf)
    status = torch.where(
        nonfinite | ~solution_finite | ~metrics_finite,
        torch.full_like(ok, LinearSolveStatus.NONFINITE, dtype=torch.int8),
        torch.where(
            factorization_failed,
            torch.full_like(ok, failure_status, dtype=torch.int8),
            torch.full_like(ok, LinearSolveStatus.SUCCESS, dtype=torch.int8),
        ),
    )
    return LinearSolveResult(
        solution=torch.where(ok[..., None], solution, torch.zeros_like(solution)),
        converged=ok,
        finite=finite,
        ok=ok,
        iterations=torch.zeros_like(ok, dtype=torch.int64),
        residual_norm=torch.where(metrics_finite, residual_norm, infinity),
        relative_residual=torch.where(metrics_finite, relative, infinity),
        status=status,
    )


class Cholesky:
    """Dense batched Cholesky solver."""

    supported_systems = frozenset(("dense",))

    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        matrix = _regularized(A, ridge)
        factor = torch.linalg.cholesky(matrix)
        return torch.cholesky_solve(b.to(A.dtype).unsqueeze(-1), factor).squeeze(-1).to(b.dtype)

    def solve_with_info(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
        *,
        initial: torch.Tensor | None = None,
    ) -> LinearSolveResult:
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

        matrix, rhs = _regularized(A, ridge), b.to(A.dtype)
        factor, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
        factor_ok = info == 0
        identity = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        safe_factor = torch.where(factor_ok[..., None, None], factor, identity)
        solution = torch.cholesky_solve(rhs.unsqueeze(-1), safe_factor).squeeze(-1)
        solution = torch.where(factor_ok[..., None], solution, torch.zeros_like(solution))
        residual = (matrix @ solution.unsqueeze(-1)).squeeze(-1) - rhs
        nonfinite = ~torch.isfinite(matrix).all(dim=(-2, -1)) | ~torch.isfinite(rhs).all(dim=-1)
        result = _direct_result(solution, residual, rhs, factor_ok, nonfinite, ~factor_ok & ~nonfinite)
        return result._replace(solution=result.solution.to(b.dtype))


class LU:
    """Dense batched LU solver for general square systems."""

    supported_systems = frozenset(("dense",))

    @staticmethod
    def _validate(A: torch.Tensor, b: torch.Tensor, initial: torch.Tensor | None) -> None:
        if initial is not None:
            raise ValueError("LU does not accept an initial solution")
        if not isinstance(A, torch.Tensor):
            raise TypeError("LU requires a dense torch.Tensor system")
        if not isinstance(b, torch.Tensor):
            raise TypeError("b must be a torch.Tensor")
        if A.ndim < 2 or A.shape[-2] != A.shape[-1]:
            raise ValueError(f"A must have shape (B..., n, n); received {tuple(A.shape)}")
        expected = (*A.shape[:-2], A.shape[-1])
        if tuple(b.shape) != expected:
            raise ValueError(f"b must have shape {expected}, got {tuple(b.shape)}")
        if b.device != A.device:
            raise ValueError(f"b must share A's device {A.device}, got {b.device}")

    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        self._validate(A, b, None)
        matrix, rhs = _regularized(A, ridge), b.to(A.dtype)
        factor, pivots = torch.linalg.lu_factor(matrix)
        return torch.linalg.lu_solve(factor, pivots, rhs.unsqueeze(-1)).squeeze(-1).to(b.dtype)

    def solve_with_info(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
        *,
        initial: torch.Tensor | None = None,
    ) -> LinearSolveResult:
        self._validate(A, b, initial)
        matrix, rhs = _regularized(A, ridge), b.to(A.dtype)
        factor, pivots, info = torch.linalg.lu_factor_ex(matrix, check_errors=False)
        factor_ok = info == 0
        identity = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        safe_factor = torch.where(factor_ok[..., None, None], factor, identity)
        identity_pivots = torch.arange(1, A.shape[-1] + 1, dtype=pivots.dtype, device=A.device)
        safe_pivots = torch.where(factor_ok[..., None], pivots, identity_pivots)
        solution = torch.linalg.lu_solve(safe_factor, safe_pivots, rhs.unsqueeze(-1)).squeeze(-1)
        solution = torch.where(factor_ok[..., None], solution, torch.zeros_like(solution))
        residual = (matrix @ solution.unsqueeze(-1)).squeeze(-1) - rhs
        nonfinite = ~torch.isfinite(matrix).all(dim=(-2, -1)) | ~torch.isfinite(rhs).all(dim=-1)
        result = _direct_result(
            solution,
            residual,
            rhs,
            factor_ok,
            nonfinite,
            ~factor_ok & ~nonfinite,
            failure_status=LinearSolveStatus.BREAKDOWN,
        )
        return result._replace(solution=result.solution.to(b.dtype))


class BandedCholesky:
    """Direct Cholesky solver for padded lower block bands."""

    supported_systems = frozenset(("banded",))
    supports_initial = False

    @staticmethod
    def _validate(A: BlockBandedMatrix, b: torch.Tensor, initial: torch.Tensor | None) -> None:
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
        return self.solve_with_info(A, b, ridge).solution

    def solve_with_info(  # noqa: PLR0915
        self,
        A: BlockBandedMatrix,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
        *,
        initial: torch.Tensor | None = None,
    ) -> LinearSolveResult:
        self._validate(A, b, initial)
        bands = A.bands
        rhs = b.to(bands.dtype).reshape(*A.batch_shape, A.time_length, A.block_size)
        ridge_value = _broadcast_ridge(ridge, A.batch_shape, bands)
        eye = torch.eye(A.block_size, dtype=bands.dtype, device=bands.device)
        zero_block, zero_vector = torch.zeros_like(bands[..., 0, 0, :, :]), torch.zeros_like(rhs[..., 0, :])
        input_finite = A.finite & torch.isfinite(rhs).all(dim=(-2, -1))
        if ridge_value is not None:
            input_finite = input_finite & torch.isfinite(ridge_value)
        running, saw_nonfinite = input_finite, ~input_finite
        saw_not_spd = torch.zeros_like(input_finite)
        factors = [[zero_block for _ in range(A.bandwidth + 1)] for _ in range(A.time_length)]

        # Lists keep earlier autograd operands independent of later factor writes.
        for time in range(A.time_length):
            schur = bands[..., time, 0, :, :]
            if ridge_value is not None:
                schur = schur + ridge_value[..., None, None] * eye
            for offset in range(1, min(A.bandwidth, time) + 1):
                schur = schur - factors[time][offset] @ factors[time][offset].mT
            schur_finite = torch.isfinite(schur).all(dim=(-2, -1))
            factor, info = torch.linalg.cholesky_ex(
                torch.where((running & schur_finite)[..., None, None], schur, eye), check_errors=False
            )
            saw_not_spd = saw_not_spd | (running & schur_finite & (info != 0))
            saw_nonfinite = saw_nonfinite | (running & ~schur_finite)
            running = running & schur_finite & (info == 0)
            factor = torch.where(running[..., None, None], factor, eye)
            factors[time][0] = factor

            for row in range(time + 1, min(A.time_length, time + A.bandwidth + 1)):
                offset = row - time
                block = bands[..., row, offset, :, :]
                for shared in range(max(0, row - A.bandwidth, time - A.bandwidth), time):
                    block = block - factors[row][row - shared] @ factors[time][time - shared].mT
                block_finite = torch.isfinite(block).all(dim=(-2, -1))
                solved = torch.linalg.solve_triangular(
                    factor,
                    torch.where((running & block_finite)[..., None, None], block, zero_block).mT,
                    upper=False,
                ).mT
                solved_finite = torch.isfinite(solved).all(dim=(-2, -1))
                saw_nonfinite = saw_nonfinite | (running & (~block_finite | ~solved_finite))
                running = running & block_finite & solved_finite
                factors[row][offset] = torch.where(running[..., None, None], solved, zero_block)

        factor_ok = running & ~saw_nonfinite & ~saw_not_spd
        forward: list[torch.Tensor] = []
        for time in range(A.time_length):
            value = torch.where(factor_ok[..., None], rhs[..., time, :], zero_vector)
            for offset in range(1, min(A.bandwidth, time) + 1):
                value = value - (factors[time][offset] @ forward[time - offset].unsqueeze(-1)).squeeze(-1)
            value = torch.linalg.solve_triangular(factors[time][0], value.unsqueeze(-1), upper=False).squeeze(-1)
            forward.append(torch.where(factor_ok[..., None], value, zero_vector))

        reverse = [zero_vector for _ in range(A.time_length)]
        for time in range(A.time_length - 1, -1, -1):
            value = forward[time]
            for row in range(time + 1, min(A.time_length, time + A.bandwidth + 1)):
                value = value - (factors[row][row - time].mT @ reverse[row].unsqueeze(-1)).squeeze(-1)
            value = torch.linalg.solve_triangular(factors[time][0].mT, value.unsqueeze(-1), upper=True).squeeze(-1)
            reverse[time] = torch.where(factor_ok[..., None], value, zero_vector)

        solution = torch.stack(reverse, dim=-2).reshape(*A.batch_shape, A.size)
        solution = torch.where(factor_ok[..., None], solution, torch.zeros_like(solution))
        residual = A.matvec(solution) - b.to(bands.dtype)
        if ridge_value is not None:
            residual += ridge_value[..., None] * solution
        result = _direct_result(solution, residual, b.to(bands.dtype), factor_ok, saw_nonfinite, saw_not_spd)
        return result._replace(solution=result.solution.to(b.dtype))


__all__ = [
    "BandedCholesky",
    "Cholesky",
    "InformativeLinearSolver",
    "LU",
    "LinearSolveResult",
    "LinearSolveStatus",
    "LinearSolver",
    "LinearSystem",
]
