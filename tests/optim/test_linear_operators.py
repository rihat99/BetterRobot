"""Matrix-free normal-CG and informative dense-solver contracts."""

from __future__ import annotations

import pytest
import torch

from better_robot.optim.solvers import (
    BandedCholesky,
    Cholesky,
    InformativeLinearSolver,
    LinearSolveStatus,
    LinearSolver,
    NormalCG,
)
from better_robot.optim.structure import NormalOperator


def _operator(matrix: torch.Tensor, *, precondition: bool = True) -> NormalOperator:
    diagonal = matrix.diagonal(dim1=-2, dim2=-1)
    return NormalOperator(
        size=matrix.shape[-1],
        matvec=lambda vector: (matrix @ vector.unsqueeze(-1)).squeeze(-1),
        preconditioner=(lambda vector: vector / diagonal) if precondition else None,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normal_cg_matches_dense_with_ridge_and_batch_axes(dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260718)
    raw = torch.randn(2, 3, 7, 7, dtype=dtype, generator=generator)
    matrix = raw.mT @ raw + 0.5 * torch.eye(7, dtype=dtype)
    rhs = torch.randn(2, 3, 7, dtype=dtype, generator=generator)
    ridge = torch.tensor([[0.0, 0.05, 0.1], [0.2, 0.4, 0.8]], dtype=dtype)
    solver = NormalCG(max_iter=80, rtol=2e-6 if dtype == torch.float32 else 1e-11)

    result = solver.solve_with_info(_operator(matrix), rhs, ridge=ridge)
    regularized = matrix.clone()
    regularized.diagonal(dim1=-2, dim2=-1).add_(ridge[..., None])
    expected = torch.linalg.solve(regularized, rhs)

    tolerance = 3e-4 if dtype == torch.float32 else 2e-9
    torch.testing.assert_close(result.solution, expected, rtol=tolerance, atol=tolerance * 0.1)
    assert result.ok.all()
    assert result.converged.all()
    assert result.finite.all()
    assert (result.status == LinearSolveStatus.SUCCESS).all()
    assert (result.iterations > 0).all()
    assert (result.iterations <= solver.max_iter).all()
    assert torch.all(result.relative_residual <= solver.rtol * 1.1)


def test_exact_warm_start_converges_without_an_iteration() -> None:
    matrix = torch.tensor([[3.0, 0.4], [0.4, 2.0]], dtype=torch.float64)
    expected = torch.tensor([0.2, -0.7], dtype=torch.float64)
    rhs = matrix @ expected

    result = NormalCG(max_iter=10, rtol=1e-12).solve_with_info(
        _operator(matrix),
        rhs,
        initial=expected,
    )

    torch.testing.assert_close(result.solution, expected, rtol=0.0, atol=0.0)
    assert bool(result.ok)
    assert int(result.iterations) == 0
    assert int(result.status) == LinearSolveStatus.SUCCESS


def test_cg_failure_statuses_are_isolated_and_solutions_are_safe() -> None:
    matrix = torch.stack(
        (
            torch.eye(2),
            -torch.eye(2),
            torch.full((2, 2), torch.nan),
        )
    ).to(dtype=torch.float64)
    rhs = torch.ones(3, 2, dtype=torch.float64)

    result = NormalCG(max_iter=5, rtol=1e-12).solve_with_info(_operator(matrix), rhs)

    torch.testing.assert_close(result.solution[0], rhs[0])
    assert torch.equal(result.solution[1:], torch.zeros_like(result.solution[1:]))
    torch.testing.assert_close(result.ok, torch.tensor([True, False, False]))
    torch.testing.assert_close(result.converged, torch.tensor([True, False, False]))
    torch.testing.assert_close(result.finite, torch.tensor([True, True, False]))
    torch.testing.assert_close(
        result.status,
        torch.tensor(
            [LinearSolveStatus.SUCCESS, LinearSolveStatus.BREAKDOWN, LinearSolveStatus.NONFINITE],
            dtype=torch.int8,
        ),
    )


def test_iteration_cap_reports_max_iter_and_returns_safe_zero() -> None:
    matrix = torch.tensor([[2.0, 0.5], [0.5, 1.5]], dtype=torch.float64)
    rhs = torch.ones(2, dtype=torch.float64)

    result = NormalCG(max_iter=0, rtol=0.0, atol=0.0).solve_with_info(_operator(matrix), rhs)

    assert not bool(result.ok)
    assert bool(result.finite)
    assert int(result.iterations) == 0
    assert int(result.status) == LinearSolveStatus.MAX_ITER
    assert torch.equal(result.solution, torch.zeros_like(rhs))


def test_cg_validates_operator_and_warm_start_contracts() -> None:
    rhs = torch.ones(2, 3)
    wrong_shape = NormalOperator(size=3, matvec=lambda vector: vector[..., :-1])
    wrong_dtype = NormalOperator(size=3, matvec=lambda vector: vector.double())

    with pytest.raises(ValueError, match="matvec must return shape"):
        NormalCG(max_iter=1).solve(wrong_shape, rhs)
    with pytest.raises(ValueError, match="preserve dtype/device"):
        NormalCG(max_iter=1).solve(wrong_dtype, rhs)
    with pytest.raises(ValueError, match="initial must have shape"):
        NormalCG(max_iter=1).solve_with_info(
            NormalOperator(size=3, matvec=lambda vector: vector),
            rhs,
            initial=torch.zeros(3),
        )
    with pytest.raises(ValueError, match="share A's dtype and device"):
        NormalCG(max_iter=1).solve(
            NormalOperator(size=3, matvec=lambda vector: vector),
            rhs,
            ridge=torch.ones(2, dtype=torch.float64),
        )


def test_cholesky_informative_path_is_strict_but_public_fallback_survives() -> None:
    matrix = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 4.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[2.0, 8.0], [2.0, 2.0]], dtype=torch.float64)
    solver = Cholesky()

    public_solution = solver.solve(matrix, rhs)
    result = solver.solve_with_info(matrix, rhs)

    assert torch.isfinite(public_solution).all()
    torch.testing.assert_close(matrix[1] @ public_solution[1], rhs[1], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(result.solution[0], torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert torch.equal(result.solution[1], torch.zeros_like(result.solution[1]))
    torch.testing.assert_close(result.ok, torch.tensor([True, False]))
    torch.testing.assert_close(result.finite, torch.tensor([True, True]))
    torch.testing.assert_close(
        result.status,
        torch.tensor([LinearSolveStatus.SUCCESS, LinearSolveStatus.NOT_SPD], dtype=torch.int8),
    )


def test_new_solvers_preserve_structural_protocols() -> None:
    assert isinstance(BandedCholesky(), LinearSolver)
    assert isinstance(BandedCholesky(), InformativeLinearSolver)
    assert isinstance(Cholesky(), LinearSolver)
    assert isinstance(Cholesky(), InformativeLinearSolver)
    assert isinstance(NormalCG(), LinearSolver)
    assert isinstance(NormalCG(), InformativeLinearSolver)
