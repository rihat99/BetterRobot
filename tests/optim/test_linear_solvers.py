"""Dense batched linear-solver contract tests for M2b."""

from __future__ import annotations

import pytest
import torch

from better_robot.optim.solvers import Cholesky, LSTSQ
from better_robot.optim.solvers.base import LinearSolver


@pytest.mark.parametrize("solver_cls", [Cholesky, LSTSQ])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dense_solvers_accept_arbitrary_batch_axes(solver_cls: type[LinearSolver], dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(17)
    raw = torch.randn(2, 3, 4, 4, generator=generator, dtype=dtype)
    matrix = raw.mT @ raw + 0.5 * torch.eye(4, dtype=dtype)
    rhs = torch.randn(2, 3, 4, generator=generator, dtype=dtype)

    actual = solver_cls().solve(matrix, rhs)
    expected = torch.linalg.solve(matrix, rhs)

    assert actual.shape == rhs.shape
    assert actual.dtype == dtype
    assert torch.allclose(actual, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("solver_cls", [Cholesky, LSTSQ])
def test_dense_solvers_apply_per_element_ridge(
    solver_cls: type[LinearSolver],
) -> None:
    matrix = torch.tensor(
        [
            [[[2.0, 0.2], [0.2, 1.0]]],
            [[[1.5, -0.1], [-0.1, 3.0]]],
        ],
        dtype=torch.float64,
    ).expand(2, 3, 2, 2)
    rhs = torch.tensor([1.0, -2.0], dtype=torch.float64).expand(2, 3, 2)
    ridge = torch.tensor([[0.0, 0.25, 0.5], [1.0, 1.5, 2.0]], dtype=torch.float64)
    matrix_before = matrix.clone()

    actual = solver_cls().solve(matrix, rhs, ridge=ridge)
    expected_matrix = matrix.clone()
    expected_matrix.diagonal(dim1=-2, dim2=-1).add_(ridge[..., None])
    expected = torch.linalg.solve(expected_matrix, rhs)

    assert torch.equal(matrix, matrix_before)
    assert actual.shape == rhs.shape
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("solver_cls", [Cholesky, LSTSQ])
def test_dense_solvers_accept_scalar_ridge_and_legacy_two_arg_call(
    solver_cls: type[LinearSolver],
) -> None:
    matrix = torch.eye(3, dtype=torch.float32).expand(4, 3, 3)
    rhs = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    assert torch.equal(solver_cls().solve(matrix, rhs), rhs)
    assert torch.allclose(solver_cls().solve(matrix, rhs, ridge=1.0), rhs / 2)


@pytest.mark.parametrize("solver_cls", [Cholesky, LSTSQ])
def test_tensor_ridge_must_share_working_dtype_and_batch_shape(
    solver_cls: type[LinearSolver],
) -> None:
    matrix = torch.eye(2, dtype=torch.float32).expand(3, 2, 2)
    rhs = torch.ones(3, 2, dtype=torch.float32)

    with pytest.raises(ValueError, match="share A's dtype and device"):
        solver_cls().solve(matrix, rhs, ridge=torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="not broadcastable"):
        solver_cls().solve(matrix, rhs, ridge=torch.ones(4, dtype=torch.float32))


def test_cholesky_fallback_is_independent_per_batch_element() -> None:
    matrix = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 4.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[2.0, 8.0], [2.0, 2.0]], dtype=torch.float64)

    actual = Cholesky().solve(matrix, rhs)

    assert torch.allclose(actual[0], torch.tensor([1.0, 2.0], dtype=actual.dtype))
    assert torch.allclose(matrix[1] @ actual[1], rhs[1], rtol=1e-10, atol=1e-12)
    assert torch.isfinite(actual).all()
