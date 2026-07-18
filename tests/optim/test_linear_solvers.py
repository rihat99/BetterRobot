"""Dense batched Cholesky contract tests."""

from __future__ import annotations

import pytest
import torch

from better_robot.optim import Cholesky


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_accepts_arbitrary_batch_axes(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(17)
    raw = torch.randn(2, 3, 4, 4, generator=generator, dtype=dtype)
    matrix = raw.mT @ raw + 0.5 * torch.eye(4, dtype=dtype)
    rhs = torch.randn(2, 3, 4, generator=generator, dtype=dtype)

    actual = Cholesky().solve(matrix, rhs)
    expected = torch.linalg.solve(matrix, rhs)

    assert actual.shape == rhs.shape
    assert actual.dtype == dtype
    assert torch.allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test_cholesky_applies_per_element_ridge() -> None:
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

    actual = Cholesky().solve(matrix, rhs, ridge=ridge)
    expected_matrix = matrix.clone()
    expected_matrix.diagonal(dim1=-2, dim2=-1).add_(ridge[..., None])
    expected = torch.linalg.solve(expected_matrix, rhs)

    assert torch.equal(matrix, matrix_before)
    assert actual.shape == rhs.shape
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_cholesky_accepts_scalar_ridge_and_two_arg_call() -> None:
    matrix = torch.eye(3, dtype=torch.float32).expand(4, 3, 3)
    rhs = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    assert torch.equal(Cholesky().solve(matrix, rhs), rhs)
    assert torch.allclose(Cholesky().solve(matrix, rhs, ridge=1.0), rhs / 2)


def test_tensor_ridge_must_share_working_dtype_and_batch_shape() -> None:
    matrix = torch.eye(2, dtype=torch.float32).expand(3, 2, 2)
    rhs = torch.ones(3, 2, dtype=torch.float32)

    with pytest.raises(ValueError, match="share A's dtype and device"):
        Cholesky().solve(matrix, rhs, ridge=torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="not broadcastable"):
        Cholesky().solve(matrix, rhs, ridge=torch.ones(4, dtype=torch.float32))
