"""Parity and failure-isolation tests for block-banded Cholesky."""

from __future__ import annotations

import pytest
import torch

from better_robot.optim.solvers import BandedCholesky, LinearSolveStatus
from better_robot.optim.temporal import BlockBandedMatrix


def _banded_spd(
    *,
    batch_shape: tuple[int, ...],
    time_length: int,
    block_size: int,
    bandwidth: int,
    dtype: torch.dtype,
) -> tuple[BlockBandedMatrix, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260718)
    size = time_length * block_size
    lower = torch.zeros(*batch_shape, size, size, dtype=dtype)
    identity = torch.eye(block_size, dtype=dtype)
    for time in range(time_length):
        row = slice(time * block_size, (time + 1) * block_size)
        for earlier in range(max(0, time - bandwidth), time + 1):
            column = slice(earlier * block_size, (earlier + 1) * block_size)
            block = 0.1 * torch.randn(
                *batch_shape,
                block_size,
                block_size,
                dtype=dtype,
                generator=generator,
            )
            if earlier == time:
                block = torch.tril(block) + 1.5 * identity
            lower[..., row, column] = block
    dense = lower @ lower.mT
    bands = torch.zeros(*batch_shape, time_length, bandwidth + 1, block_size, block_size, dtype=dtype)
    for time in range(time_length):
        row = slice(time * block_size, (time + 1) * block_size)
        for offset in range(min(bandwidth, time) + 1):
            column_time = time - offset
            column = slice(column_time * block_size, (column_time + 1) * block_size)
            bands[..., time, offset, :, :] = dense[..., row, column]
    return BlockBandedMatrix(bands, bandwidth), dense


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("time_length", "block_size", "bandwidth"),
    [(3, 2, 1), (8, 7, 2), (32, 2, 2), (5, 2, 0)],
)
def test_banded_cholesky_matches_dense_with_arbitrary_batch_axes(
    dtype: torch.dtype,
    time_length: int,
    block_size: int,
    bandwidth: int,
) -> None:
    matrix, dense = _banded_spd(
        batch_shape=(2, 3),
        time_length=time_length,
        block_size=block_size,
        bandwidth=bandwidth,
        dtype=dtype,
    )
    generator = torch.Generator(device="cpu").manual_seed(31)
    rhs = torch.randn(2, 3, matrix.size, dtype=dtype, generator=generator)
    ridge = torch.tensor([[0.0, 0.01, 0.1], [0.2, 0.4, 0.8]], dtype=dtype)
    bands_before = matrix.bands.clone()

    result = BandedCholesky().solve_with_info(matrix, rhs, ridge=ridge)
    expected_matrix = dense.clone()
    expected_matrix.diagonal(dim1=-2, dim2=-1).add_(ridge[..., None])
    expected = torch.linalg.solve(expected_matrix, rhs)

    tolerance = 4e-4 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(result.solution, expected, rtol=tolerance, atol=tolerance * 0.1)
    assert torch.equal(matrix.bands, bands_before)
    assert result.solution.shape == rhs.shape
    assert result.ok.shape == (2, 3)
    assert result.ok.all()
    assert result.converged.all()
    assert result.finite.all()
    assert (result.status == LinearSolveStatus.SUCCESS).all()
    assert (result.iterations == 0).all()


def test_non_spd_element_isolated_and_returns_safe_zero() -> None:
    bands = torch.zeros(2, 4, 2, 1, 1, dtype=torch.float64)
    bands[..., :, 0, 0, 0] = 2.0
    bands[..., 1:, 1, 0, 0] = -0.2
    bands[1, 0, 0, 0, 0] = -1.0
    rhs = torch.ones(2, 4, dtype=torch.float64)
    matrix = BlockBandedMatrix(bands, bandwidth=1)

    result = BandedCholesky().solve_with_info(matrix, rhs)
    solo = BandedCholesky().solve(BlockBandedMatrix(bands[0], 1), rhs[0])

    torch.testing.assert_close(result.solution[0], solo, rtol=0.0, atol=0.0)
    assert torch.equal(result.solution[1], torch.zeros_like(result.solution[1]))
    torch.testing.assert_close(result.ok, torch.tensor([True, False]))
    torch.testing.assert_close(result.converged, torch.tensor([True, False]))
    torch.testing.assert_close(result.finite, torch.tensor([True, True]))
    torch.testing.assert_close(
        result.status,
        torch.tensor([LinearSolveStatus.SUCCESS, LinearSolveStatus.NOT_SPD], dtype=torch.int8),
    )
    assert torch.equal(BandedCholesky().solve(matrix, rhs)[1], torch.zeros_like(rhs[1]))


def test_banded_solver_never_calls_densify(monkeypatch: pytest.MonkeyPatch) -> None:
    matrix, _dense = _banded_spd(
        batch_shape=(),
        time_length=8,
        block_size=3,
        bandwidth=2,
        dtype=torch.float64,
    )
    rhs = torch.ones(matrix.size, dtype=torch.float64)

    def forbidden(_self):
        raise AssertionError("production banded solve must not densify")

    monkeypatch.setattr(BlockBandedMatrix, "densify", forbidden)
    result = BandedCholesky().solve_with_info(matrix, rhs)

    assert bool(result.ok)
    assert torch.isfinite(result.solution).all()


def test_banded_solve_is_differentiable_on_spd_input() -> None:
    matrix, _dense = _banded_spd(
        batch_shape=(),
        time_length=5,
        block_size=2,
        bandwidth=2,
        dtype=torch.float64,
    )
    bands = matrix.bands.detach().requires_grad_(True)
    rhs = torch.linspace(-1.0, 1.0, matrix.size, dtype=torch.float64)

    solution = BandedCholesky().solve(BlockBandedMatrix(bands, matrix.bandwidth), rhs)
    gradient = torch.autograd.grad(solution.square().sum(), bands)[0]

    assert torch.isfinite(solution).all()
    assert torch.isfinite(gradient).all()


def test_banded_solver_rejects_warm_start_and_bad_ridge() -> None:
    matrix, _dense = _banded_spd(
        batch_shape=(2,),
        time_length=3,
        block_size=2,
        bandwidth=1,
        dtype=torch.float32,
    )
    rhs = torch.ones(2, matrix.size)

    with pytest.raises(ValueError, match="does not accept an initial"):
        BandedCholesky().solve_with_info(matrix, rhs, initial=torch.zeros_like(rhs))
    with pytest.raises(ValueError, match="share A's dtype and device"):
        BandedCholesky().solve(matrix, rhs, ridge=torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="not broadcastable"):
        BandedCholesky().solve(matrix, rhs, ridge=torch.ones(3))
