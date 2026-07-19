"""Informative direct-solver contracts."""

from __future__ import annotations

import pytest
import torch

from better_robot.optim.solvers import (
    BandedCholesky,
    Cholesky,
    InformativeLinearSolver,
    LU,
    LinearSolveStatus,
    LinearSolver,
)


def test_cholesky_informative_path_reports_per_element_failure() -> None:
    matrix = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 4.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[2.0, 8.0], [2.0, 2.0]], dtype=torch.float64)
    solver = Cholesky()

    with pytest.raises(torch.linalg.LinAlgError):
        solver.solve(matrix, rhs)
    result = solver.solve_with_info(matrix, rhs)

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
    assert isinstance(LU(), LinearSolver)
    assert isinstance(LU(), InformativeLinearSolver)
