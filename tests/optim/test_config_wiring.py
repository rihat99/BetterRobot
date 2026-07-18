"""Verify the live ``OptimizerConfig`` factories.

See ``docs/concepts/residuals_costs_and_solvers.md``.
"""

from __future__ import annotations

import pytest
from better_robot.optim import Cholesky, Huber, L2
from better_robot.tasks.ik import (
    _make_linear_solver,
    _make_robust_kernel,
)


def test_linear_solver_factory_returns_correct_types() -> None:
    assert isinstance(_make_linear_solver("cholesky"), Cholesky)
    with pytest.raises(ValueError, match="Unknown linear_solver"):
        _make_linear_solver("cg")
    with pytest.raises(ValueError, match="Unknown linear_solver"):
        _make_linear_solver("does_not_exist")


def test_robust_kernel_factory_returns_correct_types() -> None:
    assert isinstance(_make_robust_kernel("l2"), L2)
    assert isinstance(_make_robust_kernel("huber"), Huber)
    with pytest.raises(ValueError, match="Unknown kernel"):
        _make_robust_kernel("does_not_exist")
