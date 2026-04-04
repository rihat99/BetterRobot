"""Tests for solvers (stubs — fail until implemented)."""

import pytest
import torch


def test_lm_solver_placeholder() -> None:
    """Placeholder: verify LM solve raises NotImplementedError until implemented."""
    from better_robot.solvers import LevenbergMarquardt, Problem

    problem = Problem(variables=torch.zeros(7))
    solver = LevenbergMarquardt()
    with pytest.raises(NotImplementedError):
        solver.solve(problem)
