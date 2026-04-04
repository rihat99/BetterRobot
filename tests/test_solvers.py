"""Tests for solvers."""

import torch
import pytest
from better_robot.solvers import LevenbergMarquardt, Problem, CostTerm


def test_lm_solves_simple_quadratic():
    """LM should minimize (x - 2)^2 + (y - 3)^2 to find [2, 3]."""
    target = torch.tensor([2.0, 3.0])

    def residual(x):
        return x - target

    problem = Problem(
        variables=torch.zeros(2),
        costs=[CostTerm(residual_fn=residual, weight=1.0, kind="soft")]
    )

    solver = LevenbergMarquardt()
    solution = solver.solve(problem, max_iter=50)

    assert solution.shape == (2,)
    assert torch.allclose(solution, target, atol=1e-4), f"Expected {target}, got {solution}"


def test_lm_returns_correct_shape():
    def residual(x):
        return x ** 2

    problem = Problem(
        variables=torch.ones(5),
        costs=[CostTerm(residual_fn=residual, weight=1.0)]
    )
    solver = LevenbergMarquardt()
    sol = solver.solve(problem, max_iter=20)
    assert sol.shape == (5,)


def test_problem_total_residual():
    def r1(x): return x[:2]
    def r2(x): return x[2:]

    problem = Problem(
        variables=torch.zeros(4),
        costs=[
            CostTerm(residual_fn=r1, weight=2.0, kind="soft"),
            CostTerm(residual_fn=r2, weight=0.5, kind="soft"),
            CostTerm(residual_fn=r1, weight=1.0, kind="constraint_leq_zero"),
        ]
    )
    x = torch.ones(4)
    r = problem.total_residual(x)
    # r1 gives [1,1]*2.0 = [2,2], r2 gives [1,1]*0.5 = [0.5,0.5]
    assert r.shape == (4,)
    assert torch.allclose(r[:2], torch.tensor([2.0, 2.0]))
    assert torch.allclose(r[2:], torch.tensor([0.5, 0.5]))


def test_problem_constraint_residual():
    def r1(x): return x[:2]
    problem = Problem(
        variables=torch.zeros(4),
        costs=[
            CostTerm(residual_fn=r1, weight=1.0, kind="soft"),
            CostTerm(residual_fn=r1, weight=3.0, kind="constraint_leq_zero"),
        ]
    )
    x = torch.ones(4)
    cr = problem.constraint_residual(x)
    assert cr.shape == (2,)
    assert torch.allclose(cr, torch.tensor([3.0, 3.0]))
