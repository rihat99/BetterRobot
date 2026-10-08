"""Solver diagnostics in ``State.extra``, warm starts through ``Result.state``, and the dense-fallback warning."""

import warnings

import pytest
import torch

from better_robot import optim


def _rosenbrock(batch: int = 2, shift: float = 0.0) -> optim.Problem:
    x0 = torch.tensor([[-1.2, 1.0]], dtype=torch.float64).repeat(batch, 1)
    problem = optim.Problem({"x": x0}, data={"shift": torch.full((batch, 1), shift, dtype=torch.float64)})
    problem.residual("valley", lambda x: 10.0 * (x[:, 1:] - x[:, :1] ** 2))
    problem.residual("anchor", lambda x, shift: 1.0 - shift - x[:, :1])
    return problem


@pytest.mark.parametrize(
    "solver, keys",
    [
        (optim.LevenbergMarquardt(), {"grad_norm", "step_norm", "accepted", "damping"}),
        (optim.GaussNewton(damping=1e-3), {"grad_norm", "step_norm", "accepted"}),
        (optim.LBFGS(), {"grad_norm", "step_norm", "accepted"}),
    ],
    ids=["lm", "gn", "lbfgs"],
)
def test_solvers_write_diagnostics_and_run_stacks_them(solver, keys) -> None:
    result = optim.solve(_rosenbrock(), solver, iters=5)
    diagnostics = result.diagnostics["phase"]
    assert set(diagnostics) == keys and keys <= set(result.state.extra)
    steps = result.history["phase"].shape[0]
    for key in keys:
        assert diagnostics[key].shape == (steps, 2)
    assert diagnostics["accepted"].dtype == torch.bool and bool(diagnostics["accepted"].any(0).all())
    assert bool((diagnostics["step_norm"][~diagnostics["accepted"]] == 0).all())


def test_warm_started_lm_takes_fewer_iterations() -> None:
    first = optim.solve(_rosenbrock(), optim.LevenbergMarquardt(damping=1e3), iters=200)
    assert bool(first.done.all())
    moved = _rosenbrock(shift=0.05)
    cold = optim.solve(moved, optim.LevenbergMarquardt(damping=1e3), iters=200, x=first.x)
    warm = optim.solve(moved, optim.LevenbergMarquardt(damping=1e3), iters=200, state=first.state)
    assert bool(cold.done.all()) and bool(warm.done.all())
    assert warm.iters < cold.iters
    torch.testing.assert_close(warm["x"], cold["x"], atol=1e-6, rtol=0)


def test_warm_start_rejects_wrong_batch_and_non_state() -> None:
    first = optim.solve(_rosenbrock(batch=1), optim.LevenbergMarquardt(), iters=5)
    wide = _rosenbrock(batch=4)
    with pytest.raises(ValueError, match="batch"):
        optim.solve(wide, optim.LevenbergMarquardt(), iters=5, x=wide.variables, state=first.state)
    with pytest.raises(TypeError, match="State"):
        optim.solve(wide, optim.LevenbergMarquardt(), iters=5, state=first)


def test_warm_started_lbfgs_keeps_its_memory() -> None:
    first = optim.solve(_rosenbrock(), optim.LBFGS(), iters=10)
    solver = optim.LBFGS()
    warm = solver.init(_rosenbrock(), first.x)
    assert bool((warm.extra["rho"] == 0).all())  # a cold init has no curvature pairs
    resumed = optim.solve(_rosenbrock(), solver, iters=1, state=first.state)
    assert resumed.iters == 1 and bool((resumed.state.extra["rho"] != 0).any())


def test_time_problem_without_window_warns_on_dense_fallback() -> None:
    path = torch.zeros(1, 4, 2, dtype=torch.float64)
    problem = optim.Problem({"path": path}, time=("path",))
    problem.residual("local", lambda path: path - 1.0, window=1)
    problem.residual("global", lambda path: path.sum(1))
    with pytest.warns(UserWarning, match="global"):
        problem.linearize(problem.variables)
    quiet = optim.Problem({"path": path}, time=("path",), dense=True)
    quiet.residual("local", lambda path: path - 1.0, window=1)
    quiet.residual("global", lambda path: path.sum(1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet.linearize(quiet.variables)


def test_lbfgs_done_uses_gradient_at_new_point() -> None:
    problem = optim.Problem({"x": torch.full((1, 2), 3.0, dtype=torch.float64)})
    problem.residual("fit", lambda x: x - 1.0)
    result = optim.solve(problem, optim.LBFGS(), iters=50)
    assert bool(result.done.all())
    assert float(result.diagnostics["phase"]["grad_norm"][-1].max()) < 1e-8


def test_solve_constrained_rejects_zero_outer_rounds() -> None:
    problem = _rosenbrock()
    with pytest.raises(ValueError, match="outer"):
        optim.solve_constrained(problem, optim.LevenbergMarquardt(), equalities=["anchor"], outer=0)
