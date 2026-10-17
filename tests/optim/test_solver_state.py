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


def _counted_problem(*, with_info: bool) -> tuple[optim.Problem, list[int]]:
    """Rosenbrock with a scalar term that may return ``(value, info)``; ``calls`` counts forward passes."""
    problem, calls = _rosenbrock(), []

    @problem.cached
    def square(x):
        calls.append(1)
        return x**2

    def small(x, square):
        value = 1e-2 * square.sum(-1)
        return (value, {"largest": x.abs().amax(-1), "label": "x"}) if with_info else value

    problem.scalar("small", small)
    return problem, calls


def test_term_info_and_per_term_losses_come_from_the_step_evaluation() -> None:
    problem, calls = _counted_problem(with_info=True)
    seen = []

    def log(view, state):
        seen.append((len(calls), state.extra["terms"], state.extra["info"], state.loss))

    adam = optim.TorchOptim(torch.optim.Adam, lr=1e-2)
    result = optim.run(problem, [optim.Phase(adam, iters=3, stop_every=0, on_step=log)])
    assert [count for count, *_ in seen] == [1, 2, 3]  # one forward per step: logging re-evaluates nothing
    _, terms, info, loss = seen[-1]
    assert set(terms) == {"valley", "anchor", "small"} and terms["small"].shape == (2,)
    torch.testing.assert_close(sum(terms.values()), loss)
    assert info["small"]["label"] == "x" and not info["small"]["largest"].requires_grad
    plain, _ = _counted_problem(with_info=False)
    reference = optim.run(plain, [optim.Phase(optim.TorchOptim(torch.optim.Adam, lr=1e-2), iters=3, stop_every=0)])
    torch.testing.assert_close(result["x"], reference["x"], atol=0, rtol=0)  # info never enters the loss
    total, per_term, term_info = problem.evaluate(result.x)
    torch.testing.assert_close(total, problem.loss(result.x))
    assert set(per_term) == set(terms) and set(term_info) == {"small"}


@pytest.mark.parametrize("solver", [optim.LevenbergMarquardt(), optim.GaussNewton(damping=1e-3), optim.LBFGS()])
def test_second_order_solvers_report_per_term_losses_at_the_accepted_point(solver) -> None:
    problem, _ = _counted_problem(with_info=True)
    result = optim.solve(problem, solver, iters=4)
    terms = result.state.extra["terms"]
    expected = problem.losses(result.state.x)
    assert set(terms) == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(terms[name], value.detach())


def test_lbfgs_with_bounds_reaches_the_nonnegative_least_squares_optimum() -> None:
    # KKT by construction: even entries sit on the bound with the gradient pushing against it (multipliers mu > 0),
    # odd entries are free. A line search that judged a clipped step by the unclipped slope stalled on this.
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(2, 40, 20, generator=generator, dtype=torch.float64)
    odd = torch.arange(20) % 2
    x_star = (0.1 + torch.rand(2, 20, generator=generator, dtype=torch.float64)) * odd
    mu = (0.1 + torch.rand(2, 20, generator=generator, dtype=torch.float64)) * (1 - odd)
    r = a @ torch.linalg.solve(a.mT @ a, mu.unsqueeze(-1))  # a^T r = mu
    b = (a @ x_star.unsqueeze(-1) - r).squeeze(-1)
    zero = torch.zeros(2, 20, dtype=torch.float64)
    problem = optim.Problem({"x": zero}, bounds={"x": (zero, torch.full_like(zero, torch.inf))}, data={"a": a, "b": b})
    problem.residual("fit", lambda x, a, b: (a @ x.unsqueeze(-1)).squeeze(-1) - b)
    result = optim.solve(problem, optim.LBFGS(line_search=4), iters=200)
    torch.testing.assert_close(result["x"], x_star, atol=1e-6, rtol=0)


def test_lbfgs_line_search_stops_once_every_problem_has_a_step() -> None:
    calls = []
    problem = optim.Problem({"x": torch.zeros(3, 4, dtype=torch.float64)})

    def fit(x):
        calls.append(1)
        return x - torch.arange(4.0, dtype=x.dtype)

    problem.residual("fit", fit)
    solver = optim.LBFGS(line_search=8)
    state = solver.init(problem, dict(problem.variables))
    for _ in range(3):
        before = len(calls)
        state = solver.step(problem, state)
        assert len(calls) - before == 2  # the first trial is accepted, then one gradient


def test_lbfgs_is_done_when_a_plain_gradient_step_finds_no_decrease() -> None:
    # The single trial, a unit-length gradient step, overshoots (x: 1e-3 -> -0.999); every later step would repeat it.
    problem = optim.Problem({"x": torch.full((1, 1), 1e-3, dtype=torch.float64)})
    problem.residual("steep", lambda x: 1e3 * x)
    result = optim.solve(problem, optim.LBFGS(line_search=1), iters=50)
    assert result.iters == 1 and bool(result.done.all())
    assert float(result["x"][0, 0]) == 1e-3


def test_lbfgs_rejects_a_clipped_step_that_raises_the_loss() -> None:
    # The bound x0 >= 0 clips the unit trial along the descent direction (-10, 1) to (-0.5, 1), along which the linear
    # model rises (g . step > 0) and so does the loss, by 1.25e-5; an Armijo test with that positive slope took it.
    # The direction is forced: an L-BFGS direction with mixed-sign products g_i d_i is hard to set up from history.
    class Forced(optim.LBFGS):
        def _direction(self, g, s, y, rho):
            return torch.tensor([[-10.0, 1.0]], dtype=g.dtype)

    x0 = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
    lower = torch.tensor([[0.0, -torch.inf]], dtype=torch.float64)
    problem = optim.Problem({"x": x0}, bounds={"x": (lower, torch.full_like(x0, torch.inf))})
    problem.residual("u", lambda x: (1.0 + 0.5 * (x[:, 0] + x[:, 1]) - 0.24999 * x[:, 1] ** 2).unsqueeze(-1))
    solver = Forced(history=2)
    state = solver.init(problem, dict(problem.variables))
    state.extra["rho"][:, -1] = 1.0  # as if there were curvature history: the first trial is the unit step
    new = solver.step(problem, state)
    assert bool(new.extra["accepted"].all()) and bool((new.loss < state.loss).all())  # the half step, (-0.5, 0.5)


def test_lbfgs_gradient_step_backtracks_below_a_stiff_term_with_no_gradient() -> None:
    # At x = 0 the stiff term (x0 + x1)^2 adds no gradient but dominates the curvature along the gradient (-2, 0): the
    # plain gradient step (x0 += 1 / 2^k) first passes the Armijo test at k = 12. Eight trials stopped short and left
    # the problem done at its start point.
    problem = optim.Problem({"x": torch.zeros(1, 2, dtype=torch.float64)})
    problem.residual("fit", lambda x: x - x.new_tensor([1.0, 0.0]))
    problem.residual("stiff", lambda x: 70.71 * x.sum(-1, keepdim=True))
    result = optim.solve(problem, optim.LBFGS(), iters=50)
    x1 = -(70.71**2) / (1 + 2 * 70.71**2)  # stationary point: x0 = 1 + x1
    torch.testing.assert_close(result["x"], torch.tensor([[1 + x1, x1]], dtype=torch.float64), atol=1e-8, rtol=0)
