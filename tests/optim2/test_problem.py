"""``optim2``: linearization against autograd and finite differences, cache, bounds, masks, solvers."""

import pytest
import torch

from better_robot import optim2


def _toy_problem(kernel=None, with_scalar=False):
    """Two variables, one cached value shared by two residual terms; ``calls`` counts cache evaluations."""
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    b = torch.randn(4, 2, generator=generator, dtype=torch.float64)
    target = torch.randn(4, 5, generator=generator, dtype=torch.float64)
    problem = optim2.Problem({"a": a, "b": b}, data={"target": target})
    calls = []

    @problem.cached
    def mixed(a, b):
        calls.append(1)
        return torch.cat([a.sin() * b[:, :1], a[:, :2] * b], dim=-1)  # (B, 5)

    problem.residual("fit", lambda mixed, target: mixed - target, weight=2.0, kernel=kernel)
    problem.residual("small", lambda mixed, a: torch.cat([0.1 * mixed, a], dim=-1))
    if with_scalar:
        problem.scalar("bowl", lambda b: (b**4).sum(-1), weight=0.5)
    return problem, calls


@pytest.mark.parametrize("kernel", [None, optim2.Huber(delta=0.5)])
@pytest.mark.parametrize("with_scalar", [False, True])
def test_linearize_gradient_matches_autograd_of_loss(kernel, with_scalar):
    problem, calls = _toy_problem(kernel, with_scalar)
    normal, loss = problem.linearize()
    assert len(calls) == 1, "the cached value must be computed once per linearization"

    x = {name: value.clone().requires_grad_(True) for name, value in problem.variables.items()}
    total = problem.loss(x)
    torch.testing.assert_close(loss, total.detach())
    grads = torch.autograd.grad(total.sum(), list(x.values()))
    torch.testing.assert_close(normal.g, torch.cat([g.flatten(1) for g in grads], dim=1), atol=1e-10, rtol=0.0)


def test_linearize_hessian_matches_finite_difference_jacobian():
    problem, _ = _toy_problem()
    normal, _ = problem.linearize()

    def stacked(x):
        out = problem.residuals(x)
        return torch.cat([out["fit"] * 2.0**0.5, out["small"]], dim=-1)

    eps, columns = 1e-6, []
    for index in range(problem.tangent_dim):
        delta = torch.zeros(4, problem.tangent_dim, dtype=torch.float64)
        delta[:, index] = eps
        plus = stacked(problem.retract(problem.variables, delta))
        minus = stacked(problem.retract(problem.variables, -delta))
        columns.append((plus - minus) / (2 * eps))
    jacobian = torch.stack(columns, dim=-1)
    torch.testing.assert_close(normal.H, jacobian.mT @ jacobian, atol=1e-7, rtol=0.0)


def test_analytic_blocks_equal_autodiff_and_free_mask_freezes():
    matrix = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]], dtype=torch.float64)
    x0 = torch.zeros(3, 3, dtype=torch.float64)
    goal = torch.tensor([[1.0, 2.0]], dtype=torch.float64).expand(3, 2)
    problem = optim2.Problem({"x": x0}, data={"goal": goal})
    term = optim2.TermFn(
        fn=lambda x, goal: x @ matrix.T - goal, reads=("x", "goal"), jac=lambda x, goal: {"x": matrix.expand(3, 2, 3)}
    )
    problem.residual("linear", term, weight=3.0)
    problem.residual("prior", lambda x: 0.1 * x)
    assert optim2.check_jacobian(problem)["linear"] < 1e-12

    frozen = problem.with_(free={"x": torch.tensor([True, False, True])})
    result = optim2.solve(frozen, optim2.LevenbergMarquardt(), iters=30)
    assert torch.all(result.x["x"][:, 1] == 0.0)
    assert float(result.loss.max()) < float(problem.loss().min())


def test_same_problem_under_adam_gauss_newton_and_lm_with_active_bound():
    generator = torch.Generator().manual_seed(1)
    matrix = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    goal = torch.randn(5, 6, generator=generator, dtype=torch.float64)
    upper = torch.tensor([0.05, 10.0, 10.0], dtype=torch.float64)
    problem = optim2.Problem(
        {"x": torch.zeros(5, 3, dtype=torch.float64)}, bounds={"x": (-upper.new_full((3,), 10.0), upper)}
    )
    problem.residual("fit", lambda x: x @ matrix.T - goal)

    lm = optim2.solve(problem, optim2.LevenbergMarquardt(), iters=50)
    gn = optim2.solve(problem, optim2.GaussNewton(), iters=50)
    adam = optim2.solve(problem, optim2.TorchOptim(torch.optim.Adam, lr=0.05), iters=3000)
    assert bool(lm.done.all())
    assert float(lm.x["x"][:, 0].max()) <= 0.05
    torch.testing.assert_close(gn.x["x"], lm.x["x"], atol=1e-8, rtol=0.0)
    torch.testing.assert_close(adam.x["x"], lm.x["x"], atol=1e-4, rtol=0.0)


def test_unknown_read_fails_at_registration():
    problem = optim2.Problem({"x": torch.zeros(2, 3)})
    with pytest.raises(KeyError, match="typo"):
        problem.residual("bad", lambda typo: typo)
