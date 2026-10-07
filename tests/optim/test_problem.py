"""``optim``: linearization against autograd and finite differences, cache, bounds, masks, solvers."""

import pytest
import torch

from better_robot import optim


def _toy_problem(kernel=None, with_scalar=False):
    """Two variables, one cached value shared by two residual terms; ``calls`` counts cache evaluations."""
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    b = torch.randn(4, 2, generator=generator, dtype=torch.float64)
    target = torch.randn(4, 5, generator=generator, dtype=torch.float64)
    problem = optim.Problem({"a": a, "b": b}, data={"target": target})
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


@pytest.mark.parametrize("kernel", [None, optim.Huber(delta=0.5)])
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
    problem = optim.Problem({"x": x0}, data={"goal": goal})
    term = optim.TermFn(
        fn=lambda x, goal: x @ matrix.T - goal, reads=("x", "goal"), jac=lambda x, goal: {"x": matrix.expand(3, 2, 3)}
    )
    problem.residual("linear", term, weight=3.0)
    problem.residual("prior", lambda x: 0.1 * x)
    assert optim.check_jacobian(problem)["linear"] < 1e-12

    frozen = problem.with_(free={"x": torch.tensor([True, False, True])})
    result = optim.solve(frozen, optim.LevenbergMarquardt(), iters=30)
    assert torch.all(result.x["x"][:, 1] == 0.0)
    assert float(result.loss.max()) < float(problem.loss().min())


def test_same_problem_under_adam_gauss_newton_and_lm_with_active_bound():
    generator = torch.Generator().manual_seed(1)
    matrix = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    goal = torch.randn(5, 6, generator=generator, dtype=torch.float64)
    upper = torch.tensor([0.05, 10.0, 10.0], dtype=torch.float64)
    problem = optim.Problem(
        {"x": torch.zeros(5, 3, dtype=torch.float64)}, bounds={"x": (-upper.new_full((3,), 10.0), upper)}
    )
    problem.residual("fit", lambda x: x @ matrix.T - goal)

    lm = optim.solve(problem, optim.LevenbergMarquardt(), iters=50)
    gn = optim.solve(problem, optim.GaussNewton(), iters=50)
    adam = optim.solve(problem, optim.TorchOptim(torch.optim.Adam, lr=0.05), iters=3000)
    assert bool(lm.done.all())
    assert float(lm.x["x"][:, 0].max()) <= 0.05
    torch.testing.assert_close(gn.x["x"], lm.x["x"], atol=1e-8, rtol=0.0)
    torch.testing.assert_close(adam.x["x"], lm.x["x"], atol=1e-4, rtol=0.0)


def test_unknown_read_fails_at_registration():
    problem = optim.Problem({"x": torch.zeros(2, 3)})
    with pytest.raises(KeyError, match="typo"):
        problem.residual("bad", lambda typo: typo)


def test_time_local_terms_match_row_by_row_autodiff():
    generator = torch.Generator().manual_seed(2)
    path = torch.randn(2, 7, 3, generator=generator, dtype=torch.float64)
    scale = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    problem = optim.Problem({"path": path, "scale": scale}, time=("path",))
    problem.cached(lambda path: path.sin(), name="warped")
    problem.residual("per_knot", lambda warped, path: warped * path.roll(1, -1), window=1)
    problem.residual("accel", lambda path: (path[:, 2:] - 2 * path[:, 1:-1] + path[:, :-2]).tanh(), window=3,
                     kernel=optim.Huber(delta=0.3), weight=4.0)
    velocity = optim.TermFn(
        fn=lambda path: path[:, 1:] - path[:, :-1],
        reads=("path",),
        jac=lambda path: {
            "path": (torch.tensor([-1.0, 1.0], dtype=path.dtype)[:, None, None] * torch.eye(3, dtype=path.dtype))
            .permute(1, 0, 2).expand(2, 6, 3, 2, 3)
        },
        window=2,
    )
    problem.residual("velocity", velocity)
    problem.residual("coupled", lambda path, scale: path.sum(1) * scale[:, :3] + scale[:, 3:])
    problem.residual("scaled", lambda path, scale: (path[:, 1:] - path[:, :-1]) * scale[:, None, :3].exp(), window=2)
    errors = optim.check_jacobian(problem)
    assert set(errors) == {"per_knot", "accel", "velocity", "scaled"}
    assert max(errors.values()) < 1e-10, errors


@pytest.mark.parametrize("knots", [9, 10])
def test_block_tridiagonal_path_equals_dense(knots):
    generator = torch.Generator().manual_seed(3)
    path = torch.randn(3, knots, 2, generator=generator, dtype=torch.float64)
    force = torch.randn(3, knots, 1, generator=generator, dtype=torch.float64)
    anchor = torch.randn(3, knots, 2, generator=generator, dtype=torch.float64)

    def build(dense):
        problem = optim.Problem(
            {"path": path, "force": force},
            data={"anchor": anchor},
            time=("path", "force"),
            dense=dense,
            bounds={"force": (torch.tensor(-0.2, dtype=torch.float64), torch.tensor(0.2, dtype=torch.float64))},
        )
        problem.residual("anchor", lambda path, anchor: (path - anchor).sin(), window=1, weight=0.3)
        problem.residual("accel", lambda path: path[:, 2:] - 2 * path[:, 1:-1] + path[:, :-2], window=3)
        problem.residual("push", lambda path, force: (path[:, 1:] - path[:, :-1]).sum(-1, keepdim=True) - force[:, :-1],
                         window=2, kernel=optim.Huber(delta=0.5))
        problem.residual("effort", lambda force: force, window=1, weight=0.1)
        return problem

    banded, dense = build(False), build(True)
    normal_b, loss_b = banded.linearize()
    normal_d, loss_d = dense.linearize()
    assert isinstance(normal_b, optim.BandedNormal) and isinstance(normal_d, optim.Normal)
    torch.testing.assert_close(loss_b, loss_d)
    torch.testing.assert_close(normal_b.dense(), normal_d.H, atol=1e-12, rtol=0.0)
    damping = torch.tensor([0.0, 1e-3, 1.0], dtype=torch.float64)
    delta = normal_d.solve(damping)
    torch.testing.assert_close(normal_b.solve(damping), delta, atol=1e-9, rtol=0.0)
    torch.testing.assert_close(normal_b.predicted_decrease(delta), normal_d.predicted_decrease(delta))

    free = {"path": torch.arange(knots * 2) % 3 != 0}
    result_b = optim.solve(banded.with_(free=free), optim.LevenbergMarquardt(), iters=40)
    result_d = optim.solve(dense.with_(free=free), optim.LevenbergMarquardt(), iters=40)
    torch.testing.assert_close(result_b.x["path"], result_d.x["path"], atol=1e-7, rtol=0.0)
    torch.testing.assert_close(result_b.x["force"], result_d.x["force"], atol=1e-7, rtol=0.0)
    assert float(result_b.x["force"].abs().max()) <= 0.2


@pytest.mark.parametrize("knots", [9, 10])
def test_shared_variables_border_the_band_and_equal_dense(knots):
    generator = torch.Generator().manual_seed(4)
    path = torch.randn(3, knots, 2, generator=generator, dtype=torch.float64)
    shape = 0.3 * torch.randn(3, 3, generator=generator, dtype=torch.float64)
    anchor = torch.randn(3, knots, 2, generator=generator, dtype=torch.float64)

    def build(dense):
        # "shape" sits between the time variables on purpose: column order must not matter.
        problem = optim.Problem(
            {"path": path, "shape": shape, "force": path[..., :1].clone()},
            data={"anchor": anchor},
            time=("path", "force"),
            dense=dense,
            bounds={"shape": (torch.tensor(-0.4, dtype=torch.float64), torch.tensor(0.4, dtype=torch.float64))},
        )
        problem.cached(lambda path, shape: path * shape[:, None, :2].exp(), name="scaled")
        problem.residual("anchor", lambda scaled, anchor: (scaled - anchor).sin(), window=1, kernel=optim.Huber(delta=0.7))
        problem.residual("accel", lambda path: path[:, 2:] - 2 * path[:, 1:-1] + path[:, :-2], window=3)
        push = optim.TermFn(
            fn=lambda path, force, shape: (path[:, 1:] - path[:, :-1]).sum(-1, keepdim=True)
            - force[:, :-1] * shape[:, None, 2:],
            reads=("path", "force", "shape"),
            jac=lambda path, force, shape: {
                "path": torch.tensor([-1.0, 1.0], dtype=path.dtype)[:, None].expand(*path[:, 1:, :1].shape, 2, 2),
                "force": torch.stack([-shape[:, None, 2:].expand_as(force[:, :-1]), torch.zeros_like(force[:, :-1])], -2)
                .unsqueeze(2),
                "shape": torch.cat([torch.zeros_like(path[:, 1:]), -force[:, :-1]], -1).unsqueeze(2),
            },
            window=2,
        )
        problem.residual("push", push, weight=2.0)
        problem.residual("effort", lambda force: force, window=1, weight=0.1)
        problem.residual("shape_prior", lambda shape: shape, weight=0.05)
        return problem

    arrow, dense = build(False), build(True)
    assert max(optim.check_jacobian(arrow).values()) < 1e-10
    normal_a, loss_a = arrow.linearize()
    normal_d, loss_d = dense.linearize()
    assert isinstance(normal_a, optim.ArrowNormal) and isinstance(normal_d, optim.Normal)
    torch.testing.assert_close(loss_a, loss_d)
    torch.testing.assert_close(normal_a.g, normal_d.g, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(normal_a.dense(), normal_d.H, atol=1e-12, rtol=0.0)
    damping = torch.tensor([0.0, 1e-3, 1.0], dtype=torch.float64)
    delta = normal_d.solve(damping)
    torch.testing.assert_close(normal_a.solve(damping), delta, atol=1e-9, rtol=0.0)
    torch.testing.assert_close(normal_a.predicted_decrease(delta), normal_d.predicted_decrease(delta))

    free = {"path": torch.arange(knots * 2) % 3 != 0, "shape": torch.tensor([True, False, True])}
    result_a = optim.solve(arrow.with_(free=free), optim.LevenbergMarquardt(), iters=40)
    result_d = optim.solve(dense.with_(free=free), optim.LevenbergMarquardt(), iters=40)
    for name in ("path", "shape", "force"):
        torch.testing.assert_close(result_a.x[name], result_d.x[name], atol=1e-7, rtol=0.0)
    torch.testing.assert_close(result_a.x["shape"][:, 1], shape[:, 1].clamp(-0.4, 0.4))
    assert float(result_a.x["shape"].abs().max()) <= 0.4


def _curve_problem(goal):
    """Fit ``x`` so that a smooth nonlinear map of it matches ``goal``; the upper bound is active for some rows."""
    problem = optim.Problem(
        {"x": torch.zeros(goal.shape[0], 3, dtype=goal.dtype)},
        data={"goal": goal},
        bounds={"x": (torch.full((3,), -5.0, dtype=goal.dtype), torch.tensor([5.0, 5.0, 0.3], dtype=goal.dtype))},
    )
    problem.residual("fit", lambda x, goal: torch.cat([x + 0.2 * x.roll(1, -1) ** 2, x.sum(-1, keepdim=True)], -1) - goal)
    return problem


def test_batched_lbfgs_matches_lm():
    goal = torch.randn(6, 4, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    problem = _curve_problem(goal)
    lm = optim.solve(problem, optim.LevenbergMarquardt(), iters=100)
    lbfgs = optim.solve(problem, optim.LBFGS(), iters=300)
    torch.testing.assert_close(lbfgs.loss, lm.loss, atol=1e-9, rtol=0.0)
    torch.testing.assert_close(lbfgs.x["x"], lm.x["x"], atol=1e-4, rtol=0.0)


def test_solve_layer_gradient_matches_finite_differences():
    goal = 0.5 * torch.randn(4, 3, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    goal[:, 2] = goal[:, 2].clamp(max=0.1)  # rows 1-3 stay clear of the upper bound of 0.3
    goal[0, 2] = 2.0  # pushes x[0, 2] onto that bound

    def solution(value):
        problem = optim.Problem(
            {"x": torch.zeros(4, 3, dtype=torch.float64)},
            data={"goal": value},
            bounds={"x": (torch.full((3,), -5.0, dtype=torch.float64), torch.tensor([5.0, 5.0, 0.3], dtype=torch.float64))},
        )
        problem.residual("fit", lambda x, goal: x + 0.2 * x.roll(1, -1) ** 2 - goal)
        return optim.solve_layer(problem, optim.LevenbergMarquardt(), iters=200).x["x"]

    weights = torch.randn(4, 3, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    attached = goal.clone().requires_grad_(True)
    out = solution(attached)
    assert float(out[0, 2]) == pytest.approx(0.3)
    (analytic,) = torch.autograd.grad((out * weights).sum(), attached)
    numeric = torch.zeros_like(goal)
    for index in range(goal.numel()):
        step = torch.zeros(goal.numel(), dtype=torch.float64)
        step[index] = 1e-6
        step = step.view_as(goal)
        numeric.view(-1)[index] = ((solution(goal + step) - solution(goal - step)) * weights).sum() / 2e-6
    # Rows 1-3 are solved exactly (zero residual), so the implicit gradient is exact there.
    torch.testing.assert_close(analytic[1:], numeric[1:], atol=1e-5, rtol=0.0)
    # Row 0 sits on its bound: the clamped coordinate does not react to its own target.
    (bound_sensitivity,) = torch.autograd.grad(solution(attached)[0, 2], attached)
    assert float(bound_sensitivity.abs().max()) == 0.0


def test_review_regressions():
    base = optim.Problem({"x": torch.zeros(2, 2, dtype=torch.float64)}, data={"a": torch.ones(2, 2, dtype=torch.float64)})
    base.residual("fit", lambda x, a: x - a)
    view = base.with_(weights={"fit": 2.0})
    view.residual("extra", lambda x: x)
    assert "extra" not in base.terms, "a view must not register terms on its base problem"

    base.residual("bad_weight", lambda x: x.unsqueeze(1).expand(2, 3, 2), weight=torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="weight of 'bad_weight'"):
        base.loss()

    scalar = optim.Problem({"x": torch.zeros(2, 1, dtype=torch.float64)}, data={"a": torch.ones(2, 1, dtype=torch.float64)})
    scalar.scalar("bowl", lambda x, a: ((x - a) ** 2).sum(-1))
    with pytest.raises(ValueError, match="scalar"):
        optim.solve_layer(scalar, optim.LBFGS(), iters=5)


def test_phase_hooks_scheduler_and_per_knot_free_mask():
    path = torch.zeros(2, 5, 3, dtype=torch.float64)
    problem = optim.Problem({"path": path}, data={"goal": torch.ones(2, 5, 3, dtype=torch.float64)}, time=("path",))
    problem.residual("fit", lambda path, goal: path - goal, window=1)
    rates, seen = [], []

    def retarget(view, x):
        view.data["goal"] = view.data["goal"] * 2.0

    adam = optim.TorchOptim(
        torch.optim.Adam, lr=0.1, scheduler=lambda o: torch.optim.lr_scheduler.MultiStepLR(o, milestones=[2], gamma=0.1)
    )
    free = {"path": torch.tensor([True, False, True])}
    phases = [
        optim.Phase(adam, iters=4, free=free, stop_every=0,
                     on_step=lambda view, state: rates.append(state.extra["optimizer"].param_groups[0]["lr"])),
        optim.Phase(optim.LevenbergMarquardt(), iters=10, free=free, on_start=retarget,
                     on_step=lambda view, state: seen.append(view.losses(state.x)["fit"])),
    ]
    result = optim.run(problem, phases)
    assert rates == pytest.approx([0.1, 0.01, 0.01, 0.01])
    assert seen and seen[-1].shape == (2,)
    torch.testing.assert_close(result.x["path"][..., 1], path[..., 1])
    torch.testing.assert_close(result.x["path"][..., 0], torch.full((2, 5), 2.0, dtype=torch.float64))
    assert float(problem.data["goal"].max()) == 1.0


def test_augmented_lagrangian_reaches_the_constrained_optimum():
    a = torch.tensor([[0.9, 0.5, 0.2], [-0.4, 0.3, 0.5]], dtype=torch.float64)
    problem = optim.Problem({"x": torch.zeros(2, 3, dtype=torch.float64)}, data={"a": a})
    problem.residual("fit", lambda x, a: x - a)
    problem.residual("total", lambda x: x.sum(-1, keepdim=True) - 1.0)
    cap = optim.TermFn(fn=lambda x: x[:, :1] - 0.2, reads=("x",))
    problem.residual("cap", cap)
    result = optim.solve_constrained(
        problem, optim.LevenbergMarquardt(), equalities=("total",), inequalities=("cap",), outer=20, tolerance=1e-9
    )
    # Row 0: the cap is active (x0 = 0.2, the rest shares 0.8). Row 1: only the sum constraint acts.
    expected = torch.tensor([[0.2, 0.55, 0.25], [-0.2, 0.5, 0.7]], dtype=torch.float64)
    torch.testing.assert_close(result.x["x"], expected, atol=1e-7, rtol=0.0)
    assert bool(result.done.all()) and result.history["violation"].shape[1] == 2
    assert "total.shift" not in problem.data and problem.terms["cap"].fn is cap
    torch.testing.assert_close(result.loss, 0.5 * (expected - a).square().sum(-1), atol=1e-7, rtol=0.0)


def test_shared_columns_through_a_once_differentiable_op_fail_loudly():
    class Once(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            return value.sin()

        @staticmethod
        def backward(ctx, grad):
            with torch.no_grad():
                return grad * ctx.saved_tensors[0].cos()

    path = torch.linspace(0.1, 1.0, 12, dtype=torch.float64).reshape(1, 4, 3)
    problem = optim.Problem({"path": path, "cam": torch.full((1, 3), 0.4, dtype=torch.float64)}, time=("path",))
    problem.residual("view", lambda path, cam: path.cos() * Once.apply(cam)[:, None], window=1)
    with pytest.raises(ValueError, match="differentiated twice"):
        problem.linearize()


def test_cached_returns_handle_and_unknown_cached_read_fails_at_registration():
    problem = optim.Problem({"x": torch.zeros(2, 3)})
    ref = problem.cached(lambda x: 2.0 * x, name="doubled")
    assert isinstance(ref, optim.CacheRef) and str(ref) == "doubled"
    problem.residual("fit", lambda doubled: doubled, reads=(str(ref),))
    with pytest.raises(KeyError, match="tripled"):
        problem.residual("bad", lambda tripled: tripled, reads=("tripled",))
