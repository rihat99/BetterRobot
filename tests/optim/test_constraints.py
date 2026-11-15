"""Equality and inequality terms on ``Problem``, and ``SequentialQP``, the solver that holds them."""

import pytest
import torch

from better_robot import exceptions, optim
from better_robot.optim.linearization import linearize_rows

PLANE = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)


def _fit(case: str = "") -> tuple[optim.Problem, torch.Tensor]:
    """``(problem, optimum)``: 64 points of three variables, each pulled to its target, under the constraint of
    ``case`` (a box, a plane, the unit ball, the unit sphere); the optimum is the closed form."""
    generator = torch.Generator().manual_seed(0)
    target = torch.randn(64, 3, dtype=torch.float64, generator=generator) * 2
    start = torch.randn(64, 3, dtype=torch.float64, generator=generator) * 0.3
    norm = target.norm(dim=-1, keepdim=True)
    if case == "box":
        bounds = {"x": (torch.full((3,), -0.5, dtype=torch.float64), torch.full((3,), 0.7, dtype=torch.float64))}
        problem = optim.Problem({"x": start.clamp(-0.5, 0.7)}, data={"target": target}, bounds=bounds)
    else:
        problem = optim.Problem({"x": start}, data={"target": target})
    problem.residual("fit", lambda x, target: x - target)
    if case == "box":
        return problem, target.clamp(-0.5, 0.7)
    if case == "plane":
        problem.equality("plane", lambda x: (x * PLANE).sum(-1, keepdim=True) - 0.5)
        return problem, target - PLANE * (((target * PLANE).sum(-1, keepdim=True) - 0.5) / (PLANE * PLANE).sum())
    if case == "ball":
        problem.inequality("ball", lambda x: (x * x).sum(-1, keepdim=True), upper=1.0)
        return problem, torch.where(norm > 1, target / norm, target)
    if case == "sphere":
        problem.equality("sphere", lambda x: (x * x).sum(-1, keepdim=True) - 1.0)
        return problem, target / norm
    return problem, target


def test_constraint_terms_are_rows_scaled_by_their_weight_and_no_part_of_the_loss() -> None:
    problem, _ = _fit()
    x = problem.variables["x"]
    problem.inequality("ball", lambda x: (x * x).sum(-1, keepdim=True), upper=1.0, weight=2.0)
    low = torch.tensor([-0.5, -0.25], dtype=torch.float64)
    problem.inequality(
        "corner",
        optim.TermFn(
            fn=lambda x: x[:, :2],
            reads=("x",),
            jacobian=lambda x: {"x": torch.eye(3, dtype=x.dtype)[:2].expand(x.shape[0], 2, 3)},
        ),
        lower=low,
        weight=torch.tensor([1.0, 3.0], dtype=torch.float64),
    )
    problem.equality("sum", lambda x: x.sum(-1, keepdim=True))
    assert [problem.terms[name].kind for name in ("ball", "corner", "sum")] == ["inequality", "inequality", "equality"]
    _, loss, losses, rows = linearize_rows(problem)
    torch.testing.assert_close(loss, 0.5 * ((x - problem.data["target"]) ** 2).sum(-1))
    assert set(losses) == {"fit"} and rows is not None
    torch.testing.assert_close(
        rows.value,
        torch.cat([2 * (x * x).sum(-1, keepdim=True), x[:, :1], 3 * x[:, 1:2], x.sum(-1, keepdim=True)], dim=1),
    )
    torch.testing.assert_close(rows.jacobian[:, 0], 4 * x)  # autodiff, weighted
    torch.testing.assert_close(rows.jacobian[:, 2], torch.tensor([0.0, 3.0, 0.0], dtype=x.dtype).expand(64, 3))
    torch.testing.assert_close(rows.lower[:, 1:], torch.tensor([-0.5, -0.75, 0.0], dtype=x.dtype).expand(64, 3))
    torch.testing.assert_close(rows.upper[:, [0, 3]], torch.tensor([2.0, 0.0], dtype=x.dtype).expand(64, 2))
    assert bool((rows.lower[:, 0] == -torch.inf).all()) and bool((rows.upper[:, 1:3] == torch.inf).all())
    unweighted = torch.stack([(x * x).sum(-1) - 1, -0.5 - x[:, 0], -0.25 - x[:, 1], x.sum(-1).abs()]).amax(0)
    torch.testing.assert_close(rows.violation, unweighted.clamp_min(0.0))
    assert max(optim.check_jacobian(problem).values()) < 1e-6
    frozen = problem.with_(free={"x": torch.tensor([True, False, True])}, weights={"ball": 0.0})
    held = linearize_rows(frozen)[3]
    assert held is not None and held.value.shape == (64, 3) and bool((held.jacobian[:, :, 1] == 0).all())


@pytest.mark.parametrize(("case", "steps"), [("box", 5), ("plane", 25), ("ball", 50), ("sphere", 70)])
def test_sequential_qp_reaches_the_closed_form_optimum(case: str, steps: int) -> None:
    """Within ``steps`` steps every problem is within 1e-6 of its optimum and of its constraint. The ball and the
    sphere press on the solution with a curvature the rows do not carry (multipliers up to 3): the damping finds
    it."""
    problem, optimum = _fit(case)
    result = optim.run(problem, [optim.Phase(optim.SequentialQP(), steps, stop_every=0)])
    assert float((result.x["x"] - optimum).norm(dim=-1).max()) < 1e-6
    assert float(result.diagnostics["phase"]["violation"][-1].max()) < 1e-6


def test_the_reported_point_never_leaves_the_tolerance_while_the_iterate_does() -> None:
    problem, _ = _fit("ball")
    worst, outside = [], []

    def record(_: optim.Problem, state: optim.State) -> None:
        worst.append(((state.x["x"] ** 2).sum(-1) - 1.0).max())
        outside.append(((state.extra["iterate"][0]["x"] ** 2).sum(-1) - 1.0).max())

    optim.run(problem, [optim.Phase(optim.SequentialQP(), 30, stop_every=0, on_step=record)])
    assert float(torch.stack(worst).max()) <= 1e-4 < float(torch.stack(outside).max())


def test_only_sequential_qp_takes_constraint_terms() -> None:
    problem, _ = _fit("ball")
    cost_only = [
        optim.LevenbergMarquardt(),
        optim.GaussNewton(),
        optim.LBFGS(),
        optim.TorchOptim(torch.optim.Adam),
    ]
    for solver in cost_only:
        with pytest.raises(exceptions.ArgumentError, match="SequentialQP"):
            optim.solve(problem, solver, iterations=1)
        optim.solve(problem.with_(weights={"ball": 0.0}), solver, iterations=1)
    problem.residual("wall", lambda x: x[:, :1] - 0.2)
    with pytest.raises(exceptions.ArgumentError, match="registered as inequality"):
        optim.solve_constrained(problem, optim.LevenbergMarquardt(), inequalities=["wall"])
    with pytest.raises(exceptions.ArgumentError, match="residual terms only"):
        optim.solve_layer(problem, optim.SequentialQP())
    with pytest.raises(exceptions.ArgumentError, match="neither"):
        optim.solve(_fit()[0], optim.SequentialQP(), iterations=1)
    with pytest.raises(exceptions.ArgumentError, match="lower"):
        _fit()[0].inequality("open", lambda x: x)
    with pytest.raises(exceptions.ArgumentError, match="gain"):
        optim.SequentialQP(gain=0.0)
