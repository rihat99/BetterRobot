"""Equality and inequality terms on ``Problem``, and ``SequentialQP``, the solver that holds them."""

import pytest
import torch

from better_robot import exceptions, optim
from better_robot.optim.linearization import linearize_rows

PLANE = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)


#: ``(seed, problems, spread of the targets, spread of the starts)`` of eight draws at two scales.
DRAWS = tuple((100 + seed, 256, *spread) for seed in range(8) for spread in ((2.0, 0.3), (5.0, 1.0)))


def _square(x: torch.Tensor) -> torch.Tensor:
    return (x * x).sum(-1, keepdim=True)


def _fit(
    case: str = "", draws: tuple[tuple[int, int, float, float], ...] = ((0, 64, 2.0, 0.3),)
) -> tuple[optim.Problem, torch.Tensor]:
    """``(problem, optimum)``: points of three variables (64 unless ``draws`` says otherwise), each pulled to its
    target, under the constraint of ``case`` (a box, a plane, the unit ball, the unit sphere); the optimum is the
    closed form."""
    targets, starts = [], []
    for seed, count, far, near in draws:
        generator = torch.Generator().manual_seed(seed)
        targets.append(torch.randn(count, 3, dtype=torch.float64, generator=generator) * far)
        starts.append(torch.randn(count, 3, dtype=torch.float64, generator=generator) * near)
    target, start = torch.cat(targets), torch.cat(starts)
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


@pytest.mark.parametrize(("case", "steps"), [("box", 5), ("plane", 25), ("ball", 50), ("sphere", 75)])
def test_the_step_budget_of_a_closed_form_holds_on_every_draw(case: str, steps: int) -> None:
    """Eight draws of 256 problems at two scales: the slowest draw is within 1e-6 of its optimum after 2, 22, 44 and
    65 steps."""
    problem, optimum = _fit(case, DRAWS)
    result = optim.run(problem, [optim.Phase(optim.SequentialQP(), steps, stop_every=0)])
    assert float((result.x["x"] - optimum).norm(dim=-1).max()) < 1e-6


@pytest.mark.parametrize("weight", [1.0, 1e4])
def test_a_step_that_a_row_forces_is_shortened_until_it_is_taken(weight: float) -> None:
    """From near the centre of the unit sphere the linearized row asks for a step of 50, which no damping shortens
    at a weight where the row does not give way: the radius does."""
    start = torch.tensor([[0.01, 0.0, 0.0], [0.05, 0.02, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64).expand(4, 3)
    problem = optim.Problem({"x": start}, data={"target": target})
    problem.residual("fit", lambda x, target: x - target)
    problem.equality("sphere", lambda x: _square(x) - 1.0, weight=weight)
    result = optim.run(problem, [optim.Phase(optim.SequentialQP(), 100, stop_every=0)])
    assert float((result.x["x"] - target / target.norm(dim=-1, keepdim=True)).norm(dim=-1).max()) < 1e-6
    assert bool(result.done.all())


def test_no_problem_is_done_outside_a_constraint_it_can_meet() -> None:
    """The unit circle with ``y <= 0.5``, from starts as far as the targets: every problem ends on the circle and
    under the line."""
    generator = torch.Generator().manual_seed(1)
    target = torch.randn(256, 2, dtype=torch.float64, generator=generator) * 2
    start = torch.randn(256, 2, dtype=torch.float64, generator=generator) * 2
    problem = optim.Problem({"x": start}, data={"target": target})
    problem.residual("fit", lambda x, target: x - target)
    problem.equality("circle", lambda x: _square(x) - 1.0)
    problem.inequality("below", lambda x: x[:, 1:], upper=0.5)
    x = optim.run(problem, [optim.Phase(optim.SequentialQP(), 60, stop_every=0)]).x["x"]
    assert float((_square(x) - 1.0).abs().max()) < 1e-6 and float(x[:, 1].max()) < 0.5 + 1e-6


def test_rows_that_say_the_same_thing_twice_do_not_stall_the_steps() -> None:
    """The unit ball and the unit sphere hold the same surface, so their multipliers are not unique and drift apart
    (to hundreds); the merit's price does not follow them."""
    problem, optimum = _fit("sphere")
    problem.inequality("ball", _square, upper=1.0)
    result = optim.run(problem, [optim.Phase(optim.SequentialQP(), 150, stop_every=0)])
    assert float((result.x["x"] - optimum).norm(dim=-1).max()) < 1e-6


def test_a_zero_weight_switches_a_row_off_and_a_negative_number_is_refused() -> None:
    """Each problem holds one of ``x0 <= 0.5`` and ``x1 <= 0.5`` and has the other switched off by a zero entry of
    the weight: the row that is off holds nothing and adds no violation (it was 0 / 0, which froze ``State.x``)."""
    start = torch.full((2, 3), 0.1, dtype=torch.float64)
    problem = optim.Problem({"x": start}, data={"target": torch.full((2, 3), 2.0, dtype=torch.float64)})
    problem.residual("fit", lambda x, target: x - target)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    problem.inequality("corner", lambda x: x[:, :2], upper=0.5, weight=weight)
    result = optim.run(problem, [optim.Phase(optim.SequentialQP(), 30, stop_every=0)])
    expected = torch.tensor([[0.5, 2.0, 2.0], [2.0, 0.5, 2.0]], dtype=torch.float64)
    torch.testing.assert_close(result.x["x"], expected, atol=1e-6, rtol=0.0)
    assert float(result.diagnostics["phase"]["violation"].max()) < 1e-6
    with pytest.raises(exceptions.ArgumentError, match="at least 0"):
        problem.with_(weights={"corner": -1.0})
    with pytest.raises(exceptions.ArgumentError, match="at least 0"):
        _fit()[0].equality("sum", lambda x: x.sum(-1, keepdim=True), weight=-1.0)


def test_the_reported_point_never_leaves_the_tolerance_while_the_iterate_does() -> None:
    problem, _ = _fit("ball")
    worst, outside = [], []

    def record(_: optim.Problem, state: optim.State) -> None:
        worst.append(((state.x["x"] ** 2).sum(-1) - 1.0).max())
        outside.append(((state.extra["iterate"][0]["x"] ** 2).sum(-1) - 1.0).max())

    optim.run(problem, [optim.Phase(optim.SequentialQP(), 30, stop_every=0, on_step=record)])
    assert float(torch.stack(worst).max()) <= 1e-4 < float(torch.stack(outside).max())


def test_a_problem_is_not_done_while_its_iterate_is_outside_the_tolerance() -> None:
    """With tolerances loose enough to stop at any step, a problem still runs until its iterate is within
    ``tolerance`` of the ball it had left."""
    problem, _ = _fit("ball")
    left = []
    solver = optim.SequentialQP(gradient_tolerance=1e3, step_tolerance=1e3)
    record = lambda _, state: left.append(state.extra["rows"].violation.max())  # noqa: E731
    result = optim.run(problem, [optim.Phase(solver, 30, stop_every=0, on_step=record)])
    assert result.state is not None and float(torch.stack(left).max()) > 1e-4
    assert bool(result.done.all()) and float(result.state.extra["rows"].violation.max()) <= 1e-4


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
    off = problem.with_(weights={"ball": 0.0})
    held = optim.solve_constrained(off, optim.LevenbergMarquardt(), inequalities=["wall"])
    assert float((held.x["x"][:, 0] - 0.2).max()) < 1e-4 and float((held.x["x"] ** 2).sum(-1).max()) > 1.0
    optim.solve_layer(off, optim.LevenbergMarquardt())
    with pytest.raises(exceptions.ArgumentError, match="neither"):
        optim.solve(_fit()[0], optim.SequentialQP(), iterations=1)
    with pytest.raises(exceptions.ArgumentError, match="lower"):
        _fit()[0].inequality("open", lambda x: x)
    with pytest.raises(exceptions.ArgumentError, match="gain"):
        optim.SequentialQP(gain=0.0)
    with pytest.raises(exceptions.ArgumentError, match="tolerance"):
        optim.SequentialQP(tolerance=0.0)
