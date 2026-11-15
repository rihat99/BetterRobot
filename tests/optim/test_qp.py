"""`solve_qp`: closed forms, the optimality conditions, rows that cannot hold, padding, failures and tracing."""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import ShapeError
from better_robot.optim import QPSolution, solve_qp

INF = float("inf")


def _problems(batch: int, size: int, count: int, *, seed: int) -> tuple[torch.Tensor, ...]:
    """Feasible float64 problems with a few active rows: ``H, g, rows, lower, upper``."""
    generator = torch.Generator().manual_seed(seed)
    root = torch.randn(batch, size, size, generator=generator, dtype=torch.float64)
    H = root.mT @ root + 0.1 * torch.eye(size, dtype=torch.float64)
    g = torch.randn(batch, size, generator=generator, dtype=torch.float64)
    rows = torch.randn(batch, count, size, generator=generator, dtype=torch.float64)
    inside = 0.1 * torch.randn(batch, size, generator=generator, dtype=torch.float64)
    centre = (rows @ inside.unsqueeze(-1)).squeeze(-1)
    upper = centre + 0.5 * torch.rand(batch, count, generator=generator, dtype=torch.float64)
    lower = centre - 0.5 * torch.rand(batch, count, generator=generator, dtype=torch.float64)
    lower[:, count // 2 :] = -INF
    return H, g, rows, lower, upper


def _repeat(problem: tuple[torch.Tensor, ...], calls: int, **kwargs) -> QPSolution:
    H, g, rows, lower, upper = problem
    solution = None
    for _ in range(calls):
        solution = solve_qp(H, g, rows=rows, lower=lower, upper=upper, warm=solution, **kwargs)
    return solution


def test_box_and_equality_closed_forms() -> None:
    """A box under the identity clamps the free minimiser; one equality row projects onto its plane."""
    g = torch.tensor([[2.0, -0.05, -3.0]], dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64)[None]
    low, high = torch.full((1, 3), -0.5, dtype=torch.float64), torch.full((1, 3), 1.0, dtype=torch.float64)
    box = solve_qp(eye, g, rows=eye, lower=low, upper=high)
    torch.testing.assert_close(box.x, torch.tensor([[-0.5, 0.05, 1.0]], dtype=torch.float64), atol=1e-9, rtol=0)
    # The multiplier is positive on `upper`, negative on `lower`, zero on a row that does not press.
    torch.testing.assert_close(box.dual, torch.tensor([[-1.5, 0.0, 2.0]], dtype=torch.float64), atol=1e-8, rtol=0)

    row = torch.tensor([[[1.0, 2.0, 2.0]]], dtype=torch.float64)
    bound = torch.tensor([[3.0]], dtype=torch.float64)
    plane = solve_qp(eye, torch.zeros(1, 3, dtype=torch.float64), rows=row, lower=bound, upper=bound)
    torch.testing.assert_close(plane.x, row[:, 0] / 3.0, atol=1e-9, rtol=0)
    assert plane.primal_residual.item() < 1e-9 and plane.dual_residual.item() < 1e-9


def test_warm_calls_reach_the_optimality_conditions() -> None:
    """Repeated warm calls land every element on its KKT point: rows, stationarity, signs, complementarity."""
    problem = _problems(64, 6, 14, seed=0)
    H, g, rows, lower, upper = problem
    first = solve_qp(H, g, rows=rows, lower=lower, upper=upper)
    solution = _repeat(problem, 12)
    assert solution.primal_residual.max().item() < 1e-8
    assert solution.dual_residual.max().item() < 1e-8
    # The active-set step makes one cold call exact on more than half of the elements (39 of 64 when written).
    assert ((first.primal_residual < 1e-8) & (first.dual_residual < 1e-8)).sum().item() >= 32
    value = (rows @ solution.x.unsqueeze(-1)).squeeze(-1)
    slack_upper = torch.where(torch.isfinite(upper), upper - value, torch.zeros_like(value))
    slack_lower = torch.where(torch.isfinite(lower), value - lower, torch.zeros_like(value))
    pressing_up, pressing_down = solution.dual.clamp_min(0.0), (-solution.dual).clamp_min(0.0)
    assert (pressing_up * slack_upper).abs().max().item() < 1e-8
    assert (pressing_down * slack_lower).abs().max().item() < 1e-8
    # The residuals are the ones the caller would compute.
    violation = torch.maximum(value - upper, lower - value).clamp_min(0.0).amax(-1)
    gradient = (H @ solution.x.unsqueeze(-1)).squeeze(-1) + g + (rows.mT @ solution.dual.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(solution.primal_residual, violation, atol=1e-12, rtol=0)
    torch.testing.assert_close(solution.dual_residual, gradient.abs().amax(-1), atol=1e-9, rtol=0)


def test_rows_in_conflict_give_the_least_squares_compromise_in_the_rows_units() -> None:
    """``x0 <= -1`` against ``x0 >= 1``: the middle, and the heavier side once a row is scaled; ``x1`` is untouched."""
    H = torch.eye(2, dtype=torch.float64)
    g = torch.tensor([0.3, -0.5], dtype=torch.float64)
    rows = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    lower = torch.tensor([-INF, 1.0, -INF], dtype=torch.float64)
    upper = torch.tensor([-1.0, INF, 2.0], dtype=torch.float64)
    even = solve_qp(H, g, rows=rows, lower=lower, upper=upper, iterations=200)
    torch.testing.assert_close(even.x, torch.tensor([0.0, 0.5], dtype=torch.float64), atol=1e-8, rtol=0)
    assert even.primal_residual.item() == pytest.approx(1.0, abs=1e-8)
    # Scaling a row and its bound by 3 weights its violation by 9: argmin (x + 1)^2 + 9 (1 - x)^2 = 0.8.
    scaled = solve_qp(
        H, g, rows=rows * torch.tensor([[1.0], [3.0], [1.0]]), lower=lower * 3, upper=upper, iterations=200
    )
    torch.testing.assert_close(scaled.x, torch.tensor([0.8, 0.5], dtype=torch.float64), atol=1e-8, rtol=0)
    # The multipliers of the rows in conflict grow with the steps; the point does not.
    longer = solve_qp(H, g, rows=rows, lower=lower, upper=upper, iterations=400)
    assert longer.dual[0].item() > 1.5 * even.dual[0].item()
    torch.testing.assert_close(longer.x, even.x, atol=1e-8, rtol=0)


def test_padding_and_repeated_rows_do_not_move_the_solution() -> None:
    problem = _problems(16, 5, 8, seed=1)
    H, g, rows, lower, upper = problem
    plain = _repeat(problem, 12)
    pad = torch.zeros(16, 3, 5, dtype=torch.float64)
    open_bound = torch.full((16, 3), INF, dtype=torch.float64)
    padded = _repeat(
        (H, g, torch.cat([rows, pad], 1), torch.cat([lower, -open_bound], 1), torch.cat([upper, open_bound], 1)), 12
    )
    torch.testing.assert_close(padded.x, plain.x, atol=1e-8, rtol=0)
    assert padded.dual[:, 8:].abs().max().item() == 0.0
    doubled = _repeat((H, g, rows.repeat(1, 2, 1), lower.repeat(1, 2), upper.repeat(1, 2)), 12)
    torch.testing.assert_close(doubled.x, plain.x, atol=1e-8, rtol=0)


def test_more_active_rows_than_variables() -> None:
    """Forty half-spaces tangent to a ball around a vertex the objective pulls past: the vertex."""
    generator = torch.Generator().manual_seed(2)
    normals = torch.nn.functional.normalize(torch.randn(40, 4, generator=generator, dtype=torch.float64), dim=-1)
    vertex = 0.2 * torch.randn(4, generator=generator, dtype=torch.float64)
    solution = solve_qp(
        torch.eye(4, dtype=torch.float64),
        -(vertex + 5.0 * normals.sum(0)),
        rows=normals,
        lower=torch.full((40,), -INF, dtype=torch.float64),
        upper=normals @ vertex,
        iterations=100,
    )
    assert solution.primal_residual.item() < 1e-6
    value = normals @ solution.x
    assert (value <= normals @ vertex + 1e-6).all()


def test_a_failed_element_is_flagged_and_leaves_the_others_alone() -> None:
    H, g, rows, lower, upper = _problems(3, 4, 6, seed=3)
    good = solve_qp(H, g, rows=rows, lower=lower, upper=upper)
    H, g = H.clone(), g.clone()
    H[0] = -torch.eye(4, dtype=torch.float64)
    g[1, 0] = float("nan")
    solution = solve_qp(H, g, rows=rows, lower=lower, upper=upper)
    for bad in (0, 1):
        assert solution.x[bad].abs().max().item() == 0.0 and solution.dual[bad].abs().max().item() == 0.0
        assert solution.primal_residual[bad].item() == INF and solution.dual_residual[bad].item() == INF
        assert solution.penalty[bad].item() == 1.0
    torch.testing.assert_close(solution.x[2], good.x[2])
    with pytest.raises(ShapeError, match="at least one row"):
        solve_qp(H, g, rows=rows[:, :0], lower=lower[:, :0], upper=upper[:, :0])
    with pytest.raises(ShapeError, match=r"H \(B\.\.\., N, N\)"):
        solve_qp(H, g, rows=rows, lower=lower, upper=upper[:, :-1])


def test_batch_axes_dtype_and_gradient() -> None:
    H, g, rows, lower, upper = (t.float() for t in _problems(6, 4, 7, seed=4))
    flat = solve_qp(H, g, rows=rows, lower=lower, upper=upper)
    grid = solve_qp(
        H.reshape(2, 3, 4, 4), g.reshape(2, 3, 4).requires_grad_(), rows=rows.reshape(2, 3, 7, 4),
        lower=lower.reshape(2, 3, 7), upper=upper.reshape(2, 3, 7),
    )  # fmt: skip
    single = solve_qp(H[0], g[0], rows=rows[0], lower=lower[0], upper=upper[0])
    assert grid.x.shape == (2, 3, 4) and grid.dual.shape == (2, 3, 7) and grid.penalty.shape == (2, 3)
    assert single.x.shape == (4,) and single.primal_residual.shape == ()
    assert grid.x.dtype == torch.float32 and not grid.x.requires_grad
    torch.testing.assert_close(grid.x.reshape(6, 4), flat.x)
    torch.testing.assert_close(single.x, flat.x[0])


def test_solve_traces_as_one_graph_with_no_host_read() -> None:
    """A cold solve and a warm one each compile whole, equal the eager result, and read no tensor back."""
    problem = _problems(8, 5, 9, seed=5)

    def twice(H, g, rows, lower, upper):
        first = solve_qp(H, g, rows=rows, lower=lower, upper=upper)
        second = solve_qp(H, g, rows=rows, lower=lower, upper=upper, warm=first)
        return second.x, second.dual, second.primal_residual, second.dual_residual, second.penalty

    torch._dynamo.reset()
    expected = twice(*problem)
    actual = torch.compile(twice, fullgraph=True)(*problem)
    for ours, theirs in zip(actual[:-1], expected[:-1], strict=True):
        torch.testing.assert_close(ours, theirs, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(actual[-1], expected[-1], rtol=1e-3, atol=0)  # the penalty is a ratio of small residuals
    with torch.profiler.profile() as profile:
        twice(*problem)
    reads = {"aten::_local_scalar_dense", "aten::nonzero", "aten::item"}
    assert not [event.name for event in profile.events() if event.name in reads]
