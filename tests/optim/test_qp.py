"""`solve_qp`: closed forms, the optimality conditions, rows that cannot hold, padding, failures and tracing."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.exceptions import ArgumentError, ShapeError
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


def _cap(H: torch.Tensor, g: torch.Tensor, weight: float = 100.0) -> torch.Tensor:
    """Where the multipliers stop ``(B, 1)``: ``weight`` times the larger of ``|g|`` and the diagonal of ``H``."""
    return weight * torch.maximum(g.abs().amax(-1), H.diagonal(dim1=-2, dim2=-1).amax(-1)).unsqueeze(-1)


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


def test_equality_rows_alone_are_a_linear_solve() -> None:
    """Four equalities on seven variables: every element is exact after one cold call, whatever its multiplier signs."""
    generator = torch.Generator().manual_seed(0)
    g = torch.randn(256, 7, generator=generator, dtype=torch.float64)
    rows = torch.randn(256, 4, 7, generator=generator, dtype=torch.float64)
    bound = torch.randn(256, 4, generator=generator, dtype=torch.float64)
    H = torch.eye(7, dtype=torch.float64).expand(256, 7, 7)
    for iterations in (1, 20):
        solution = solve_qp(H, g, rows=rows, lower=bound, upper=bound, iterations=iterations)
        assert solution.primal_residual.max().item() < 1e-8 and solution.dual_residual.max().item() < 1e-8
        # The closed form: x = -g - rows' dual with (rows rows') dual = -(rows g + bound).
        dual = torch.linalg.solve(rows @ rows.mT, -((rows @ g.unsqueeze(-1)).squeeze(-1) + bound))
        torch.testing.assert_close(solution.dual, dual, atol=1e-8, rtol=1e-8)
        assert (dual > 0.1).any() and (dual < -0.1).any()


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
    assert solution.exact.all() and first.exact.sum().item() >= 32
    assert (first.primal_residual[first.exact] < 1e-8).all() and (first.dual_residual[first.exact] < 1e-8).all()
    value = (rows @ solution.x.unsqueeze(-1)).squeeze(-1)
    slack_upper = torch.where(torch.isfinite(upper), upper - value, torch.zeros_like(value))
    slack_lower = torch.where(torch.isfinite(lower), value - lower, torch.zeros_like(value))
    pressing_up, pressing_down = solution.dual.clamp_min(0.0), (-solution.dual).clamp_min(0.0)
    assert (pressing_up * slack_upper).abs().max().item() < 1e-8
    assert (pressing_down * slack_lower).abs().max().item() < 1e-8
    # The residuals are the ones the caller would compute, also on a step that is not the solution: each row against
    # the bound its multiplier presses on, or against its bounds; the largest violation is never above it.
    for step in (solution, first):
        value = (rows @ step.x.unsqueeze(-1)).squeeze(-1)
        held = torch.where(step.dual > 0, upper, torch.where(step.dual < 0, lower, torch.clamp(value, lower, upper)))
        violation = torch.maximum(value - upper, lower - value).clamp_min(0.0).amax(-1)
        gradient = (H @ step.x.unsqueeze(-1)).squeeze(-1) + g + (rows.mT @ step.dual.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(step.primal_residual, (value - held).abs().amax(-1), atol=1e-12, rtol=0)
        torch.testing.assert_close(step.dual_residual, gradient.abs().amax(-1), atol=1e-9, rtol=0)
        assert (step.primal_residual >= violation - 1e-12).all()
    assert (first.primal_residual > 1e-3).any()


def test_rows_in_conflict_give_the_elastic_minimiser_in_the_rows_units() -> None:
    """``x0 <= -1`` against ``x0 >= 1``: any ``x0`` between costs the same, so the objective picks; ``x1`` is untouched."""
    H = torch.eye(2, dtype=torch.float64)
    g = torch.tensor([0.3, -0.5], dtype=torch.float64)
    rows = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    lower = torch.tensor([-INF, 1.0, -INF], dtype=torch.float64)
    upper = torch.tensor([-1.0, INF, 2.0], dtype=torch.float64)
    even = _repeat((H, g, rows, lower, upper), 30)
    torch.testing.assert_close(even.x, torch.tensor([-0.3, 0.5], dtype=torch.float64), atol=1e-8, rtol=0)
    # Both rows give way: their multipliers sit at the cap (the weight of 100 times the unit diagonal of H) and
    # cancel; the residual is the larger violation.
    torch.testing.assert_close(even.dual, torch.tensor([1e2, -1e2, 0.0], dtype=torch.float64), atol=1e-6, rtol=0)
    assert even.primal_residual.item() == pytest.approx(1.3, abs=1e-8)
    assert even.dual_residual.item() < 1e-8
    # Scaling a row and its bound by 3 counts its violation three times: that row holds, the other gives way.
    scaled = _repeat((H, g, rows * torch.tensor([[1.0], [3.0], [1.0]]), lower * 3, upper), 30)
    torch.testing.assert_close(scaled.x, torch.tensor([1.0, 0.5], dtype=torch.float64), atol=1e-8, rtol=0)
    expected = torch.tensor([1e2, -(1e2 + 1.3) / 3, 0.0], dtype=torch.float64)
    torch.testing.assert_close(scaled.dual, expected, atol=1e-6, rtol=0)
    assert scaled.primal_residual.item() == pytest.approx(2.0, abs=1e-8)


def test_a_row_gives_way_at_the_cap() -> None:
    """A box whose multipliers would be -1.5 and 2, stopped at the cap 1: a weight of 1/3 times the largest entry of g, 3."""
    g = torch.tensor([[2.0, -0.05, -3.0]], dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64)[None]
    low, high = torch.full((1, 3), -0.5, dtype=torch.float64), torch.full((1, 3), 1.0, dtype=torch.float64)
    soft = _repeat((eye, g, eye, low, high), 10, weight=1.0 / 3.0)
    torch.testing.assert_close(soft.x, torch.tensor([[-1.0, 0.05, 2.0]], dtype=torch.float64), atol=1e-8, rtol=0)
    torch.testing.assert_close(soft.dual, torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float64), atol=1e-8, rtol=0)
    assert soft.primal_residual.item() == pytest.approx(1.0, abs=1e-8) and soft.dual_residual.item() < 1e-8


def _conflict(kind: str, call: int, *, conflict: bool, batch: int = 48, size: int = 7) -> tuple[torch.Tensor, ...]:
    """Two rows in conflict inside a 0.3 box, ``H`` with a 1e-4 floor; without ``conflict`` the two rows hold.

    ``drift``: two different rows whose normals move about 1 % per call; ``turn``: one row asked both ways, turning
    slowly, the gradient moving; ``fixed``: the same row, constant.
    """
    generator = torch.Generator().manual_seed(1)
    draw = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)  # noqa: E731
    root = draw(batch, size, size)
    H, g = root.mT @ root / size + 1e-4 * torch.eye(size, dtype=torch.float64), draw(batch, size)
    first, other, turn, turn_other = (
        draw(batch, 1, size),
        draw(batch, 1, size),
        draw(batch, 1, size),
        draw(batch, 1, size),
    )
    box = torch.eye(size, dtype=torch.float64).expand(batch, size, size)
    edge, open_side = (
        torch.full((batch, size), 0.3, dtype=torch.float64),
        torch.full((batch, 1), INF, dtype=torch.float64),
    )
    one = torch.ones(batch, 1, dtype=torch.float64)
    side = -one if conflict else one
    if kind == "drift":  # first x <= -1 and second x <= -1 with second close to -first
        second = -first + 0.3 * other
        pair = torch.cat(
            [first + 0.05 * math.sin(0.01 * call) * turn, second + 0.05 * math.cos(0.01 * call) * turn_other], 1
        )
        return (
            H,
            g,
            torch.cat([pair, box], 1),
            torch.cat([-open_side, -open_side, -edge], 1),
            torch.cat([side, side, edge], 1),
        )
    if kind == "turn":
        first, g = first + 0.02 * math.sin(0.1 * call) * turn, g + 0.01 * math.sin(0.05 * call)
    pair = torch.cat([first, first], 1)  # first x <= -1 against first x >= 1
    return (
        H,
        g,
        torch.cat([pair, box], 1),
        torch.cat([-open_side, -side, -edge], 1),
        torch.cat([side, open_side, edge], 1),
    )


@pytest.mark.parametrize(("kind", "stretch"), [("drift", 100), ("turn", 100), ("fixed", 50)])
def test_a_long_stretch_of_rows_in_conflict_stays_bounded_and_lets_go(kind: str, stretch: int) -> None:
    """Warm calls through rows in conflict keep a bounded step; ten calls after the rows can hold, every step is right.

    No step wrong by more than 1e-3 is flagged exact on the way. With hard rows the multipliers and the carried penalty grew through the stretch: steps of 47 against a box of
    0.3, and afterwards wrong steps that reported both residuals under 1e-4 (found in review).
    """
    solution = None
    for call in range(stretch):
        H, g, rows, lower, upper = _conflict(kind, call, conflict=True)
        solution = solve_qp(H, g, rows=rows, lower=lower, upper=upper, warm=solution)
        # An elastic minimiser may leave the box (by up to 1.76 in review): one box row
        # gives way where both rows can then hold.
        assert solution.x.abs().max().item() < 2.0
        assert (solution.dual.abs() <= _cap(H, g) * (1 + 1e-12)).all()
    for call in range(stretch, stretch + 10):
        problem = _conflict(kind, call, conflict=False)
        H, g, rows, lower, upper = problem
        solution = solve_qp(H, g, rows=rows, lower=lower, upper=upper, warm=solution)
        reference = _repeat(problem, 25, iterations=100)
        assert reference.primal_residual.max().item() < 1e-8 and reference.dual_residual.max().item() < 1e-7
        wrong = (solution.x - reference.x).abs().amax(-1) > 1e-3
        assert not (wrong & solution.exact).any()
        assert not (wrong & (solution.primal_residual < 1e-4) & (solution.dual_residual < 1e-4)).any()
    assert not wrong.any()


def test_a_warm_call_at_the_elastic_minimiser_of_rows_in_conflict_stays_there() -> None:
    """Twenty warm calls put every element on its elastic minimiser, certified here; one more call leaves it there.

    A warm start that pulled the rows inside their bounds moved 78 of 512 such minimisers by up to 0.27 (found in
    review), so some elements never settled.
    """
    problem = _conflict("fixed", 0, conflict=True)
    H, g, rows, lower, upper = problem
    solution = _repeat(problem, 20)
    # The certificate: the gradient vanishes, and each multiplier is the slope of `weight` times the violation at
    # its row: zero inside the bounds, of the side's sign on a bound, `weight` beyond it.
    gradient = (H @ solution.x.unsqueeze(-1)).squeeze(-1) + g + (rows.mT @ solution.dual.unsqueeze(-1)).squeeze(-1)
    reach = (rows @ solution.x.unsqueeze(-1)).squeeze(-1) + solution.dual
    slope = torch.clamp(reach - torch.clamp(reach, lower, upper), -_cap(H, g), _cap(H, g))
    assert gradient.abs().max().item() < 1e-8 and (slope - solution.dual).abs().max().item() < 1e-8
    assert (solution.primal_residual > 0.5).all()  # the rows cannot all hold
    again = solve_qp(H, g, rows=rows, lower=lower, upper=upper, warm=solution)
    assert (again.x - solution.x).abs().max().item() < 1e-8


def test_an_exact_element_is_a_minimiser_where_the_curvature_is_weak() -> None:
    """Problems shaped like damped inverse kinematics, damping 1e-6: what is flagged exact meets the optimality conditions.

    The active-set step used to cut a multiplier of the wrong sign to zero and keep its point where the gradient
    left was under 1e-6 of its largest term. In these 52 calls, 8 kept points had a gradient up to 2.6e-6 and sat
    3.7e-3 to 5.8e-2 from the minimiser, along a curvature of 5e-6 (found in review). Such a point is
    now refused; the gradient of a kept one is the floor of the solve, 1e-9 of the step.
    """
    generator = torch.Generator().manual_seed(3)
    draw = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)  # noqa: E731
    jacobian, residual, turn, shift, pair = (
        draw(512, 6, 7),
        draw(512, 6),
        draw(512, 6, 7),
        draw(512, 6),
        draw(512, 1, 7),
    )
    rows = torch.cat([pair, pair, torch.eye(7, dtype=torch.float64).expand(512, 7, 7)], 1)
    edge = torch.full((512, 7), 0.3, dtype=torch.float64)
    open_side = torch.full((512, 1), INF, dtype=torch.float64)
    solution, kept = None, 0
    for call in range(52):
        J = jacobian + 0.1 * math.sin(0.05 * call) * turn
        H = J.mT @ J
        H = H + 1e-6 * torch.diag_embed(H.diagonal(dim1=-2, dim2=-1))
        g = (J.mT @ (residual + 0.2 * math.sin(0.07 * call) * shift).unsqueeze(-1)).squeeze(-1)
        side = torch.full((512, 1), -0.05 if call < 40 else 0.05, dtype=torch.float64)  # 40 calls in conflict
        lower, upper = torch.cat([-open_side, -side, -edge], 1), torch.cat([side, open_side, edge], 1)
        solution = solve_qp(H, g, rows=rows, lower=lower, upper=upper, warm=solution)
        gradient = (H @ solution.x.unsqueeze(-1)).squeeze(-1) + g + (rows.mT @ solution.dual.unsqueeze(-1)).squeeze(-1)
        reach = (rows @ solution.x.unsqueeze(-1)).squeeze(-1) + solution.dual
        slope = torch.clamp(reach - torch.clamp(reach, lower, upper), -_cap(H, g), _cap(H, g))
        assert (gradient.abs().amax(-1) * solution.exact).max().item() < 1e-7
        assert ((slope - solution.dual).abs().amax(-1) * solution.exact).max().item() < 1e-9
        kept += solution.exact.sum().item()
    assert kept > 20000  # 21645 of 26624 when written


@pytest.mark.parametrize("factor", [1e-2, 1e3])
def test_the_weight_follows_the_scale_of_the_objective(factor: float) -> None:
    """Scaling ``H`` and ``g`` scales the multipliers and leaves ``x``, where rows give way and where they hold.

    Under one fixed cap of 1e3, rows in conflict let go slowly at a hundredth of the objective (498 of 512
    problems wrong ten calls later) and feasible rows gave way at a thousand times it (found in review).
    """
    H, g, rows, lower, upper = problem = _conflict("fixed", 0, conflict=True, batch=16)
    base, scaled = _repeat(problem, 20), _repeat((H * factor, g * factor, rows, lower, upper), 20)
    assert base.exact.all() and scaled.exact.all() and (base.primal_residual > 0.5).all()
    torch.testing.assert_close(scaled.x, base.x, atol=1e-7, rtol=0)
    torch.testing.assert_close(scaled.dual, base.dual * factor, atol=0, rtol=1e-6)
    # A row that asks for a motion holds although the gradient vanishes: the cap follows the diagonal of H too.
    eye = factor * torch.eye(3, dtype=torch.float64)
    push = solve_qp(
        eye,
        torch.zeros(3, dtype=torch.float64),
        rows=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        lower=torch.tensor([0.1], dtype=torch.float64),
        upper=torch.tensor([INF], dtype=torch.float64),
    )
    torch.testing.assert_close(push.x, torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64), atol=1e-9, rtol=0)
    torch.testing.assert_close(push.dual, torch.tensor([-0.1 * factor], dtype=torch.float64), atol=0, rtol=1e-6)


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


def test_a_zero_row_that_cannot_hold_stays_bounded_and_crossed_bounds_are_an_equality() -> None:
    problem = _problems(16, 5, 8, seed=1)
    H, g, rows, lower, upper = problem
    plain = _repeat(problem, 12)
    # 0 <= -0.3 can never hold: the multiplier of that row stops at the weight and the others are solved as before.
    zero, never = torch.zeros(16, 1, 5, dtype=torch.float64), torch.full((16, 1), -0.3, dtype=torch.float64)
    stuck_problem = (
        H,
        g,
        torch.cat([rows, zero], 1),
        torch.cat([lower, torch.full_like(never, -INF)], 1),
        torch.cat([upper, never], 1),
    )
    stuck = _repeat(stuck_problem, 12)
    torch.testing.assert_close(stuck.x, plain.x, atol=1e-8, rtol=0)
    torch.testing.assert_close(stuck.dual[:, -1], _cap(H, g).squeeze(-1), atol=1e-9, rtol=0)
    torch.testing.assert_close(stuck.primal_residual, torch.full((16,), 0.3, dtype=torch.float64), atol=1e-9, rtol=0)
    first = solve_qp(*stuck_problem[:2], rows=stuck_problem[2], lower=stuck_problem[3], upper=stuck_problem[4])
    assert (stuck.penalty <= 1e4 * first.penalty).all()  # within 1e3 of the cold penalty, which one call moves by 10

    # lower = 1 above upper = -1 on x0: the row is the equality x0 = -1.
    crossed = solve_qp(
        torch.eye(2, dtype=torch.float64),
        torch.tensor([0.3, -0.5], dtype=torch.float64),
        rows=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        lower=torch.tensor([1.0], dtype=torch.float64),
        upper=torch.tensor([-1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(crossed.x, torch.tensor([-1.0, 0.5], dtype=torch.float64), atol=1e-9, rtol=0)
    torch.testing.assert_close(crossed.dual, torch.tensor([0.7], dtype=torch.float64), atol=1e-8, rtol=0)
    assert crossed.primal_residual.item() < 1e-9


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
        assert solution.penalty[bad].item() == 1.0 and not solution.exact[bad].item()
    torch.testing.assert_close(solution.x[2], good.x[2])
    with pytest.raises(ShapeError, match="at least one row"):
        solve_qp(H, g, rows=rows[:, :0], lower=lower[:, :0], upper=upper[:, :0])
    with pytest.raises(ShapeError, match=r"hessian \(B\.\.\., N, N\)"):
        solve_qp(H, g, rows=rows, lower=lower, upper=upper[:, :-1])
    # A warm start of another layout is refused: one row fewer, and the same number of entries in another shape.
    with pytest.raises(ShapeError, match=r"dual \(3, 5\).*got \(3, 4\), \(3, 6\), \(3,\)"):
        solve_qp(H, g, rows=rows[:, :5], lower=lower[:, :5], upper=upper[:, :5], warm=good)
    reshaped = QPSolution(
        good.x.reshape(2, 6), good.dual.reshape(2, 9), primal_residual=good.primal_residual,
        dual_residual=good.dual_residual, penalty=good.penalty[:2], exact=good.exact[:2],
    )  # fmt: skip
    with pytest.raises(ShapeError, match="warm start of the same problem layout"):
        solve_qp(H, g, rows=rows, lower=lower, upper=upper, warm=reshaped)
    # The weight is a positive Python number: a tensor is refused by its type, with no read of its value.
    for weight in (0.0, -1.0, float("nan"), True, "100", torch.tensor(100.0)):
        with pytest.raises(ArgumentError, match="positive Python number as weight"):
            solve_qp(H, g, rows=rows, lower=lower, upper=upper, weight=weight)


def test_batch_axes_dtype_and_gradient() -> None:
    H, g, rows, lower, upper = (t.float() for t in _problems(6, 4, 7, seed=4))
    flat = solve_qp(H, g, rows=rows, lower=lower, upper=upper)
    grid = solve_qp(
        H.reshape(2, 3, 4, 4), g.reshape(2, 3, 4).requires_grad_(), rows=rows.reshape(2, 3, 7, 4),
        lower=lower.reshape(2, 3, 7), upper=upper.reshape(2, 3, 7),
    )  # fmt: skip
    single = solve_qp(H[0], g[0], rows=rows[0], lower=lower[0], upper=upper[0])
    assert grid.x.shape == (2, 3, 4) and grid.dual.shape == (2, 3, 7) and grid.penalty.shape == (2, 3)
    assert grid.exact.shape == (2, 3) and grid.exact.dtype == torch.bool
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
        return second.x, second.dual, second.primal_residual, second.dual_residual, second.exact, second.penalty

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
