"""A batch of independent quadratic programs with two-sided linear rows: one warm-startable step of a fixed cost.

``solve_qp`` minimises ``½ xᵀH x + gᵀx`` subject to ``lower <= rows x <= upper`` per batch element, with elastic
rows (a row gives way once its multiplier reaches a cap that ``weight`` sets). An equality is a row with
``lower == upper``, a one-sided row has an infinite bound, a box on ``x`` is identity rows. The method is the
alternating direction method of multipliers in the form OSQP uses (one factorization of ``H + penalty · rowsᵀ rows``
per call, then ``iterations`` steps of three matrix-vector products each), followed by one active-set step that is
kept where it lands on the exact solution (decision record 0054). Every batch element runs the same steps, with no
Python branch on a tensor value and no host read, so the whole solve traces as one graph. It is made to be called
once per step of an outer loop and started from its own last answer: accuracy comes from those repeated warm calls,
not from one call.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import final, override

import torch

from ..exceptions import ArgumentError, ShapeError

_FLOOR = (
    1e-9  # added to the unit-diagonal H, as Normal.solve does: a variable the objective does not see stays solvable
)
_RELAXATION = 1.6  # over-relaxation of each step (the default of OSQP)
_PENALTIES = (1e-6, 1e6)  # range of the cold penalty, in unit-diagonal variables
_RANGE = 1e3  # a penalty stays within this factor of the cold one of the problem it is used on
_RETUNE = 10.0  # largest factor by which one solve changes the penalty it hands to the next
_SETTLED = 1e-10  # relative residual under which a side counts as converged when the penalty is retuned
_STIFFNESS = 1e6  # weight of an active row in the active-set step, in units of the cold penalty
_UPDATES = 2  # multiplier updates of the active-set step: the active rows hold to rounding
_EXACT = 1e-6  # relative size of both residuals under which the active-set step counts as the solution


@final
@dataclass(frozen=True)
class QPSolution:
    """What `solve_qp` returns, per batch element; pass it back as ``warm=`` to start the next solve from it."""

    x: torch.Tensor  #: ``(B..., N)`` the minimiser where ``exact``, elsewhere the last step towards it.
    dual: torch.Tensor  #: ``(B..., M)`` multiplier of each row: positive presses on ``upper``, negative on ``lower``.
    _: KW_ONLY
    #: ``(B...,)`` largest distance of a row from where its multiplier holds it, in that row's unit: from ``upper``
    #: where ``dual > 0``, from ``lower`` where ``dual < 0``, from the bounds where ``dual == 0`` (the violation).
    primal_residual: torch.Tensor
    dual_residual: torch.Tensor  #: ``(B...,)`` largest entry of ``H x + g + rowsᵀ dual``.
    penalty: torch.Tensor  #: ``(B...,)`` weight of the rows against ``H`` that the next warm solve starts with.
    #: ``(B...,)`` bool: ``x`` and ``dual`` are the active-set point, the minimiser of the elastic problem. Where
    #: rows give way, ``primal_residual`` stays positive at the minimiser, and small residuals do not place a step
    #: near the minimiser along weak curvature: this is the sign of having it.
    exact: torch.Tensor

    @override
    def __repr__(self) -> str:
        dtype = str(self.x.dtype).removeprefix("torch.")
        batch, size, count = tuple(self.x.shape[:-1]), self.x.shape[-1], self.dual.shape[-1]
        return f"QPSolution(B={batch}, N={size}, M={count}, {dtype}, {self.x.device})"


@final
@dataclass(frozen=True)
class _Scaled:
    """One batch of problems in unit-diagonal variables ``x = scale · y``; the rows keep the caller's units.

    A vector of the variables is a row ``(B, 1, N)``: a row times a batch of matrices is the fast product, several
    times faster on one CPU thread than the matrices times a column, and it needs no transposed copy of ``A``.
    """

    H: torch.Tensor  # (B, N, N) unit diagonal plus the floor
    c: torch.Tensor  # (B, 1, N)
    A: torch.Tensor  # (B, M, N)
    lower: torch.Tensor  # (B, M)
    upper: torch.Tensor  # (B, M)

    def values(self, y: torch.Tensor) -> torch.Tensor:
        """The rows at ``y: (B, 1, N)``: ``(B, M)``."""
        return torch.bmm(y, self.A.mT).squeeze(-2)

    def measure(self, y: torch.Tensor, value: torch.Tensor, dual: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """How far each row value is from where ``dual`` holds it ``(B, M)``, the gradient ``(B, 1, N)``, its relative size.

        A positive multiplier holds its row on ``upper``, a negative one on ``lower``, a zero one between the bounds:
        the distance is the row's violation, or how far a row that is pressed on sits off its bound.
        """
        inside = torch.clamp(value, self.lower, self.upper)
        gap = value - torch.where(dual > 0, self.upper, torch.where(dual < 0, self.lower, inside))
        curve = torch.bmm(y, self.H)
        back = torch.bmm(dual.unsqueeze(-2), self.A)
        gradient = curve + self.c + back
        parts = torch.maximum(torch.maximum(curve.abs().amax(-1), back.abs().amax(-1)), self.c.abs().amax(-1))
        return gap, gradient, (gradient.abs().amax(-1) / parts.clamp_min(1e-300)).squeeze(-1)


def _solve(factor: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Two triangular solves, not ``cholesky_solve``: that one reads its status back to the host."""
    return torch.linalg.solve_triangular(
        factor.mT, torch.linalg.solve_triangular(factor, right, upper=False), upper=True
    )


def _active_set_step(
    problem: _Scaled, v: torch.Tensor, dual: torch.Tensor, yields: torch.Tensor, cold: torch.Tensor, cap: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """The rows that ``v: (B, M)`` puts on a bound solved as equalities: ``y: (B, 1, N)``, ``dual``, where it is exact.

    The equalities are held by a stiff penalty and `_UPDATES` multiplier updates. A row that ``yields`` (its
    multiplier is at ``cap: (B, 1)``) keeps that multiplier and is not held. The pair is the solution when its gradient
    vanishes, every row is where its multiplier holds it (a row that yields may sit beyond its bound) and no held
    row asks for a multiplier of the wrong sign or beyond ``cap``. Such a multiplier is not cut to fit: the
    gradient that the cut leaves is small, but along a direction of weak curvature a small gradient is a long way
    from the minimiser. An equality row (``lower == upper``) is always held and its multiplier has either sign.
    """
    equal = problem.lower == problem.upper
    above, below = (v >= problem.upper) | equal, (v <= problem.lower) | equal
    bound = torch.clamp(v, problem.lower, problem.upper)
    stiff = _STIFFNESS * cold.unsqueeze(-1) * ((above | below) & ~yields)
    weighted = (problem.A * stiff.unsqueeze(-1)).mT
    factor, info = torch.linalg.cholesky_ex(problem.H + torch.bmm(weighted, problem.A))
    pull, force = stiff * bound, dual
    y = _solve(factor, (torch.bmm((pull - force).unsqueeze(-2), problem.A) - problem.c).mT).mT
    value = problem.values(y)
    for _ in range(_UPDATES):
        force = force + stiff * (value - bound)
        y = _solve(factor, (torch.bmm((pull - force).unsqueeze(-2), problem.A) - problem.c).mT).mT
        value = problem.values(y)
    nothing = torch.zeros_like(v)
    lowest, highest = torch.where(below, -cap, nothing), torch.where(above, cap, nothing)
    asked = force + stiff * (value - bound)
    force = torch.clamp(asked, lowest, highest)
    gap, slope, ahead = problem.measure(y, value, force)
    strict = torch.where(yields & (gap * dual > 0), nothing, gap)
    reach = torch.maximum(value.abs().amax(-1), torch.clamp(value, problem.lower, problem.upper).abs().amax(-1))
    allowed = ((asked == force) | (stiff == 0)).all(-1)
    exact = (info == 0) & allowed & (strict.abs().amax(-1) <= _EXACT * reach) & (ahead <= _EXACT)
    return y, force, gap, slope, exact


def _checked(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    rows: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weight: float,
    warm: QPSolution | None,
) -> tuple[torch.Size, int, int]:
    """The batch shape, ``N`` and ``M`` of inputs that fit each other: the one check at the boundary, shapes and types only."""
    if isinstance(weight, bool) or not isinstance(weight, int | float) or not weight > 0:
        raise ArgumentError(f"solve_qp takes a positive Python number as weight, got {weight!r}")
    batch, size, count = gradient.shape[:-1], gradient.shape[-1], lower.shape[-1]
    if hessian.shape != (*batch, size, size) or rows.shape != (*batch, count, size) or upper.shape != (*batch, count):
        raise ShapeError(
            f"solve_qp takes hessian (B..., N, N), gradient (B..., N), rows (B..., M, N) and lower, upper (B..., M); got "
            f"{tuple(hessian.shape)}, {tuple(gradient.shape)}, {tuple(rows.shape)}, {tuple(lower.shape)}, {tuple(upper.shape)}"
        )
    if count == 0:
        raise ShapeError(
            "solve_qp needs at least one row; without rows the minimiser is Normal(hessian, gradient).solve()"
        )
    if warm is not None and (warm.x.shape, warm.dual.shape, warm.penalty.shape) != (gradient.shape, lower.shape, batch):
        raise ShapeError(
            f"solve_qp takes a warm start of the same problem layout: x {tuple(gradient.shape)}, dual "
            f"{tuple(lower.shape)}, penalty {tuple(batch)}; got {tuple(warm.x.shape)}, {tuple(warm.dual.shape)}, "
            f"{tuple(warm.penalty.shape)}"
        )
    return batch, size, count


def solve_qp(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    *,
    rows: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    iterations: int = 20,
    weight: float = 100.0,
    warm: QPSolution | None = None,
) -> QPSolution:
    """One warm-startable step towards the minimiser of ``½ xᵀH x + gᵀx`` subject to ``lower <= rows x <= upper``.

    ``hessian: (B..., N, N)`` is ``H``, symmetric positive semi-definite (the ``H`` of a `Normal` with its damping
    added), ``gradient: (B..., N)`` is ``g``, ``rows: (B..., M, N)`` with ``M >= 1``, ``lower, upper: (B..., M)``.
    All share the batch axes ``B...``; each batch element is a problem of its own. A bound may be infinite; a
    padding row is a zero row with both bounds infinite; a row whose ``lower`` is above its ``upper`` is the
    equality ``rows x = upper``. The factorizations run in float64 whatever the dtype, which the result keeps; the
    result carries no gradient.

    A call runs ``iterations`` steps, each moving ``x`` towards the rows and ``dual`` towards the multipliers, then
    solves the rows the multipliers press on as equalities. That point is returned, with ``exact`` set, where it
    puts every row where its multiplier holds it and makes the gradient vanish, both to a relative ``1e-6`` (of the
    largest row value and of the largest term of the gradient), and where no row it holds asks for a multiplier of
    the wrong sign or beyond the cap (below). A point that would need such a multiplier cut is refused: the gradient
    the cut leaves is small, but along a direction of weak curvature a small gradient is far from the minimiser.
    Elsewhere the last step is returned with ``exact`` unset, and ``primal_residual`` and ``dual_residual`` say how
    far it still is, in the caller's units.

    Small residuals bound the gap in the objective, not the distance to the minimiser along weak curvature. On
    problems shaped like damped inverse kinematics (damping 1e-4), steps up to 1.6e-2 from the minimiser reported
    ``primal_residual`` 0 and ``dual_residual`` under 1e-4, with an objective at most 1.4e-6 above the minimum.
    Between equality rows in conflict, steps up to 9e-2 from the elastic minimiser show a ``dual_residual`` of 1e-11
    to 1e-10 and a positive ``primal_residual``, as the minimiser does (2 and 64 of 256 such problems after 60 warm
    calls, none of them ``exact``). ``exact`` is the sign that a step is the minimiser; where rows give way it is the
    only one.

    One call is a step, not a solve: accuracy comes from calling again with ``warm=`` set to the last solution, not
    from more ``iterations``, because the penalty that sets the pace of the steps is tuned between calls. On random
    feasible problems of 7 variables and 30 rows, half of them active, ``cond(H) = 1e4``, a cold call solves (both
    residuals under 1e-8) none of 256 at 20 to 500 iterations (median violation 0.33 at 20, 0.062 at 500), while
    warm calls of 20 iterations on the same problems solve 47, 181 and 247 of them after 5, 12 and 40 calls (35
    variables and 150 rows: 44 of 128 after 40 calls). An outer loop whose problem changes little between its
    steps gets the same from one warm call per step. ``warm`` is the solution of a nearby problem with the same row
    layout; it carries ``x``, ``dual`` and the ``penalty`` the last call tuned. The penalty is kept within a factor
    of 1000 of the one a cold call of the new problem starts with; each row starts from its value at ``x``, so a
    warm call started at an elastic minimiser stays there (it moved by under 1e-9 on problems of 7 variables, 1.7e-6
    on problems shaped like inverse kinematics at damping 1e-6); and nothing in it needs a reset. A cold call starts
    from ``x = 0``.

    The rows bound the step only in the limit. A cold call on an ill-conditioned ``H`` can return a step far outside
    a box of identity rows: 35 against a box of 0.3 at ``cond(H) = 1e8`` and 20 iterations, 3.4 at 200 iterations,
    0.30 after 5 warm calls; ``primal_residual`` reports it.

    The rows are elastic. Each batch element has a cap, ``weight`` times the larger of ``|g|∞`` and the largest
    diagonal entry of ``H``; a row gives way once holding it would take a multiplier beyond the cap, so the problem
    solved is ``½ xᵀH x + gᵀx + cap · (sum of the rows' violations)`` and every ``dual`` stays within ``±cap``. Where
    the rows can all hold with multipliers under the cap, its minimiser is the one under hard rows. Where they
    cannot, it is the L1-elastic minimiser for that cap, with ``primal_residual`` positive: a row gives way where
    holding it would cost the objective more than the cap per unit of violation. Only as ``weight`` grows does that
    point tend to the least sum of violations and, among such points, the least objective. It may break one row by
    much where a least-squares compromise breaks many by little, and that row may be a box row: two rows in
    conflict inside a box of 0.3 can both hold once one box row gives way, at steps up to 1.76. Scale a box's rows up
    when the box must win.

    The cap follows the objective: scaling ``H`` and ``g`` by ``s`` scales the cap and every multiplier by ``s`` and
    leaves ``x``, so ``weight`` is a pure number. It compares a multiplier (objective per unit of a row) with a
    gradient (objective per unit of a variable), which assumes rows whose coefficients are of order one and
    variables in natural units. Scaling a row and its bounds by ``t`` divides its multiplier by ``t`` and counts its
    violation ``t`` times: that is how a row is weighted against the others. The unit-diagonal scaling of the
    variables inside the solve changes neither. With the default ``100.0`` no row of a feasible problem had to give
    way in the problems measured (shaped like inverse kinematics, with the objective scaled by 1 to 3000, with
    near-opposed rows, and with rows that ask for a motion where ``g`` vanishes: the multipliers needed stay at least
    28 times under the cap), and rows that had been in conflict let go within 7 warm calls in the review's problems
    of 7 variables (23 at 35 variables and 150 rows). A smaller ``weight`` lets go sooner (5 calls at 30) with less
    margin. A zero row whose bounds exclude 0 can never hold: its multiplier goes to the cap and stays there, and the
    other rows are solved as without it.

    A batch element whose ``H + penalty · rowsᵀ rows`` is not positive definite, or whose data are not finite, gets
    ``x = 0``, ``dual = 0``, both residuals ``inf`` and ``penalty = 1``.
    """
    batch, size, count = _checked(hessian, gradient, rows, lower, upper, weight, warm)
    matrix = hessian.detach().double().reshape(-1, size, size)
    diagonal = matrix.diagonal(dim1=-2, dim2=-1)
    scale = torch.where(diagonal > 0, diagonal.clamp_min(1e-300).rsqrt(), torch.ones_like(diagonal))
    identity = torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    linear = gradient.detach().double().reshape(-1, size)
    # The multipliers stop at `weight` times the size of the objective's gradient: at x = 0, or one unit of a
    # variable away where that is larger, so that a gradient that vanishes does not let every row go.
    cap = weight * torch.maximum(linear.abs().amax(-1), diagonal.amax(-1)).unsqueeze(-1)
    problem = _Scaled(
        H=matrix * scale.unsqueeze(-1) * scale.unsqueeze(-2) + _FLOOR * identity,
        c=(linear * scale).unsqueeze(-2),
        A=rows.detach().double().reshape(-1, count, size) * scale.unsqueeze(-2),
        lower=torch.minimum(lower, upper).detach().double().reshape(-1, count),
        upper=upper.detach().double().reshape(-1, count),
    )
    gram = torch.bmm(problem.A.mT, problem.A)
    # The rows weigh as much as the objective: trace(cold · gram) = trace(unit-diagonal H) = N.
    cold = (size / gram.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-300)).clamp(*_PENALTIES)
    if warm is None:
        penalty = cold
        y, dual = torch.zeros_like(problem.c), torch.zeros_like(problem.lower)
    else:
        penalty = torch.clamp(warm.penalty.detach().double().reshape(-1), cold / _RANGE, cold * _RANGE)
        y = (warm.x.detach().double().reshape(-1, size) / scale).unsqueeze(-2)
        dual = torch.clamp(warm.dual.detach().double().reshape(-1, count), -cap, cap)
    rho = penalty.unsqueeze(-1)
    most = cap / rho

    # The inverse of H + penalty · rowsᵀ rows, by N right-hand sides: each step is then three products with a vector.
    factor, info = torch.linalg.cholesky_ex(problem.H + rho.unsqueeze(-1) * gram)
    inverse = _solve(factor, identity.expand_as(gram))
    free, spread = -torch.bmm(problem.c, inverse), rho.unsqueeze(-1) * inverse
    # State v = z + give: z is where the rows are held, give = dual / penalty. A row is held inside its bounds until
    # its multiplier reaches `cap`; from there it gives way and z follows v beyond the bound (the L1-elastic row).
    # A start takes z from the rows as they are (OSQP's warm start): pulled inside the bounds, the z of a row that
    # gives way would leave the minimiser a warm call starts at.
    value = problem.values(y)
    z, give = value, dual / rho
    v = z + give
    for _ in range(iterations):
        y = torch.baddbmm(free, torch.bmm((z - give).unsqueeze(-2), problem.A), spread)
        value = problem.values(y)
        v = torch.add(v, value - z, alpha=_RELAXATION)
        give = torch.clamp(v - torch.clamp(v, problem.lower, problem.upper), -most, most)
        z = v - give
    dual = rho * give
    gap, slope, ahead = problem.measure(y, value, dual)
    # The next solve weighs the rows more when they lag behind the multipliers, less when they lead (OSQP's rule), by
    # the residuals of the elastic problem: they vanish at its minimiser even when the rows cannot all hold, so a
    # long stretch of such problems leaves a penalty the next problem can use. Both sides are floored at `_SETTLED`:
    # a pair that has converged keeps its penalty instead of following rounding.
    behind = (value - z).abs().amax(-1) / torch.maximum(value.abs().amax(-1), z.abs().amax(-1)).clamp_min(1e-300)
    change = (behind.clamp_min(_SETTLED) / ahead.clamp_min(_SETTLED)).sqrt().clamp(1.0 / _RETUNE, _RETUNE)
    penalty = torch.clamp(penalty * change, cold / _RANGE, cold * _RANGE)

    held, force, held_gap, held_slope, exact = _active_set_step(problem, v, dual, give.abs() >= most, cold, cap)
    exact = exact.unsqueeze(-1)
    y = torch.where(exact, held.squeeze(-2), y.squeeze(-2))
    dual = torch.where(exact, force, dual)
    primal = torch.where(exact, held_gap, gap).abs().amax(-1)
    slope = torch.where(exact, held_slope.squeeze(-2), slope.squeeze(-2))

    x = y * scale
    stationarity = ((slope - _FLOOR * y) / scale).abs().amax(-1)  # of the caller's H, without the floor
    ok = (info == 0) & torch.isfinite(x).all(-1) & torch.isfinite(dual).all(-1)
    failed = torch.full_like(primal, float("inf"))
    return QPSolution(
        x=torch.where(ok.unsqueeze(-1), x, torch.zeros_like(x)).to(gradient.dtype).reshape(*batch, size),
        dual=torch.where(ok.unsqueeze(-1), dual, torch.zeros_like(dual)).to(gradient.dtype).reshape(*batch, count),
        primal_residual=torch.where(ok, primal, failed).to(gradient.dtype).reshape(batch),
        dual_residual=torch.where(ok, stationarity, failed).to(gradient.dtype).reshape(batch),
        penalty=torch.where(ok, penalty, torch.ones_like(penalty)).to(gradient.dtype).reshape(batch),
        exact=(exact.squeeze(-1) & ok).reshape(batch),
    )


__all__ = ["QPSolution", "solve_qp"]
