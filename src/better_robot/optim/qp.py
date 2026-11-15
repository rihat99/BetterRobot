"""A batch of independent quadratic programs with two-sided linear rows, solved by a fixed number of cheap steps.

``solve_qp`` minimises ``½ xᵀH x + gᵀx`` subject to ``lower <= rows x <= upper`` per batch element. An equality is a
row with ``lower == upper``, a one-sided row has an infinite bound, a box on ``x`` is identity rows. The method is the
alternating direction method of multipliers in the form OSQP uses (one factorization of ``H + penalty · rowsᵀ rows``
per call, then ``iterations`` steps of three matrix-vector products each), followed by one active-set step that is
kept where it lands on the exact solution (decision record 0054). Every batch element runs the same steps, with
no Python branch on a tensor value and no host read, so the whole solve traces as one graph. It is made to be called
once per step of an outer loop and started from its own last answer.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import final, override

import torch

from ..exceptions import ShapeError

_FLOOR = (
    1e-9  # added to the unit-diagonal H, as Normal.solve does: a variable the objective does not see stays solvable
)
_RELAXATION = 1.6  # over-relaxation of each step (the default of OSQP)
_PENALTIES = (1e-6, 1e6)  # range of the penalty, in unit-diagonal variables
_RETUNE = 10.0  # largest factor by which one solve changes the penalty it hands to the next
_SETTLED = 1e-10  # relative residual under which a side counts as converged when the penalty is retuned
_STIFFNESS = 1e6  # weight of an active row in the active-set step, in units of the cold penalty
_UPDATES = 2  # multiplier updates of the active-set step: the active rows hold to rounding
_EXACT = 1e-6  # relative size of both residuals under which the active-set step counts as the solution


@final
@dataclass(frozen=True)
class QPSolution:
    """What `solve_qp` returns, per batch element; pass it back as ``warm=`` to start the next solve from it."""

    x: torch.Tensor  #: ``(B..., N)`` the minimiser, exact or approximate (the residuals say which).
    dual: torch.Tensor  #: ``(B..., M)`` multiplier of each row: positive presses on ``upper``, negative on ``lower``.
    _: KW_ONLY
    primal_residual: torch.Tensor  #: ``(B...,)`` largest violation of a row by ``x``, in that row's unit.
    dual_residual: torch.Tensor  #: ``(B...,)`` largest entry of ``H x + g + rowsᵀ dual``.
    penalty: torch.Tensor  #: ``(B...,)`` weight of the rows against ``H`` that the next warm solve starts with.

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
        """Largest violation ``(B,)`` of the row values ``value``, the gradient ``(B, 1, N)`` and its relative size."""
        primal = torch.maximum(value - self.upper, self.lower - value).clamp_min(0.0).amax(-1)
        curve = torch.bmm(y, self.H)
        back = torch.bmm(dual.unsqueeze(-2), self.A)
        gradient = curve + self.c + back
        parts = torch.maximum(torch.maximum(curve.abs().amax(-1), back.abs().amax(-1)), self.c.abs().amax(-1))
        return primal, gradient, (gradient.abs().amax(-1) / parts.clamp_min(1e-300)).squeeze(-1)


def _solve(factor: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Two triangular solves, not ``cholesky_solve``: that one reads its status back to the host."""
    return torch.linalg.solve_triangular(
        factor.mT, torch.linalg.solve_triangular(factor, right, upper=False), upper=True
    )


def _active_set_step(
    problem: _Scaled, v: torch.Tensor, dual: torch.Tensor, cold: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The rows that ``v: (B, M)`` puts on a bound solved as equalities: ``y: (B, 1, N)``, its row values, ``dual``, status.

    The equalities are held by a stiff penalty and `_UPDATES` multiplier updates; a multiplier of the wrong sign is
    dropped, so the pair is a solution exactly when it also satisfies the rows that were left out.
    """
    above, below = v >= problem.upper, v <= problem.lower
    active = (above | below).to(v.dtype)
    bound = torch.where(above, problem.upper, torch.where(below, problem.lower, torch.zeros_like(v)))
    stiff = _STIFFNESS * cold.unsqueeze(-1) * active
    weighted = (problem.A * stiff.unsqueeze(-1)).mT
    factor, info = torch.linalg.cholesky_ex(problem.H + torch.bmm(weighted, problem.A))
    pull, force = stiff * bound, dual * active
    y = _solve(factor, (torch.bmm((pull - force).unsqueeze(-2), problem.A) - problem.c).mT).mT
    value = problem.values(y)
    for _ in range(_UPDATES):
        force = force + stiff * (value - bound)
        y = _solve(factor, (torch.bmm((pull - force).unsqueeze(-2), problem.A) - problem.c).mT).mT
        value = problem.values(y)
    force = force + stiff * (value - bound)
    signed = torch.where(above, force.clamp_min(0.0), force.clamp_max(0.0))
    return y, value, torch.where(above & below, force, signed) * active, info


def solve_qp(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    *,
    rows: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    iterations: int = 20,
    warm: QPSolution | None = None,
) -> QPSolution:
    """Minimise ``½ xᵀH x + gᵀx`` subject to ``lower <= rows x <= upper``, per batch element.

    ``hessian: (B..., N, N)`` is ``H``, symmetric positive semi-definite (the ``H`` of a `Normal` with its damping
    added), ``gradient: (B..., N)`` is ``g``, ``rows: (B..., M, N)`` with ``M >= 1``, ``lower, upper: (B..., M)``; a
    bound may be infinite, and a padding row is a zero row with both bounds infinite. All share the batch axes ``B...``. The factorizations run in float64
    whatever the dtype, which the result keeps; the result carries no gradient.

    Each of the ``iterations`` steps moves ``x`` towards the rows and ``dual`` towards the multipliers; then the rows
    the multipliers press on are solved as equalities, and where that point satisfies every row and every sign it
    is returned: the exact solution, with both residuals at rounding. Elsewhere the last step is returned, and
    ``primal_residual`` and ``dual_residual`` say how far it still is, in the caller's units. ``warm`` starts from the
    solution of a nearby problem with the same row layout (the last step of a solver or of a control loop); it
    carries ``x``, ``dual`` and the ``penalty`` the last solve tuned, so the steps of consecutive calls add up while
    the problem changes slowly. A cold solve starts from ``x = 0``.

    When the rows cannot all hold, ``x`` tends to the point that violates them least, in the sum of squares of the
    violations in the units the rows are given in (scale a row and its bounds to weight it), and to the minimiser
    of the objective among such points; ``primal_residual`` stays positive and the ``dual`` of the rows in conflict
    grows with every step. A batch element whose ``H + penalty · rowsᵀ rows`` is not positive definite, or whose
    data are not finite, gets ``x = 0``, ``dual = 0``, both residuals ``inf`` and ``penalty = 1``.
    """
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
    matrix = hessian.detach().double().reshape(-1, size, size)
    diagonal = matrix.diagonal(dim1=-2, dim2=-1)
    scale = torch.where(diagonal > 0, diagonal.clamp_min(1e-300).rsqrt(), torch.ones_like(diagonal))
    identity = torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    problem = _Scaled(
        H=matrix * scale.unsqueeze(-1) * scale.unsqueeze(-2) + _FLOOR * identity,
        c=(gradient.detach().double().reshape(-1, size) * scale).unsqueeze(-2),
        A=rows.detach().double().reshape(-1, count, size) * scale.unsqueeze(-2),
        lower=lower.detach().double().reshape(-1, count),
        upper=upper.detach().double().reshape(-1, count),
    )
    gram = torch.bmm(problem.A.mT, problem.A)
    # The rows weigh as much as the objective: trace(cold · gram) = trace(unit-diagonal H) = N.
    cold = (size / gram.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-300)).clamp(*_PENALTIES)
    if warm is None:
        penalty = cold
        y, dual = torch.zeros_like(problem.c), torch.zeros_like(problem.lower)
    else:
        penalty = warm.penalty.detach().double().reshape(-1)
        y = (warm.x.detach().double().reshape(-1, size) / scale).unsqueeze(-2)
        dual = warm.dual.detach().double().reshape(-1, count)
    rho = penalty.unsqueeze(-1)

    # The inverse of H + penalty · rowsᵀ rows, by N right-hand sides: each step is then three products with a vector.
    factor, info = torch.linalg.cholesky_ex(problem.H + rho.unsqueeze(-1) * gram)
    inverse = _solve(factor, identity.expand_as(gram))
    free, spread = -torch.bmm(problem.c, inverse), rho.unsqueeze(-1) * inverse
    # State v = z + dual / penalty: z = clamp(v) is the rows' value inside the bounds, the rest is the multiplier.
    value = problem.values(y)
    z = torch.clamp(value, problem.lower, problem.upper)
    v = z + dual / rho
    for _ in range(iterations):
        y = torch.baddbmm(free, torch.bmm((z + z - v).unsqueeze(-2), problem.A), spread)
        value = problem.values(y)
        v = torch.add(v, value - z, alpha=_RELAXATION)
        z = torch.clamp(v, problem.lower, problem.upper)
    dual = rho * (v - z)
    primal, slope, ahead = problem.measure(y, value, dual)
    # The next solve weighs the rows more when they lag behind the multipliers, less when they lead (OSQP's rule).
    # Both sides are floored at `_SETTLED`: a pair that has converged keeps its penalty instead of following rounding.
    behind = (value - z).abs().amax(-1) / torch.maximum(value.abs().amax(-1), z.abs().amax(-1)).clamp_min(1e-300)
    change = (behind.clamp_min(_SETTLED) / ahead.clamp_min(_SETTLED)).sqrt().clamp(1.0 / _RETUNE, _RETUNE)
    penalty = (penalty * change).clamp(*_PENALTIES)

    held, held_value, force, held_info = _active_set_step(problem, v, dual, cold)
    held_primal, held_slope, held_ahead = problem.measure(held, held_value, force)
    reach = torch.maximum(
        held_value.abs().amax(-1), torch.clamp(held_value, problem.lower, problem.upper).abs().amax(-1)
    )
    exact = ((held_info == 0) & (held_primal <= _EXACT * reach) & (held_ahead <= _EXACT)).unsqueeze(-1)
    y = torch.where(exact, held.squeeze(-2), y.squeeze(-2))
    dual = torch.where(exact, force, dual)
    primal = torch.where(exact.squeeze(-1), held_primal, primal)
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
    )


__all__ = ["QPSolution", "solve_qp"]
