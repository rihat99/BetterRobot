"""A batch of independent quadratic programs with two-sided linear rows, solved by a fixed number of cheap steps.

``solve_qp`` minimises ``½ xᵀH x + gᵀx`` subject to ``lower <= rows x <= upper`` per batch element. An equality is a
row with ``lower == upper``, a one-sided row has an infinite bound, a box on ``x`` is identity rows. The method is the
alternating direction method of multipliers in the form OSQP uses (one factorization of ``H + penalty · rowsᵀ rows``
per call, then ``iterations`` steps of two matrix-vector products each), followed by one active-set step that is
kept where it lands on the exact solution (decision record 0053). Every batch element runs the same steps, with
no Python branch on a tensor value and no host read, so the whole solve traces as one graph. It is made to be called
once per step of an outer loop and started from its own last answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

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
    primal_residual: torch.Tensor  #: ``(B...,)`` largest violation of a row by ``x``, in that row's unit.
    dual_residual: torch.Tensor  #: ``(B...,)`` largest entry of ``H x + g + rowsᵀ dual``.
    penalty: torch.Tensor  #: ``(B...,)`` weight of the rows against ``H`` that the next warm solve starts with.


@final
@dataclass(frozen=True)
class _Scaled:
    """One batch of problems in unit-diagonal variables ``x = scale · y``; the rows keep the caller's units."""

    H: torch.Tensor  # (B, N, N) unit diagonal plus the floor
    c: torch.Tensor  # (B, N, 1)
    A: torch.Tensor  # (B, M, N)
    At: torch.Tensor  # (B, N, M)
    lower: torch.Tensor  # (B, M)
    upper: torch.Tensor  # (B, M)

    def measure(self, y: torch.Tensor, dual: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Row values ``(B, M)``, their largest violation ``(B,)``, the gradient ``(B, N)`` and its relative size."""
        value = torch.bmm(self.A, y).squeeze(-1)
        primal = torch.maximum(value - self.upper, self.lower - value).clamp_min(0.0).amax(-1)
        curve = torch.bmm(self.H, y).squeeze(-1)
        back = torch.bmm(self.At, dual.unsqueeze(-1)).squeeze(-1)
        gradient = curve + self.c.squeeze(-1) + back
        parts = torch.maximum(curve.abs().amax(-1), back.abs().amax(-1))
        parts = torch.maximum(parts, self.c.abs().amax(-2).squeeze(-1))
        return value, primal, gradient, gradient.abs().amax(-1) / parts.clamp_min(1e-300)


def _solve(factor: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Two triangular solves, not ``cholesky_solve``: that one reads its status back to the host."""
    return torch.linalg.solve_triangular(
        factor.mT, torch.linalg.solve_triangular(factor, right, upper=False), upper=True
    )


def _active_set_step(
    problem: _Scaled, v: torch.Tensor, dual: torch.Tensor, cold: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The rows that ``v: (B, M)`` puts on a bound solved as equalities: ``y: (B, N, 1)``, ``dual: (B, M)``, status.

    The equalities are held by a stiff penalty and `_UPDATES` multiplier updates; a multiplier of the wrong sign is
    dropped, so the pair is a solution exactly when it also satisfies the rows that were left out.
    """
    above, below = v >= problem.upper, v <= problem.lower
    active = (above | below).to(v.dtype)
    bound = torch.where(above, problem.upper, torch.where(below, problem.lower, torch.zeros_like(v)))
    stiff = _STIFFNESS * cold.unsqueeze(-1) * active
    factor, info = torch.linalg.cholesky_ex(problem.H + torch.bmm(problem.At * stiff.unsqueeze(-2), problem.A))
    force = dual * active
    for _ in range(_UPDATES + 1):
        y = _solve(factor, torch.bmm(problem.At, (stiff * bound - force).unsqueeze(-1)) - problem.c)
        force = force + stiff * (torch.bmm(problem.A, y).squeeze(-1) - bound)
    signed = torch.where(above, force.clamp_min(0.0), force.clamp_max(0.0))
    return y, torch.where(above & below, force, signed) * active, info


def solve_qp(
    H: torch.Tensor,
    g: torch.Tensor,
    *,
    rows: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    iterations: int = 20,
    warm: QPSolution | None = None,
) -> QPSolution:
    """Minimise ``½ xᵀH x + gᵀx`` subject to ``lower <= rows x <= upper``, per batch element.

    ``H: (B..., N, N)`` symmetric positive semi-definite (a Gauss-Newton ``H`` with its damping), ``g: (B..., N)``,
    ``rows: (B..., M, N)`` with ``M >= 1``, ``lower, upper: (B..., M)``; a bound may be infinite, and a padding row is
    a zero row with both bounds infinite. All share the batch axes ``B...``. The factorizations run in float64
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
    batch, size, count = g.shape[:-1], g.shape[-1], lower.shape[-1]
    if H.shape != (*batch, size, size) or rows.shape != (*batch, count, size) or upper.shape != (*batch, count):
        raise ShapeError(
            f"solve_qp takes H (B..., N, N), g (B..., N), rows (B..., M, N) and lower, upper (B..., M); got "
            f"{tuple(H.shape)}, {tuple(g.shape)}, {tuple(rows.shape)}, {tuple(lower.shape)}, {tuple(upper.shape)}"
        )
    if count == 0:
        raise ShapeError("solve_qp needs at least one row; without rows the minimiser is Normal(H, g).solve()")
    H64 = H.detach().double().reshape(-1, size, size)
    diagonal = H64.diagonal(dim1=-2, dim2=-1)
    scale = torch.where(diagonal > 0, diagonal.clamp_min(1e-300).rsqrt(), torch.ones_like(diagonal))
    A = rows.detach().double().reshape(-1, count, size) * scale.unsqueeze(-2)
    problem = _Scaled(
        H=H64 * scale.unsqueeze(-1) * scale.unsqueeze(-2)
        + _FLOOR * torch.eye(size, dtype=H64.dtype, device=H64.device),
        c=(g.detach().double().reshape(-1, size) * scale).unsqueeze(-1),
        A=A,
        At=A.mT.contiguous(),
        lower=lower.detach().double().reshape(-1, count),
        upper=upper.detach().double().reshape(-1, count),
    )
    gram = torch.bmm(problem.At, A)
    # The rows weigh as much as the objective: trace(cold · gram) = trace(unit-diagonal H) = N.
    cold = (size / gram.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-300)).clamp(*_PENALTIES)
    if warm is None:
        penalty = cold
        y, dual = torch.zeros_like(problem.c), torch.zeros_like(problem.lower)
    else:
        penalty = warm.penalty.detach().double().reshape(-1)
        y = (warm.x.detach().double().reshape(-1, size) / scale).unsqueeze(-1)
        dual = warm.dual.detach().double().reshape(-1, count)
    rho = penalty.unsqueeze(-1)

    factor, info = torch.linalg.cholesky_ex(problem.H + rho.unsqueeze(-1) * gram)
    right = _solve(factor, torch.cat([rho.unsqueeze(-1) * problem.At, -problem.c], dim=-1))
    lift, free = right[..., :count].contiguous(), right[..., count:].contiguous()
    # State v = z + dual / penalty: z = clamp(v) is the rows' value inside the bounds, the rest is the multiplier.
    z = torch.clamp(torch.bmm(A, y).squeeze(-1), problem.lower, problem.upper)
    v = z + dual / rho
    for _ in range(iterations):
        y = torch.baddbmm(free, lift, (z + z - v).unsqueeze(-1))
        v = torch.add(v, torch.bmm(A, y).squeeze(-1) - z, alpha=_RELAXATION)
        z = torch.clamp(v, problem.lower, problem.upper)
    dual = rho * (v - z)
    value, primal, gradient, ahead = problem.measure(y, dual)
    # The next solve weighs the rows more when they lag behind the multipliers, less when they lead (OSQP's rule).
    # Both sides are floored at `_SETTLED`: a pair that has converged keeps its penalty instead of following rounding.
    behind = (value - z).abs().amax(-1) / torch.maximum(value.abs().amax(-1), z.abs().amax(-1)).clamp_min(1e-300)
    change = (behind.clamp_min(_SETTLED) / ahead.clamp_min(_SETTLED)).sqrt().clamp(1.0 / _RETUNE, _RETUNE)
    penalty = (penalty * change).clamp(*_PENALTIES)

    held, force, held_info = _active_set_step(problem, v, dual, cold)
    held_value, held_primal, held_gradient, held_ahead = problem.measure(held, force)
    reach = torch.maximum(
        held_value.abs().amax(-1), torch.clamp(held_value, problem.lower, problem.upper).abs().amax(-1)
    )
    exact = ((held_info == 0) & (held_primal <= _EXACT * reach) & (held_ahead <= _EXACT)).unsqueeze(-1)
    y, dual = torch.where(exact, held.squeeze(-1), y.squeeze(-1)), torch.where(exact, force, dual)
    primal = torch.where(exact.squeeze(-1), held_primal, primal)
    gradient = torch.where(exact, held_gradient, gradient)

    x = y * scale
    stationarity = ((gradient - _FLOOR * y) / scale).abs().amax(-1)  # of the caller's H, without the floor
    ok = (info == 0) & torch.isfinite(x).all(-1) & torch.isfinite(dual).all(-1)
    failed = torch.full_like(primal, float("inf"))
    return QPSolution(
        x=torch.where(ok.unsqueeze(-1), x, torch.zeros_like(x)).to(g.dtype).reshape(*batch, size),
        dual=torch.where(ok.unsqueeze(-1), dual, torch.zeros_like(dual)).to(g.dtype).reshape(*batch, count),
        primal_residual=torch.where(ok, primal, failed).to(g.dtype).reshape(batch),
        dual_residual=torch.where(ok, stationarity, failed).to(g.dtype).reshape(batch),
        penalty=torch.where(ok, penalty, torch.ones_like(penalty)).to(g.dtype).reshape(batch),
    )


__all__ = ["QPSolution", "solve_qp"]
