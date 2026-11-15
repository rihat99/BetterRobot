"""A batch of independent quadratic programs with two-sided linear rows, solved by a fixed number of Newton steps.

``solve_qp`` minimises ``½ xᵀH x + gᵀx`` subject to ``lower <= rows x <= upper`` per batch element. An equality is a
row with ``lower == upper``, a one-sided row has an infinite bound, a box on ``x`` is identity rows. The method is a
primal-dual interior point (Mehrotra's predictor-corrector) whose slack equation carries the proximal term of the
method of multipliers, so a row that cannot hold costs a bounded force instead of breaking the solve (decision record
0053). Every batch element runs the same ``iterations``, with no Python branch on a tensor value and no host read, so
the whole solve traces as one graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch

from ..exceptions import ShapeError

_FLOOR = 1e-9  # added to the unit-diagonal H before its factorization, as Normal.solve does
# In the solver's units (unit rows, unit problem scale):
_CENTRE = 1e-10  # smallest complementarity the steps aim at: steps past convergence change nothing
_SOFTEST = 1e-8  # smallest give per unit of multiplier of the row that is shortest in the metric of H
_BAND = 1e-4  # a longer row gives less, by its squared length, down to this fraction: violations compare as given
_WARM = 1e-3  # complementarity a warm start is pushed back to


@final
@dataclass(frozen=True)
class QPSolution:
    """What `solve_qp` returns, per batch element; pass it back as ``warm=`` to start the next solve from it."""

    x: torch.Tensor  #: ``(B..., N)`` the minimiser.
    dual: torch.Tensor  #: ``(B..., M)`` multiplier of each row: positive presses on ``upper``, negative on ``lower``.
    primal_residual: torch.Tensor  #: ``(B...,)`` largest violation of a row, in that row's unit.
    dual_residual: torch.Tensor  #: ``(B...,)`` largest entry of ``H x + g + rowsᵀ dual``.


def solve_qp(
    H: torch.Tensor,
    g: torch.Tensor,
    *,
    rows: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    iterations: int = 12,
    warm: QPSolution | None = None,
) -> QPSolution:
    """Minimise ``½ xᵀH x + gᵀx`` subject to ``lower <= rows x <= upper``, per batch element.

    ``H: (B..., N, N)`` symmetric positive definite (add the damping before the call), ``g: (B..., N)``,
    ``rows: (B..., M, N)`` with ``M >= 1``, ``lower, upper: (B..., M)``; a bound may be infinite, and a padding row is
    a zero row with both bounds infinite. All share the batch axes ``B...``. The factorizations run in float64
    whatever the dtype, which the result keeps; the result carries no gradient.

    The rows are hard: at the default ``iterations`` a problem that has a solution is solved with every row inside
    its bounds to about ``1e-7`` of the row's unit. When the rows cannot all hold, ``x`` is the point that violates
    them least, in the sum of squares of the violations with each row scaled to unit length in the metric of ``H``,
    and ``primal_residual`` reports the largest violation left; the multipliers of the rows in conflict then grow
    with ``iterations``. ``dual_residual`` near zero says the active rows have settled. A batch element whose ``H`` is
    not positive definite, or whose data are not finite, gets ``x = 0``, ``dual = 0`` and both residuals ``inf``.

    ``warm`` starts from a previous solution of a nearby problem with the same row layout (the last step of a
    control loop) and needs fewer ``iterations`` for the same accuracy.
    """
    batch, size, count = g.shape[:-1], g.shape[-1], lower.shape[-1]
    if H.shape != (*batch, size, size) or rows.shape != (*batch, count, size) or upper.shape != (*batch, count):
        raise ShapeError(
            f"solve_qp takes H (B..., N, N), g (B..., N), rows (B..., M, N) and lower, upper (B..., M); got "
            f"{tuple(H.shape)}, {tuple(g.shape)}, {tuple(rows.shape)}, {tuple(lower.shape)}, {tuple(upper.shape)}"
        )
    if count == 0:
        raise ShapeError("solve_qp needs at least one row; without rows the minimiser is Normal(H, g).solve()")
    H64, g64, rows64 = H.detach().double(), g.detach().double(), rows.detach().double()
    lower64, upper64 = lower.detach().double(), upper.detach().double()

    # Least-distance form: x = scale · L⁻ᵀ y with H = L Lᵀ after unit-diagonal scaling, so the objective is
    # ½|y|² + cᵀy and every row has unit length; one unit then measures the objective, the slacks and the multipliers.
    diagonal = H64.diagonal(dim1=-2, dim2=-1)
    scale = torch.where(diagonal > 0, diagonal.clamp_min(1e-300).rsqrt(), torch.ones_like(diagonal))
    eye = torch.eye(size, dtype=H64.dtype, device=H64.device)
    factor, info = torch.linalg.cholesky_ex(H64 * scale.unsqueeze(-1) * scale.unsqueeze(-2) + _FLOOR * eye)
    c = torch.linalg.solve_triangular(factor, (g64 * scale).unsqueeze(-1), upper=False).squeeze(-1)
    A = torch.linalg.solve_triangular(factor, (rows64 * scale.unsqueeze(-2)).mT, upper=False).mT
    norm = torch.linalg.vector_norm(A, dim=-1)
    present = (norm > 0) & (torch.isfinite(lower64) | torch.isfinite(upper64))
    shortest = torch.where(present, norm, torch.full_like(norm, float("inf"))).amin(-1, keepdim=True)
    norm = torch.where(norm > 0, norm, torch.ones_like(norm))
    A = A / norm.unsqueeze(-1)
    give = (shortest / norm).square().clamp(_BAND, 1.0)
    give = torch.cat([give, give], dim=-1)
    # One-sided rows [A; -A] y + s = h with slacks s >= 0 and multipliers lam >= 0; an infinite bound has no row.
    h = torch.cat([upper64 / norm, -lower64 / norm], dim=-1)
    finite = torch.isfinite(h)
    on = finite.to(h.dtype)
    h = torch.where(finite, h, torch.zeros_like(h))
    pairs = on.sum(-1, keepdim=True).clamp_min(1.0)
    reach = torch.maximum(c.abs().amax(-1), (-h).clamp_min(0.0).amax(-1)).clamp_min(1e-30).unsqueeze(-1)
    c, h = c / reach, h / reach

    def along(y: torch.Tensor) -> torch.Tensor:
        """``[A; -A] y``: ``(B..., N)`` to ``(B..., 2M)``."""
        value = (A @ y.unsqueeze(-1)).squeeze(-1)
        return torch.cat([value, -value], dim=-1)

    def back(lam: torch.Tensor) -> torch.Tensor:
        """``[A; -A]ᵀ lam``: ``(B..., 2M)`` to ``(B..., N)``."""
        return (A.mT @ (lam[..., :count] - lam[..., count:]).unsqueeze(-1)).squeeze(-1)

    if warm is None:
        y = -c
        s = (h - along(y)).clamp_min(1.0)
        lam = on
    else:
        y = (factor.mT @ (warm.x.detach().double() / scale).unsqueeze(-1)).squeeze(-1) / reach
        lam = warm.dual.detach().double() * norm / reach
        lam = torch.cat([lam.clamp_min(0.0), (-lam).clamp_min(0.0)], dim=-1)
        # Each pair starts at complementarity _WARM or above: an active row keeps its multiplier, a clear one its slack.
        s = torch.maximum((h - along(y)).clamp_min(0.0), _WARM / lam.clamp_min(_WARM**0.5))
        lam = torch.maximum(lam, _WARM / s) * on
    far = torch.full_like(s, 1e30)
    centre = torch.full_like(reach, _CENTRE)

    def longest(dlam: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        """Largest step ``(B..., 1)`` that keeps every slack and multiplier non-negative."""
        ratios = torch.minimum(torch.where(ds < 0, -s / ds, far), torch.where(dlam < 0, -lam / dlam, far))
        return ratios.amin(-1, keepdim=True)

    for _ in range(iterations):
        mu = (lam * s).sum(-1, keepdim=True) / pairs
        stationarity = y + c + back(lam)
        feasibility = (along(y) + s - h) * on
        # A row gives by `soft` per unit of multiplier (the proximal term of the method of multipliers). The give
        # follows the complementarity down, so a row that cannot hold is found while the steps are still long.
        soft = mu.clamp(_SOFTEST, 1.0) * give
        theta = on / (s + lam * soft)
        weight = theta * lam
        gram = (A.mT * (weight[..., :count] + weight[..., count:]).unsqueeze(-2)) @ A
        newton = torch.linalg.cholesky_ex(eye + gram)[0]

        def direction(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Newton step towards complementarity ``lam s = target`` at zero stationarity and feasibility."""
            base = theta * (target + lam * feasibility)
            # Two triangular solves, not cholesky_solve: that one reads its status back to the host on CPU.
            half = torch.linalg.solve_triangular(newton, (-stationarity - back(base)).unsqueeze(-1), upper=False)
            dy = torch.linalg.solve_triangular(newton.mT, half, upper=True).squeeze(-1)
            moved = along(dy)
            dlam = base + weight * moved
            return dy, dlam, (dlam * soft - feasibility - moved) * on

        dy, dlam, ds = direction(-lam * s)  # predictor: straight at the solution
        step = longest(dlam, ds).clamp_max(1.0)
        reached = ((lam + step * dlam) * (s + step * ds)).sum(-1, keepdim=True) / pairs
        target = torch.maximum((reached / mu.clamp_min(1e-300)).pow(3) * mu, centre)
        dy, dlam, ds = direction(target - lam * s - ds * dlam)  # corrector: centred, second-order
        step = (0.99 * longest(dlam, ds)).clamp_max(1.0)
        y, lam, s = y + step * dy, lam + step * dlam, s + step * ds

    x = scale * torch.linalg.solve_triangular(factor.mT, (y * reach).unsqueeze(-1), upper=True).squeeze(-1)
    dual = (lam[..., :count] - lam[..., count:]) * reach / norm
    value = (rows64 @ x.unsqueeze(-1)).squeeze(-1)
    primal = torch.maximum(value - upper64, lower64 - value).clamp_min(0.0).amax(-1)
    gradient = (H64 @ x.unsqueeze(-1)).squeeze(-1) + g64 + (rows64.mT @ dual.unsqueeze(-1)).squeeze(-1)
    ok = (info == 0) & torch.isfinite(x).all(-1) & torch.isfinite(dual).all(-1)
    failed = torch.full_like(primal, float("inf"))
    return QPSolution(
        x=torch.where(ok.unsqueeze(-1), x, torch.zeros_like(x)).to(g.dtype),
        dual=torch.where(ok.unsqueeze(-1), dual, torch.zeros_like(dual)).to(g.dtype),
        primal_residual=torch.where(ok, primal, failed).to(g.dtype),
        dual_residual=torch.where(ok, gradient.abs().amax(-1), failed).to(g.dtype),
    )


__all__ = ["QPSolution", "solve_qp"]
