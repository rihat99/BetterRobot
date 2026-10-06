"""Normal equations of a batch of independent least-squares problems."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Normal:
    """Gauss-Newton model ``loss(delta) ≈ loss + g·delta + ½ delta·H·delta`` per batch element."""

    H: torch.Tensor  #: ``(B, N, N)`` symmetric positive semi-definite.
    g: torch.Tensor  #: ``(B, N)`` gradient of the loss in tangent coordinates.
    lower: torch.Tensor | None = None  #: ``(B, N)`` most negative allowed step, or ``None``.
    upper: torch.Tensor | None = None  #: ``(B, N)`` most positive allowed step, or ``None``.

    def restrict(self, lower: torch.Tensor, upper: torch.Tensor, *, tolerance: float = 1e-9) -> Normal:
        """Box-constrain the step to ``[lower, upper]`` (projected Newton, one active-set pass).

        A coordinate that sits on a bound and whose gradient pushes it outward is frozen: its
        gradient entry becomes 0 and its row and column of ``H`` become the identity. The step
        of the other coordinates is clamped to the box by :meth:`solve`.
        """
        blocked = ((lower >= -tolerance) & (self.g > 0)) | ((upper <= tolerance) & (self.g < 0))
        keep = (~blocked).to(self.g.dtype)
        H = self.H * keep.unsqueeze(-1) * keep.unsqueeze(-2) + torch.diag_embed(1.0 - keep)
        return Normal(H=H, g=self.g * keep, lower=lower, upper=upper)

    def solve(self, damping: torch.Tensor | float = 0.0, *, floor: float = 1e-9) -> torch.Tensor:
        """Solve ``(H + damping · diag(H)) delta = -g``. Returns ``delta: (B, N)``.

        Columns are scaled to unit diagonal and the factorization runs in float64, because a
        float32 Cholesky fails on the rank-deficient ``H`` of redundant robots. A batch element
        whose factorization still fails gets ``delta = 0`` (a rejected step for LM).
        """
        H, g = self.H.double(), self.g.double()
        diagonal = H.diagonal(dim1=-2, dim2=-1)
        scale = torch.where(diagonal > 0, diagonal.clamp_min(1e-300).rsqrt(), torch.ones_like(diagonal))
        scaled = H * scale.unsqueeze(-1) * scale.unsqueeze(-2)
        shift = torch.as_tensor(damping, dtype=H.dtype, device=H.device).expand(H.shape[0]) + floor
        scaled = scaled + torch.diag_embed(shift.unsqueeze(-1).expand_as(diagonal))
        factor, info = torch.linalg.cholesky_ex(scaled)
        delta = torch.cholesky_solve(-(g * scale).unsqueeze(-1), factor).squeeze(-1) * scale
        ok = (info == 0).unsqueeze(-1) & torch.isfinite(delta).all(dim=-1, keepdim=True)
        delta = torch.where(ok, delta, torch.zeros_like(delta)).to(self.g.dtype)
        if self.lower is not None:
            delta = torch.minimum(torch.maximum(delta, self.lower), self.upper)
        return delta

    def predicted_decrease(self, delta: torch.Tensor) -> torch.Tensor:
        """Loss decrease the quadratic model predicts for ``delta: (B, N)``. Returns ``(B,)``."""
        curvature = (delta.unsqueeze(-2) @ self.H @ delta.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return -(self.g * delta).sum(-1) - 0.5 * curvature


__all__ = ["Normal"]
