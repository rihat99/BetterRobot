"""``StateMultibody`` — state manifold of a rigid-body system (Crocoddyl pattern).

Wraps ``model.integrate`` / ``model.difference`` into the state-manifold
interface that DDP / iLQR consume. The state is the concatenation
``x = [q (nq), v (nv)]``; the tangent is ``dx = [dq (nv), dv (nv)]``.

See ``docs/concepts/dynamics.md §7``.
"""

from __future__ import annotations

import torch

from ..data_model.model import Model


class StateMultibody:
    """State manifold ``x = (q, v)`` of a multibody system.

    ``nx = nq + nv`` (representation dim), ``ndx = 2 * nv`` (tangent dim).
    """

    def __init__(self, model: Model) -> None:
        self.model = model
        self.nq = model.nq
        self.nv = model.nv
        self.nx: int = model.nq + model.nv
        self.ndx: int = 2 * model.nv

    def zero(self) -> torch.Tensor:
        """Return ``x = (q_neutral, 0)``."""
        q0 = self.model.q_neutral
        v0 = torch.zeros(self.nv, dtype=q0.dtype, device=q0.device)
        return torch.cat([q0, v0], dim=-1)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[..., : self.nq], x[..., self.nq :]

    def _split_tangent(self, dx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return dx[..., : self.nv], dx[..., self.nv :]

    def integrate(self, x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """``x ⊕ dx``: configuration retraction + linear velocity update."""
        q, v = self._split(x)
        dq, dv = self._split_tangent(dx)
        q_new = self.model.integrate(q, dq)
        v_new = v + dv
        return torch.cat([q_new, v_new], dim=-1)

    def diff(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """``x1 ⊖ x0``: configuration tangent + velocity delta."""
        q0, v0 = self._split(x0)
        q1, v1 = self._split(x1)
        dq = self.model.difference(q0, q1)
        dv = v1 - v0
        return torch.cat([dq, dv], dim=-1)

    def jacobian_integrate(
        self, x: torch.Tensor, dx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``∂integrate/∂x``, ``∂integrate/∂dx`` — autograd-derived."""
        def _f(x_, dx_):
            return self.integrate(x_, dx_)
        return torch.autograd.functional.jacobian(_f, (x.detach(), dx.detach()))

    def jacobian_diff(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``∂diff/∂x0``, ``∂diff/∂x1`` — autograd-derived."""
        def _f(x0_, x1_):
            return self.diff(x0_, x1_)
        return torch.autograd.functional.jacobian(_f, (x0.detach(), x1.detach()))
