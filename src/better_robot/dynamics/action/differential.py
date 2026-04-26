"""``DifferentialActionModel`` — continuous-time dynamics + per-knot cost.

``DifferentialActionModelFreeFwd`` is the no-contact forward-dynamics
flavour: ``ẋ = (v, ABA(q, v, u))``, plus an optional running cost. The
control ``u`` is a ``(nv,)`` joint torque vector.

Derivatives are obtained via autograd through ``calc``. The pure-PyTorch
ABA is fully differentiable; running costs are expected to also be
differentiable. See ``docs/design/06_DYNAMICS.md §6``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from ...data_model.model import Model
from ..aba import aba
from ..state_manifold import StateMultibody
from .action import ActionData


def _quadratic_cost(
    state: StateMultibody,
    *,
    x_ref: torch.Tensor | None = None,
    u_ref: torch.Tensor | None = None,
    Wx: float = 1e-3,
    Wu: float = 1e-3,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build a simple quadratic running cost ``Wx · ||x⊖x_ref||² + Wu · ||u−u_ref||²``."""

    def cost(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if x_ref is None:
            x_err = torch.zeros(state.ndx, dtype=x.dtype, device=x.device)
        else:
            x_err = state.diff(x_ref, x)
        u_err = u if u_ref is None else (u - u_ref)
        return 0.5 * (Wx * (x_err * x_err).sum(dim=-1) + Wu * (u_err * u_err).sum(dim=-1))

    return cost


@dataclass
class DifferentialActionModel:
    """Base for continuous-time action models.

    Subclasses override :meth:`forward_dynamics` to plug in their own
    ``q̈ = f(q, v, u)``. The cost callable defaults to a small quadratic
    on state-tangent and control magnitude.
    """

    model: Model
    state: StateMultibody
    cost: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    nu: int = 0

    def __post_init__(self) -> None:
        if self.nu == 0:
            self.nu = self.state.nv
        if self.cost is None:
            self.cost = _quadratic_cost(self.state)

    def create_data(self) -> ActionData:
        return ActionData()

    def forward_dynamics(self, q: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """``q̈ = f(q, v, u)``. Override in subclasses."""
        raise NotImplementedError

    def calc(self, data: ActionData, x: torch.Tensor, u: torch.Tensor) -> None:
        q = x[..., : self.state.nq]
        v = x[..., self.state.nq :]
        ddq = self.forward_dynamics(q, v, u)
        # ẋ on the tangent: dq = v, dv = ddq.
        data.xnext = torch.cat([v, ddq], dim=-1)
        data.cost = self.cost(x, u)

    def calc_diff(self, data: ActionData, x: torch.Tensor, u: torch.Tensor) -> None:
        x_d = x.detach()
        u_d = u.detach()

        def _f(x_, u_):
            q_ = x_[..., : self.state.nq]
            v_ = x_[..., self.state.nq :]
            return torch.cat([v_, self.forward_dynamics(q_, v_, u_)], dim=-1)

        Fx, Fu = torch.autograd.functional.jacobian(_f, (x_d, u_d), vectorize=False)
        data.fx, data.fu = Fx, Fu

        def _g(x_, u_):
            return self.cost(x_, u_)

        lx, lu = torch.autograd.functional.jacobian(_g, (x_d, u_d), vectorize=False)
        data.lx, data.lu = lx, lu


class DifferentialActionModelFreeFwd(DifferentialActionModel):
    """Free-forward dynamics: ``q̈ = ABA(q, v, u)``."""

    def forward_dynamics(self, q: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return aba(self.model, self.model.create_data(), q, v, u)
