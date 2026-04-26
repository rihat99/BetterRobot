"""``IntegratedActionModel`` — discrete-time wrapper around a differential model.

Implements explicit Euler and RK4 over the multibody state manifold.
Both expose the ``ActionModel`` protocol used by an OC solver: ``calc``
fills ``data.xnext`` and ``data.cost``; ``calc_diff`` adds the
state/control Jacobians.

See ``docs/concepts/dynamics.md §6``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..state_manifold import StateMultibody
from .action import ActionData, ActionModel
from .differential import DifferentialActionModel


@dataclass
class IntegratedActionModelEuler(ActionModel):
    differential: DifferentialActionModel
    dt: float = 0.01
    with_cost_residual: bool = True

    def __post_init__(self) -> None:
        self.state: StateMultibody = self.differential.state
        self.nu: int = self.differential.nu

    def create_data(self) -> ActionData:
        return ActionData()

    def calc(self, data: ActionData, x: torch.Tensor, u: torch.Tensor) -> None:
        # Continuous-time tangent: f = (v, ddq).
        diff_data = ActionData()
        self.differential.calc(diff_data, x, u)
        f = diff_data.xnext  # actually ẋ on the tangent
        data.xnext = self.state.integrate(x, self.dt * f)
        data.cost = self.dt * diff_data.cost

    def calc_diff(self, data: ActionData, x: torch.Tensor, u: torch.Tensor) -> None:
        x_d = x.detach()
        u_d = u.detach()

        def _step(x_, u_):
            d = ActionData()
            self.differential.calc(d, x_, u_)
            return self.state.integrate(x_, self.dt * d.xnext)

        Fx, Fu = torch.autograd.functional.jacobian(_step, (x_d, u_d), vectorize=False)
        data.fx, data.fu = Fx, Fu

        def _cost(x_, u_):
            d = ActionData()
            self.differential.calc(d, x_, u_)
            return self.dt * d.cost

        lx, lu = torch.autograd.functional.jacobian(_cost, (x_d, u_d), vectorize=False)
        data.lx, data.lu = lx, lu


@dataclass
class IntegratedActionModelRK4(ActionModel):
    differential: DifferentialActionModel
    dt: float = 0.01
    with_cost_residual: bool = True

    def __post_init__(self) -> None:
        self.state: StateMultibody = self.differential.state
        self.nu: int = self.differential.nu

    def create_data(self) -> ActionData:
        return ActionData()

    def _f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        d = ActionData()
        self.differential.calc(d, x, u)
        return d.xnext  # tangent ẋ

    def _step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        k1 = self._f(x, u)
        x2 = self.state.integrate(x, 0.5 * self.dt * k1)
        k2 = self._f(x2, u)
        x3 = self.state.integrate(x, 0.5 * self.dt * k2)
        k3 = self._f(x3, u)
        x4 = self.state.integrate(x, self.dt * k3)
        k4 = self._f(x4, u)
        avg = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return self.state.integrate(x, self.dt * avg)

    def calc(self, data: ActionData, x: torch.Tensor, u: torch.Tensor) -> None:
        data.xnext = self._step(x, u)
        d = ActionData()
        self.differential.calc(d, x, u)
        data.cost = self.dt * d.cost

    def calc_diff(self, data: ActionData, x: torch.Tensor, u: torch.Tensor) -> None:
        x_d = x.detach()
        u_d = u.detach()

        Fx, Fu = torch.autograd.functional.jacobian(self._step, (x_d, u_d), vectorize=False)
        data.fx, data.fu = Fx, Fu

        def _cost(x_, u_):
            d = ActionData()
            self.differential.calc(d, x_, u_)
            return self.dt * d.cost

        lx, lu = torch.autograd.functional.jacobian(_cost, (x_d, u_d), vectorize=False)
        data.lx, data.lu = lx, lu
