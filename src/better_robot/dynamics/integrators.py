"""Time integrators over the state manifold.

``integrate_q`` is implementable today (it uses only ``Model.integrate``
and therefore the per-joint ``JointModel.integrate`` routines). The dynamics
integrator entry points remain reserved and are currently unsupported.

See ``docs/concepts/dynamics.md`` ("Integrators").
"""

from __future__ import annotations

import torch

from ..data_model.data import Data
from ..data_model.model import Model


def integrate_q(
    model: Model,
    q: torch.Tensor,
    v: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Retract ``q`` by ``dt * v`` via ``model.integrate``. ``q ⊕ dt v``.

    Manifold-aware: per-joint ``JointModel.integrate`` handles SE(3) for
    free-flyer and SO(3) for spherical joints. Revolute / prismatic
    components fall through to Euclidean addition.

    Parameters
    ----------
    model : Model
    q : Tensor
        ``(B..., nq)`` configuration.
    v : Tensor
        ``(B..., nv)`` generalised velocity.
    dt : float
        Timestep.

    Returns
    -------
    Tensor
        ``(B..., nq)`` retracted configuration.
    """
    return model.integrate(q, dt * v)


def semi_implicit_euler(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    dt: float,
    *,
    fext: torch.Tensor | None = None,
):
    """Reserved semi-implicit Euler entry point; currently unsupported."""
    raise NotImplementedError("semi_implicit_euler is not implemented")


def symplectic_euler(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    dt: float,
    *,
    fext: torch.Tensor | None = None,
):
    """Reserved symplectic Euler entry point; currently unsupported."""
    raise NotImplementedError("symplectic_euler is not implemented")


def rk4(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    dt: float,
    *,
    fext: torch.Tensor | None = None,
):
    """Reserved fourth-order Runge-Kutta entry point; currently unsupported."""
    raise NotImplementedError("rk4 is not implemented")
