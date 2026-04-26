"""Time integrators over the state manifold.

``integrate_q`` is implementable today (it uses only ``Model.integrate``
and therefore the per-joint ``JointModel.integrate`` routines). The
dynamics integrators wait on ``aba``.

See ``docs/concepts/dynamics.md §8``.
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
    """Semi-implicit Euler: ``v_{k+1} = v_k + dt * aba(...)``, ``q_{k+1} = q_k ⊕ dt * v_{k+1}``.

    TODO(milestone D4+). See docs/concepts/dynamics.md §8.
    """
    raise NotImplementedError("TODO(milestone D4+) — see docs/concepts/dynamics.md §8")


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
    """Symplectic Euler variant.

    TODO(milestone D4+). See docs/concepts/dynamics.md §8.
    """
    raise NotImplementedError("TODO(milestone D4+) — see docs/concepts/dynamics.md §8")


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
    """4th-order Runge-Kutta over the state manifold.

    TODO(milestone D4+). See docs/concepts/dynamics.md §8.
    """
    raise NotImplementedError("TODO(milestone D4+) — see docs/concepts/dynamics.md §8")
