"""Time integrators over the state manifold.

``integrate_q`` uses ``Model.integrate`` and therefore the per-joint
``JointModel.integrate`` routines. Full physics integration belongs to a
simulation layer and is not advertised by this package.

See ``docs/concepts/dynamics.md`` ("Integrators").
"""

from __future__ import annotations

import torch

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
