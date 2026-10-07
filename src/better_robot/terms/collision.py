"""Collision terms: clearance of robot spheres to obstacles and to each other.

Residuals are ``margin - distance``, hinged at zero by default so a clear configuration costs nothing.
With ``hinge=False`` the plain ``margin - distance`` is the inequality ``<= 0`` for
:func:`~better_robot.optim.solve_constrained`. Jacobians come from autodiff.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..collision import RobotSpheres, self_distances
from ..kinematics.state import KinematicState
from ..optim import TermFn


def sphere_clearance(
    spheres: RobotSpheres,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    margin: float,
    hinge: bool = True,
    state: str = "state",
) -> TermFn:
    """Clearance of every sphere to static obstacles.

    ``distance_fn(centers: (B..., S, 3), radii: (S,))`` returns signed distances ``(B..., S)`` or
    ``(B..., S, K)`` for ``K`` obstacles. Residual ``(B..., S * K)``; with ``window=1`` on a time axis
    it is ``(B, T, S * K)``.
    """

    def residual(kinematics: KinematicState) -> torch.Tensor:
        centers = spheres.centers(kinematics)
        gap = margin - distance_fn(centers, spheres.radii).flatten(centers.dim() - 2)
        return gap.clamp_min(0.0) if hinge else gap

    return TermFn(fn=residual, reads=(state,))


def self_clearance(
    spheres: RobotSpheres, pairs: torch.Tensor, *, margin: float, hinge: bool = True, state: str = "state"
) -> TermFn:
    """Clearance between the sphere pairs ``pairs: (P, 2)`` (see ``collision.self_collision_pairs``).

    Residual ``(B..., P)``.
    """

    def residual(kinematics: KinematicState) -> torch.Tensor:
        gap = margin - self_distances(spheres, kinematics, pairs)
        return gap.clamp_min(0.0) if hinge else gap

    return TermFn(fn=residual, reads=(state,))
