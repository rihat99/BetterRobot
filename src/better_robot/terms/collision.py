"""Collision terms: clearance of robot spheres to obstacles and to each other.

Residuals are ``margin - distance``, hinged at zero by default so a clear configuration costs nothing. The hinge is
C1: ``gap² / (2 margin)`` while the gap ``margin - distance`` is below ``margin``, then ``gap - margin / 2``, so the
Gauss-Newton model does not jump when a contact starts. With ``hinge=False`` the plain ``margin - distance`` is the
inequality ``<= 0`` for ``better_robot.optim.solve_constrained``.

Jacobians are analytic: the gradient of each distance with respect to its sphere centre times
``kinematics.point_jacobian``. For ``sphere_clearance`` that gradient is the reverse-mode derivative of
``distance_fn`` alone (``torch.func.vjp``), so every obstacle shape gets it without code of its own: the SDF, and a
caller's function that meets the two requirements `sphere_clearance` states. On a time axis register the terms with
``window=1``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..collision import RobotSpheres, self_distances
from ..exceptions import ArgumentError
from ..kinematics import KinematicState, point_jacobian
from ..optim import TermFn
from .robot import _band, _hinge, _hinge_slope


def sphere_clearance(
    spheres: RobotSpheres,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    margin: float,
    hinge: bool = True,
    state: str = "state",
    q: str = "q",
) -> TermFn:
    """Clearance of every sphere to static obstacles.

    ``distance_fn(centers: (B..., S, 3), radii: (S,))`` returns signed distances ``(B..., S)`` or
    ``(B..., S, K)`` for ``K`` obstacles. Residual ``(B..., S * K)``; with ``window=1`` on a time axis it is
    ``(B, T, S * K)``. With ``hinge=True`` each residual is zero while ``distance >= margin``, ``margin / 2`` at
    contact (``distance = 0``) and ``margin / 2 - distance`` beyond it; with ``hinge=False`` it is
    ``margin - distance``.

    The Jacobian, with respect to the variable ``q`` that the cached ``state`` reads, pulls back through
    ``distance_fn`` alone, so the function must meet two requirements (C-EXT-8): entry ``s`` depends on centre ``s``
    only, and ``torch.func`` can transform it (a ``torch.autograd.Function`` defines ``setup_context``). For any other
    function register ``optim.TermFn(fn=term.fn, reads=term.reads)``: the same residual, differentiated by autodiff.
    """
    frames = spheres.frame_ids.tolist()  # bench-ok: once per term; point_jacobian takes frame ids as ints
    band = _band(torch.tensor(max(margin, 0.0), dtype=spheres.radii.dtype, device=spheres.radii.device))

    def residual(kinematics: KinematicState) -> torch.Tensor:
        centers = spheres.centers(kinematics)
        gap = margin - distance_fn(centers, spheres.radii).flatten(centers.dim() - 2)
        return _hinge(gap, band) if hinge else gap

    def jacobian(kinematics: KinematicState) -> dict[str, torch.Tensor]:
        centers = spheres.centers(kinematics)
        distance, pull, *_ = torch.func.vjp(
            lambda at: distance_fn(at, spheres.radii).reshape(*at.shape[:-1], -1), centers
        )  # (B..., S, K)
        count = distance.shape[-1]
        if count == 1:
            gradient = pull(torch.ones_like(distance))[0].unsqueeze(-2)
        else:  # one pullback per obstacle k, batched: seed k picks column k
            seeds = torch.eye(count, dtype=distance.dtype, device=distance.device)
            seeds = seeds.reshape(count, *(1,) * (distance.dim() - 1), count).expand(count, *distance.shape)
            gradient = torch.func.vmap(pull)(seeds)[0].movedim(0, -2)  # (B..., S, K, 3)
        points = point_jacobian(kinematics.model, kinematics, frames=frames, points=spheres.offsets)
        rows = gradient @ points  # (B..., S, K, nv)
        if hinge:
            rows = _hinge_slope(margin - distance, band)[..., None] * rows
        return {q: -rows.flatten(-3, -2)}

    return TermFn(fn=residual, reads=(state,), jacobian=jacobian)


def self_clearance(
    spheres: RobotSpheres,
    *,
    pairs: torch.Tensor,
    margin: float,
    hinge: bool = True,
    closest: int | None = None,
    state: str = "state",
    q: str = "q",
) -> TermFn:
    """Clearance between the sphere pairs ``pairs: (P, 2)`` (see ``collision.self_collision_pairs``).

    Residual ``(B..., P)``, or ``(B..., closest)``: the ``closest`` pairs of each configuration, nearest first, when
    ``closest`` is below ``P``; hinged as in `sphere_clearance`. Keeping a fixed number of rows bounds the cost on robots with thousands of pairs; a
    pair outside them costs nothing, so choose ``closest`` above the number of pairs that can be inside the margin
    at once. The Jacobian is with respect to the variable ``q`` that the cached ``state`` reads.
    """
    if closest is not None and closest < 1:
        raise ArgumentError(f"closest must be at least 1, got {closest}")
    frames = spheres.frame_ids.tolist()  # bench-ok: once per term; point_jacobian takes frame ids as ints
    band = _band(torch.tensor(max(margin, 0.0), dtype=spheres.radii.dtype, device=spheres.radii.device))
    keep = pairs.shape[0] if closest is None else min(closest, pairs.shape[0])

    def nearest(kinematics: KinematicState) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Distances ``(B..., keep)`` and, when only some pairs are kept, their indices into ``pairs``."""
        distance = self_distances(spheres, kinematics, pairs=pairs)
        if keep == pairs.shape[0]:
            return distance, None
        distance, index = distance.topk(keep, dim=-1, largest=False)
        return distance, index

    def residual(kinematics: KinematicState) -> torch.Tensor:
        gap = margin - nearest(kinematics)[0]
        return _hinge(gap, band) if hinge else gap

    def jacobian(kinematics: KinematicState) -> dict[str, torch.Tensor]:
        distance, index = nearest(kinematics)
        first, second = pairs[:, 0], pairs[:, 1]
        if index is not None:
            first, second = first[index], second[index]  # (B..., keep)
        first, second = first.expand(distance.shape), second.expand(distance.shape)
        centers = spheres.centers(kinematics)  # (B..., S, 3)
        points = point_jacobian(kinematics.model, kinematics, frames=frames, points=spheres.offsets)
        batch = centers.shape[:-2]

        def gather(table: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
            """Rows ``rows (B..., keep)`` of ``table (B..., S, *event)``."""
            index = rows.reshape(*rows.shape, *(1,) * (table.dim() - len(batch) - 1))
            return torch.take_along_dim(table, index, dim=len(batch))

        delta = gather(centers, first) - gather(centers, second)  # (B..., keep, 3)
        # The gradient of collision's clamped norm: zero, not NaN, for coincident centres.
        length = delta.square().sum(-1, keepdim=True)
        normal = torch.where(length > 1e-24, delta / length.clamp_min(1e-24).sqrt(), 0.0)
        rows = (normal.unsqueeze(-2) @ (gather(points, first) - gather(points, second))).squeeze(-2)
        if hinge:
            rows = _hinge_slope(margin - distance, band)[..., None] * rows
        return {q: -rows}

    return TermFn(fn=residual, reads=(state,), jacobian=jacobian)
