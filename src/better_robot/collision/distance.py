"""Signed distances from spheres to spheres, capsules, half-spaces, oriented boxes and sampled SDF grids, and between
two capsules.

Each function takes a sphere ``center (..., 3)``, ``radius (...)`` and one shape (``capsule_capsule`` two capsules),
and returns ``(...)``: positive is the gap between surfaces, negative is penetration depth. Leading batch axes
broadcast; all inputs are in one common frame (usually ``WORLD``). The norm has a zero (not NaN) gradient at a zero
vector, so gradients stay finite even when a centre lies exactly on a segment or on another centre.
"""

from __future__ import annotations

import itertools

import torch

from ..exceptions import ShapeError
from ..lie._checks import check_tensor, check_tensors


def _norm(vector: torch.Tensor) -> torch.Tensor:
    return vector.square().sum(-1).clamp_min(1e-24).sqrt()


def sphere_sphere(
    center_a: torch.Tensor, radius_a: torch.Tensor, *, center_b: torch.Tensor, radius_b: torch.Tensor
) -> torch.Tensor:
    """Distance between spheres ``center_*: (..., 3)``, ``radius_*: (...)``. Returns ``(...)``."""
    check_tensors(
        center_a=(center_a, (3,)), radius_a=(radius_a, ()), center_b=(center_b, (3,)), radius_b=(radius_b, ())
    )
    return sphere_sphere_raw(center_a, radius_a, center_b=center_b, radius_b=radius_b)


def sphere_sphere_raw(
    center_a: torch.Tensor, radius_a: torch.Tensor, *, center_b: torch.Tensor, radius_b: torch.Tensor
) -> torch.Tensor:
    """`sphere_sphere` without input checks, for checked inputs: ``(..., 3), (...), (..., 3), (...) → (...)``."""
    return _norm(center_a - center_b) - radius_a - radius_b


def sphere_capsule(
    center: torch.Tensor, radius: torch.Tensor, *, start: torch.Tensor, end: torch.Tensor, capsule_radius: torch.Tensor
) -> torch.Tensor:
    """Sphere ``center: (..., 3)``, ``radius: (...)`` to the capsule around segment ``start, end: (..., 3)`` with
    ``capsule_radius: (...)``.

    A zero-length segment is a sphere at ``start``. Returns ``(...)``.
    """
    check_tensors(
        center=(center, (3,)),
        radius=(radius, ()),
        start=(start, (3,)),
        end=(end, (3,)),
        capsule_radius=(capsule_radius, ()),
    )
    axis = end - start
    t = ((center - start) * axis).sum(-1) / axis.square().sum(-1).clamp_min(1e-24)
    closest = start + t.clamp(0.0, 1.0)[..., None] * axis
    return _norm(center - closest) - radius - capsule_radius


def _closest_parameters(
    axis_a: torch.Tensor, axis_b: torch.Tensor, offset: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parameters ``s, t (...)`` in ``[0, 1]`` of the closest points ``start_a + s axis_a``, ``start_b + t axis_b``.

    ``offset = start_a - start_b``. Ericson (*Real-Time Collision Detection* 5.1.9) without branches: ``s`` from the
    infinite lines, ``t`` closest to that point, ``s`` closest to the clamped ``t``. A zero denominator (a point, or
    parallel segments) gives a clamped arbitrary value that the next step makes exact.
    """
    tiny = torch.finfo(offset.dtype).tiny
    length_a, length_b = axis_a.square().sum(-1), axis_b.square().sum(-1)
    cross, along_a, along_b = (axis_a * axis_b).sum(-1), (axis_a * offset).sum(-1), (axis_b * offset).sum(-1)
    denominator = (length_a * length_b - cross.square()).clamp_min(tiny)
    s = ((cross * along_b - along_a * length_b) / denominator).clamp(0.0, 1.0)
    t = ((cross * s + along_b) / length_b.clamp_min(tiny)).clamp(0.0, 1.0)
    s = ((cross * t - along_a) / length_a.clamp_min(tiny)).clamp(0.0, 1.0)
    return s, t


def capsule_capsule(
    start_a: torch.Tensor,
    end_a: torch.Tensor,
    *,
    radius_a: torch.Tensor,
    start_b: torch.Tensor,
    end_b: torch.Tensor,
    radius_b: torch.Tensor,
) -> torch.Tensor:
    """Distance between the capsules around segments ``start_*, end_*: (..., 3)`` with radii ``radius_*: (...)``.

    A zero-length segment is a sphere; parallel segments use one of their closest pairs. Returns ``(...)``.
    """
    check_tensors(
        start_a=(start_a, (3,)),
        end_a=(end_a, (3,)),
        radius_a=(radius_a, ()),
        start_b=(start_b, (3,)),
        end_b=(end_b, (3,)),
        radius_b=(radius_b, ()),
    )
    return capsule_capsule_raw(start_a, end_a, radius_a=radius_a, start_b=start_b, end_b=end_b, radius_b=radius_b)


def capsule_capsule_raw(
    start_a: torch.Tensor,
    end_a: torch.Tensor,
    *,
    radius_a: torch.Tensor,
    start_b: torch.Tensor,
    end_b: torch.Tensor,
    radius_b: torch.Tensor,
) -> torch.Tensor:
    """`capsule_capsule` without input checks, for checked inputs: segments ``(..., 3)``, radii ``(...) → (...)``."""
    axis_a, axis_b, offset = end_a - start_a, end_b - start_b, start_a - start_b
    # The distance is stationary in s and t at the closest points, so detached parameters give its exact gradient
    # and keep the near-parallel division out of every derivative.
    s, t = _closest_parameters(axis_a.detach(), axis_b.detach(), offset.detach())
    return _norm(offset + s[..., None] * axis_a - t[..., None] * axis_b) - radius_a - radius_b


def sphere_halfspace(
    center: torch.Tensor, radius: torch.Tensor, *, normal: torch.Tensor, offset: torch.Tensor | float
) -> torch.Tensor:
    """Sphere ``center: (..., 3)``, ``radius: (...)`` to the solid ``{x : normal . x <= offset}``.

    ``normal: (..., 3)`` has unit length, ``offset`` a number or ``(...)``. Returns ``(...)``.
    """
    check_tensors(center=(center, (3,)), radius=(radius, ()), normal=(normal, (3,)))
    if isinstance(offset, torch.Tensor):
        check_tensor("offset", offset, shape=(), dtype=center.dtype, device=center.device, owner="center")
    return (center * normal).sum(-1) - offset - radius


def sphere_box(
    center: torch.Tensor,
    radius: torch.Tensor,
    *,
    box_center: torch.Tensor,
    box_rotation: torch.Tensor,
    half_extents: torch.Tensor,
) -> torch.Tensor:
    """Sphere ``center: (..., 3)``, ``radius: (...)`` to an oriented box ``box_center: (..., 3)``,
    ``box_rotation: (..., 3, 3)``, ``half_extents: (..., 3)``.

    Exact Euclidean distance outside the box; inside, minus the distance to the nearest face. Returns ``(...)``.
    """
    check_tensors(
        center=(center, (3,)),
        radius=(radius, ()),
        box_center=(box_center, (3,)),
        box_rotation=(box_rotation, (3, 3)),
        half_extents=(half_extents, (3,)),
    )
    local = ((center - box_center)[..., None, :] @ box_rotation)[..., 0, :]
    excess = local.abs() - half_extents
    return _norm(excess.clamp_min(0.0)) + excess.amax(-1).clamp_max(0.0) - radius


def sphere_sdf(
    center: torch.Tensor, radius: torch.Tensor, *, grid: torch.Tensor, origin: torch.Tensor, voxel_size: float
) -> torch.Tensor:
    """Sphere ``center: (..., 3)``, ``radius: (...)`` to a sampled signed distance field, by trilinear interpolation.

    ``grid: (X, Y, Z)`` (each at least 2) holds the field at ``origin + voxel_size * (i, j, k)``,
    ``origin: (3,)``. Points outside the grid are clamped to its boundary. Returns ``(...)``.
    """
    check_tensors(center=(center, (3,)), radius=(radius, ()), grid=(grid, ()), origin=(origin, (3,)))
    if grid.ndim != 3 or min(grid.shape) < 2 or origin.ndim != 1:
        raise ShapeError(
            f"grid must have shape (X, Y, Z) with each size at least 2 and origin (3,), got {tuple(grid.shape)} and "
            f"{tuple(origin.shape)}"
        )
    size = torch.tensor(grid.shape, dtype=center.dtype, device=center.device)
    index = torch.minimum(((center - origin) / voxel_size).clamp_min(0.0), size - 1)
    lower = torch.minimum(index.detach().floor(), size - 2)
    weight = index - lower
    lower = lower.long()
    value = torch.zeros_like(radius * center[..., 0])
    for corner in itertools.product((0, 1), repeat=3):
        sample = grid[lower[..., 0] + corner[0], lower[..., 1] + corner[1], lower[..., 2] + corner[2]]
        factor = torch.ones_like(sample)
        for axis, bit in enumerate(corner):
            factor = factor * (weight[..., axis] if bit else 1.0 - weight[..., axis])
        value = value + factor * sample
    return value - radius
