"""Signed distances from spheres to spheres, capsules, half-spaces, oriented boxes and sampled SDF grids.

Each function takes a sphere ``center (..., 3)``, ``radius (...)`` and one shape, and returns ``(...)``: positive is
the gap between surfaces, negative is penetration depth. Leading batch axes broadcast; all inputs are in one common
frame (usually ``WORLD``). The norm has a zero (not NaN) gradient at a zero vector, so gradients stay finite even
when a centre lies exactly on a segment or on another centre.
"""

from __future__ import annotations

import itertools

import torch


def _norm(vector: torch.Tensor) -> torch.Tensor:
    return vector.square().sum(-1).clamp_min(1e-24).sqrt()


def sphere_sphere(
    center_a: torch.Tensor, radius_a: torch.Tensor, center_b: torch.Tensor, radius_b: torch.Tensor
) -> torch.Tensor:
    """Distance between spheres ``center_*: (..., 3)``, ``radius_*: (...)``. Returns ``(...)``."""
    return _norm(center_a - center_b) - radius_a - radius_b


def sphere_capsule(
    center: torch.Tensor, radius: torch.Tensor, start: torch.Tensor, end: torch.Tensor, capsule_radius: torch.Tensor
) -> torch.Tensor:
    """Sphere ``center: (..., 3)``, ``radius: (...)`` to the capsule around segment ``start, end: (..., 3)``.

    A zero-length segment is a sphere at ``start``. Returns ``(...)``.
    """
    axis = end - start
    t = ((center - start) * axis).sum(-1) / axis.square().sum(-1).clamp_min(1e-24)
    closest = start + t.clamp(0.0, 1.0)[..., None] * axis
    return _norm(center - closest) - radius - capsule_radius


def sphere_halfspace(
    center: torch.Tensor, radius: torch.Tensor, normal: torch.Tensor, offset: torch.Tensor | float
) -> torch.Tensor:
    """Sphere to the solid ``{x : normal . x <= offset}``; ``normal: (..., 3)`` has unit length. Returns ``(...)``."""
    return (center * normal).sum(-1) - offset - radius


def sphere_box(
    center: torch.Tensor,
    radius: torch.Tensor,
    box_center: torch.Tensor,
    box_rotation: torch.Tensor,
    half_extents: torch.Tensor,
) -> torch.Tensor:
    """Sphere to an oriented box ``box_center: (..., 3)``, ``box_rotation: (..., 3, 3)``, ``half_extents: (..., 3)``.

    Exact Euclidean distance outside the box; inside, minus the distance to the nearest face. Returns ``(...)``.
    """
    local = ((center - box_center)[..., None, :] @ box_rotation)[..., 0, :]
    excess = local.abs() - half_extents
    return _norm(excess.clamp_min(0.0)) + excess.amax(-1).clamp_max(0.0) - radius


def sphere_sdf(
    center: torch.Tensor, radius: torch.Tensor, grid: torch.Tensor, origin: torch.Tensor, voxel_size: float
) -> torch.Tensor:
    """Sphere to a sampled signed distance field, by trilinear interpolation.

    ``grid: (X, Y, Z)`` (each at least 2) holds the field at ``origin + voxel_size * (i, j, k)``,
    ``origin: (3,)``. Points outside the grid are clamped to its boundary. Returns ``(...)``.
    """
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
