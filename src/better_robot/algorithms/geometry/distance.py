"""Collision distance dispatcher and SDF smoothing."""
from __future__ import annotations
from typing import Callable
import torch
from .primitives import CollGeom, Sphere, Capsule, HalfSpace
from .distance_pairs import (
    sphere_sphere, sphere_capsule, capsule_capsule,
    halfspace_sphere, halfspace_capsule,
)

DISTANCE_FUNCTIONS: dict[tuple[type, type], Callable] = {
    (Sphere, Sphere): sphere_sphere,
    (Sphere, Capsule): sphere_capsule,
    (Capsule, Capsule): capsule_capsule,
    (HalfSpace, Sphere): halfspace_sphere,
    (HalfSpace, Capsule): halfspace_capsule,
}


def compute_distance(geom_a: CollGeom, geom_b: CollGeom) -> torch.Tensor:
    """Dispatch to the appropriate distance function for (geom_a, geom_b).

    Positive = separated, negative = penetrating.
    Tries both (type_a, type_b) and (type_b, type_a) orderings.
    """
    key = (type(geom_a), type(geom_b))
    fn = DISTANCE_FUNCTIONS.get(key)
    if fn is not None:
        return fn(geom_a, geom_b)

    key_swapped = (type(geom_b), type(geom_a))
    fn_swapped = DISTANCE_FUNCTIONS.get(key_swapped)
    if fn_swapped is not None:
        return fn_swapped(geom_b, geom_a)

    raise NotImplementedError(
        f"No distance function for ({type(geom_a).__name__}, {type(geom_b).__name__})"
    )


def colldist_from_sdf(dist: torch.Tensor, activation_dist: float | torch.Tensor) -> torch.Tensor:
    """Convert signed distance to optimization-friendly collision cost.

    Based on https://arxiv.org/pdf/2310.17274#page=7.39 (PyRoki pattern).
    Returns values <= 0: 0 when dist >= activation_dist, increasingly negative for penetration.

    Args:
        dist: Signed distances (positive = separated, negative = penetrating).
        activation_dist: Margin below which cost activates.

    Returns:
        Transformed distances (<= 0). Use -result as a cost (positive = bad).
    """
    dist = torch.minimum(dist, torch.full_like(dist, float(activation_dist)))
    dist = torch.where(
        dist < 0,
        dist - 0.5 * activation_dist,
        -0.5 / (float(activation_dist) + 1e-6) * (dist - activation_dist) ** 2,
    )
    dist = torch.minimum(dist, torch.zeros_like(dist))
    return dist
