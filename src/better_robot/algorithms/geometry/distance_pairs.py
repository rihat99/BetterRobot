"""Pairwise geometry distance functions (PyTorch port of PyRoki JAX implementations)."""
from __future__ import annotations
import torch
from .primitives import Sphere, Capsule, HalfSpace
from . import _utils


def sphere_sphere(s1: Sphere, s2: Sphere) -> torch.Tensor:
    """Signed distance between two spheres. Positive = separated."""
    _, dist_center = _utils.normalize_with_norm(s2.center - s1.center)
    return dist_center - (s1.radius + s2.radius)


def sphere_capsule(sphere: Sphere, capsule: Capsule) -> torch.Tensor:
    """Signed distance between sphere and capsule. Positive = separated."""
    pt_on_axis = _utils.closest_segment_point(capsule.point_a, capsule.point_b, sphere.center)
    _, dist_center = _utils.normalize_with_norm(sphere.center - pt_on_axis)
    return dist_center - (sphere.radius + capsule.radius)


def capsule_capsule(c1: Capsule, c2: Capsule) -> torch.Tensor:
    """Signed distance between two capsules. Positive = separated."""
    pt1, pt2 = _utils.closest_segment_to_segment_points(
        c1.point_a, c1.point_b, c2.point_a, c2.point_b
    )
    _, dist_center = _utils.normalize_with_norm(pt2 - pt1)
    return dist_center - (c1.radius + c2.radius)


def halfspace_sphere(hs: HalfSpace, sphere: Sphere) -> torch.Tensor:
    """Signed distance from halfspace to sphere. Positive = sphere outside halfspace."""
    dist = (sphere.center - hs.point).mul(hs.normal).sum(dim=-1) - sphere.radius
    return dist


def halfspace_capsule(hs: HalfSpace, capsule: Capsule) -> torch.Tensor:
    """Signed distance from halfspace to capsule. Positive = capsule outside halfspace."""
    dist1 = (capsule.point_a - hs.point).mul(hs.normal).sum(dim=-1) - capsule.radius
    dist2 = (capsule.point_b - hs.point).mul(hs.normal).sum(dim=-1) - capsule.radius
    return torch.minimum(dist1, dist2)
