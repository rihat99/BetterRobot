"""Collision geometry primitives."""
from __future__ import annotations
from dataclasses import dataclass
import torch


class CollGeom:
    """Base class for all collision geometry."""


@dataclass
class Sphere(CollGeom):
    """Sphere collision geometry."""
    center: torch.Tensor
    """Shape (3,). Center position in world frame."""
    radius: float
    """Sphere radius in meters."""


@dataclass
class Capsule(CollGeom):
    """Capsule collision geometry (cylinder with hemispherical caps)."""
    point_a: torch.Tensor
    """Shape (3,). First endpoint."""
    point_b: torch.Tensor
    """Shape (3,). Second endpoint."""
    radius: float
    """Capsule radius in meters."""


@dataclass
class Box(CollGeom):
    """Axis-aligned box collision geometry."""
    position: torch.Tensor
    """Shape (3,). Box center in world frame."""
    extent: torch.Tensor
    """Shape (3,). Full extents [width, depth, height] in meters."""


@dataclass
class HalfSpace(CollGeom):
    """Half-space (infinite plane) collision geometry."""
    point: torch.Tensor
    """Shape (3,). Any point on the plane."""
    normal: torch.Tensor
    """Shape (3,). Outward unit normal."""

    @staticmethod
    def from_point_and_normal(point: torch.Tensor, normal: torch.Tensor) -> "HalfSpace":
        """Create a HalfSpace from a point on the plane and outward normal."""
        n = normal / normal.norm()
        return HalfSpace(point=point, normal=n)


@dataclass
class Heightmap(CollGeom):
    """Heightmap collision geometry."""
    heights: torch.Tensor
    """Shape (H, W). Height values on a regular grid."""
    origin: torch.Tensor
    """Shape (3,). World-space position of the grid origin (corner)."""
    resolution: float
    """Grid cell size in meters."""
