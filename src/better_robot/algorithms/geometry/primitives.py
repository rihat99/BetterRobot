"""Collision geometry primitives."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from ...math.so3 import so3_rotation_matrix

__all__ = [
    "CollGeom",
    "Sphere",
    "Capsule",
    "Box",
    "HalfSpace",
    "Heightmap",
]


class CollGeom:
    """Base class for all collision geometry."""

    def transform(self, pose: torch.Tensor) -> "CollGeom":
        """Apply SE3 transform [tx, ty, tz, qx, qy, qz, qw] to geometry.

        Returns a new CollGeom with the geometry in the transformed frame.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement transform()")


@dataclass
class Sphere(CollGeom):
    """Sphere collision geometry."""
    center: torch.Tensor
    """Shape (3,). Center position in world frame."""
    radius: float
    """Sphere radius in meters."""

    def transform(self, pose: torch.Tensor) -> "Sphere":
        """Apply SE3 pose to sphere center. Radius is invariant to transforms."""
        # Apply rotation+translation to center point
        R = so3_rotation_matrix(pose[3:7])  # (3,3)
        t = pose[:3]
        new_center = R @ self.center + t
        return Sphere(center=new_center, radius=self.radius)

    @property
    def position(self) -> torch.Tensor:
        """Alias for center."""
        return self.center


@dataclass
class Capsule(CollGeom):
    """Capsule collision geometry (cylinder with hemispherical caps)."""
    point_a: torch.Tensor
    """Shape (3,). First endpoint."""
    point_b: torch.Tensor
    """Shape (3,). Second endpoint."""
    radius: float
    """Capsule radius in meters."""

    def transform(self, pose: torch.Tensor) -> "Capsule":
        """Apply SE3 pose to both endpoints. Radius is invariant."""
        R = so3_rotation_matrix(pose[3:7])  # (3,3)
        t = pose[:3]
        new_a = R @ self.point_a + t
        new_b = R @ self.point_b + t
        return Capsule(point_a=new_a, point_b=new_b, radius=self.radius)

    @property
    def center(self) -> torch.Tensor:
        """Midpoint of the capsule segment."""
        return (self.point_a + self.point_b) / 2.0

    @staticmethod
    def from_endpoints(p0: torch.Tensor, p1: torch.Tensor, radius: float) -> "Capsule":
        """Create capsule from two endpoint positions."""
        return Capsule(point_a=p0, point_b=p1, radius=radius)

    @staticmethod
    def from_trimesh(mesh: "trimesh.Trimesh") -> "Capsule":
        """Create capsule as the minimum bounding cylinder of a trimesh.

        If the mesh is empty, returns a degenerate zero-radius capsule at the origin.
        The capsule endpoints are derived from the cylinder axis (Z-column of the
        bounding-cylinder transform) and half-height.

        Args:
            mesh: Input trimesh. Typically the merged collision geometry of one robot link.

        Returns:
            Capsule in the mesh's local frame.
        """
        try:
            import trimesh as tm
        except ImportError as exc:
            raise ImportError(
                "trimesh is required for Capsule.from_trimesh(). "
                "Install it with: pip install trimesh"
            ) from exc

        if mesh.is_empty:
            z = torch.zeros(3, dtype=torch.float32)
            return Capsule(point_a=z, point_b=z.clone(), radius=0.0)

        result = tm.bounds.minimum_cylinder(mesh)
        radius = float(result["radius"])
        height = float(result["height"])
        tf = result["transform"]  # 4x4 ndarray; cylinder center at tf[:3,3], axis = tf[:3,2]
        center = torch.tensor(tf[:3, 3], dtype=torch.float32)
        axis = torch.tensor(tf[:3, 2], dtype=torch.float32)  # Z-column
        # Inset endpoints by radius so the hemispherical caps don't extend
        # beyond the bounding cylinder.  When height < 2*radius the capsule
        # degenerates to a sphere at the center.
        half_seg = max(0.0, height / 2.0 - radius) * axis
        return Capsule(point_a=center - half_seg, point_b=center + half_seg, radius=radius)

    def decompose_to_spheres(self, n_segments: int = 5) -> list[Sphere]:
        """Decompose capsule into n_segments spheres along the axis.

        Spheres have the same radius as the capsule, placed at evenly
        spaced positions from point_a to point_b.

        Args:
            n_segments: Number of spheres to generate (>= 1).

        Returns:
            List of Sphere objects.
        """
        if n_segments < 1:
            raise ValueError("n_segments must be >= 1")
        spheres = []
        for i in range(n_segments):
            t = i / max(n_segments - 1, 1)
            center = self.point_a + t * (self.point_b - self.point_a)
            spheres.append(Sphere(center=center, radius=self.radius))
        return spheres


@dataclass
class Box(CollGeom):
    """Axis-aligned box collision geometry."""
    position: torch.Tensor
    """Shape (3,). Box center in world frame."""
    extent: torch.Tensor
    """Shape (3,). Full extents [width, depth, height] in meters."""

    def transform(self, pose: torch.Tensor) -> "Box":
        """Apply SE3 pose to box center. Extent is invariant (axis-aligned approximation)."""
        R = so3_rotation_matrix(pose[3:7])  # (3,3)
        t = pose[:3]
        new_pos = R @ self.position + t
        return Box(position=new_pos, extent=self.extent)

    @property
    def center(self) -> torch.Tensor:
        """Alias for position."""
        return self.position


@dataclass
class HalfSpace(CollGeom):
    """Half-space (infinite plane) collision geometry."""
    point: torch.Tensor
    """Shape (3,). Any point on the plane."""
    normal: torch.Tensor
    """Shape (3,). Outward unit normal."""

    def transform(self, pose: torch.Tensor) -> "HalfSpace":
        """Apply SE3 pose to halfspace point and normal."""
        R = so3_rotation_matrix(pose[3:7])  # (3,3)
        t = pose[:3]
        new_point = R @ self.point + t
        new_normal = R @ self.normal
        return HalfSpace(point=new_point, normal=new_normal)

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

    def transform(self, pose: torch.Tensor) -> "Heightmap":
        """Apply SE3 pose to heightmap origin. Heights stay in local frame."""
        R = so3_rotation_matrix(pose[3:7])
        t = pose[:3]
        new_origin = R @ self.origin + t
        return Heightmap(heights=self.heights, origin=new_origin, resolution=self.resolution)
