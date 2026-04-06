"""RobotCollision: sphere decomposition of robot links."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    import yourdfpy

from ...models.robot_model import RobotModel
from ...math.so3 import so3_rotation_matrix
from .primitives import CollGeom, Sphere
from .distance import compute_distance

__all__ = [
    "RobotCollision",
]


@dataclass
class RobotCollision:
    """Robot collision model using sphere decomposition.

    Each robot link is approximated as a set of spheres.
    Sphere centers are stored in link-local frame and transformed by FK at query time.
    """
    # Centers in local link frame: (num_spheres, 3)
    _local_centers: torch.Tensor
    # Radii: (num_spheres,)
    _radii: torch.Tensor
    # Which link each sphere belongs to: (num_spheres,)
    _link_indices: torch.Tensor
    # Active pairs for self-collision: list of (i, j) sphere index pairs
    _active_pairs_i: tuple[int, ...]
    _active_pairs_j: tuple[int, ...]

    @staticmethod
    def from_sphere_decomposition(
        sphere_decomposition: dict,
        model: RobotModel,
    ) -> "RobotCollision":
        """Create a RobotCollision from a sphere decomposition dict.

        Args:
            sphere_decomposition: Dict mapping link_name to list of
                {'center': [x,y,z], 'radius': r} dicts, OR to
                {'centers': [[x,y,z], ...], 'radii': [r, ...]} dicts.
            model: RobotModel instance (for link name/index mapping).

        Returns:
            RobotCollision instance.
        """
        link_name_to_idx = {name: idx for idx, name in enumerate(model.links.names)}

        all_centers: list[list[float]] = []
        all_radii: list[float] = []
        sphere_link_indices: list[int] = []
        geom_counts: list[int] = [0] * model.links.num_links

        for link_name, data in sphere_decomposition.items():
            if link_name not in link_name_to_idx:
                continue
            link_idx = link_name_to_idx[link_name]

            # Support both single sphere {'center', 'radius'} and multiple {'centers', 'radii'}
            if 'centers' in data:
                centers = data['centers']
                radii = data['radii']
            else:
                centers = [data['center']]
                radii = [data['radius']]

            for center, radius in zip(centers, radii):
                all_centers.append(list(center))
                all_radii.append(float(radius))
                sphere_link_indices.append(link_idx)
                geom_counts[link_idx] += 1

        if not all_centers:
            raise ValueError("No spheres found in sphere_decomposition.")

        local_centers = torch.tensor(all_centers, dtype=torch.float32)
        radii = torch.tensor(all_radii, dtype=torch.float32)
        link_indices = torch.tensor(sphere_link_indices, dtype=torch.long)

        # Compute active self-collision pairs (exclude adjacent links)
        # For simplicity: exclude spheres on the same link
        num_spheres = len(all_centers)
        idx_i: list[int] = []
        idx_j: list[int] = []
        for i in range(num_spheres):
            for j in range(i + 1, num_spheres):
                # Skip same-link pairs
                if sphere_link_indices[i] != sphere_link_indices[j]:
                    idx_i.append(i)
                    idx_j.append(j)

        return RobotCollision(
            _local_centers=local_centers,
            _radii=radii,
            _link_indices=link_indices,
            _active_pairs_i=tuple(idx_i),
            _active_pairs_j=tuple(idx_j),
        )

    def _get_world_spheres(self, model: RobotModel, q: torch.Tensor) -> list[Sphere]:
        """Get all spheres transformed to world frame given joint config."""
        fk = model.forward_kinematics(q)  # (num_links, 7)
        spheres = []
        for i in range(len(self._radii)):
            link_idx = int(self._link_indices[i].item())
            T = fk[link_idx]  # (7,)
            R = so3_rotation_matrix(T[3:7])
            t = T[:3]
            world_center = R @ self._local_centers[i].to(T.device) + t
            spheres.append(Sphere(center=world_center, radius=float(self._radii[i].item())))
        return spheres

    def compute_self_collision_distance(
        self,
        model: RobotModel,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute signed distances for all active self-collision pairs.

        Returns:
            Shape (num_active_pairs,). Positive = separated, negative = penetrating.
        """
        spheres = self._get_world_spheres(model, q)
        dists = []
        for i, j in zip(self._active_pairs_i, self._active_pairs_j):
            dists.append(compute_distance(spheres[i], spheres[j]))
        if not dists:
            return torch.zeros(0)
        return torch.stack(dists)

    def compute_world_collision_distance(
        self,
        model: RobotModel,
        q: torch.Tensor,
        world_geom: list[CollGeom],
    ) -> torch.Tensor:
        """Compute signed distances from each robot sphere to all world geometries.

        Returns:
            Shape (num_robot_spheres * len(world_geom),). Negative = penetrating.
        """
        spheres = self._get_world_spheres(model, q)
        dists = []
        for sphere in spheres:
            for wg in world_geom:
                dists.append(compute_distance(wg, sphere))
        if not dists:
            return torch.zeros(0)
        return torch.stack(dists)
