"""RobotCollision: sphere decomposition of robot links."""

from __future__ import annotations

import torch

from ..core._robot import Robot
from ._geometry import CollGeom


class RobotCollision:
    """Robot collision model using sphere decomposition.

    Each robot link is approximated as a set of spheres.
    Sphere centers are transformed by FK at query time.
    """

    @staticmethod
    def from_sphere_decomposition(
        sphere_decomposition: dict,
        urdf: object,
    ) -> "RobotCollision":
        """Create a RobotCollision from a sphere decomposition dict.

        Args:
            sphere_decomposition: Dict mapping link name to list of
                {'center': [x,y,z], 'radius': r} dicts.
            urdf: yourdfpy.URDF instance (used to resolve link frame transforms).

        Returns:
            RobotCollision instance.
        """
        raise NotImplementedError

    def compute_self_collision_distance(
        self,
        robot: Robot,
        cfg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute signed distances for all active self-collision pairs.

        Args:
            robot: Robot instance.
            cfg: Shape (num_actuated_joints,). Current config.

        Returns:
            Shape (num_active_pairs,). Signed distance per pair
            (negative = penetrating).
        """
        raise NotImplementedError

    def compute_world_collision_distance(
        self,
        robot: Robot,
        cfg: torch.Tensor,
        world_geom: list[CollGeom],
    ) -> torch.Tensor:
        """Compute signed distances from each robot sphere to all world geometries.

        Args:
            robot: Robot instance.
            cfg: Shape (num_actuated_joints,). Current config.
            world_geom: List of world collision geometry objects.

        Returns:
            Shape (num_robot_spheres * len(world_geom),). Signed distance per
            sphere-geometry pair (negative = penetrating).
        """
        raise NotImplementedError
