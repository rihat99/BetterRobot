"""A robot approximated by spheres rigidly attached to its frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..data_model.model import Model
from ..kinematics.state import KinematicState
from .distance import sphere_sphere


@dataclass(frozen=True)
class RobotSpheres:
    """Spheres ``s`` with centre ``frame_ids[s]`` pose applied to ``offsets[s]``.

    ``frame_ids: (S,)`` long, ``offsets: (S, 3)`` in the frame, ``radii: (S,)``.
    """

    frame_ids: torch.Tensor
    offsets: torch.Tensor
    radii: torch.Tensor

    @classmethod
    def from_frames(
        cls, model: Model, spheres: Mapping[str | int, Sequence[tuple[Sequence[float], float]]]
    ) -> RobotSpheres:
        """Build from ``{frame name or id: [(offset xyz, radius), ...]}``, in the model's dtype and device."""
        like = model.lower_pos_limit
        frame_ids, offsets, radii = [], [], []
        for frame, entries in spheres.items():
            frame_id = model.frame_id(frame) if isinstance(frame, str) else frame
            for offset, radius in entries:
                frame_ids.append(frame_id)
                offsets.append(list(offset))
                radii.append(radius)
        return cls(
            frame_ids=torch.tensor(frame_ids, dtype=torch.long, device=like.device),
            offsets=torch.tensor(offsets, dtype=like.dtype, device=like.device).reshape(-1, 3),
            radii=torch.tensor(radii, dtype=like.dtype, device=like.device),
        )

    def centers(self, state: KinematicState) -> torch.Tensor:
        """World centres ``(B..., S, 3)`` from ``state.frame_transform: (B..., nframes, 3, 4)``."""
        rotation = state.frame_rotation[..., self.frame_ids, :, :]
        translation = state.frame_translation[..., self.frame_ids, :]
        return translation + (rotation @ self.offsets[..., None])[..., 0]


def self_collision_pairs(model: Model, spheres: RobotSpheres) -> torch.Tensor:
    """Sphere index pairs ``(P, 2)`` (``i < j``) that can move relative to each other.

    Excluded: spheres on the same rigid body (frames joined only by fixed joints) and on bodies
    joined by one moving joint, which touch by construction.
    """
    structure = model.structure
    parents = model.parents

    def body(joint: int) -> int:
        # The nearest ancestor joint with coordinates; fixed joints do not separate bodies.
        while joint > 0 and structure.nvs_full[joint] == 0:
            joint = parents[joint]
        return joint

    body_of = [body(joint) for joint in structure.frame_parent_joint_ids]
    pairs = []
    frame_ids = spheres.frame_ids.tolist()
    for i, frame_i in enumerate(frame_ids):
        for j in range(i + 1, len(frame_ids)):
            body_i, body_j = body_of[frame_i], body_of[frame_ids[j]]
            if body_i != body_j and body(parents[body_i]) != body_j and body(parents[body_j]) != body_i:
                pairs.append((i, j))
    return torch.tensor(pairs, dtype=torch.long, device=spheres.frame_ids.device).reshape(-1, 2)


def self_distances(spheres: RobotSpheres, state: KinematicState, pairs: torch.Tensor) -> torch.Tensor:
    """Signed distances ``(B..., P)`` between the sphere pairs ``pairs: (P, 2)``."""
    centers = spheres.centers(state)
    first, second = pairs[:, 0], pairs[:, 1]
    return sphere_sphere(centers[..., first, :], spheres.radii[first], centers[..., second, :], spheres.radii[second])
