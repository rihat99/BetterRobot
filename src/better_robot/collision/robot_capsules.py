"""A robot approximated by capsules rigidly attached to its frames, and self-collision distances between them.

``RobotCapsules`` stores, per capsule, a frame id, the two end points of its segment in that frame and a radius; equal
end points make a sphere. ``self_collision_pairs`` filters the pairs with the same rules as for spheres;
``self_capsule_distances`` evaluates the signed capsule-capsule distances for the remaining pairs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..data_model.model import Model
from ..kinematics.state import KinematicState
from .distance import capsule_capsule


@dataclass(frozen=True)
class RobotCapsules:
    """``C`` capsules, each rigidly attached to a model frame.

    Capsule ``c`` is every point within ``radii[c]`` of the segment from ``starts[c]`` to ``ends[c]``, both in the
    coordinates of frame ``frame_ids[c]``. Shapes: ``frame_ids (C,)`` long, ``starts, ends (C, 3)``, ``radii (C,)``;
    build it with ``from_frames`` so dtype and device match the model.

    ```python
    capsules = RobotCapsules.from_frames(model, {"body_left_knee_link": [((0.01, 0, 0), (0.01, 0, -0.15), 0.045)]})
    starts, ends = capsules.segments(forward_kinematics_state(model, q))  # (B..., C, 3) each, WORLD
    ```
    """

    frame_ids: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor
    radii: torch.Tensor

    @classmethod
    def from_frames(
        cls, model: Model, capsules: Mapping[str | int, Sequence[tuple[Sequence[float], Sequence[float], float]]]
    ) -> RobotCapsules:
        """Build from ``{frame name or id: [(start xyz, end xyz, radius), ...]}``, in the model's dtype and device."""
        like = model.lower_pos_limit
        frame_ids, starts, ends, radii = [], [], [], []
        for frame, entries in capsules.items():
            for start, end, radius in entries:
                frame_ids.append(model.frame_id(frame))
                starts.append(list(start))
                ends.append(list(end))
                radii.append(radius)
        return cls(
            frame_ids=torch.tensor(frame_ids, dtype=torch.long, device=like.device),
            starts=torch.tensor(starts, dtype=like.dtype, device=like.device).reshape(-1, 3),
            ends=torch.tensor(ends, dtype=like.dtype, device=like.device).reshape(-1, 3),
            radii=torch.tensor(radii, dtype=like.dtype, device=like.device),
        )

    def segments(self, state: KinematicState) -> tuple[torch.Tensor, torch.Tensor]:
        """End points ``(starts, ends)``, each ``(B..., C, 3)`` in ``WORLD``, from the frame poses in ``state``."""
        # Advanced indexing, as in RobotSpheres.centers: index_select's CUDA backward is not bitwise reproducible.
        rotation = state.frame_rotation[..., self.frame_ids, :, :]
        translation = state.frame_translation[..., self.frame_ids, :]
        starts = translation + (rotation * self.starts[..., None, :]).sum(-1)
        return starts, translation + (rotation * self.ends[..., None, :]).sum(-1)


def self_capsule_distances(capsules: RobotCapsules, state: KinematicState, *, pairs: torch.Tensor) -> torch.Tensor:
    """Signed distances ``(B..., P)`` between the capsule pairs ``pairs: (P, 2)`` (see ``self_collision_pairs``)."""
    starts, ends = capsules.segments(state)
    first, second = pairs[:, 0], pairs[:, 1]
    return capsule_capsule(
        starts[..., first, :],
        ends[..., first, :],
        radius_a=capsules.radii[first],
        start_b=starts[..., second, :],
        end_b=ends[..., second, :],
        radius_b=capsules.radii[second],
    )
