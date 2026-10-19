"""A robot approximated by spheres rigidly attached to its frames, and self-collision distances between them.

``RobotSpheres`` stores, per sphere, a frame id, an offset in that frame and a radius. ``self_collision_pairs``
filters pairs that cannot collide (same rigid body, bodies joined by one moving joint, or frame pairs the caller
ignores); ``self_distances`` evaluates the signed sphere-sphere distances for the remaining pairs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..data_model.model import Model
from ..kinematics.state import KinematicState
from .distance import sphere_sphere

if TYPE_CHECKING:
    from .robot_capsules import RobotCapsules


@dataclass(frozen=True)
class RobotSpheres:
    """``S`` spheres, each rigidly attached to a model frame.

    Sphere ``s`` has its centre at ``offsets[s]`` in the coordinates of frame ``frame_ids[s]`` and radius
    ``radii[s]``. Shapes: ``frame_ids (S,)`` long, ``offsets (S, 3)`` or batched ``(B..., S, 3)`` (per-sample
    offsets, broadcast against the state's batch axes), ``radii (S,)``; build it with ``from_frames`` so dtype and
    device match the model, and move it with ``to``.

    ```python
    spheres = RobotSpheres.from_frames(model, {"body_panda_link7": [((0.0, 0.0, 0.05), 0.06)]})
    centers = spheres.centers(forward_kinematics_state(model, q))  # (B..., S, 3), WORLD
    ```
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

    def to(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> RobotSpheres:
        """Copy on ``device`` with floating ``offsets``/``radii`` in ``dtype``; ``frame_ids`` stay ``long``."""
        return RobotSpheres(
            frame_ids=self.frame_ids.to(device=device),
            offsets=self.offsets.to(device=device, dtype=dtype),
            radii=self.radii.to(device=device, dtype=dtype),
        )

    def centers(self, state: KinematicState) -> torch.Tensor:
        """Sphere centres ``(B..., S, 3)`` in ``WORLD``, from the frame poses in ``state`` (``KinematicState``)."""
        # A product-sum: a batched 3x3 matmul was 2.5x slower forward + backward. Advanced indexing, not index_select,
        # whose CUDA backward adds with atomics and so is not bitwise reproducible.
        rotation = state.frame_rotation[..., self.frame_ids, :, :]
        translation = state.frame_translation[..., self.frame_ids, :]
        return translation + (rotation * self.offsets[..., None, :]).sum(-1)


def self_collision_pairs(
    model: Model,
    spheres: RobotSpheres | RobotCapsules,
    *,
    ignore: Iterable[tuple[int | str, int | str]] | None = None,
) -> torch.Tensor:
    """Sphere (or capsule) index pairs ``(P, 2)`` (``i < j``) that can move relative to each other.

    Excluded: spheres on the same rigid body (frames joined only by fixed joints), on bodies
    joined by one moving joint, which touch by construction, and on the frame pairs in ``ignore``
    (frame names or frame ids, either order; body ids are not accepted, since an int is read as a
    frame id), e.g. links whose meshes overlap at every reachable ``q``.
    """
    ignored = {frozenset((model.frame_id(a), model.frame_id(b))) for a, b in (ignore or ())}
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
            if frozenset((frame_i, frame_ids[j])) in ignored:
                continue
            if body_i != body_j and body(parents[body_i]) != body_j and body(parents[body_j]) != body_i:
                pairs.append((i, j))
    return torch.tensor(pairs, dtype=torch.long, device=spheres.frame_ids.device).reshape(-1, 2)


def self_distances(spheres: RobotSpheres, state: KinematicState, *, pairs: torch.Tensor) -> torch.Tensor:
    """Signed distances ``(B..., P)`` between the sphere pairs ``pairs: (P, 2)``."""
    centers = spheres.centers(state)
    first, second = pairs[:, 0], pairs[:, 1]
    return sphere_sphere(
        centers[..., first, :], spheres.radii[first], center_b=centers[..., second, :], radius_b=spheres.radii[second]
    )
