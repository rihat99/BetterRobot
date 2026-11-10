"""A robot approximated by spheres rigidly attached to its frames, and self-collision distances between them.

``RobotSpheres`` stores, per sphere, a frame id, an offset in that frame and a radius. ``self_collision_pairs``
filters pairs that cannot collide (same rigid body, bodies joined by one moving joint, or frame pairs the caller
ignores); ``self_distances`` evaluates the signed sphere-sphere distances for the remaining pairs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import torch

from ..exceptions import ArgumentError, DeviceMismatchError, DtypeMismatchError, ShapeError
from ..kinematics import KinematicState
from ..model.model import Model
from .distance import sphere_sphere_raw

if TYPE_CHECKING:
    from .robot_capsules import RobotCapsules


@final
@dataclass(frozen=True, kw_only=True)
class RobotSpheres:
    """``S`` spheres, each rigidly attached to a model frame.

    Sphere ``s`` has its centre at ``offsets[s]`` in the coordinates of frame ``frame_ids[s]`` and radius
    ``radii[s]``. Shapes: ``frame_ids (S,)`` long, ``offsets (S, 3)`` or batched ``(B..., S, 3)`` (per-sample
    offsets, broadcast against the state's batch axes), ``radii (S,)``; build it with ``from_frames`` so dtype and
    device match the model, and move it with ``to``.

    ```python
    spheres = RobotSpheres.from_frames(model, {"panda_link7": [((0.0, 0.0, 0.05), 0.06)]})
    centers = spheres.centers(forward_kinematics(model, q))  # (B..., S, 3), WORLD
    ```
    """

    frame_ids: torch.Tensor
    offsets: torch.Tensor
    radii: torch.Tensor

    def __repr__(self) -> str:
        dtype = str(self.radii.dtype).removeprefix("torch.")
        return f"RobotSpheres({self.frame_ids.shape[0]} spheres, {dtype}, {self.radii.device})"

    @classmethod
    def from_frames(
        cls, model: Model, spheres: Mapping[str | int, Sequence[tuple[Sequence[float], float]]]
    ) -> RobotSpheres:
        """Build from ``{frame name or id: [(offset xyz, radius), ...]}``, in the model's dtype and device.

        Frames resolve through `Model.frame_id`: an unknown name raises ``UnknownNameError``, an id outside
        ``[0, nframes)`` raises ``ArgumentError``.
        """
        frame_ids, offsets, radii = [], [], []
        for frame, entries in spheres.items():
            frame_id = model.frame_id(frame)
            for offset, radius in entries:
                frame_ids.append(frame_id)
                offsets.append(list(offset))
                radii.append(radius)
        return cls(
            frame_ids=torch.tensor(frame_ids, dtype=torch.long, device=model.device),
            offsets=torch.tensor(offsets, dtype=model.dtype, device=model.device).reshape(-1, 3),
            radii=torch.tensor(radii, dtype=model.dtype, device=model.device),
        )

    def to(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> RobotSpheres:
        """Copy on ``device`` with floating ``offsets``/``radii`` in ``dtype``; ``frame_ids`` stay ``long``."""
        return RobotSpheres(
            frame_ids=self.frame_ids.to(device=device),
            offsets=self.offsets.to(device=device, dtype=dtype),
            radii=self.radii.to(device=device, dtype=dtype),
        )

    def centers(self, state: KinematicState) -> torch.Tensor:
        """Sphere centres ``(B..., S, 3)`` in ``WORLD``, from the frame poses in ``state`` (``KinematicState``).

        Raises ``DtypeMismatchError`` or ``DeviceMismatchError`` when the spheres are not in the state's dtype and on
        its device (move them with ``to``).
        """
        _check_attached("spheres", self.offsets, state)
        # A product-sum: a batched 3x3 matmul was 2.5x slower forward + backward. Advanced indexing, not index_select,
        # whose CUDA backward adds with atomics and so is not bitwise reproducible.
        rotation = state.frame_rotations[..., self.frame_ids, :, :]
        translation = state.frame_translations[..., self.frame_ids, :]
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
    parents = model.parent_joint_ids

    def body(joint: int) -> int:
        # The nearest ancestor joint with coordinates; fixed joints do not separate bodies.
        while joint > 0 and model.nvs_full[joint] == 0:
            joint = parents[joint]
        return joint

    body_of = [body(joint) for joint in model.frame_parent_joint_ids]
    pairs = []
    frame_ids = spheres.frame_ids.tolist()
    if any(not 0 <= frame < model.nframes for frame in frame_ids):
        raise ArgumentError(
            f"spheres.frame_ids must lie in [0, {model.nframes}) for this model; build them with from_frames(model, ...)"
        )
    for i, frame_i in enumerate(frame_ids):
        for j in range(i + 1, len(frame_ids)):
            body_i, body_j = body_of[frame_i], body_of[frame_ids[j]]
            if frozenset((frame_i, frame_ids[j])) in ignored:
                continue
            if body_i != body_j and body(parents[body_i]) != body_j and body(parents[body_j]) != body_i:
                pairs.append((i, j))
    return torch.tensor(pairs, dtype=torch.long, device=spheres.frame_ids.device).reshape(-1, 2)


def self_distances(spheres: RobotSpheres, state: KinematicState, *, pairs: torch.Tensor) -> torch.Tensor:
    """Signed distances ``(B..., P)`` between the sphere pairs ``pairs: (P, 2)``. Raises as `RobotSpheres.centers`, and
    ``ShapeError``/``DtypeMismatchError``/``DeviceMismatchError`` for ``pairs`` that is not a ``(P, 2)`` long tensor
    on the spheres' device."""
    _check_pairs(pairs, spheres.frame_ids.device)
    centers = spheres.centers(state)
    first, second = pairs[:, 0], pairs[:, 1]
    return sphere_sphere_raw(
        centers[..., first, :], spheres.radii[first], center_b=centers[..., second, :], radius_b=spheres.radii[second]
    )


def _check_attached(name: str, values: torch.Tensor, state: KinematicState) -> None:
    """Raise unless ``values`` (a sphere or capsule table) is in ``state``'s dtype and on its device."""
    if not isinstance(state, KinematicState):
        raise TypeError(f"state must be a KinematicState, got {type(state).__name__}")
    dtype, device = state.frame_transforms.dtype, state.frame_transforms.device
    if values.dtype != dtype:
        raise DtypeMismatchError(f"{name}.dtype={values.dtype} != state.dtype={dtype}; pass {name}.to(dtype={dtype})")
    if values.device != device:
        raise DeviceMismatchError(
            f"{name}.device={values.device} != state.device={device}; pass {name}.to(device={device})"
        )


def _check_pairs(pairs: object, device: torch.device) -> None:
    """Raise unless ``pairs`` is a ``(P, 2)`` long tensor on ``device``."""
    if not isinstance(pairs, torch.Tensor):
        raise TypeError(f"pairs must be a torch.Tensor, got {type(pairs).__name__}")
    if pairs.dtype != torch.long:
        raise DtypeMismatchError(f"pairs.dtype={pairs.dtype} is not torch.int64; pass pairs.long()")
    if pairs.ndim != 2 or pairs.shape[-1] != 2:
        raise ShapeError(f"pairs must have shape (P, 2), got {tuple(pairs.shape)}")
    if pairs.device != device:
        raise DeviceMismatchError(f"pairs.device={pairs.device} != spheres.device={device}; pass pairs.to({device})")
