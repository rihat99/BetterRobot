"""Forward kinematics: `forward_kinematics` turns a configuration into a frozen `KinematicState` of world transforms."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, final

import torch

from ..exceptions import QuaternionNormError
from ..lie import so3
from ..model._checks import batch_shape, check_tensor, expand_batch
from ..model.model import Model
from ._fk import placement_matrices, world_transforms
from ._jacobian import world_column_table
from ._plans import fk_plan, jacobian_plan
from .configuration import full_configuration

_T = TypeVar("_T")

#: Tolerance of the opt-in quaternion norm check.
_QUATERNION_NORM_TOLERANCE = 0.1


def _poses(transforms: torch.Tensor) -> torch.Tensor:
    """``[R | p]: (..., 3, 4)`` to ``[x y z qx qy qz qw]: (..., 7)`` with canonical-sign quaternions."""
    return torch.cat([transforms[..., 3], so3.from_matrix(transforms[..., :3])], dim=-1)


@final
@dataclass(frozen=True, eq=False)
class KinematicState:
    """World placements of every joint and frame at one configuration batch, as ``[R | p]`` matrices.

    Poses, rotations, translations and the Jacobian column table are derived from the transforms on first use.
    Derived tensors are memoised under ``(name, grad mode)`` by the one writer `_memoised`, so a value made under
    ``torch.no_grad`` never serves a differentiable call; ``dataclasses.replace`` starts with an empty memo.
    """

    model: Model  #: The model the state was computed from.
    q: torch.Tensor  #: ``(B..., nq)`` configuration the state was computed at.
    joint_transforms: torch.Tensor  #: ``(B..., njoints, 3, 4)`` world ``[R | p]`` of every joint.
    frame_transforms: torch.Tensor  #: ``(B..., nframes, 3, 4)`` world ``[R | p]`` of every frame.
    _memo: dict[tuple[str, bool], Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def joint_rotations(self) -> torch.Tensor:
        """``(B..., njoints, 3, 3)`` world joint rotation matrices, a view of `joint_transforms`."""
        return self.joint_transforms[..., :3]

    @property
    def joint_translations(self) -> torch.Tensor:
        """``(B..., njoints, 3)`` world joint origins, a view of `joint_transforms`."""
        return self.joint_transforms[..., 3]

    @property
    def frame_rotations(self) -> torch.Tensor:
        """``(B..., nframes, 3, 3)`` world frame rotation matrices, a view of `frame_transforms`."""
        return self.frame_transforms[..., :3]

    @property
    def frame_translations(self) -> torch.Tensor:
        """``(B..., nframes, 3)`` world frame origins, a view of `frame_transforms`."""
        return self.frame_transforms[..., 3]

    @property
    def joint_poses(self) -> torch.Tensor:
        """``(B..., njoints, 7)`` world poses ``[x y z qx qy qz qw]``, converted on first use."""
        return self._memoised("joint_poses", lambda: _poses(self.joint_transforms))

    @property
    def frame_poses(self) -> torch.Tensor:
        """``(B..., nframes, 7)`` world poses ``[x y z qx qy qz qw]``, converted on first use."""
        return self._memoised("frame_poses", lambda: _poses(self.frame_transforms))

    def joint_pose_of(self, joint: int | str) -> torch.Tensor:
        """``(B..., 7)`` world pose of one joint (id or name), without converting the others."""
        return _poses(self.joint_transforms[..., self.model.joint_id(joint), :, :])

    def frame_pose_of(self, frame: int | str) -> torch.Tensor:
        """``(B..., 7)`` world pose of one frame (id or name), without converting the others."""
        return _poses(self.frame_transforms[..., self.model.frame_id(frame), :, :])

    @property
    def jacobian_columns(self) -> torch.Tensor:
        """``(B..., 6, nv_full)`` world Jacobian column table shared by every joint and frame, built on first use.

        Column ``c`` holds ``Ad(oM_a) S_a[:, k]`` for its owner joint ``a`` and local column ``k``; a joint's world
        Jacobian is this table masked by the joint's support and reduced to the public ``nv`` (`jacobians`).
        """
        plan = jacobian_plan(self.model, dtype=self.q.dtype, device=self.q.device)
        return self._memoised(
            "jacobian_columns", lambda: world_column_table(self.joint_rotations, self.joint_translations, plan)
        )

    def _memoised(self, name: str, derive: Callable[[], _T]) -> _T:
        """``derive()`` once per ``(name, grad mode)``: the only writer of ``_memo``."""
        key = (name, torch.is_grad_enabled())
        if key not in self._memo:
            self._memo[key] = derive()
        return self._memo[key]


def _check_quaternion_norms(model: Model, q: torch.Tensor) -> None:
    """Raise `QuaternionNormError` when any spherical or free-flyer quaternion in ``q`` is far from unit norm.

    Converts a tensor predicate to a Python ``bool`` and so synchronises accelerators: called only behind the
    opt-in flag of `forward_kinematics`.
    """
    columns = fk_plan(model, dtype=q.dtype, device=q.device).quaternion_columns  # (n_quat, 4)
    if columns.numel() == 0:
        return
    norms = q[..., columns].norm(dim=-1)  # (B..., n_quat)
    tolerance = _QUATERNION_NORM_TOLERANCE
    if bool(((norms - 1.0).abs() > tolerance).any()):  # bench-ok: opt-in public-boundary debug validation
        observed = float(norms.min()), float(norms.max())  # bench-ok: opt-in public-boundary debug validation
        raise QuaternionNormError(
            f"quaternion norm of a spherical or free-flyer joint outside [{1 - tolerance}, {1 + tolerance}] "
            f"(observed range {observed}). Normalise q before passing it."
        )


def forward_kinematics(model: Model, q: torch.Tensor, *, check_quaternion_norm: bool = False) -> KinematicState:
    """Forward kinematics. ``q: (B..., nq)`` -> `KinematicState` of every joint and frame at batch ``broadcast(B..., I...)``.

    ``I...`` is the model's value batch; a ``q`` batch that does not broadcast with it raises `ShapeError`.
    ``check_quaternion_norm=True`` raises `QuaternionNormError` when the quaternion of any spherical or free-flyer
    joint is farther than 0.1 from unit norm. It syncs accelerators, so it is off by default.
    """
    check_tensor("q", q, shape=(model.nq,), dtype=model.dtype, device=model.device)
    batch = batch_shape(model, "q", q, (model.nq,))
    q = expand_batch("q", q, batch, (model.nq,))
    if check_quaternion_norm:
        _check_quaternion_norms(model, q)
    placements, frames = placement_matrices(model)
    count = math.prod(batch)  # explicit: a model without coordinates has zero-width q
    nj, nf = model.njoints, model.nframes
    # Unbatched model values broadcast inside the matmuls; batched ones are flattened with q.
    if placements.ndim > 3:
        placements = placements.broadcast_to(*batch, nj, 4, 4).reshape(count, nj, 4, 4)
    if frames.ndim > 3:
        frames = frames.broadcast_to(*batch, nf, 4, 4).reshape(count, nf, 4, 4)
    q_full = full_configuration(model, q).reshape(count, model.nq_full)
    joints, frames = world_transforms(model, q_full, placements, frames)
    return KinematicState(
        model=model,
        q=q,
        joint_transforms=joints.view(*batch, nj, 3, 4),
        frame_transforms=frames.view(*batch, nf, 3, 4),
    )


__all__ = ["KinematicState", "forward_kinematics"]
