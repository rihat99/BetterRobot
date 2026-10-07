"""Forward kinematics — one topological scan per ``Model``.

Replaces the legacy split between ``_fk_impl`` and the
``_solve_floating_*`` family. FK is a single function that walks
``model.topo_order`` and calls ``joint_models[j].joint_transform``. A
free-flyer root is not a special case — it's the joint at index 1.

See ``docs/concepts/kinematics_and_jacobians.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data_model import KinematicsLevel
from ..data_model.data import Data
from ..data_model.execution_batch import broadcast_to_execution_batch
from ..data_model.model import Model
from ..data_model.model_structure import ModelStructure
from ..data_model.model_values import ModelValues
from ..lie import se3
from . import _fk_matrix
from .state import _validate_free_flyer_quaternion_norm, _validate_q


@dataclass(frozen=True)
class FKResult:
    """Fresh world- and parent-frame joint placements from raw FK."""

    joint_pose_world: torch.Tensor
    joint_pose_local: torch.Tensor


@dataclass(frozen=True)
class FramePlacementsResult:
    """Fresh world-frame placements from :func:`frame_placements_raw`."""

    frame_pose_world: torch.Tensor


def forward_kinematics_raw(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> FKResult:
    """Tensor-only FK primitive returning named world/local placements.

    Runs the batched matrix lane in :mod:`._fk_matrix`: kind-grouped local
    transforms, one batched ``4x4`` matmul per joint over the topological order,
    and a single batched matrix-to-quaternion pass. Autograd- and
    ``torch.func``-safe (jacrev and jacfwd).

    Parameters
    ----------
    structure : ModelStructure
    values : ModelValues
    q : (B..., nq)

    Returns
    -------
    FKResult
        ``joint_pose_world`` has shape ``(B..., njoints, 7)`` and contains
        world-frame joint placements (the quantity historically called
        ``oMi``). ``joint_pose_local`` has the same shape and contains
        parent-frame joint placements (``liMi``).

    Notes
    -----
    ``structure``, ``values``, and ``q`` must already satisfy the public FK
    boundary contracts; this internal primitive does not revalidate them.
    Free-flyer quaternions are assumed to be pre-normalized.  Use
    :func:`forward_kinematics` with ``check_quaternion_norm=True`` to run the
    opt-in debug check before calling this hot-path primitive.

    The matrix lane emits canonical-sign quaternions, so an output quaternion
    may differ in overall sign (``q`` vs ``-q``, the same rotation) from a
    quaternion-composition lane; sign-align before comparing raw quaternions.

    See docs/concepts/kinematics_and_jacobians.md and docs/conventions/naming.md.
    """
    joint_pose_world, joint_pose_local = _fk_matrix.matrix_forward(structure, values, q)
    return FKResult(
        joint_pose_world=joint_pose_world,
        joint_pose_local=joint_pose_local,
    )


def forward_kinematics(  # noqa: PLR0912 - validates and routes one public FK request
    model: Model,
    q_or_data: torch.Tensor | Data,
    *,
    compute_frames: bool = False,
    check_quaternion_norm: bool = False,
    use_compile: bool = False,
) -> Data:
    """Compute joint (and optionally frame) placements.

    Parameters
    ----------
    model : Model
        Immutable kinematic tree.
    q_or_data : torch.Tensor | Data
        Either a flat configuration tensor of shape ``(B..., nq)`` or a
        pre-allocated ``Data`` whose ``q`` field is populated.
    compute_frames : bool
        If true, also populate ``data.frame_pose_world``.
    check_quaternion_norm : bool
        If true, verify that a free-flyer quaternion has norm within 0.1 of
        one and raise :class:`~better_robot.exceptions.QuaternionNormError`
        otherwise.  This opt-in debug check synchronizes accelerator tensors;
        the default hot path assumes free-flyer quaternions are pre-normalized.
    use_compile : bool
        Opt into a ``torch.compile``d Torch FK lane, cached per model on first
        use. The first call pays a multi-second compilation; later calls reuse
        the cached callable, and ``dynamic=True`` lets the batch size vary
        without a recompile.
    Returns
    -------
    Data
        ``Data`` with ``joint_pose_local`` and ``joint_pose_world`` populated
        (and ``frame_pose_world`` if ``compute_frames=True``).

    See docs/concepts/kinematics_and_jacobians.md.
    """
    if isinstance(q_or_data, Data):
        data = q_or_data
        q = data.q
    else:
        q = q_or_data
        data = None

    # Validate exactly once at the public boundary. The raw torch and
    # frame passes below trust these inputs. Keep the tensor-to-Python norm
    # check opt-in so the default path is sync-free.
    _validate_q(model.structure, model.values, q)
    batch_shape = model.values._execution_batch_shape(q)
    if data is None:
        data = model.create_data(
            batch_shape=batch_shape,
            device=q.device,
            dtype=q.dtype,
        )
    q = broadcast_to_execution_batch(
        q,
        batch_shape,
        (model.nq,),
        name="q",
    )
    data.q = q
    if check_quaternion_norm:
        _validate_free_flyer_quaternion_norm(model, q)
    if use_compile:
        joint_pose_world, joint_pose_local = _fk_matrix.matrix_forward_compiled(model.structure, model.values, q)
    else:
        fk_result = forward_kinematics_raw(model.structure, model.values, q)
        joint_pose_world = fk_result.joint_pose_world
        joint_pose_local = fk_result.joint_pose_local
    data.joint_pose_world = joint_pose_world
    data.joint_pose_local = joint_pose_local
    object.__setattr__(data, "_kinematics_level", KinematicsLevel.PLACEMENTS)

    if compute_frames:
        data.frame_pose_world = frame_placements_raw(
            model.structure,
            model.values,
            joint_pose_world,
        ).frame_pose_world

    return data


def update_frame_placements(model: Model, data: Data) -> Data:
    """Populate ``data.frame_pose_world`` from ``data.joint_pose_world``
    and the model's frame metadata.

    Requires ``data.joint_pose_world`` to be populated (call
    :func:`forward_kinematics` first).

    See docs/concepts/kinematics_and_jacobians.md.
    """
    joint_pose_world = data.joint_pose_world
    assert joint_pose_world is not None, "call forward_kinematics before update_frame_placements"

    data.frame_pose_world = frame_placements_raw(
        model.structure,
        model.values,
        joint_pose_world,
    ).frame_pose_world
    return data


def frame_placements_raw(
    structure: ModelStructure,
    values: ModelValues,
    joint_pose_world: torch.Tensor,
) -> FramePlacementsResult:
    """Place frames from prevalidated structure/value tensors."""

    batch_shape = tuple(joint_pose_world.shape[:-2])
    parents = structure.frame_parent_joints
    parent_poses = joint_pose_world.index_select(-2, parents.to(torch.int64))
    local = broadcast_to_execution_batch(
        values.frame_placements,
        batch_shape,
        (structure.nframes, 7),
        name="frame_placements",
    )
    return FramePlacementsResult(frame_pose_world=se3.compose(parent_poses, local))
