"""Forward kinematics — one topological scan per ``Model``.

Replaces the legacy split between ``_fk_impl`` and the
``_solve_floating_*`` family. FK is a single function that walks
``model.topo_order`` and calls ``joint_models[j].joint_transform``. A
free-flyer root is not a special case — it's the joint at index 1.

See ``docs/concepts/kinematics_and_jacobians.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch

from ..data_model import KinematicsLevel
from ..data_model.data import Data
from ..data_model.execution_batch import broadcast_to_execution_batch
from ..data_model.joint_dispatch import joint_transform
from ..data_model.joint_models import JointFreeFlyer
from ..data_model.model import Model
from ..data_model.model_structure import ModelStructure
from ..data_model.model_values import ModelValues
from ..data_model.reduced_coordinates import expand_configuration
from ..exceptions import (
    DeviceMismatchError,
    DtypeMismatchError,
    QuaternionNormError,
    ShapeError,
)
from ..lie import se3


#: Tolerance used by the opt-in free-flyer quaternion debug check.
_QUAT_NORM_TOL = 0.1
_WARNED_WARP_FALLBACKS: set[str] = set()


def _warn_warp_fallback(reason_key: str, reason: str) -> None:
    """Warn once when an explicit Warp FK request uses the Torch lane."""

    if reason_key in _WARNED_WARP_FALLBACKS:
        return
    _WARNED_WARP_FALLBACKS.add(reason_key)
    warnings.warn(
        f"better_robot: use_warp=True requested the Warp FK lane, but {reason}; using the Torch FK lane instead.",
        RuntimeWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class FKResult:
    """Fresh world- and parent-frame joint placements from raw FK."""

    joint_pose_world: torch.Tensor
    joint_pose_local: torch.Tensor


@dataclass(frozen=True)
class FramePlacementsResult:
    """Fresh world-frame placements from :func:`frame_placements_raw`."""

    frame_pose_world: torch.Tensor


def _validate_q(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> None:
    """Check the shape and device of a configuration tensor.

    Raises
    ------
    ShapeError
        If ``q.shape[-1] != model.nq``.
    DeviceMismatchError
        If ``q.device`` differs from the model's tensor device.

    See ``docs/conventions/contracts.md §1``.
    """
    if q.ndim < 1 or q.shape[-1] != structure.nq:
        raise ShapeError(f"q has shape {tuple(q.shape)}; expected trailing size model.nq={structure.nq}")
    if q.dtype not in (torch.float32, torch.float64):
        raise DtypeMismatchError(f"q.dtype={q.dtype} is unsupported; use torch.float32 or torch.float64")
    model_dtype = values.joint_placements.dtype
    if q.dtype != model_dtype:
        raise DtypeMismatchError(
            f"q.dtype={q.dtype} != model.dtype={model_dtype}. Cast q or call model.to(dtype=q.dtype) before evaluation."
        )
    model_device = values.joint_placements.device
    if q.device != model_device:
        raise DeviceMismatchError(
            f"q.device={q.device} != model.device={model_device}. Call model.to(q.device) or q.to(model.device) first."
        )


def _validate_free_flyer_quaternion_norm(model: Model, q: torch.Tensor) -> None:
    """Debug-check the free-flyer quaternion norm.

    This check converts a tensor predicate to a Python ``bool`` and therefore
    synchronizes accelerators.  It must only be called from an explicitly
    requested public-boundary check, never from tensor-only primitives.
    """
    if model.njoints >= 2 and isinstance(model.joint_models[1], JointFreeFlyer):
        # Free-flyer q layout: [tx, ty, tz, qx, qy, qz, qw]
        quat = q[..., 3:7]
        norm = quat.norm(dim=-1)
        if bool(((norm - 1.0).abs() > _QUAT_NORM_TOL).any()):  # bench-ok: opt-in public-boundary debug validation
            bad = float(norm.min()), float(norm.max())  # bench-ok: opt-in public-boundary debug validation
            raise QuaternionNormError(
                f"free-flyer quaternion norm outside [{1 - _QUAT_NORM_TOL}, "
                f"{1 + _QUAT_NORM_TOL}] (observed range {bad}). "
                f"Normalise before passing."
            )


def forward_kinematics_raw(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> FKResult:
    """Tensor-only FK primitive returning named world/local placements.

    Autograd-safe: uses list accumulation + ``torch.stack`` instead of
    in-place writes so the backward pass can trace through every SE3
    composition cleanly.

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

    See docs/concepts/kinematics_and_jacobians.md and docs/conventions/naming.md.
    """
    batch_shape = values._execution_batch_shape(q)
    q = broadcast_to_execution_batch(
        q,
        batch_shape,
        (structure.nq,),
        name="q",
    )
    q_full = expand_configuration(structure, q)
    placements = broadcast_to_execution_batch(
        values.joint_placements,
        batch_shape,
        (structure.njoints, 7),
        name="joint_placements",
    )
    axes = structure.joint_axes
    pitches = structure.joint_pitches

    world_list: list[torch.Tensor] = [None] * structure.njoints  # type: ignore[list-item]
    local_list: list[torch.Tensor] = [None] * structure.njoints  # type: ignore[list-item]

    for j in structure.topo_order:
        nq_j = structure.nqs_full[j]
        q_j = q_full[
            ...,
            structure.idx_qs_full[j] : structure.idx_qs_full[j] + nq_j,
        ]
        T_j = joint_transform(
            structure.joint_models[j],
            structure.joint_kind_codes[j],
            axes[j],
            pitches[j],
            q_j,
        )

        # joint_pose_local[j] = T_placement ∘ T_j  (parent-frame placement)
        local_j = se3.compose(placements[..., j, :], T_j)
        local_list[j] = local_j

        parent = structure.parents[j]
        if parent < 0:
            world_list[j] = local_j
        else:
            world_list[j] = se3.compose(world_list[parent], local_j)  # (B..., 7)

    stack_dim = len(batch_shape)
    joint_pose_world = torch.stack(world_list, dim=stack_dim)  # (B..., njoints, 7)
    joint_pose_local = torch.stack(local_list, dim=stack_dim)
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
    use_warp: bool = False,
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
    use_warp : bool
        Opt into the CUDA-validated fused Warp FK lane when the optional ``warp``
        extra, joint kinds, dtype, and layout are supported. Unsupported
        inputs use the torch lane and emit a one-shot warning naming the
        reason.
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

    # Validate exactly once at the public boundary. The raw torch, Warp, and
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
    warp_result = None
    if use_warp:
        try:
            from ._warp_bridge import try_warp_forward_kinematics  # noqa: PLC0415
        except ModuleNotFoundError as error:
            if error.name != "warp":
                raise
            if q.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "better_robot: Warp FK cannot fall back to the Torch lane while CUDA "
                    "graph capture is active because the optional Warp runtime is unavailable. "
                    "Install the Warp extra before capture, or disable graph capture."
                ) from error
            _warn_warp_fallback(
                "optional-runtime",
                "the optional Warp runtime is unavailable",
            )
        else:
            warp_result = try_warp_forward_kinematics(model.structure, model.values, q)
    if warp_result is None:
        fk_result = forward_kinematics_raw(model.structure, model.values, q)
        joint_pose_world = fk_result.joint_pose_world
        joint_pose_local = fk_result.joint_pose_local
    else:
        joint_pose_world, joint_pose_local = warp_result.world, warp_result.local
    data.joint_pose_world = joint_pose_world
    data.joint_pose_local = joint_pose_local
    object.__setattr__(data, "_kinematics_level", KinematicsLevel.PLACEMENTS)

    if compute_frames and warp_result is not None:
        data.frame_pose_world = warp_result.frames
    elif compute_frames:
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
