"""Pose-format forward kinematics: the lean matrix pass converted to ``[t, q]`` poses.

:func:`~better_robot.kinematics.forward.forward_kinematics_raw` delegates here. The joint
transforms and the world chain come from :mod:`._fk_lean` (one affine map, one matmul per
tree level); this module broadcasts the inputs and converts world and parent-relative
rotations to scalar-last quaternions in one batched pass.

The matrix-to-quaternion stage emits canonical-sign quaternions (Shepperd selection), so a
quaternion may differ in overall sign (``q`` vs ``-q``) from a quaternion-composition
lane; both represent the same rotation. Callers that compare raw quaternions across lanes
must sign-align.

See ``docs/concepts/kinematics_and_jacobians.md``.
"""

from __future__ import annotations

import math

import torch

from ..data_model.execution_batch import broadcast_to_execution_batch
from ..data_model.model_structure import ModelStructure
from ..data_model.model_values import ModelValues
from ..data_model.reduced_coordinates import expand_configuration
from ..lie import se3, so3
from ._fk_lean import lean_core


def matrix_core(
    structure: ModelStructure,
    q_full: torch.Tensor,
    placements: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the batched matrix FK on prepared inputs.

    :param q_full: full (mimic-expanded) configuration ``(B..., nq_full)``.
    :param placements: parent-frame joint placements ``(B..., njoints, 7)``.
    :returns: ``(world, local)`` joint placements, each ``(B..., njoints, 7)``
        in ``[tx, ty, tz, qx, qy, qz, qw]`` layout.
    """

    batch_shape = tuple(placements.shape[:-2])
    nj = structure.njoints
    count = math.prod(batch_shape)  # explicit: a model without coordinates has zero-width q
    world, local = lean_core(structure, q_full.reshape(count, -1), se3.to_matrix(placements.reshape(count, nj, 7)))
    both = torch.cat([world, local], dim=1)  # (N, 2 nj, 3, 4)
    pose = torch.cat([both[..., 3], so3.from_matrix(both[..., :3])], dim=-1).view(*batch_shape, 2 * nj, 7)
    return pose[..., :nj, :], pose[..., nj:, :]


def prepare_inputs(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast and mimic-expand FK inputs to the execution batch.

    :returns: ``(q_full, placements)`` broadcast to the shared execution batch.
    """

    batch_shape = values._execution_batch_shape(q)
    q = broadcast_to_execution_batch(q, batch_shape, (structure.nq,), name="q")
    q_full = expand_configuration(structure, q)
    placements = broadcast_to_execution_batch(
        values.joint_placements,
        batch_shape,
        (structure.njoints, 7),
        name="joint_placements",
    )
    return q_full, placements


def matrix_forward(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager matrix FK. Returns ``(world, local)`` placements ``(B..., njoints, 7)``."""

    q_full, placements = prepare_inputs(structure, values, q)
    return matrix_core(structure, q_full, placements)


def _compiled_core(structure: ModelStructure):
    """Return the ``torch.compile``d matrix core cached on ``structure``.

    The core closes over the structure and takes only the
    two differentiable tensors, so ``dynamic=True`` lets the batch size vary
    without a recompile. The multi-second first-call compilation is paid once
    per structure (and per dtype/device, since :meth:`ModelStructure.to` mints a
    fresh instance).
    """

    core = getattr(structure, "_fk_matrix_compiled_core", None)
    if core is None:

        def run(q_full: torch.Tensor, placements: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return matrix_core(structure, q_full, placements)

        core = torch.compile(run, dynamic=True)
        object.__setattr__(structure, "_fk_matrix_compiled_core", core)
    return core


def matrix_forward_compiled(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled matrix FK. Returns ``(world, local)`` placements ``(B..., njoints, 7)``."""

    q_full, placements = prepare_inputs(structure, values, q)
    return _compiled_core(structure)(q_full, placements)


__all__ = [
    "matrix_core",
    "matrix_forward",
    "matrix_forward_compiled",
    "prepare_inputs",
]
