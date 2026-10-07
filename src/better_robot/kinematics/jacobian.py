"""Raw Jacobian time-variation kernels behind :mod:`.state`.

See ``docs/concepts/kinematics_and_jacobians.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..data_model.execution_batch import broadcast_to_execution_batch
from ..data_model.model_structure import ModelStructure
from ..data_model.model_values import ModelValues
from ..data_model.reduced_coordinates import reduce_jacobian
from ..lie import se3
from ..lie.tangents import hat_so3
from ._jacobian_columns import get_plan, world_column_table

_ReferenceFrame = Literal["world", "local", "local_world_aligned"]


@dataclass(frozen=True)
class JointJacobiansTimeVariationResult:
    """World-frame joint Jacobian time variation from
    :func:`joint_jacobians_time_variation_raw`.

    Both fields are WORLD-frame and share shape ``(B..., njoints, 6, nv)``.
    ``joint_jacobians`` is the ordinary joint Jacobian, surfaced as a byproduct
    (or echoed from a reused input) so a caller can fill both caches from one
    pass without rebuilding the column table.
    """

    joint_jacobians: torch.Tensor  #: WORLD joint Jacobian ``J`` (byproduct/reuse).
    joint_jacobians_dot: torch.Tensor  #: WORLD joint Jacobian time variation ``J̇``.


def joint_jacobians_time_variation_raw(
    structure: ModelStructure,
    q: torch.Tensor,
    v: torch.Tensor,
    joint_pose_world: torch.Tensor,
    *,
    joint_jacobians: torch.Tensor | None = None,
) -> JointJacobiansTimeVariationResult:
    """Tensor-only WORLD joint-Jacobian time-variation primitive.

    Returns ``J̇`` such that a joint's WORLD spatial acceleration is
    ``a_spatial = J v̇ + J̇ v``. The pass is transport-only: every shipped joint
    has a configuration-constant local motion subspace (no ``dS/dt`` term; see
    the zero-valued subspace-derivative hook
    :func:`~better_robot.data_model.joint_models.base.joint_motion_subspace_derivative`),
    so ``J̇`` is the frame drift of a static column table.

    Each owner joint ``a``'s full-width column block transports as
    ``dJ[:, cols(a)] = ad(ov_a) @ J[:, cols(a)]`` with
    ``ad([v; w]) = [[ŵ, v̂], [0, ŵ]]`` and ``ov_a = J_a @ v`` the WORLD spatial
    velocity of joint ``a``. ``ov`` comes from the reduced per-joint Jacobian ×
    reduced ``v`` (identical to the full-width product); the owner-keyed
    transport is valid on **full-width** columns only — a reduced mimic column
    blends distinct owners — so the shared full-width table
    (:func:`._jacobian_columns.world_column_table`) always drives it and mimic
    reduction is applied last, exactly like the plain joint Jacobian.

    A supplied ``joint_jacobians`` (reduced, ``(B..., njoints, 6, nv)``) is
    reused for ``ov`` and echoed back; it never replaces the full-width build.
    ``joint_pose_world`` is ``(B..., njoints, 7)`` and both result fields are
    ``(B..., njoints, 6, nv)``.
    """
    batch = tuple(joint_pose_world.shape[:-2])
    q = broadcast_to_execution_batch(q, batch, (structure.nq,), name="q")
    v = broadcast_to_execution_batch(v, batch, (structure.nv,), name="v")
    device, dtype = q.device, q.dtype

    plan = get_plan(structure)
    columns = world_column_table(structure, joint_pose_world, plan, device=device, dtype=dtype)
    if joint_jacobians is None:
        joint_jacobians = reduce_jacobian(structure, columns.unsqueeze(-3) * plan.support_mask)

    # WORLD spatial velocity of every joint (reduced J × reduced v).
    ov = torch.einsum("...jck,...k->...jc", joint_jacobians, v)  # (B..., njoints, 6)

    # ad(ov_owner(c)) @ column(c), per full-width column, via the ad-cross form.
    ov_col = ov.index_select(-2, plan.column_owner)  # (B..., nv_full, 6)
    cols = columns.transpose(-1, -2)  # (B..., nv_full, 6)
    ov_lin, ov_ang = ov_col[..., :3], ov_col[..., 3:]
    col_lin, col_ang = cols[..., :3], cols[..., 3:]
    dot_lin = torch.linalg.cross(ov_ang, col_lin) + torch.linalg.cross(ov_lin, col_ang)
    dot_ang = torch.linalg.cross(ov_ang, col_ang)
    dot_columns = torch.cat((dot_lin, dot_ang), dim=-1).transpose(-1, -2)  # (B..., 6, nv_full)

    dot_full = dot_columns.unsqueeze(-3) * plan.support_mask
    return JointJacobiansTimeVariationResult(
        joint_jacobians=joint_jacobians,
        joint_jacobians_dot=reduce_jacobian(structure, dot_full),
    )


def _time_variation_in_reference(
    joint_jacobian_world: torch.Tensor,
    joint_jacobian_dot_world: torch.Tensor,
    pose_world: torch.Tensor,
    v: torch.Tensor,
    reference: _ReferenceFrame,
) -> torch.Tensor:
    """Derive a ref's ``J̇`` in ``reference`` from its WORLD ``J`` / ``J̇``.

    Closed forms verified against Pinocchio to machine precision. ``p`` is the
    ref origin ``pose_world[..., :3]`` and ``ov = J v`` is the ref's WORLD
    spatial velocity (identical for a joint and any frame rigidly attached to
    it). All tensors carry ``(B..., 6, nv)`` Jacobians / ``(B..., 7)`` poses /
    ``(B..., nv)`` velocity.

    - ``"world"``: the cache, ``a_spatial = J v̇ + J̇ v``.
    - ``"local_world_aligned"``: ``J̇_ang`` unchanged;
      ``J̇_lin = dJ_lin − p × dJ_ang − v_pt × J_W_ang`` with
      ``v_pt = ov.lin + ov.ang × p`` the world velocity of the moving point at
      the origin. The final ``− v_pt × J_W_ang`` term is what a static-style
      translation misses (O(1) error); ``a_classical = J v̇ + J̇ v``.
    - ``"local"``: ``J̇ = Ad(oMref)⁻¹ dJ − ad(v_ref) (Ad(oMref)⁻¹ J)`` with
      ``v_ref = Ad(oMref)⁻¹ ov`` the ref's LOCAL spatial velocity;
      ``a_spatial = J v̇ + J̇ v``.
    """
    if reference == "world":
        return joint_jacobian_dot_world

    ov = torch.matmul(joint_jacobian_world, v.unsqueeze(-1)).squeeze(-1)  # (B..., 6)

    if reference == "local_world_aligned":
        p = pose_world[..., :3]
        v_pt = ov[..., :3] + torch.linalg.cross(ov[..., 3:], p)
        dot_ang = joint_jacobian_dot_world[..., 3:, :]
        dot_lin = (
            joint_jacobian_dot_world[..., :3, :]
            - torch.matmul(hat_so3(p), dot_ang)
            - torch.matmul(hat_so3(v_pt), joint_jacobian_world[..., 3:, :])
        )
        return torch.cat((dot_lin, dot_ang), dim=-2)

    if reference == "local":
        adjoint_inv = se3.adjoint_inv(pose_world)  # (B..., 6, 6)
        j_local = torch.matmul(adjoint_inv, joint_jacobian_world)
        dot_local = torch.matmul(adjoint_inv, joint_jacobian_dot_world)
        v_local = torch.matmul(adjoint_inv, ov.unsqueeze(-1)).squeeze(-1)  # (B..., 6)
        skew_ang, skew_lin = hat_so3(v_local[..., 3:]), hat_so3(v_local[..., :3])
        j_local_lin, j_local_ang = j_local[..., :3, :], j_local[..., 3:, :]
        correction = torch.cat(
            (
                torch.matmul(skew_ang, j_local_lin) + torch.matmul(skew_lin, j_local_ang),
                torch.matmul(skew_ang, j_local_ang),
            ),
            dim=-2,
        )
        return dot_local - correction

    raise ValueError(f"Unsupported reference frame: {reference!r}")


def frame_jacobian_time_variation_raw(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
    v: torch.Tensor,
    joint_pose_world: torch.Tensor,
    frame_id: int,
    *,
    reference: _ReferenceFrame = "local_world_aligned",
    joint_jacobians: torch.Tensor | None = None,
    joint_jacobians_dot: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one frame's Jacobian time variation from raw tensors.

    A rigidly attached frame shares its parent joint's WORLD ``J`` and ``J̇``;
    the closed forms in :func:`_time_variation_in_reference` then translate/rotate
    to the frame origin ``oMframe = oMparent · placement``. Acceleration pairing:
    ``a_spatial = J v̇ + J̇ v`` for ``"world"``/``"local"``, ``a_classical`` for
    ``"local_world_aligned"``. Supplying compatible ``joint_jacobians`` and
    ``joint_jacobians_dot`` reuses an existing pass; either missing triggers a
    fresh :func:`joint_jacobians_time_variation_raw` build.
    """
    batch = tuple(joint_pose_world.shape[:-2])
    q = broadcast_to_execution_batch(q, batch, (structure.nq,), name="q")
    v = broadcast_to_execution_batch(v, batch, (structure.nv,), name="v")
    if joint_jacobians is None or joint_jacobians_dot is None:
        result = joint_jacobians_time_variation_raw(
            structure, q, v, joint_pose_world, joint_jacobians=joint_jacobians
        )
        joint_jacobians = result.joint_jacobians
        joint_jacobians_dot = result.joint_jacobians_dot

    parent_index = structure.frame_parent_joints[frame_id : frame_id + 1]
    j_parent = torch.index_select(joint_jacobians, -3, parent_index).squeeze(-3)
    dot_parent = torch.index_select(joint_jacobians_dot, -3, parent_index).squeeze(-3)
    T_local = broadcast_to_execution_batch(
        values.frame_placements[..., frame_id, :],
        batch,
        (7,),
        name="frame_placements row",
    )
    T_parent = torch.index_select(joint_pose_world, -2, parent_index).squeeze(-2)
    T_frame = se3.compose(T_parent, T_local)
    return _time_variation_in_reference(j_parent, dot_parent, T_frame, v, reference)
