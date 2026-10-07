"""World-frame dynamics core behind the ``KinematicState`` dynamics API.

Everything is expressed in the world frame, so every ancestor or descendant sum over the
tree is one matmul with a constant support matrix instead of a Python loop over joints:

* ``C (N, 6, nv_full)`` — world Jacobian columns, shared with ``frame_jacobian``;
* body velocity ``V_i = Σ_{a supports i} C_a v_a``;
* body acceleration ``A_i = -g + Σ_{a supports i} (C_a a_a + (V_a × C_a) v_a)``
  (``V_a × C_a`` is the time derivative of a world column);
* body wrench ``f_i = I_i A_i + V_i ×* I_i V_i - fext_i`` with ``I_i`` the world spatial inertia;
* torque ``τ_a = C_a · Σ_{i in subtree(a)} f_i``;
* mass matrix ``M[r, c] = C_r · I^c_{owner(c)} C_c`` for ``r`` on the path of ``c``, ``I^c`` the
  composite inertia of the subtree (one matmul with the support matrix).

Operator counts are independent of the joint count; the support matmuls cost
``O(njoints²)`` flops per batch element, negligible up to a few hundred joints. Forward
dynamics solves ``M ddq = τ - b`` by Cholesky: the articulated-body recursion is sequential in
the tree depth and gains nothing in batched torch for ``nv`` up to a few hundred.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..data_model.model_structure import ModelStructure
from ..data_model.model_values import ModelValues, inertia_about_com, packed_inertias_to_6x6, spatial_inertia_matrix
from ..kinematics import jacobian_columns
from ..lie.tangents import hat_so3


@dataclass(frozen=True)
class DynamicsPlan:
    """Constant support masks derived from the topology (memoised on the structure)."""

    jacobian: jacobian_columns.JointJacobianPlan
    support: (
        torch.Tensor
    )  #: ``(njoints, njoints)`` bool — ``[i, a]`` True when joint ``a`` is on the path of joint ``i``.
    column_support: (
        torch.Tensor
    )  #: ``(njoints, nv_full)`` bool — ``[i, c]`` True when the owner of column ``c`` supports joint ``i``.
    upper: (
        torch.Tensor
    )  #: ``(nv_full, nv_full)`` bool — ``[r, c]`` True when the owner of ``r`` supports the owner of ``c``.
    lower: (
        torch.Tensor
    )  #: ``(nv_full, nv_full)`` bool — the transpose of ``upper`` minus the blocks ``upper`` already covers.


def get_dynamics_plan(structure: ModelStructure) -> DynamicsPlan:
    """Memoised `DynamicsPlan` for ``structure``."""
    plan = getattr(structure, "_dynamics_plan", None)
    if plan is None:
        jacobian = jacobian_columns.get_plan(structure)
        device = jacobian.column_owner.device
        support = torch.zeros((structure.njoints, structure.njoints), dtype=torch.bool, device=device)
        for joint in range(structure.njoints):
            support[joint, list(structure.supports[joint])] = True
        support[0, 0] = False  # the universe transmits its children's wrenches only, never its own fext
        column_support = jacobian.support_mask.squeeze(-2)
        upper = column_support.index_select(0, jacobian.column_owner).mT.contiguous()
        plan = DynamicsPlan(jacobian, support, column_support, upper, upper.mT & ~upper)
        object.__setattr__(structure, "_dynamics_plan", plan)
    return plan


def cross_motion(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``ad(v) u`` for twists ``(..., 6)`` in ``[linear, angular]`` layout."""
    v_lin, v_ang = v[..., :3], v[..., 3:]
    u_lin, u_ang = u[..., :3], u[..., 3:]
    return torch.cat(
        (torch.linalg.cross(v_ang, u_lin) + torch.linalg.cross(v_lin, u_ang), torch.linalg.cross(v_ang, u_ang)), dim=-1
    )


def cross_force(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """``ad*(v) f`` for a twist and a wrench ``(..., 6)`` in ``[linear, angular]`` layout."""
    v_lin, v_ang = v[..., :3], v[..., 3:]
    f_lin, f_ang = f[..., :3], f[..., 3:]
    return torch.cat(
        (torch.linalg.cross(v_ang, f_lin), torch.linalg.cross(v_ang, f_ang) + torch.linalg.cross(v_lin, f_lin)), dim=-1
    )


def adjoint_inverse(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """``Ad(T⁻¹)`` of the transform with world rotation ``(..., 3, 3)`` and origin ``(..., 3)``: ``(..., 6, 6)``."""
    rotation_t = rotation.mT
    top = torch.cat((rotation_t, -(rotation_t @ hat_so3(translation))), dim=-1)
    bottom = torch.cat((torch.zeros_like(rotation_t), rotation_t), dim=-1)
    return torch.cat((top, bottom), dim=-2)


@dataclass(frozen=True)
class WorldGeometry:
    """Model-independent part of `WorldKinematics` (memoised on a ``KinematicState``)."""

    columns: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor
    origin: torch.Tensor


@dataclass(frozen=True)
class WorldKinematics:
    """World-frame quantities of one configuration that every dynamics pass reads.

    Twists and wrenches move between the world and a joint frame with the rotation and origin
    directly (two ``3x3`` products and one cross product), never through a ``6x6`` adjoint.

    All world quantities are taken about ``origin``, the first joint's position in each sample,
    not the world origin: lever arms stay of the robot's size, so float32 accuracy does not
    degrade when the robot is far from the world origin. Torques, mass and centroidal matrices
    are invariant to this choice; callers shifting twists or points must use ``origin``.
    """

    columns: torch.Tensor  #: ``(N, 6, nv_full)`` world Jacobian columns about ``origin``.
    rotation: torch.Tensor  #: ``(N, njoints, 3, 3)`` world rotation of every joint frame.
    translation: torch.Tensor  #: ``(N, njoints, 3)`` position of every joint frame relative to ``origin``.
    origin: torch.Tensor  #: ``(N, 3)`` reference point of the world quantities.
    inertia: torch.Tensor  #: ``(njoints, 6, 6)`` or ``(N, njoints, 6, 6)`` spatial inertia in the joint frame.
    packed: (
        torch.Tensor
    )  #: ``(njoints, 10)`` or ``(N, njoints, 10)`` packed body inertias (mass, com, symmetric inertia).

    def to_joint_frames(self, motion: torch.Tensor) -> torch.Tensor:
        """Express world twists ``(N, njoints, 6)`` in each joint's own frame (``Ad(oMi⁻¹)``)."""
        rotation_t = self.rotation.mT
        linear = motion[..., :3] - torch.linalg.cross(self.translation, motion[..., 3:])
        return torch.cat((rotation_t @ linear.unsqueeze(-1), rotation_t @ motion[..., 3:, None]), dim=-2).squeeze(-1)

    def to_joint_frames_force(self, wrench: torch.Tensor) -> torch.Tensor:
        """Express world wrenches ``(N, njoints, 6)`` in each joint's own frame (``Ad(oMi)ᵀ``)."""
        rotation_t = self.rotation.mT
        angular = wrench[..., 3:] - torch.linalg.cross(self.translation, wrench[..., :3])
        return torch.cat((rotation_t @ wrench[..., :3, None], rotation_t @ angular.unsqueeze(-1)), dim=-2).squeeze(-1)

    def to_world_force(self, wrench: torch.Tensor) -> torch.Tensor:
        """Express joint-frame wrenches ``(N, njoints, 6)`` in the world frame (``Ad(oMi)⁻ᵀ``)."""
        linear = (self.rotation @ wrench[..., :3, None]).squeeze(-1)
        angular = (self.rotation @ wrench[..., 3:, None]).squeeze(-1) + torch.linalg.cross(self.translation, linear)
        return torch.cat((linear, angular), dim=-1)

    def world_inertia(self) -> torch.Tensor:
        """Spatial inertia of every body about the world origin, ``(N, njoints, 6, 6)``.

        Assembled from the rotated centre of mass and ``R I_com Rᵀ`` with ``3x3`` products; the
        ``6x6`` congruence with the adjoint costs several times more memory traffic.
        """
        mass, com, sym = self.packed[..., 0], self.packed[..., 1:4], self.packed[..., 4:10]
        com_world = (self.rotation @ com.unsqueeze(-1)).squeeze(-1) + self.translation
        inertia_com = self.rotation @ inertia_about_com(sym) @ self.rotation.mT
        return spatial_inertia_matrix(mass.expand(com_world.shape[:-1]), com_world, inertia_com)


def world_geometry(structure: ModelStructure, transform: torch.Tensor, plan: DynamicsPlan) -> WorldGeometry:
    """Columns, rotations and shifted origins from world joint transforms ``(N, njoints, 3, 4)``."""
    rotation = transform[..., :3]
    origin = transform[:, min(1, structure.njoints - 1), :, 3]
    translation = transform[..., 3] - origin.unsqueeze(-2)
    columns = jacobian_columns.world_column_table_from_matrices(
        structure, rotation, translation, plan.jacobian, device=transform.device, dtype=transform.dtype
    )
    return WorldGeometry(columns, rotation, translation, origin)


def spatial_inertias(values: ModelValues) -> torch.Tensor:
    """``values.spatial_inertias()`` cached on the values object, invalidated by in-place edits.

    Same policy as ``kinematics._fk_lean.placement_matrices``: no cache when the inertias require
    grad, are ``torch.func``-wrapped or inference tensors.
    """
    packed = values.body_inertias
    if packed.requires_grad or torch._C._functorch.is_functorch_wrapped_tensor(packed):
        return packed_inertias_to_6x6(packed)
    if torch.is_inference_mode_enabled() or packed.is_inference():
        return packed_inertias_to_6x6(packed)
    cached = getattr(values, "_spatial_inertias_cache", None)
    if cached is None or cached[0] != packed._version:
        cached = (packed._version, packed_inertias_to_6x6(packed))
        object.__setattr__(values, "_spatial_inertias_cache", cached)
    return cached[1]


def world_kinematics(
    geometry: WorldGeometry, values: ModelValues, batch: tuple[int, ...], structure: ModelStructure
) -> WorldKinematics:
    """Attach the model's inertias (broadcast to the flattened batch ``N``) to a `WorldGeometry`."""
    packed, inertia = values.body_inertias, spatial_inertias(values)
    if packed.ndim > 2:
        count, nj = math.prod(batch), structure.njoints
        packed = packed.broadcast_to(*batch, nj, 10).reshape(count, nj, 10)
        inertia = inertia.broadcast_to(*batch, nj, 6, 6).reshape(count, nj, 6, 6)
    return WorldKinematics(geometry.columns, geometry.rotation, geometry.translation, geometry.origin, inertia, packed)


def apply_inertia(inertia: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    """``I v`` for ``(njoints, 6, 6)`` or ``(N, njoints, 6, 6)`` inertias and ``(N, njoints, 6)`` twists."""
    if inertia.ndim == 3:
        # einsum batches over joints without materialising the broadcast inertia.
        return torch.einsum("jab,njb->nja", inertia, motion)
    return (inertia @ motion.unsqueeze(-1)).squeeze(-1)


def body_velocities(kin: WorldKinematics, plan: DynamicsPlan, v_full: torch.Tensor) -> torch.Tensor:
    """World spatial velocity of every joint frame, ``(N, njoints, 6)``."""
    weighted = kin.columns * v_full.unsqueeze(-2)  # (N, 6, nv_full)
    return (weighted @ plan.column_support.to(weighted.dtype).mT).mT


def inverse_dynamics_full(
    kin: WorldKinematics,
    plan: DynamicsPlan,
    gravity: torch.Tensor,
    v_full: torch.Tensor,
    a_full: torch.Tensor,
    fext: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse dynamics in the world frame.

    :param gravity: ``(N, 6)`` spatial gravity about the world origin; the base accelerates at ``-gravity``.
    :param fext: ``(N, njoints, 6)`` external wrench per joint in the joint frame, or ``None``.
    :returns: ``(tau_full (N, nv_full), V, A, f)`` — world body velocities, accelerations and the
        wrench each joint transmits to its subtree, all ``(N, njoints, 6)``.
    """
    column_support = plan.column_support.to(kin.columns.dtype)
    velocity = body_velocities(kin, plan, v_full)
    owner_velocity = velocity.index_select(-2, plan.jacobian.column_owner)  # (N, nv_full, 6)
    column_rate = cross_motion(owner_velocity, kin.columns.mT).mT  # (N, 6, nv_full)
    weighted = kin.columns * a_full.unsqueeze(-2) + column_rate * v_full.unsqueeze(-2)
    # Gravity is given about the world origin; move its linear part to the reference origin.
    gravity = torch.cat((gravity[:, :3] + torch.linalg.cross(gravity[:, 3:], kin.origin), gravity[:, 3:]), dim=-1)
    acceleration = (weighted @ column_support.mT).mT - gravity.unsqueeze(-2)
    # Body wrenches in the joint frames (the inertia lives there), then back to the world frame.
    local_velocity = kin.to_joint_frames(velocity)
    local_acceleration = kin.to_joint_frames(acceleration)
    momentum = apply_inertia(kin.inertia, local_velocity)
    local_wrench = apply_inertia(kin.inertia, local_acceleration) + cross_force(local_velocity, momentum)
    if fext is not None:
        local_wrench = local_wrench - fext
    wrench = kin.to_world_force(local_wrench)
    # Right-multiplying by the constant support matrix folds the batch into one large mm.
    subtree_wrench = (wrench.mT @ plan.support.to(wrench.dtype)).mT  # (N, njoints, 6)
    column_wrench = subtree_wrench.index_select(-2, plan.jacobian.column_owner)  # (N, nv_full, 6)
    tau_full = (kin.columns.mT * column_wrench).sum(-1)
    return tau_full, velocity, acceleration, subtree_wrench


def momentum_columns(kin: WorldKinematics, plan: DynamicsPlan) -> torch.Tensor:
    """World spatial momentum of the subtree per unit velocity of each column, ``(N, 6, nv_full)``.

    Column ``c`` is ``I^c_{owner(c)} C_c`` with ``I^c`` the composite inertia of the subtree.
    """
    inertia = kin.world_inertia()
    composite = (inertia.flatten(-2, -1).mT @ plan.support.to(inertia.dtype)).mT  # (N, njoints, 36)
    composite = composite.reshape(*inertia.shape).index_select(-3, plan.jacobian.column_owner)  # (N, nv_full, 6, 6)
    return (composite @ kin.columns.mT.unsqueeze(-1)).squeeze(-1).mT


def mass_matrix_full(kin: WorldKinematics, plan: DynamicsPlan) -> torch.Tensor:
    """Joint-space inertia matrix over the full tangent, ``(N, nv_full, nv_full)``."""
    products = kin.columns.mT @ momentum_columns(kin, plan)  # (N, nv_full, nv_full)
    # Two passes over the (large) matrix: keep the entries whose row is on the column's path,
    # mirror them below, zero the rest.
    return torch.where(plan.upper, products, torch.where(plan.lower, products.mT, 0.0))


def centroidal_map_full(kin: WorldKinematics, plan: DynamicsPlan, com: torch.Tensor) -> torch.Tensor:
    """Centroidal momentum matrix over the full tangent, ``(N, 6, nv_full)``.

    World momentum columns moved to the frame with world axes at the centre of mass ``com (N, 3)``
    (given in world coordinates).
    """
    columns = momentum_columns(kin, plan)
    linear = columns[..., :3, :]
    lever = (com - kin.origin).unsqueeze(-1)
    angular = columns[..., 3:, :] - torch.linalg.cross(lever, linear, dim=-2)
    return torch.cat((linear, angular), dim=-2)


__all__ = [
    "DynamicsPlan",
    "WorldGeometry",
    "WorldKinematics",
    "adjoint_inverse",
    "apply_inertia",
    "body_velocities",
    "centroidal_map_full",
    "cross_force",
    "cross_motion",
    "get_dynamics_plan",
    "inverse_dynamics_full",
    "mass_matrix_full",
    "momentum_columns",
    "spatial_inertias",
    "world_geometry",
    "world_kinematics",
]
