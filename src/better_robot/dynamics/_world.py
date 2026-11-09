"""World-frame dynamics core behind `inverse_dynamics`, `mass_matrix`, `forward_dynamics` and `centroidal_map`.

Everything is expressed in the world frame, so every ancestor or descendant sum over the tree is one matmul with a
constant support matrix (`DynamicsPlan`, a plan on the model) instead of a Python loop over joints:

* ``C (N, 6, nv_full)``: world Jacobian columns, the state's `KinematicState.jacobian_columns`;
* body velocity ``V_i = Σ_{a supports i} C_a v_a``;
* body acceleration ``A_i = -g + Σ_{a supports i} (C_a a_a + (V_a × C_a) v_a)`` (``V_a × C_a`` is the time
  derivative of a world column);
* body wrench ``f_i = I_i A_i + V_i ×* I_i V_i - f_ext,i`` with ``I_i`` the spatial inertia of body ``i``;
* torque ``τ_a = C_a · Σ_{i in subtree(a)} f_i``;
* mass matrix ``M[r, c] = C_r · I^c_{owner(c)} C_c`` for ``r`` on the path of ``c``, ``I^c`` the composite inertia of
  the subtree (one matmul with the support matrix).

Operator counts are independent of the joint count; the support matmuls cost ``O(njoints²)`` flops per batch element,
negligible up to a few hundred joints. Forward dynamics solves ``M ddq = τ - b`` by Cholesky: the articulated-body
recursion is sequential in the tree depth and gains nothing in batched torch for ``nv`` up to a few hundred, and with
mimic joints the projected equations are the only correct form.

The world geometry (`KinematicState.world_geometry`) is the state's column table (about the world origin) moved to a
reference origin with one cross product, with the joint rotations and origins; kinematics memoises it on the state,
keyed by grad mode, so dynamics calls on one state (`forward_dynamics` makes two) derive it once (record 0041). All
world quantities are taken about that origin, the first joint's position in each
sample, so lever arms stay of the robot's size and float32 accuracy does not degrade far from the world origin (G1 at
100 m: mass matrix within 4.8e-6 of float64).

* RNEA forms wrenches in the joint frames, where the inertia is constant; CRBA needs the world inertia (`spatial`'s
  packed inertia rotated by ``R I_c Rᵀ``, then its 6x6 matrix).
* Twists and wrenches change frame with the joint's rotation matrix and origin (two ``3x3`` products and a cross
  product), not through a ``6x6`` adjoint and not through `spatial.transform_motion`/`transform_force`, which take
  quaternion poses: on one CPU thread those cost 2-3x the matrix form (41 joints at batch 4096: 30.5 against 15.7 ms
  per twist change, 18.9 against 7.4 ms per wrench change) plus 24.8 ms to convert the state's matrices to
  quaternions (0.16 ms at batch 1).
* Cross products and inertia matrices are `spatial`'s unchecked ``*_raw`` twins: the public wrappers check every input
  once, and the checked forms add 2.2-3.1 us per call at batch 1.
* Constant matrices multiply from the right (``x.mT @ support``) so torch folds the batch into one ``mm``; a left
  multiply becomes a slow batched ``bmm``.

Cost model: at batch 1 and in backward the operator count dominates. At large CPU batches every operator streams an
``(N, njoints, 6)`` tensor through memory, so a per-joint loop on cache-resident slices is not much slower in forward
mode; the mass matrix ``(N, nv, nv)`` is then the cost (tree200 at 4096 is 655 MB).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..kinematics import KinematicState
from ..kinematics._plans import JacobianPlan, jacobian_plan
from ..model.model import Model
from ..model.topology import build_supports
from ..spatial.algebra import cross_force_raw, cross_motion_raw
from ..spatial.inertia import inertia_matrix_raw, rotational_inertia_raw


@dataclass(frozen=True)
class DynamicsPlan:
    """Constant support masks derived from the topology (a plan on the model)."""

    jacobian: JacobianPlan
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


def build_plan(model: Model, dtype: torch.dtype, device: torch.device) -> DynamicsPlan:
    """Derive the support masks from the topology, on top of the Jacobian plan."""
    jacobian = jacobian_plan(model, dtype=dtype, device=device)
    support = torch.zeros((model.njoints, model.njoints), dtype=torch.bool, device=device)
    for joint, chain in enumerate(build_supports(model.parent_joint_ids)):
        support[joint, list(chain)] = True
    support[0, 0] = False  # the universe transmits its children's wrenches only, never its own external force
    column_support = jacobian.support_mask.squeeze(-2)
    upper = column_support.index_select(0, jacobian.column_owner).mT.contiguous()
    return DynamicsPlan(jacobian, support, column_support, upper, upper.mT & ~upper)


def dynamics_plan(model: Model, *, dtype: torch.dtype, device: torch.device) -> DynamicsPlan:
    """The model's `DynamicsPlan` in ``dtype`` on ``device`` (built once per model, dtype and device)."""
    return model._plan("dynamics", build_plan, dtype=dtype, device=device)


@dataclass(frozen=True)
class WorldKinematics:
    """World-frame quantities of one state that every dynamics pass reads, the batch flattened to ``N``.

    All world quantities are taken about ``origin``, the first joint's position in each sample, not the world origin.
    Torques, mass and centroidal matrices are invariant to this choice; callers shifting twists or points use
    ``origin``.
    """

    batch: tuple[int, ...]  #: the state's batch shape ``B...``; ``N = prod(batch)``.
    plan: DynamicsPlan  #: support masks of the model's tree.
    columns: torch.Tensor  #: ``(N, 6, nv_full)`` world Jacobian columns about ``origin``.
    rotations: torch.Tensor  #: ``(N, njoints, 3, 3)`` world rotation of every joint frame.
    translations: torch.Tensor  #: ``(N, njoints, 3)`` position of every joint frame relative to ``origin``.
    origin: torch.Tensor  #: ``(N, 3)`` reference point of the world quantities.
    inertias: torch.Tensor  #: ``(njoints, 6, 6)`` or ``(N, njoints, 6, 6)`` spatial inertia in each joint frame.
    body_inertias: torch.Tensor  #: ``(njoints, 10)`` or ``(N, njoints, 10)`` packed body inertias.

    def to_joint_frames(self, motions: torch.Tensor) -> torch.Tensor:
        """Express world twists ``(N, njoints, 6)`` in each joint's own frame (``Ad(oMi⁻¹)``)."""
        rotations_t = self.rotations.mT
        linear = motions[..., :3] - torch.linalg.cross(self.translations, motions[..., 3:])
        return torch.cat((rotations_t @ linear.unsqueeze(-1), rotations_t @ motions[..., 3:, None]), dim=-2).squeeze(-1)

    def to_world_force(self, wrenches: torch.Tensor) -> torch.Tensor:
        """Express joint-frame wrenches ``(N, njoints, 6)`` in the world frame (``Ad(oMi)⁻ᵀ``)."""
        linear = (self.rotations @ wrenches[..., :3, None]).squeeze(-1)
        angular = (self.rotations @ wrenches[..., 3:, None]).squeeze(-1) + torch.linalg.cross(self.translations, linear)
        return torch.cat((linear, angular), dim=-1)

    def world_inertias(self) -> torch.Tensor:
        """Spatial inertia of every body about ``origin`` in world axes, ``(N, njoints, 6, 6)``.

        The packed inertia's centre of mass and rotational inertia rotated with ``3x3`` products (``R c + p``,
        ``R I_c Rᵀ``); the ``6x6`` congruence with the adjoint costs several times more memory traffic. The rotated
        inertia is packed from its diagonal and upper triangle: `inertia_from_mass_com_raw`'s symmetrisation and
        per-entry stack cost 14 more operators per call.
        """
        com_world = (self.rotations @ self.body_inertias[..., 1:4, None]).squeeze(-1) + self.translations
        rotational = self.rotations @ rotational_inertia_raw(self.body_inertias) @ self.rotations.mT
        mass = self.body_inertias[..., :1].expand(*com_world.shape[:-1], 1)
        upper = (rotational.diagonal(dim1=-2, dim2=-1), rotational[..., 0, 1:], rotational[..., 1, 2:])
        return inertia_matrix_raw(torch.cat((mass, com_world, *upper), dim=-1))


def world_kinematics(model: Model, state: KinematicState) -> WorldKinematics:
    """World-frame quantities of ``state`` about each sample's first joint.

    The geometry is the state's `KinematicState.world_geometry`, memoised on the state by kinematics (record 0041).
    The inertias come from ``model`` on every call (their ``6x6`` form memoised on the model by grad mode,
    record 0040); a batched model's inertias are broadcast to the flattened batch.
    """
    batch = tuple(state.q.shape[:-1])
    columns, rotations, translations, origin = state.world_geometry
    count, nj = math.prod(batch), model.njoints
    body_inertias = model.body_inertias
    inertias = model._value("spatial_inertias", lambda m: inertia_matrix_raw(m.body_inertias))
    if body_inertias.ndim > 2:
        body_inertias = body_inertias.broadcast_to(*batch, nj, 10).reshape(count, nj, 10)
        inertias = inertias.broadcast_to(*batch, nj, 6, 6).reshape(count, nj, 6, 6)
    return WorldKinematics(
        batch=batch,
        plan=dynamics_plan(model, dtype=state.q.dtype, device=state.q.device),
        columns=columns,
        rotations=rotations,
        translations=translations,
        origin=origin,
        inertias=inertias,
        body_inertias=body_inertias,
    )


def apply_inertias(inertias: torch.Tensor, motions: torch.Tensor) -> torch.Tensor:
    """``I v`` for ``(njoints, 6, 6)`` or ``(N, njoints, 6, 6)`` inertias and ``(N, njoints, 6)`` twists."""
    if inertias.ndim == 3:
        # einsum batches over joints without materialising the broadcast inertia.
        return torch.einsum("jab,njb->nja", inertias, motions)
    return (inertias @ motions.unsqueeze(-1)).squeeze(-1)


def inverse_dynamics_full(
    kin: WorldKinematics,
    gravity: torch.Tensor,
    v_full: torch.Tensor,
    a_full: torch.Tensor,
    external_forces: torch.Tensor | None,
) -> torch.Tensor:
    """Joint torques over the full tangent, ``(N, nv_full)``, by the world-frame recursive Newton-Euler sums.

    :param gravity: ``(N, 6)`` spatial gravity about the world origin; the base accelerates at ``-gravity``.
    :param v_full: ``(N, nv_full)`` velocity; ``a_full`` the acceleration, same shape.
    :param external_forces: ``(N, njoints, 6)`` external wrench per joint in the joint frame, or ``None``.
    """
    plan = kin.plan
    column_support = plan.column_support.to(kin.columns.dtype)
    weighted = kin.columns * v_full.unsqueeze(-2)  # (N, 6, nv_full)
    velocities = (weighted @ column_support.mT).mT  # (N, njoints, 6)
    # Advanced indexing, not index_select: column_owner repeats a joint per column (a free flyer six times), and
    # index_select's CUDA backward adds the repeats with atomics, so its gradients are not bitwise reproducible.
    owner_velocities = velocities[..., plan.jacobian.column_owner, :]  # (N, nv_full, 6)
    column_rates = cross_motion_raw(owner_velocities, kin.columns.mT).mT  # (N, 6, nv_full)
    weighted = kin.columns * a_full.unsqueeze(-2) + column_rates * v_full.unsqueeze(-2)
    # Gravity is given about the world origin; move its linear part to the reference origin.
    gravity = torch.cat((gravity[:, :3] + torch.linalg.cross(gravity[:, 3:], kin.origin), gravity[:, 3:]), dim=-1)
    accelerations = (weighted @ column_support.mT).mT - gravity.unsqueeze(-2)
    # Body wrenches in the joint frames (the inertia lives there), then back to the world frame.
    local_velocities = kin.to_joint_frames(velocities)
    momenta = apply_inertias(kin.inertias, local_velocities)
    local_wrenches = apply_inertias(kin.inertias, kin.to_joint_frames(accelerations))
    local_wrenches = local_wrenches + cross_force_raw(local_velocities, momenta)
    if external_forces is not None:
        local_wrenches = local_wrenches - external_forces
    wrenches = kin.to_world_force(local_wrenches)
    # Right-multiplying by the constant support matrix folds the batch into one large mm.
    subtree_wrenches = (wrenches.mT @ plan.support.to(wrenches.dtype)).mT  # (N, njoints, 6)
    column_wrenches = subtree_wrenches[..., plan.jacobian.column_owner, :]  # (N, nv_full, 6)
    return (kin.columns.mT * column_wrenches).sum(-1)


def momentum_columns(kin: WorldKinematics) -> torch.Tensor:
    """World spatial momentum of the subtree per unit velocity of each column, ``(N, 6, nv_full)``.

    Column ``c`` is ``I^c_{owner(c)} C_c`` with ``I^c`` the composite inertia of the subtree.
    """
    plan = kin.plan
    inertias = kin.world_inertias()
    composite = (inertias.flatten(-2, -1).mT @ plan.support.to(inertias.dtype)).mT  # (N, njoints, 36)
    composite = composite.reshape(*inertias.shape)[..., plan.jacobian.column_owner, :, :]  # (N, nv_full, 6, 6)
    return (composite @ kin.columns.mT.unsqueeze(-1)).squeeze(-1).mT


def mass_matrix_full(kin: WorldKinematics) -> torch.Tensor:
    """Joint-space inertia matrix over the full tangent, ``(N, nv_full, nv_full)``."""
    products = kin.columns.mT @ momentum_columns(kin)  # (N, nv_full, nv_full)
    # Two passes over the (large) matrix: keep the entries whose row is on the column's path,
    # mirror them below, zero the rest.
    return torch.where(kin.plan.upper, products, torch.where(kin.plan.lower, products.mT, 0.0))


def centroidal_map_full(kin: WorldKinematics, com: torch.Tensor) -> torch.Tensor:
    """Centroidal momentum matrix over the full tangent, ``(N, 6, nv_full)``.

    World momentum columns moved to the frame with world axes at the centre of mass ``com (N, 3)`` (given in world
    coordinates).
    """
    columns = momentum_columns(kin)
    linear = columns[..., :3, :]
    lever = (com - kin.origin).unsqueeze(-1)
    angular = columns[..., 3:, :] - torch.linalg.cross(lever, linear, dim=-2)
    return torch.cat((linear, angular), dim=-2)


__all__ = [
    "DynamicsPlan",
    "WorldKinematics",
    "apply_inertias",
    "centroidal_map_full",
    "dynamics_plan",
    "inverse_dynamics_full",
    "mass_matrix_full",
    "momentum_columns",
    "world_kinematics",
]
