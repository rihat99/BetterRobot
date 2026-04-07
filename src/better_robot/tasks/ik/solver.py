"""Inverse kinematics — unified fixed-base and floating-base solver."""
from __future__ import annotations

import functools

import torch

from ...models.robot_model import RobotModel
from ...models.data import RobotData
from ...math.se3 import (
    se3_identity, se3_log, se3_compose, se3_inverse, se3_exp, se3_normalize
)
from ...math.spatial import adjoint_se3
from ...costs.pose import pose_residual
from ...costs.limits import limit_residual
from ...costs.regularization import rest_residual
from ...costs.collision import self_collision_cost, self_collision_residual
from ...costs.cost_term import CostTerm
from ...algorithms.geometry.robot_collision import RobotCollision
from ...algorithms.geometry import _utils as _geom_utils
from ...algorithms.geometry.distance import colldist_from_sdf
from ...algorithms.kinematics.jacobian import (
    compute_jacobian, limit_jacobian, rest_jacobian,
)
from ...solvers.problem import Problem
from ...solvers.registry import SOLVERS
from .config import IKConfig
from .variable import IKVariable


def solve_ik(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    config: IKConfig | None = None,
    initial_q: torch.Tensor | None = None,
    initial_base_pose: torch.Tensor | None = None,
    max_iter: int = 100,
    robot_coll: "RobotCollision | None" = None,
) -> "RobotData":
    """Solve inverse kinematics for one or more end-effector targets.

    Args:
        model: RobotModel instance.
        targets: {link_name: (7,) SE3 target [tx, ty, tz, qx, qy, qz, qw]}.
        config: IK configuration (weights, position/orientation balance).
        initial_q: (n,) starting joint config. Defaults to model._q_default.
        initial_base_pose: (7,) SE3 initial base pose. When provided, the base
            is also optimized (floating-base mode). Default None = fixed base.
        max_iter: Solver iterations.

    Returns:
        RobotData with .q populated (and .base_pose if floating-base).
    """
    if config is None:
        config = IKConfig()
    if initial_base_pose is not None:
        return _solve_floating(model, targets, config, initial_q, initial_base_pose, max_iter, robot_coll)
    return _solve_fixed(model, targets, config, initial_q, max_iter, robot_coll)


# ---------------------------------------------------------------------------
# Fixed-base path
# ---------------------------------------------------------------------------

def _solve_fixed(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    config: IKConfig,
    initial_q: torch.Tensor | None,
    max_iter: int,
    robot_coll: "RobotCollision | None" = None,
) -> torch.Tensor:
    initial = initial_q.clone() if initial_q is not None else model._q_default.clone()
    q_rest = model._q_default.clone()

    costs = []
    for link_name, target_pose in targets.items():
        link_idx = model.link_index(link_name)
        costs.append(CostTerm(
            residual_fn=functools.partial(
                pose_residual,
                model=model,
                target_link_index=link_idx,
                target_pose=target_pose,
                pos_weight=config.pos_weight,
                ori_weight=config.ori_weight,
            ),
            weight=config.pose_weight,
        ))

    costs += [
        CostTerm(
            residual_fn=functools.partial(limit_residual, model=model),
            weight=config.limit_weight,
        ),
        CostTerm(
            residual_fn=functools.partial(rest_residual, q_rest=q_rest),
            weight=config.rest_weight,
        ),
    ]

    if robot_coll is not None:
        costs.append(self_collision_cost(
            model, robot_coll,
            margin=config.collision_margin,
            weight=config.collision_weight,
        ))

    # Analytic Jacobian does not cover the collision cost — fall back to autodiff when collision is enabled.
    jac_fn = None
    if config.jacobian == "analytic" and robot_coll is None:
        jac_fn = _build_fixed_jacobian_fn(model, targets, config, q_rest)

    problem = Problem(
        variables=initial,
        costs=costs,
        lower_bounds=model.joints.lower_limits.clone(),
        upper_bounds=model.joints.upper_limits.clone(),
        jacobian_fn=jac_fn,
    )
    q_result = SOLVERS.get(config.solver)(**config.solver_params).solve(problem, max_iter=max_iter)
    return model.create_data(q=q_result)


def _build_fixed_jacobian_fn(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    config: IKConfig,
    q_rest: torch.Tensor,
):
    """Build analytical Jacobian for the full IK problem (fixed base)."""
    target_specs = [
        (model.link_index(name), pose) for name, pose in targets.items()
    ]

    def jacobian_fn(q: torch.Tensor) -> torch.Tensor:
        fk = model.forward_kinematics(q)
        rows = []
        for link_idx, target_pose in target_specs:
            J = compute_jacobian(
                model, q, link_idx, target_pose,
                config.pos_weight, config.ori_weight, fk=fk,
            )
            rows.append(J * config.pose_weight)
        rows.append(limit_jacobian(q, model) * config.limit_weight)
        rows.append(rest_jacobian(q, q_rest) * config.rest_weight)
        return torch.cat(rows, dim=0)

    return jacobian_fn


# ---------------------------------------------------------------------------
# Floating-base path (unified via IKVariable)
# ---------------------------------------------------------------------------

def _base_reg_residual(
    base: torch.Tensor,
    default_base: torch.Tensor,
    base_pos_weight: float,
    base_ori_weight: float,
) -> torch.Tensor:
    """6D residual pulling the base toward default_base."""
    T_err = se3_compose(se3_inverse(default_base), base)
    log_err = se3_log(T_err)
    return torch.cat([log_err[:3] * base_pos_weight, log_err[3:] * base_ori_weight])


def _fb_residual(
    q: torch.Tensor,
    base: torch.Tensor,
    model: RobotModel,
    target_link_indices: list[int],
    target_poses: list[torch.Tensor],
    config: IKConfig,
    q_rest: torch.Tensor,
    default_base: torch.Tensor,
    robot_coll: "RobotCollision | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (fk, residual_vector) for floating-base IK."""
    fk = model.forward_kinematics(q, base_pose=base)
    parts = []
    for link_idx, tp in zip(target_link_indices, target_poses):
        T_err = se3_compose(se3_inverse(tp), fk[link_idx])
        log_e = se3_log(T_err)
        parts.append(
            torch.cat([log_e[:3] * config.pos_weight, log_e[3:] * config.ori_weight])
            * config.pose_weight
        )
    parts.append(limit_residual(q, model) * config.limit_weight)
    parts.append(rest_residual(q, q_rest) * config.rest_weight)
    parts.append(
        _base_reg_residual(base, default_base, config.base_pos_weight, config.base_ori_weight)
    )
    if robot_coll is not None:
        parts.append(self_collision_residual(
            q, model, robot_coll,
            margin=config.collision_margin,
            weight=config.collision_weight,
            base_pose=base,
        ))
    return fk, torch.cat(parts)


def _analytic_collision_jacobian(
    q: torch.Tensor,
    base: torch.Tensor,
    fk: torch.Tensor,
    model: RobotModel,
    robot_coll: "RobotCollision",
    config: "IKConfig",
    n: int,
) -> torch.Tensor:
    """Fully analytic geometric collision Jacobian — no autodiff.

    Self-collision distances are invariant to the base pose (both endpoints in
    a pair move rigidly together when the base moves).  Therefore, only the
    joint columns (first ``n``) need to be computed; the 6 base columns are
    exactly zero and are left at zero.

    For each active pair (distance < ``config.collision_margin``):

    1. Compute closest points ``c1`` (on link_i) and ``c2`` (on link_j) and
       the unit direction ``u = (c2 - c1) / |c2 - c1|``.
    2. For each actuated joint ``m`` in the kinematic chain to link_i or link_j:

       *Revolute/continuous*:
           ``∂d / ∂q_m = u · (cross(a_m, c2 - p_m) × [m in chain_j]
                              - cross(a_m, c1 - p_m) × [m in chain_i])``

       *Prismatic*:
           ``∂d / ∂q_m = u · (a_m × [m in chain_j] - a_m × [m in chain_i])``

    where ``a_m`` is the world-frame joint axis and ``p_m`` is the world-frame
    joint origin, both derived from the already-computed FK.

    Returns:
        ``(M, n + 6)`` Jacobian.  Rows for inactive pairs and the 6 base
        columns are all zero.
    """
    from ...math.so3 import so3_act, so3_rotation_matrix

    M = len(robot_coll._active_pairs_i)
    J_coll = q.new_zeros(M, n + 6)

    if M == 0:
        return J_coll

    pairs_i_all = list(robot_coll._active_pairs_i)
    pairs_j_all = list(robot_coll._active_pairs_j)

    # ------------------------------------------------------------------ #
    # 1+2. Compute world endpoints from the PASSED fk (no extra FK call).
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        assert robot_coll._capsule_link_indices is not None
        assert robot_coll._local_points_a is not None
        assert robot_coll._local_points_b is not None
        assert robot_coll._capsule_radii is not None

        cap_li = robot_coll._capsule_link_indices
        cap_fk = fk[cap_li]                                         # (N, 7)
        R_all = so3_rotation_matrix(cap_fk[:, 3:7])                 # (N, 3, 3)
        t_all = cap_fk[:, :3]                                       # (N, 3)
        local_pa = robot_coll._local_points_a.to(dtype=fk.dtype, device=fk.device)
        local_pb = robot_coll._local_points_b.to(dtype=fk.dtype, device=fk.device)
        pa = (R_all @ local_pa.unsqueeze(-1)).squeeze(-1) + t_all  # (N, 3)
        pb = (R_all @ local_pb.unsqueeze(-1)).squeeze(-1) + t_all  # (N, 3)

        # All-pairs distances to find active set.
        pa1_all = pa[pairs_i_all]
        pb1_all = pb[pairs_i_all]
        pa2_all = pa[pairs_j_all]
        pb2_all = pb[pairs_j_all]
        c1_all, c2_all = _geom_utils.closest_segment_to_segment_points(
            pa1_all, pb1_all, pa2_all, pb2_all,
        )
        _, dist_all = _geom_utils.normalize_with_norm(c2_all - c1_all)
        r1_all = robot_coll._capsule_radii[pairs_i_all].to(dtype=fk.dtype, device=fk.device)
        r2_all = robot_coll._capsule_radii[pairs_j_all].to(dtype=fk.dtype, device=fk.device)
        raw_dist_all = dist_all - (r1_all + r2_all)

    active_idx = (raw_dist_all < config.collision_margin).nonzero(as_tuple=True)[0].tolist()
    if not active_idx:
        return J_coll

    pairs_i_list = [pairs_i_all[k] for k in active_idx]
    pairs_j_list = [pairs_j_all[k] for k in active_idx]

    with torch.no_grad():
        c1 = c1_all[active_idx]
        c2 = c2_all[active_idx]
        diff = c2 - c1
        dist_val = dist_all[active_idx]
        u = diff / dist_val.unsqueeze(-1).clamp_min(1e-7)                     # (K,3)
        r1 = r1_all[active_idx]
        r2 = r2_all[active_idx]
        raw_dist = raw_dist_all[active_idx]

    # Pre-compute world joint positions and axes (one pass, ~29 joints).
    # Joint position = parent-link world pose composed with the joint origin.
    j_pos_by_cfg: dict[int, torch.Tensor] = {}
    j_ax_by_cfg: dict[int, torch.Tensor] = {}
    j_rev_by_cfg: dict[int, bool] = {}

    with torch.no_grad():
        for joint_idx in range(len(model._fk_joint_order)):
            cfg_idx = int(model._fk_cfg_indices[joint_idx])
            if cfg_idx < 0:
                continue
            parent_link = int(model._fk_joint_parent_link[joint_idx])
            T_parent = fk[parent_link]
            T_origin = model._fk_joint_origins[joint_idx].to(dtype=fk.dtype, device=fk.device)
            T_j = se3_compose(T_parent, T_origin)
            j_pos_by_cfg[cfg_idx] = T_j[:3]
            j_ax_by_cfg[cfg_idx] = so3_act(T_j[3:7], model._fk_joint_axes[joint_idx].to(dtype=fk.dtype, device=fk.device))
            j_rev_by_cfg[cfg_idx] = model._fk_joint_types[joint_idx] in ("revolute", "continuous")

    # ------------------------------------------------------------------ #
    # 3. Chains for each link involved in active pairs.
    # ------------------------------------------------------------------ #
    cap_link_i = [int(robot_coll._capsule_link_indices[pi].item()) for pi in pairs_i_list]
    cap_link_j = [int(robot_coll._capsule_link_indices[pj].item()) for pj in pairs_j_list]

    # Cache chains keyed by link_idx (avoid re-traversing the same chain).
    chain_cache: dict[int, list[int]] = {}

    def _get_chain(link_idx: int) -> list[int]:
        if link_idx not in chain_cache:
            chain_cache[link_idx] = model.get_chain(link_idx)
        return chain_cache[link_idx]

    # ------------------------------------------------------------------ #
    # 4. Accumulate Jacobian rows for active pairs.
    # ------------------------------------------------------------------ #
    K = len(active_idx)

    # Derivative of residual w.r.t. raw capsule-capsule distance.
    #
    # residual = -colldist_from_sdf(d, margin) * weight
    # colldist_from_sdf:
    #   d >= margin  → 0
    #   0 <= d < margin → -0.5/margin * (d - margin)^2   (≤ 0)
    #   d < 0        → d - 0.5*margin                    (< 0)
    #
    # ∂residual/∂d:
    #   0 <= d < margin → -weight * (d - margin)/margin  = weight*(1 - d/margin)  > 0
    #                     BUT this is wrong sign for the chain-rule:
    #                     we need ∂(−f)/∂d = weight*(d/margin - 1)  < 0
    #   d < 0        → ∂(−f)/∂d = −weight * 1 = −weight           < 0
    #
    # In both active cases dcost_ddist is NEGATIVE (increasing distance → smaller residual).
    margin = config.collision_margin
    margin_f = float(margin)
    with torch.no_grad():
        rd = raw_dist  # already clipped to active pairs (< margin)
        dcost_ddist = torch.where(
            rd < 0,
            -torch.ones_like(rd),                        # linear region: ∂r/∂d = -weight
            rd / margin_f - 1.0,                          # quadratic:    ∂r/∂d = weight*(d/m-1)
        ) * config.collision_weight

    for k_loc in range(K):
        global_k = active_idx[k_loc]
        u_k = u[k_loc]                   # (3,) unit direction
        c1_k = c1[k_loc]                 # (3,) closest pt on link_i
        c2_k = c2[k_loc]                 # (3,) closest pt on link_j
        s_k = dcost_ddist[k_loc]         # scalar chain-rule factor

        chain_i = _get_chain(cap_link_i[k_loc])
        chain_j = _get_chain(cap_link_j[k_loc])
        set_i = set(chain_i)
        set_j = set(chain_j)

        union = set_i.union(set_j)
        for joint_idx in union:
            cfg_idx = int(model._fk_cfg_indices[joint_idx])
            if cfg_idx < 0:
                continue
            p_j = j_pos_by_cfg[cfg_idx]
            a_j = j_ax_by_cfg[cfg_idx]
            rev = j_rev_by_cfg[cfg_idx]

            in_i = joint_idx in set_i
            in_j = joint_idx in set_j

            if rev:
                dc1 = torch.linalg.cross(a_j, c1_k - p_j) if in_i else None
                dc2 = torch.linalg.cross(a_j, c2_k - p_j) if in_j else None
            else:  # prismatic
                dc1 = a_j if in_i else None
                dc2 = a_j if in_j else None

            contrib = (
                (torch.zeros(3, dtype=q.dtype, device=q.device) if dc2 is None else dc2)
                - (torch.zeros(3, dtype=q.dtype, device=q.device) if dc1 is None else dc1)
            )
            # ∂d/∂q_m = u · contrib; then chain rule with s_k
            J_coll[global_k, cfg_idx] = s_k * (u_k * contrib).sum()

    return J_coll


def _solve_floating(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    config: IKConfig,
    initial_q: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
    robot_coll: "RobotCollision | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified floating-base IK using IKVariable and our LM solver.

    Variable: x = [q (n,), base_tangent (6,)]
    base_tangent is a se3 offset applied to current base via retraction.
    """
    if config.jacobian == "analytic":
        # Analytic path handles both the no-collision and collision cases.
        # When collision is enabled a sparse Jacobian covers only active pairs.
        return _solve_floating_analytic(model, targets, config, initial_q, initial_base_pose, max_iter, robot_coll)
    return _solve_floating_autodiff(model, targets, config, initial_q, initial_base_pose, max_iter, robot_coll)


def _solve_floating_autodiff(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    config: IKConfig,
    initial_q: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
    robot_coll: "RobotCollision | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Floating-base IK with autodiff Jacobian, using IKVariable retraction."""
    base = initial_base_pose.clone().float()
    q = (
        initial_q.clone().float()
        if initial_q is not None
        else model._q_default.clone().float()
    )
    q_rest = model._q_default.clone().float()
    lo = model.joints.lower_limits.float()
    hi = model.joints.upper_limits.float()

    target_link_indices = [model.link_index(name) for name in targets]
    target_poses = [tp.float() for tp in targets.values()]
    default_base = base.clone()
    n = model.joints.num_actuated_joints
    lam = 1e-4
    factor = 2.0
    reject = 16

    for _ in range(max_iter):
        _, r = _fb_residual(q, base, model, target_link_indices, target_poses, config, q_rest, default_base, robot_coll)

        # Build residual as function of flat [q, base_tangent]
        def total_res_flat(x: torch.Tensor) -> torch.Tensor:
            q_x = x[:n]
            delta_base = x[n:]
            new_base = se3_normalize(se3_compose(se3_exp(delta_base), base))
            _, res = _fb_residual(q_x, new_base, model, target_link_indices, target_poses, config, q_rest, default_base, robot_coll)
            return res

        x0 = torch.cat([q, torch.zeros(6, dtype=q.dtype, device=q.device)])
        J = torch.func.jacrev(total_res_flat)(x0)
        JtJ = J.T @ J
        Jtr = J.T @ r

        for _ in range(reject):
            A = JtJ + lam * torch.eye(n + 6, dtype=q.dtype, device=q.device)
            delta = torch.linalg.solve(A, -Jtr)
            q_new = (q + delta[:n]).clamp(lo, hi)
            base_new = se3_normalize(se3_compose(se3_exp(delta[n:]), base))
            _, r_new = _fb_residual(q_new, base_new, model, target_link_indices, target_poses, config, q_rest, default_base, robot_coll)
            if r_new.norm() <= r.norm():
                q, base = q_new, base_new
                lam = max(lam / factor, 1e-7)
                break
            lam = min(lam * factor, 1e7)

    return model.create_data(q=q.detach(), base_pose=base.detach())


def _solve_floating_analytic(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    config: IKConfig,
    initial_q: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
    robot_coll: "RobotCollision | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Floating-base IK with analytic Jacobian (optionally with sparse collision).

    When *robot_coll* is provided, a sparse collision Jacobian is appended:
    only pairs within ``config.collision_margin`` receive non-zero rows, cutting
    the number of autodiff backward passes from O(all_pairs) to O(active_pairs).
    """
    base = initial_base_pose.clone().float()
    default_base = initial_base_pose.clone().float()
    q = (
        initial_q.clone().float()
        if initial_q is not None
        else model._q_default.clone().float()
    )
    q_rest = model._q_default.clone().float()
    lo = model.joints.lower_limits.float()
    hi = model.joints.upper_limits.float()

    target_link_indices = [model.link_index(name) for name in targets]
    target_poses = [tp.float() for tp in targets.values()]

    n = model.joints.num_actuated_joints
    lam = 1e-4
    factor = 2.0
    reject = 16
    w_vec = q.new_tensor([config.pos_weight] * 3 + [config.ori_weight] * 3)
    w_base_vec = q.new_tensor(
        [config.base_pos_weight] * 3 + [config.base_ori_weight] * 3
    )

    for _ in range(max_iter):
        fk, r = _fb_residual(
            q, base, model, target_link_indices, target_poses, config, q_rest,
            default_base, robot_coll,
        )

        J_rows = []
        for link_idx, target_pose in zip(target_link_indices, target_poses):
            J_joints = compute_jacobian(
                model, q, link_idx, target_pose,
                config.pos_weight, config.ori_weight, base_pose=base, fk=fk,
            ) * config.pose_weight
            T_ee_local = se3_compose(se3_inverse(base), fk[link_idx])
            Ad = adjoint_se3(se3_inverse(T_ee_local))
            J_base = w_vec.unsqueeze(1) * Ad * config.pose_weight
            J_rows.append(torch.cat([J_joints, J_base], dim=1))

        J_lim = torch.cat([
            limit_jacobian(q, model) * config.limit_weight,
            torch.zeros(2 * n, 6, dtype=q.dtype, device=q.device),
        ], dim=1)
        J_rest = torch.cat([
            rest_jacobian(q, q_rest) * config.rest_weight,
            torch.zeros(n, 6, dtype=q.dtype, device=q.device),
        ], dim=1)
        J_base_reg = torch.cat([
            torch.zeros(6, n, dtype=q.dtype, device=q.device),
            torch.diag(w_base_vec),
        ], dim=1)
        J_parts = [*J_rows, J_lim, J_rest, J_base_reg]

        if robot_coll is not None:
            J_coll = _analytic_collision_jacobian(q, base, fk, model, robot_coll, config, n)
            J_parts.append(J_coll)

        J = torch.cat(J_parts, dim=0)

        JtJ = J.T @ J
        Jtr = J.T @ r

        for _ in range(reject):
            A = JtJ + lam * torch.eye(n + 6, dtype=q.dtype, device=q.device)
            delta = torch.linalg.solve(A, -Jtr)

            q_new = (q + delta[:n]).clamp(lo, hi)
            base_new = se3_normalize(se3_compose(se3_exp(delta[n:]), base))

            _, r_new = _fb_residual(
                q_new, base_new, model, target_link_indices, target_poses, config, q_rest,
                default_base, robot_coll,
            )
            if r_new.norm() <= r.norm():
                q, base = q_new, base_new
                lam = max(lam / factor, 1e-7)
                break
            lam = min(lam * factor, 1e7)

    return model.create_data(q=q.detach(), base_pose=base.detach())
