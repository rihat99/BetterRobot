# costs/ — Residual Functions and Cost Terms

Differentiable cost functions. Depends on `math/`, `models/`, `algorithms/`.

## Design Pattern

Each cost module provides two levels:
1. **Raw residual** `my_residual(cfg, ...) → Tensor` — pure function, used directly or via `functools.partial`
2. **Factory function** `my_cost(...) → CostTerm` — returns a ready-to-use `CostTerm` for use in `Problem`

## Public API

```python
from better_robot.costs import (
    CostTerm,
    pose_residual, pose_cost,
    limit_residual, limit_cost,
    rest_residual, rest_cost, smoothness_residual,
    self_collision_residual, self_collision_cost,
    world_collision_residual, world_collision_cost,
    manipulability_residual, manipulability_cost,
)
```

## `CostTerm` (`cost_term.py`)

```python
@dataclass
class CostTerm:
    residual_fn: Callable[[Tensor], Tensor]    # signature: (x,) → residual vector
    weight: float = 1.0
    kind: Literal["soft", "constraint_leq_zero"] = "soft"
```

`Problem.total_residual(x)` concatenates all `soft` residuals weighted by `weight`.

## `pose.py` — `pose_residual` / `pose_cost`

**`pose_residual(cfg, robot, target_link_index, target_pose, pos_weight=1.0, ori_weight=1.0, base_pose=None) → (6,)`**

Computes `log(T_target⁻¹ @ T_actual(cfg))` in se3 tangent space.

- Weights: `pos_weight` on `[:3]`, `ori_weight` on `[3:]`
- `base_pose`: optional `(*batch, 7)` passthrough to `forward_kinematics`

**`pose_cost(robot, target_link_index, target_pose, pos_weight, ori_weight, pose_weight, base_pose) → CostTerm`**

Factory. Returns `CostTerm` with `functools.partial` already applied.

**Default weights**: `ori_weight=0.1` is intentional — Panda default config has ~173° from identity, which puts SO3 log near singularity. Higher `ori_weight` causes oscillation.

## `limits.py` — `limit_residual` / `limit_cost`

**`limit_residual(cfg, robot, weight=1.0) → (2n,)`**

Returns `[clamp(lo - cfg, min=0), clamp(cfg - hi, min=0)]`.

**Critical:** `torch.clamp(min=0)` must never be removed. Without it, residuals are nonzero even within limits, creating a centering force that overwhelms the pose cost (18 limit dims vs 6 pose dims for a 9-DOF robot). Hard projection in the LM solver (`lower_bounds/upper_bounds`) handles the geometric constraint; the soft cost provides gradients near the boundary.

Stubs (not implemented): `velocity_residual`, `acceleration_residual`, `jerk_residual`.

## `regularization.py` — `rest_residual` / `rest_cost`

**`rest_residual(cfg, rest_pose, weight=1.0) → (n,)`**

Returns `(cfg - rest_pose) * weight`. Keeps robot near a comfortable null-space pose.

**`smoothness_residual(cfg_t, cfg_t1, weight=1.0) → (n,)`**

Penalizes large joint velocity between adjacent configs. Used in trajectory optimization.

## `collision.py` — `self_collision_residual` / `world_collision_residual`

Fully implemented. Uses `RobotCollision` (sphere decomposition) and `colldist_from_sdf` from `algorithms/geometry/`.

**`self_collision_residual(cfg, robot, robot_coll, margin=0.02, weight=1.0) → (num_active_pairs,)`**

Computes per-pair sphere-sphere signed distances, applies `colldist_from_sdf`, returns violation residuals. Higher values = more violation.

**`world_collision_residual(cfg, robot, robot_coll, world_geom, margin=0.02, weight=1.0) → (num_spheres * len(world_geom),)`**

Same pattern for robot-world collisions.

**`self_collision_cost(robot, robot_coll, margin, weight) → CostTerm`**

**`world_collision_cost(robot, robot_coll, world_geom, margin, weight) → CostTerm`**

Factory functions. Return `CostTerm` with `functools.partial` already applied.

## `manipulability.py` — `manipulability_residual` / `manipulability_cost`

Fully implemented. Penalizes low Yoshikawa manipulability.

**`manipulability_residual(cfg, robot, target_link_index, weight=1.0) → (1,)`**

Manipulability = `sqrt(det(J_pos @ J_pos^T))` where `J_pos` is the 3×N position Jacobian.
Returns `weight / manipulability` — minimizing this maximizes manipulability.

**`manipulability_cost(robot, target_link_index, weight=1.0) → CostTerm`**

Factory function.

## Adding a New Cost

```python
def my_residual(cfg: Tensor, param_a, param_b) -> Tensor:
    ...

# Bind extra args with partial, wrap in CostTerm:
from functools import partial
from better_robot.costs import CostTerm

term = CostTerm(
    residual_fn=partial(my_residual, param_a=..., param_b=...),
    weight=1.0,
)
```
