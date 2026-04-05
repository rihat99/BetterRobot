# tasks/ik/ — Inverse Kinematics

Unified IK solver — fixed and floating base through a single `solve_ik()` call.

## Public API

```python
from better_robot.tasks.ik import solve_ik, IKConfig
# or:
from better_robot import solve_ik, IKConfig
```

## `solve_ik`

```python
def solve_ik(
    model: RobotModel,
    targets: dict[str, Tensor],          # {link_name: (7,) SE3 target}
    cfg: IKConfig | None = None,
    initial_cfg: Tensor | None = None,   # warm start; defaults to model._default_cfg
    initial_base_pose: Tensor | None = None,  # None → fixed base; (7,) → floating base
    max_iter: int = 100,
) -> Tensor | tuple[Tensor, Tensor]
```

**Dispatch**: `initial_base_pose is None` → `_solve_fixed` (uses `Problem` + `SOLVERS` registry).
`initial_base_pose` is set → `_run_floating_base_lm` (uses PyPose LM with `_FloatingBaseIKModule`).

**Returns**: Fixed base: `(num_actuated_joints,)` tensor. Floating base: `(base_pose (7,), cfg (n,))`.

## `IKConfig`

```python
@dataclass
class IKConfig:
    pos_weight: float = 1.0        # position component of pose error
    ori_weight: float = 0.1        # orientation component — keep low (see note)
    pose_weight: float = 1.0       # overall pose cost scale
    limit_weight: float = 0.1      # soft joint limit penalty
    rest_weight: float = 0.01      # pull toward model._default_cfg
    base_pos_weight: float = 2.0   # floating-base position regularization
    base_ori_weight: float = 0.5   # floating-base orientation regularization
    jacobian: Literal["autodiff", "analytic"] = "autodiff"
    solver: str = "lm"             # solver name from SOLVERS registry
    solver_params: dict = {}       # kwargs forwarded to solver constructor
```

**`ori_weight=0.1`**: intentionally low. Panda default config has ~173° from identity orientation, which puts SO3 log near singularity. Higher `ori_weight` causes oscillation.

**`solver` / `solver_params`**: applies only to fixed-base path. Floating-base always uses the PyPose LM (`_run_floating_base_lm`).

## Fixed-Base Path (`_solve_fixed`)

Builds a `Problem` with:
1. One `pose_residual` `CostTerm` per target link
2. One `limit_residual` `CostTerm`
3. One `rest_residual` `CostTerm`

Then calls `SOLVERS.get(ik_cfg.solver)(**ik_cfg.solver_params).solve(problem, max_iter)`.

If `ik_cfg.jacobian == "analytic"`, builds an analytical Jacobian function via `_build_fixed_jacobian_fn` using `compute_jacobian`, `limit_jacobian`, and `rest_jacobian`.

## Floating-Base Path (`_run_floating_base_lm`)

Two modes:
- `jacobian="analytic"` → `_run_floating_base_lm_analytic`: custom LM loop with block-structured `(n+6, n+6)` system. Variables: `[cfg (n,) | base_tangent (6,)]`. Base updated via SE3 retraction: `base_new = se3_exp(δ_base) @ base`.
- `jacobian="autodiff"` → uses `_FloatingBaseIKModule` (PyPose `nn.Module`) with `ppo.LevenbergMarquardt`.

**Base Jacobian trick** (analytic mode):
```python
T_ee_local = se3_compose(se3_inverse(base), fk_world[link_idx])
J_base = diag([pos_w]*3 + [ori_w]*3) * adjoint_se3(se3_inverse(T_ee_local)) * pose_weight
```
Reuses FK from `_fb_residual` — no extra FK call.

## Usage Examples

```python
model = br.load_urdf(urdf)

# Fixed-base, default config as warm start
cfg = br.solve_ik(model, targets={"panda_hand": target_pose}, max_iter=20)

# Fixed-base with analytic Jacobian
cfg = br.solve_ik(
    model,
    targets={"panda_hand": target_pose},
    cfg=IKConfig(jacobian="analytic", rest_weight=0.001),
    initial_cfg=cfg,
    max_iter=20,
)

# Floating-base (humanoid)
base_pose, cfg = br.solve_ik(
    model,
    targets={
        "left_rubber_hand":     p_lh,
        "right_rubber_hand":    p_rh,
        "left_ankle_roll_link": p_lf,
        "right_ankle_roll_link": p_rf,
    },
    initial_base_pose=torch.tensor([0., 0., 0.78, 0., 0., 0., 1.]),
    initial_cfg=cfg,
    max_iter=20,
)
```

**Initialization tip**: use robot's FK orientation at default config, not identity:
```python
fk0 = model.forward_kinematics(model._default_cfg)
nat_q = fk0[model.link_index("panda_hand"), 3:7]
target = torch.cat([position, nat_q])
```
