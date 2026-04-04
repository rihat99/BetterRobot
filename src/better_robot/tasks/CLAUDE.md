# tasks/ — High-Level IK and Optimization APIs

## Public API

| Function | Description | Returns |
|----------|-------------|---------|
| `solve_ik` | Unified IK — fixed or floating base | `Tensor` or `(Tensor, Tensor)` |

## `solve_ik` (`_ik.py`)

```python
cfg = solve_ik(
    robot,
    targets={"panda_hand": pose},       # {link_name: (7,) SE3}
    cfg=IKConfig(),                     # optional: tune weights
    initial_cfg=cfg,                    # warm start (fixed base)
    max_iter=20,
)

# Floating base (humanoid)
base_pose, cfg = solve_ik(
    robot,
    targets={
        "left_rubber_hand":     p_lh,
        "right_rubber_hand":    p_rh,
        "left_ankle_roll_link": p_lf,
        "right_ankle_roll_link": p_rf,
    },
    initial_base_pose=base_pose,        # triggers floating-base path
    initial_cfg=cfg,
    max_iter=20,
)
```

**Dispatch logic:** `initial_base_pose is None` → fixed-base path (uses `Problem`/`CostTerm`/LM); not None → floating-base path (uses `_FloatingBaseIKModule`).

Fixed base returns `(num_actuated_joints,)` tensor.
Floating base returns `(base_pose (7,), cfg (n,))`.

## `IKConfig` (`_config.py`)

```python
@dataclass
class IKConfig:
    pos_weight: float = 1.0    # position component of pose error
    ori_weight: float = 0.1    # orientation component of pose error
    pose_weight: float = 1.0   # overall pose cost scale
    limit_weight: float = 0.1  # soft joint limit penalty
    rest_weight: float = 0.01  # pull toward default config
```

All weights passed to both fixed-base and floating-base paths. Low `ori_weight=0.1` is intentional — high orientation weight causes LM to oscillate near the SO3 log singularity at 180° rotation.

## Internal structure

- `_ik.py`: `solve_ik` (public) + `_solve_fixed` (private)
- `_floating_base_ik.py`: `_FloatingBaseIKModule` + `_run_floating_base_lm` (both private)
- `_config.py`: `IKConfig` dataclass
- `solve_trajopt` and `retarget` are stubs (not yet implemented)

## Weight Defaults

All defaults give same behavior as before the unification:
- `pose_weight=1.0` with `pos_weight=1.0, ori_weight=0.1`
- `limit_weight=0.1` (soft penalty; hard clamp enforced separately in LM loop)
- `rest_weight=0.01` (keeps robot near `robot._default_cfg`)
