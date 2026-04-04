# tasks/ — High-Level IK and Optimization APIs

## Functions

| Function | Base | Targets | Returns |
|----------|------|---------|---------|
| `solve_ik` | fixed | single link | `(n,)` cfg |
| `solve_ik_multi` | fixed | dict of links | `(n,)` cfg |
| `solve_ik_floating_base` | floating | dict of links | `(7,) base_pose, (n,) cfg` |

## `solve_ik_multi` (`_ik.py`)

Builds one `CostTerm(pose_residual(...))` per target, plus shared `limit_residual`
and `rest_residual`. Uses existing `Problem`/`LevenbergMarquardt`. Fixed base only.
Always uses LM solver. No collision support (use `solve_ik` for that).

```python
cfg = solve_ik_multi(
    robot,
    targets={"panda_hand": pose_a, "panda_link6": pose_b},
    max_iter=50,
)
```

## `solve_ik_floating_base` (`_floating_base_ik.py`)

Uses a dedicated `_FloatingBaseIKModule(nn.Module)` with:
- `self.base = pp.Parameter(pp.SE3(initial_base))` — PyPose Lie parameter
- `self.cfg = nn.Parameter(initial_cfg)` — joint angles

After each LM step: joints clamped to limits, base quaternion renormalized via
`copy_` (not direct `.data` slice assignment — `pp.Parameter` is a LieTensor).
`vectorize=True` works because FK branching is on fixed joint-type data.

```python
base_pose, cfg = solve_ik_floating_base(
    robot,
    targets={
        "left_rubber_hand":      p_lh,
        "right_rubber_hand":     p_rh,
        "left_ankle_roll_link":  p_lf,
        "right_ankle_roll_link": p_rf,
    },
    initial_base_pose=torch.tensor([0., 0., 0.78, 0., 0., 0., 1.]),
    initial_cfg=cfg,  # warm start
    max_iter=20,
)
```

Warm-start loop: pass previous `base_pose` and `cfg` as `initial_*` each frame.

## Weight Defaults

All three functions share the same defaults:
- `"pose": 1.0` with `pos_weight=1.0, ori_weight=0.1`
- `"limits": 0.1` (soft penalty; hard clamp enforced separately)
- `"rest": 0.01` (keeps robot near `robot._default_cfg`)

Low `ori_weight=0.1` is intentional — high orientation weight causes LM to oscillate
near the SO3 log singularity at 180° rotation (affects humanoids in natural pose).
