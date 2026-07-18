# Inverse kinematics on Panda

This walkthrough uses Panda from the optional examples package:

```bash
python -m pip install '.[demos]'  # from the BetterRobot source checkout
```

```python
import torch
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)

target = torch.tensor(
    [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],   # [tx ty tz qx qy qz qw]
    dtype=torch.float64,
)
result = br.solve_ik(model, {"body_panda_hand": target})
print(result.q)
print(result.frame_pose("body_panda_hand"))
```

`solve_ik` returns an `IKResult` with the final `q` and convergence
diagnostics, a `fk()` helper that re-runs FK at that point, and a
`frame_pose(name)` shortcut. Check `result.converged`; exhausting the iteration
budget still returns the final candidate rather than relabelling it as a
solution.

## Choosing the optimiser

```python
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig

result = br.solve_ik(
    model,
    {"body_panda_hand": target},
    optimizer_cfg=OptimizerConfig(optimizer="lm", max_iter=200),
    cost_cfg=IKCostConfig(pos_weight=1.0, ori_weight=1.0, limit_weight=0.1),
)
```

Levenberg–Marquardt is the default. Switch to `"gn"` for plain
Gauss–Newton, `"adam"` for first-order, or `"lm_then_adam"` for a two-stage
LM seed followed by Adam refinement. The retained `"lbfgs"` and
`"lm_then_lbfgs"` spellings raise an actionable error: the named-block task
facade does not yet provide batched per-element L-BFGS histories and line
searches.

## Robust kernels

Pose / position / orientation residuals can be reweighted with a robust
kernel:

```python
result = br.solve_ik(
    model,
    {"body_panda_hand": target},
    optimizer_cfg=OptimizerConfig(kernel="huber"),
)
```

Available kernels: `l2`, `huber`, `cauchy`, `tukey`. See
{doc}`/concepts/solver_stack` for how kernels plug into the residual
chain.

## Next

For floating-base robots, see {doc}`floating_base`.
