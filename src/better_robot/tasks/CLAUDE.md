# tasks/ — High-Level Task APIs

Top-level user-facing APIs. Depends on all lower layers.

## Public API

```python
from better_robot.tasks import solve_ik, IKConfig, solve_trajopt, TrajOptConfig, retarget, RetargetConfig
# or via top-level:
import better_robot as br
cfg = br.solve_ik(model, targets={"panda_hand": pose})
```

## Subpackages

### `ik/` — Inverse kinematics

See `ik/CLAUDE.md`.

### `trajopt/` — Trajectory optimization (stub)

`solve_trajopt(model, targets, cfg, initial_traj, max_iter)` — raises `NotImplementedError`.

### `retarget/` — Motion retargeting (stub)

`retarget(model, source_motion, cfg, max_iter)` — raises `NotImplementedError`.
