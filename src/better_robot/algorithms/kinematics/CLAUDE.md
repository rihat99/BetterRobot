# algorithms/kinematics/ — Kinematics Algorithms

Free functions for forward kinematics, Jacobians, and kinematic chains.

## Public API

```python
from better_robot.algorithms.kinematics import (
    forward_kinematics,
    compute_jacobian, limit_jacobian, rest_jacobian,
    get_chain,
)
# or via top-level:
import better_robot as br
fk = br.forward_kinematics(model, cfg)
J  = br.compute_jacobian(model, cfg, link_idx, target, pos_w, ori_w)
```

## Functions

### `forward_kinematics(model, cfg, base_pose=None) → Tensor`

`forward.py` — thin delegate to `model.forward_kinematics()`.

- `cfg`: `(*batch, num_actuated_joints)`
- `base_pose`: `(*batch, 7)` optional SE3 base transform
- Returns: `(*batch, num_links, 7)` SE3 poses

### `compute_jacobian(model, cfg, target_link_index, target_pose, pos_weight, ori_weight, base_pose=None, fk=None) → Tensor`

`jacobian.py` — geometric (body-frame) Jacobian of `pose_residual` wrt `cfg`.

- Returns `(6, num_actuated_joints)`
- `target_pose` is kept for API symmetry but unused in computation
- `fk` can be passed to skip re-computing FK (performance optimization)
- **jlog approximation**: `jlog(T_err) ≈ I` — valid for errors < ~30°

**Cross-product formula** (revolute joint `j`):
```python
lin_world = cross(axis_world, p_ee - p_j)
J[:, cfg_idx] = R_ee.T @ [lin_world * pos_w, axis_world * ori_w]
```

### `limit_jacobian(cfg, model) → Tensor`

Returns `(2 * n, n)` diagonal Jacobian of `limit_residual` wrt `cfg`.
- `-1` on diagonal where `cfg < lower_limit`
- `+1` on diagonal where `cfg > upper_limit`
- `0` inside limits

### `rest_jacobian(cfg, rest_pose) → Tensor`

Returns `(n, n)` identity matrix — Jacobian of `rest_residual = cfg - rest_pose`.

### `get_chain(model, link_idx) → list[int]`

`chain.py` — returns actuated joint indices from root to `link_idx` in BFS (root→EE) order.

## Parameter Order Convention

All functions follow `(model, cfg, ...)` — model first, config second. This matches the documented public API `br.compute_jacobian(model, q, ...)`.
