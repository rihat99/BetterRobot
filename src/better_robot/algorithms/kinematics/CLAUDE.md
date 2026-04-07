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
J  = br.compute_jacobian(model, q, ...)
```

## Functions

### `forward_kinematics(model, cfg, base_pose=None) → Tensor`

`forward.py` — FULL FK implementation. Contains the complete BFS traversal over joints, quaternion-based SE3 composition, and optional base transform application. `RobotModel.forward_kinematics()` is a thin delegate that calls this function.

- `cfg`: `(*batch, num_actuated_joints)`
- `base_pose`: `(*batch, 7)` optional SE3 base transform
- Returns: `(*batch, num_links, 7)` SE3 poses

**Implementation detail**: iterates over `model._fk_joint_order` (BFS), computes per-joint SE3 transform (revolute: quaternion from axis-angle; prismatic: translation), accumulates via `se3_compose`.

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

`chain.py` — FULL implementation. Walks parent-link pointers from `link_idx` back to root, collecting actuated joint indices. Returns them in root→EE order. `RobotModel.get_chain()` is a thin delegate to this function.

- Returns actuated joint indices in BFS (root→EE) order
- Raises `ValueError` if `link_idx` is orphaned (unreachable from root)

## Parameter Order Convention

All functions follow `(model, cfg, ...)` — model first, config second. This matches the documented public API `br.compute_jacobian(model, q, ...)`.
