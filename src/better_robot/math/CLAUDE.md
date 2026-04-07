# math/ — Lie Group Mathematics

Foundation layer. No internal dependencies — imports only `torch` and `pypose`.

## Public API

```python
from better_robot.math import (
    se3_identity, se3_compose, se3_inverse, se3_log, se3_exp, se3_apply_base,
    adjoint_se3,
    qxyzw_to_wxyz, wxyz_to_qxyzw,
    so3_rotation_matrix, so3_act, so3_from_matrix,
)
```

## Files

### `se3.py` — SE3 Lie group operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `se3_identity` | `() → (7,)` | Zero translation, identity quaternion |
| `se3_compose` | `(a, b) → (7,)` | SE3 composition: `a @ b` |
| `se3_inverse` | `(t,) → (7,)` | SE3 inverse |
| `se3_log` | `(t,) → (6,)` | SE3 → se3 tangent `[tx, ty, tz, rx, ry, rz]` |
| `se3_exp` | `(v,) → (7,)` | se3 tangent → SE3 |
| `se3_apply_base` | `(base, links) → (…, N, 7)` | Apply base transform to all link poses |

All functions support batch dimensions (`...`).

### `so3.py` — SO3 Lie group operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `so3_rotation_matrix` | `(q: (4,)) → (3,3)` | Quaternion `[qx,qy,qz,qw]` → rotation matrix |
| `so3_act` | `(q: (4,), v: (3,)) → (3,)` | Rotate vector `v` by quaternion `q` |
| `so3_from_matrix` | `(R: (3,3)) → (4,)` | Rotation matrix → quaternion `[qx,qy,qz,qw]` |

Used by `algorithms/geometry/robot_collision.py` and other modules that need rotation matrix conversions without importing PyPose directly.

### `spatial.py` — Spatial algebra

| Function | Signature | Description |
|----------|-----------|-------------|
| `adjoint_se3` | `(t,) → (6, 6)` | 6×6 Adjoint matrix for SE3 transform |

Used by the floating-base IK Jacobian. Formula:
```
Ad(T) = [[R,    skew(p)@R],
         [0,    R        ]]
```
where `R` is the rotation matrix and `p` is the translation vector.

### `transforms.py` — Quaternion convention converters

| Function | Signature | Description |
|----------|-----------|-------------|
| `qxyzw_to_wxyz` | `(q,) → tuple` | PyPose `[qx,qy,qz,qw]` → viser `(w,x,y,z)` |
| `wxyz_to_qxyzw` | `(w,x,y,z) → (4,)` | viser `(w,x,y,z)` → PyPose `[qx,qy,qz,qw]` |

## Convention

| Object | Format |
|--------|--------|
| SE3 pose | `[tx, ty, tz, qx, qy, qz, qw]` — PyPose native, scalar-last |
| se3 tangent | `[tx, ty, tz, rx, ry, rz]` — PyPose native |

## Backend

PyPose (`pp.SE3`, `pp.se3`, `pp.SO3`) is used internally in `se3.py` and `so3.py` only. All other modules in the codebase import from `math/se3.py` or `math/so3.py` — they do not import pypose directly. To swap the backend, only `se3.py` and `so3.py` need changing.
