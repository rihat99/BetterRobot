# lie/ — SE3/SO3 Lie Group Operations

Functional API (no classes). Plain tensors in, plain tensors out.

## Convention (never deviate)

| Object | Shape | Format |
|--------|-------|--------|
| SE3 pose | `(..., 7)` | `[tx, ty, tz, qx, qy, qz, qw]` — scalar last |
| se3 tangent | `(..., 6)` | `[vx, vy, vz, wx, wy, wz]` — linear first |
| SO3 quaternion | `(..., 4)` | `[qx, qy, qz, qw]` — scalar last |
| SO3 tangent | `(..., 3)` | axis-angle |

## PyPose Isolation Rule

**All `import pypose` statements live in exactly one file: `_pypose_backend.py`.**
A lint test enforces this. Every other module in the library uses `se3.py` and `so3.py` wrappers. Swapping PyPose for Warp = replace `_pypose_backend.py` only.

## Known PyPose Bug

PyPose's `SE3.Log().backward()` has an incorrect factor-of-2 in the quaternion gradient. This is why `residual_jacobian` in `kinematics/` uses central finite differences instead of `torch.autograd.functional.jacobian`. FD epsilon: `1e-3` for float32, `1e-7` for float64.

## Modules

- `se3.py` — `compose`, `inverse`, `log`, `exp`, `act`, `adjoint`, `from_axis_angle`, `from_translation`, `normalize`, `sclerp`
- `so3.py` — same pattern + `from_matrix`, `to_matrix`, `slerp`
- `tangents.py` — right/left Jacobians of SO3/SE3 exp/log, `hat`/`vee` maps
- `_pypose_backend.py` — sole PyPose import point
