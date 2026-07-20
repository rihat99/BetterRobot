# lie/ — SE3/SO3 Lie Group Operations

Functional API (no classes). Plain tensors in, plain tensors out.

## Convention (never deviate)

| Object | Shape | Format |
|--------|-------|--------|
| SE3 pose | `(..., 7)` | `[tx, ty, tz, qx, qy, qz, qw]` — scalar last |
| se3 tangent | `(..., 6)` | `[vx, vy, vz, wx, wy, wz]` — linear first |
| SO3 quaternion | `(..., 4)` | `[qx, qy, qz, qw]` — scalar last |
| SO3 tangent | `(..., 3)` | axis-angle |

## Implementation

The pure-Torch implementations live directly in `so3.py` and `se3.py`; PyPose was removed in P10-D. Shared quaternion, matrix, hat, and Taylor-cutoff primitives live in `so3.py` and are imported by `se3.py` or `tangents.py` as needed. There is no runtime dispatch here: an optional Warp optimisation replaces a complete FK/RNEA-style pass at that pass's integration boundary, not individual Lie primitives.

## Numerics

`so3.py`, `se3.py`, and `tangents.py` stitch Taylor expansions at `θ → 0` via `torch.where` against dtype-aware `θ²` cutoffs. The full-formula branches use safe dummy inputs in the Taylor region so first- and second-order gradients are finite at `θ = 0`. `_matrix_to_quat` uses Shepperd 1978 four-branch selection for stable conversion when `qw → 0`. fp64 `gradcheck` and `gradgradcheck` cover the public SE3/SO3 exp/log operations and the SO3 right Jacobians at identity and near identity.

## Modules

- `se3.py` — `compose`, `inverse`, `log`, `exp`, `act`, `adjoint`, `from_matrix`, `to_matrix`, `from_axis_angle`, `from_translation`, `normalize`, `sclerp`
- `so3.py` — same pattern + fixed-convention `from_euler` / `to_euler`, `from_matrix`, `to_matrix`, `slerp`
- `alignment.py` — weighted, batched `umeyama` similarity-transform fitting with a proper-rotation reflection fix
- `tangents.py` — right/left Jacobians of SO3/SE3 exp/log, `hat`/`vee` maps
