# lie/ — SE3/SO3 Lie Group Operations

Functional API (no classes). Plain tensors in, plain tensors out.

## Convention (never deviate)

| Object | Shape | Format |
|--------|-------|--------|
| SE3 pose | `(..., 7)` | `[tx, ty, tz, qx, qy, qz, qw]` — scalar last |
| se3 tangent | `(..., 6)` | `[vx, vy, vz, wx, wy, wz]` — linear first |
| SO3 quaternion | `(..., 4)` | `[qx, qy, qz, qw]` — scalar last |
| SO3 tangent | `(..., 3)` | axis-angle |

## Backend

Pure-PyTorch SE3/SO3 implementation lives in `_torch_native_backend.py`. There is one backend; PyPose was removed in P10-D. Swapping for a future runtime (e.g. Warp) = add a same-shaped backend module and reroute `lie/se3.py` / `lie/so3.py`.

## Numerics

`_torch_native_backend.py` stitches Taylor expansions at `θ → 0` via `torch.where` against a `θ²` cutoff (1e-8) so SE3/SO3 `exp` and `log` stay smooth and differentiable across the singularity. `_matrix_to_quat` uses Shepperd 1978 four-branch selection for stable conversion when `qw → 0`. fp64 `gradcheck` covers `se3_{log,exp,inverse,compose,act}` and `so3_{exp,log}`.

## Modules

- `se3.py` — `compose`, `inverse`, `log`, `exp`, `act`, `adjoint`, `from_axis_angle`, `from_translation`, `normalize`, `sclerp`
- `so3.py` — same pattern + `from_matrix`, `to_matrix`, `slerp`
- `tangents.py` — right/left Jacobians of SO3/SE3 exp/log, `hat`/`vee` maps
- `_torch_native_backend.py` — pure-PyTorch backend (the only one)
