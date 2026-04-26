# Lie groups and spatial algebra

BetterRobot uses one storage convention everywhere:

| Object | Layout |
|--------|--------|
| SE(3) pose | `(..., 7)` `[tx, ty, tz, qx, qy, qz, qw]` (scalar last) |
| se(3) tangent | `(..., 6)` `[vx, vy, vz, wx, wy, wz]` (linear first) |
| SO(3) quaternion | `(..., 4)` `[qx, qy, qz, qw]` |
| Spatial Jacobian | `(..., 6, nv)` rows ordered `[v_lin (3), ω (3)]` |

The Lie functional API lives in `better_robot.lie.{se3, so3, tangents}`.
Every op is implemented in a pure-PyTorch backend
(`lie/_torch_native_backend.py`) — there is no PyPose dependency since
P10-D.

## Singularity handling

`se3.exp` / `se3.log` use Taylor expansions stitched in via
`torch.where` against a `θ²` cutoff. The autograd graph stays smooth
across `θ = 0`. fp64 `gradcheck` covers every public op (see
`tests/lie/test_torch_backend_gradcheck.py`).

## Quaternion double cover

`so3.log(q)` folds `q` and `−q` into the same tangent (a `qw < 0`
branch flips the sign), so the round-trip `log ∘ exp` is the identity
modulo numerical noise.
