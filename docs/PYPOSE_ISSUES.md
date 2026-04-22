# PyPose Bugs and Workarounds

This document records every known correctness issue BetterRobot has
encountered with [pypose](https://github.com/pypose/pypose), where we
work around them today, and what a permanent fix (patch pypose or
replace it) would have to do. Keep it updated as new problems surface
so a future rewrite has a precise acceptance test.

## 1. Where pypose lives

All `import pypose` statements in the codebase are confined to
`src/better_robot/lie/_pypose_backend.py` (enforced by a lint test in
`tests/test_layer_dependencies.py`). `lie/se3.py` and `lie/so3.py` are
thin wrappers that forward to the backend. To replace pypose: swap the
backend file; nothing else in the library has to change.

The functions that currently route through pypose:

| BetterRobot API | pypose call |
|-----------------|-------------|
| `se3.compose` / `se3.inverse` | `_pp.SE3(...) @ _pp.SE3(...)` / `.Inv()` |
| `se3.log` / `se3.exp` | `.Log()` / `_pp.se3(v).Exp()` |
| `se3.act` | `.Act(p)` |
| `se3.adjoint` / `se3.adjoint_inv` | uses `_pp.SO3(...).matrix()` (forward only) |
| `so3.compose` / `so3.inverse` | `_pp.SO3(...) @ _pp.SO3(...)` / `.Inv()` |
| `so3.log` / `so3.exp` | `.Log()` / `_pp.so3(w).Exp()` |
| `so3.to_matrix` / `so3.from_matrix` | `.matrix()` / `_pp.mat2SO3()` |
| `so3.act` / `so3.adjoint` | `.Act(p)` / `.matrix()` |

`lie/tangents.py` is pure PyTorch — the right/left Jacobians, hat, vee,
and small-angle handling never touch pypose, so their gradients are
reliable.

## 2. The issues — empirically verified

### 2.1 Autograd gradients are manifold-space, not ambient-space

**Symptom.** Every pypose Lie-group operation returns correct values on
the forward pass, but `torch.autograd.grad` or `.backward()` through
them produces gradients that do not match finite differences of the
same function treated as a plain tensor map.

Concretely — at `q_unit = [0.205, 0.308, 0.103, 0.923]`:

```
so3.log(q) Jacobian — autograd vs central FD wrt q (chain includes `q/‖q‖`):

autograd:                                    finite-diff:
[ 0.944,  0.066, -0.338, -0.194]             [ 2.046, -0.091, -0.030, -0.421]
[-0.150,  0.910,  0.195, -0.292]             [-0.091,  1.970, -0.045, -0.632]
[ 0.310, -0.237,  0.965, -0.097]             [-0.030, -0.045,  2.091, -0.210]

norm autograd = 1.76,   norm fd = 3.62  →  ratio ≈ 0.49
```

Not a clean factor of 2: diagonals are scaled by ≈ 0.5, off-diagonals
have wrong sign and magnitude, and the `qw` column of autograd is
identically zero (whereas FD shows it is nonzero).

The same pattern shows up on `so3.exp`, `se3.exp`, `se3.log`,
`se3.compose`, `se3.inverse` — anything that produces a Lie-group
element. On `so3.exp(ω)` at `ω = [0.3, 0.4, 0.2]`:

```
autograd[qw row] = [0, 0, 0]          finite-diff[qw row] = [-0.074, -0.099, -0.049]
autograd diagonal ≈ 0.96                finite-diff diagonal ≈ 0.49  → ratio ≈ 2.0
```

**Root cause (best-effort diagnosis).** PyPose treats SE(3)/SO(3)
tensors as living on a manifold and implements the backward pass in
terms of the tangent space at the output. Gradient components along
directions that leave the manifold (the `qw` direction on the unit
sphere, for example) are projected out. Perturbations that would be
invalid Lie-group elements — non-unit quaternions — have no lineage in
the backward graph. That is a mathematically sensible convention if
the downstream code also works in tangent space, but BetterRobot
residuals that produce plain `(..., 7)` tensors and get composed with
non-Lie ops do not have this guarantee.

**Reproduction script** — sits in this directory for convenience, and
is short enough to paste into a shell:

```python
import torch
from better_robot.lie import so3

q_raw = torch.tensor([0.2, 0.3, 0.1, 0.9], dtype=torch.float64)

def fn(q):
    return so3.log(q / q.norm())

# autograd
q_ag = q_raw.clone().requires_grad_(True)
J_ag = torch.stack(
    [torch.autograd.grad(fn(q_ag)[i], q_ag, retain_graph=True)[0] for i in range(3)]
)

# central FD
eps = 1e-7
J_fd = torch.zeros(3, 4, dtype=torch.float64)
for j in range(4):
    qp = q_raw.clone(); qp[j] += eps
    qm = q_raw.clone(); qm[j] -= eps
    J_fd[:, j] = (fn(qp) - fn(qm)) / (2 * eps)

print("autograd / fd (elementwise):")
print(J_ag / (J_fd + 1e-12))
```

### 2.2 The old framing — "factor-of-2 on quaternion backward"

Earlier documentation (`src/better_robot/lie/CLAUDE.md`,
`src/better_robot/kinematics/jacobian.py:244`,
`src/better_robot/optim/optimizers/adam.py:5`) describes the bug as a
"factor-of-2 error in the quaternion `Log().backward()`". That framing
is **incomplete**: the ≈ 0.5× discrepancy shows up on diagonals of
`so3.log`, but off-diagonals are arbitrarily wrong, and `so3.exp`,
`se3.exp`, `se3.compose`, `se3.inverse` are also affected. Treat the
broader statement above as the authoritative one; the factor-of-2
phrasing is useful as a shorthand for why gradient magnitudes drift
but should not be relied on for correctness analysis.

### 2.3 Downstream consequences

Any optimizer that expects `loss.backward()` to yield the correct
gradient of `loss = ‖r(q)‖²` wrt a `q` tensor that is composed through
pypose operations is going to step in a wrong direction. In practice
we have observed:

- IK: small degradation only — pose residuals use analytic Jacobians
  that never invoke pypose's backward (see §3.1).
- Human-motion trajectory optimization: catastrophic — with
  autograd-only gradients the optimizer reduced acceleration by 7 %
  and *increased* contact-velocity by 11 %. Switching to the analytic
  path below yielded 91 % and 95 % reductions respectively.

## 3. Current workarounds

### 3.1 Residual Jacobians

`src/better_robot/kinematics/jacobian.py::residual_jacobian` falls back
to central finite differences rather than
`torch.autograd.functional.jacobian` specifically to sidestep this
issue (lines 243–246). The existing residuals each implement
`.jacobian()` analytically so the FD path is rarely taken in practice.

FD epsilon: `1e-3` for `float32`, `1e-7` for `float64`.

### 3.2 Trajectory residuals — sparse `J^T r`

`VelocityResidual`, `AccelerationResidual`, `ReferenceTrajectoryResidual`,
and `ContactConsistencyResidual` each implement
`apply_jac_transpose(state, r) -> Tensor` that computes `J^T r` in
`O(T·nv)` or `O(K·T·nv²)` without materialising the dense Jacobian and
without invoking pypose's backward.

`CostStack.gradient(state)` aggregates these contributions into the
loss gradient. The BetterHumanForce `scripts/optimize_motion.py`
closure uses this directly and writes to `delta_v.grad` manually, so
`torch.optim.LBFGS` never calls `loss.backward()`.

### 3.3 Adam optimizer

`src/better_robot/optim/optimizers/adam.py` drives its moments from
`problem.jacobian(x)` → `J^T r` instead of `loss.backward()`, for the
same reason.

### 3.4 FK forward-pass discipline

`src/better_robot/kinematics/forward.py:72-73` explicitly avoids
in-place writes during the topological scan so that pypose's forward
graph stays connected — the forward numerics are correct even if
backward is not. A similar rule holds anywhere we compose many SE(3)
poses through `se3.compose`.

## 4. What a permanent fix needs to provide

Whether we patch pypose upstream, replace `_pypose_backend.py` with a
pure-PyTorch implementation, or port to another Lie-algebra library
(`theseus`, `torch-manifolds`, custom), the replacement must:

1. **Match central FD for every API function on random unit inputs at
   `atol = 1e-10, rtol = 1e-8` in fp64**, for Jacobians of forward
   outputs with respect to ambient tensor components. The existing
   `tests/test_pinocchio/` suite is a close proxy — expand it with
   gradient checks.
2. **Preserve tangent-space semantics where they are actually wanted**
   — the `apply_jac_transpose` + `CostStack.gradient` path depends on
   ambient-space matching because the caller applies the retraction
   itself. A replacement must document what its backward returns and
   either match the ambient convention or offer both paths.
3. **Keep the functional API** (plain `torch.Tensor` in, plain
   `torch.Tensor` out) and the shape conventions:
   - SE3 pose `[tx, ty, tz, qx, qy, qz, qw]`
   - se3 tangent `[vx, vy, vz, wx, wy, wz]`
   - SO3 quaternion `[qx, qy, qz, qw]`
4. **Survive re-exports** — `_pp.SE3(...).tensor()` pattern currently
   returns a plain tensor; replacement must too (no `TensorSubclass`
   drift). See `docs/03_LIE_AND_SPATIAL.md §5` for the
   `__torch_function__` lesson learned.

## 5. Remaining `apply_jac_transpose` coverage

When the replacement lands, the sparse-gradient path can be deleted.
Until then, these residuals *do not* have `apply_jac_transpose` and
therefore fall back to dense `J^T r` (OK for IK sizes, would OOM on
full-length motion trajectories):

- `residuals/pose.py` — `PoseResidual`, `PositionResidual`, `OrientationResidual`
- `residuals/limits.py` — `JointPositionLimit`, `JointVelocityLimit`, `JointAccelLimit`
- `residuals/regularization.py` — `RestResidual`, `NullspaceResidual` (only `ReferenceTrajectoryResidual` is covered)
- `residuals/manipulability.py` — `YoshikawaResidual`
- `residuals/collision.py` — `SelfCollisionResidual`, `WorldCollisionResidual`
- `residuals/temporal.py` — `TimeIndexedResidual` (wrapper; inherits from the inner residual)

If those get used in a trajopt cost stack that runs into Jacobian-size
problems before the pypose replacement lands, add
`apply_jac_transpose` to each. The pattern is consistent: take `r`,
reshape to the residual's output layout, multiply by the relevant
sparse J block per timestep, accumulate.

## 6. Related docs

- `src/better_robot/lie/CLAUDE.md` — pypose isolation rule
- `src/better_robot/kinematics/CLAUDE.md` — why `residual_jacobian` uses FD
- `src/better_robot/lie/se3.py` — inline note at the `log` wrapper
- `src/better_robot/optim/optimizers/adam.py` — inline note at the top of the Adam loop
- `docs/03_LIE_AND_SPATIAL.md` — the isolation design and the plan to swap the backend
