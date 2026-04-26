# 03 · Replace PyPose with a pure-PyTorch SE3/SO3 backend

★★★ **Foundational.** The PyPose autograd issue
([PYPOSE_ISSUES.md](../../status/PYPOSE_ISSUES.md)) is the single most
fragile thing in the library. Every residual ships defensive code to
work around it. The fix is overdue.

## Problem

PyPose's `SE3.Log().backward()` and friends return gradients in the
*manifold tangent space*, not the *ambient* `(..., 7)` space. The
projection drops components in directions that leave the manifold —
notably the `qw` component of an unnormalised quaternion. For
BetterRobot, where residuals are composed with non-Lie ops and the
caller applies its own retraction, this means PyPose's backward is
wrong for our autograd flow.

Concrete consequences from
[PYPOSE_ISSUES.md §2.3](../../status/PYPOSE_ISSUES.md):

- `kinematics.residual_jacobian` uses **central finite differences**
  by default (eps `1e-3` for fp32, `1e-7` for fp64) to dodge PyPose's
  backward.
- `apply_jac_transpose` exists on temporal residuals to compute
  `J^T r` without ever invoking `loss.backward()`.
- `optim.optimizers.adam` reads gradients from `problem.jacobian(x)`
  instead of `loss.backward()`.
- Every analytic residual (PoseResidual, OrientationResidual, …)
  carries an analytic `.jacobian()` partly because PyPose's autograd
  cannot be trusted there.

This is real defensive code mass. It is also a continuing tax on every
new residual, every new dynamics function, and every contributor.

## Goal

A pure-PyTorch SE3/SO3 backend that:

1. Implements every function in `LieOps` (Proposal 02).
2. Passes `torch.autograd.gradcheck` against central FD at fp64 with
   `atol=1e-8, rtol=1e-6` on random unit-quaternion inputs.
3. Matches the current PyPose-backed forward pass to fp32 ulp
   precision on the existing test suite.
4. Has identical batch / device semantics.
5. Has lower or equal latency than PyPose for the operations in
   `LieOps` (no, this is not "for free" — see Tradeoffs).

Once this lands:

- `kinematics.residual_jacobian` switches to autodiff by default; FD
  becomes opt-in.
- The `apply_jac_transpose` machinery in temporal residuals can be
  retired.
- Adam, LBFGS, and future optimisers can call `loss.backward()`
  without surprise.

## Where it goes

This is a **single-file replacement** because [Proposal 02] makes
the backend dispatch explicit.

```
backends/torch_native/lie_ops.py     # today: routes to lie/_pypose_backend
                                     # tomorrow: routes to lie/_torch_backend
lie/_pypose_backend.py               # kept; reachable via env var for diff-testing
lie/_torch_backend.py                # NEW — pure PyTorch implementation
```

A `BR_LIE_BACKEND=pypose|torch` env var (default `torch` once green,
`pypose` until then) lets us A/B-test on real workloads.

## Implementation sketch

The math is well-known and short. The full module is ~250 lines of
PyTorch. Outline:

> **Caveat — illustrative, not authoritative.** The formula sketch
> below is a high-level layout, not a copy-pasteable reference.
> SE3 `log` / `exp` near the singular limits (`θ → 0`, `θ → π`) and
> the SO3 quaternion-log sign branch are easy to get wrong. Before
> committing the implementation:
>
> 1. Re-derive every formula from a primary source — Murray, Li &
>    Sastry §A.5; Barfoot §7.1; Sola et al. *Micro-Lie theory* §1.
> 2. Lock the result with `torch.autograd.gradcheck` at fp64 and a
>    central-FD parity test against Pinocchio on randomised batches.
> 3. Add explicit tests at `θ ∈ {0, π/2, π − 1e-6}` for both `log`
>    and `exp`. Do not trust a single-test pass at `θ = π/4`.
>
> An independent re-derivation caught a bug in an earlier draft of
> this sketch — the SE3 left-Jacobian `V` was written
> `V = a · I + b · W + c · W²`, which is wrong. The correct form is
> `V = I + b · W + c · W²` with
> `b = (1 − cos θ)/θ²`, `c = (θ − sin θ)/θ³`. The corrected snippet
> below reflects that fix; the lesson is that this code path needs
> derivation discipline, not formula transcription.

```python
# src/better_robot/lie/_torch_backend.py
"""Pure-PyTorch SE3/SO3 backend. Replaces _pypose_backend.py.

Storage convention (unchanged):
    SE3 pose : (..., 7) [tx, ty, tz, qx, qy, qz, qw] (scalar last)
    SO3 quat : (..., 4) [qx, qy, qz, qw]
    se3 tan  : (..., 6) [vx, vy, vz, wx, wy, wz] (linear first)
    so3 tan  : (..., 3) [wx, wy, wz]
"""
import torch
from .tangents import hat_so3   # already pure PyTorch

EPS = 1e-7

# ==== quaternion (xyzw) primitives ====

def _q_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, scalar-last."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dim=-1)

def _q_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)

def _q_act(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Active rotation of a 3-vector by a unit quaternion. Avoids
    matrix construction; differentiable."""
    qv, qw = q[..., :3], q[..., 3:4]                 # (..., 3), (..., 1)
    t = 2.0 * torch.cross(qv, p, dim=-1)
    return p + qw * t + torch.cross(qv, t, dim=-1)

def _q_log(q: torch.Tensor) -> torch.Tensor:
    """SO3 log via stable atan2-based formula."""
    qv = q[..., :3]
    qw = q[..., 3:4].clamp(-1.0, 1.0)                # numerical safety
    sin_half = qv.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, qw.abs())
    coeff = torch.where(
        sin_half > EPS,
        angle / sin_half,
        2.0 + sin_half.pow(2) / 3.0,                 # Taylor expansion
    )
    return qv * coeff * qw.sign()

def _q_exp(omega: torch.Tensor) -> torch.Tensor:
    """SO3 exp from axis-angle, returning (..., 4) [qx, qy, qz, qw]."""
    angle = omega.norm(dim=-1, keepdim=True)
    half = angle / 2.0
    sinc_half = torch.where(
        angle > EPS,
        torch.sin(half) / angle,
        0.5 - angle.pow(2) / 48.0,                  # Taylor
    )
    qv = omega * sinc_half
    qw = torch.cos(half)
    return torch.cat([qv, qw], dim=-1)

# ==== SE3 ops ====

def se3_compose(a, b):
    ta, qa = a[..., :3], a[..., 3:7]
    tb, qb = b[..., :3], b[..., 3:7]
    t_new = ta + _q_act(qa, tb)
    q_new = _q_compose(qa, qb)
    return torch.cat([t_new, q_new], dim=-1)

def se3_inverse(t):
    tt, q = t[..., :3], t[..., 3:7]
    q_inv = _q_conjugate(q)
    t_inv = -_q_act(q_inv, tt)
    return torch.cat([t_inv, q_inv], dim=-1)

def se3_log(T):
    """SE3 log via Murray, Li, Sastry §A.5."""
    tt, q = T[..., :3], T[..., 3:7]
    omega = _q_log(q)
    angle = omega.norm(dim=-1, keepdim=True)
    # V_inv(omega) — closed form
    half = angle / 2.0
    cot = torch.where(
        angle > EPS,
        half / torch.tan(half).clamp_min(EPS),
        1.0 - angle.pow(2) / 12.0,
    )
    Wx = hat_so3(omega)
    V_inv = torch.eye(3, dtype=T.dtype, device=T.device) - 0.5 * Wx + (
        (1.0 - cot[..., None]) / angle.pow(2)[..., None].clamp_min(EPS)
    ) * (Wx @ Wx)
    v = (V_inv @ tt.unsqueeze(-1)).squeeze(-1)
    return torch.cat([v, omega], dim=-1)

def se3_exp(xi):
    """SE3 exp via the closed form for the left-Jacobian V(θ).

    Reference: Barfoot, *State Estimation for Robotics* §7.1, eq. 7.85.
    V(ω) = I + b · W + c · W²,
        b = (1 − cos θ) / θ²,
        c = (θ − sin θ) / θ³.
    The Taylor expansions cover θ → 0 (and the b/c series can be
    re-checked against Sola et al. §1.D).
    """
    v, omega = xi[..., :3], xi[..., 3:6]
    q = _q_exp(omega)
    angle = omega.norm(dim=-1, keepdim=True)
    Wx = hat_so3(omega)
    # b(θ) = (1 − cos θ) / θ². Taylor at θ = 0: 1/2 − θ²/24 + θ⁴/720 − …
    b = torch.where(
        angle > EPS,
        (1.0 - torch.cos(angle)) / angle.pow(2),
        0.5 - angle.pow(2) / 24.0,
    )
    # c(θ) = (θ − sin θ) / θ³. Taylor at θ = 0: 1/6 − θ²/120 + θ⁴/5040 − …
    c = torch.where(
        angle > EPS,
        (angle - torch.sin(angle)) / angle.pow(3),
        1.0 / 6.0 - angle.pow(2) / 120.0,
    )
    I3 = torch.eye(3, dtype=xi.dtype, device=xi.device)
    V = I3 + b[..., None] * Wx + c[..., None] * (Wx @ Wx)
    t = (V @ v.unsqueeze(-1)).squeeze(-1)
    return torch.cat([t, q], dim=-1)

def se3_act(T, p):
    return T[..., :3] + _q_act(T[..., 3:7], p)

def se3_adjoint(T):
    """6x6 spatial adjoint, linear-first row order."""
    t, q = T[..., :3], T[..., 3:7]
    R = _quat_to_R(q)
    upper = torch.cat([R, hat_so3(t) @ R], dim=-1)
    lower = torch.cat([torch.zeros_like(R), R], dim=-1)
    return torch.cat([upper, lower], dim=-2)

def se3_adjoint_inv(T):
    t, q = T[..., :3], T[..., 3:7]
    R = _quat_to_R(q)
    Rt = R.transpose(-1, -2)
    upper = torch.cat([Rt, -(Rt @ hat_so3(t))], dim=-1)
    lower = torch.cat([torch.zeros_like(R), Rt], dim=-1)
    return torch.cat([upper, lower], dim=-2)

def se3_normalize(T):
    return torch.cat([T[..., :3], T[..., 3:7] / T[..., 3:7].norm(dim=-1, keepdim=True)], dim=-1)

def _quat_to_R(q):
    qx, qy, qz, qw = q.unbind(-1)
    R = torch.stack([
        torch.stack([1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)], dim=-1),
        torch.stack([2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)], dim=-1),
        torch.stack([2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)], dim=-1),
    ], dim=-2)
    return R
```

The Taylor expansions cover the singular limit `angle → 0`. The
boundary `angle ≈ π` uses the same formulas with `clamp_min` on the
denominator; tests in `tests/lie/test_singularities.py` lock the
behaviour at `angle ∈ {0, π/2, π - 1e-6}`.

## Validation: gradcheck and FD parity

Three test suites lock correctness:

```python
# tests/lie/test_torch_backend_gradcheck.py — fp64 only
@pytest.mark.parametrize("fn", [se3_log, se3_exp, se3_compose, se3_inverse, se3_act])
def test_gradcheck(fn):
    inputs = make_random_unit_inputs(seed=0, dtype=torch.float64)
    assert torch.autograd.gradcheck(fn, inputs, atol=1e-8, rtol=1e-6)
```

```python
# tests/lie/test_torch_backend_fd_parity.py
def test_log_jacobian_matches_central_fd():
    q = sample_random_unit_quat()
    J_ag = torch.autograd.functional.jacobian(_q_log, q)
    J_fd = central_fd(_q_log, q, eps=1e-7)
    assert torch.allclose(J_ag, J_fd, atol=1e-7)
```

```python
# tests/lie/test_torch_backend_value_parity.py
def test_value_matches_pypose_backend():
    """Forward pass exact-equality (modulo 1ulp) with the legacy backend."""
    set_backend_lie("pypose")
    ref = se3_compose(a, b)
    set_backend_lie("torch")
    got = se3_compose(a, b)
    assert torch.allclose(got, ref, atol=1e-6)
```

The same tests cover `tangents.py` — but those are *already* pure
PyTorch, so they pass without changes.

## What gets retired (and what stays first-class)

Once the torch backend is the default and gradcheck-green:

| Defensive code | Status after replacement |
|----------------|-------------------------|
| `kinematics.residual_jacobian` defaults to central FD | Default flips to `torch.func.jacrev`; FD becomes opt-in (`JacobianStrategy.FINITE_DIFF`) for hand-validating analytic Jacobians during development. |
| `optim/optimizers/adam.py` reads `problem.jacobian(x)` | Optional — `loss.backward()` is now safe; both paths must be supported because matrix-free workloads (long-horizon trajopt) prefer the Jacobian-product path. |
| Temporal residuals' `apply_jac_transpose` | **Kept and first-class.** Even with correct Lie autograd, `J^T r` on long trajectories is the memory-bounded operation; materialising `J` of shape `(dim, T·nv)` does not scale. See [Proposal 16](16_optim_wiring_and_matrix_free.md). |
| Analytic `.jacobian()` on residuals | **Kept.** Sparsity, banded structure, and per-knot block patterns are first-class for solver assembly; autograd is the fallback, not the replacement. |
| `lie/_pypose_backend.py` | Kept under `BR_LIE_BACKEND=pypose` for diff-testing; deletable in v1.2. |
| Note in `lie/CLAUDE.md` about the "factor-of-2 bug" | Replaced by "this backend uses pure PyTorch; the legacy PyPose backend is reachable via env var for regression testing". |

The deletion of `_pypose_backend.py` itself is a separate decision
(we may keep it as a regression oracle).

The point of this proposal is to make autograd *safe* — so users
who reach for `loss.backward()` get the right answer — not to retire
the analytic and matrix-free paths the library was already careful
to build. The two paths coexist; the choice is per-workload.

## Performance expectation

PyPose's SE3/SO3 ops are not particularly fast — they go through
Python wrappers and a `LieTensor.__torch_function__` dispatch. The
pure-PyTorch implementation is short and stays inside torch ops that
inductor already knows how to fuse. Expectation:

- Forward pass: within 1.2× of PyPose, often faster, on CPU and CUDA.
- Backward pass: dramatically faster than PyPose's tangent-space
  projection because we avoid the indirection.
- `torch.compile`: more inlinable (no `__torch_function__` boundary).

The benchmark in `tests/bench/bench_lie.py` (new — see
[Proposal 12](12_regression_and_benchmarks.md)) freezes both
backends so we cannot regress without noticing.

## Migration plan

1. **Phase A — write the backend** (1 file, ~250 LOC).
2. **Phase B — gradcheck and value-parity tests** under
   `BR_LIE_BACKEND=torch`. Default still `pypose`.
3. **Phase C — flip the default**. Run the full test suite under both
   backends in CI for one release.
4. **Phase D — remove the FD path in `residual_jacobian`** as the
   default; FD becomes opt-in via `JacobianStrategy.FINITE_DIFF`.
5. **Phase E — drop PyPose** in v1.2 (delete `_pypose_backend.py` and
   `BR_LIE_BACKEND`); update docs; remove the `pypose` dependency
   from `pyproject.toml`.

Phases A–C land within one release; D in the next; E one after that.
External users have a full release window with both backends
available before PyPose disappears.

## Tradeoffs

| For | Against |
|-----|---------|
| Eliminates the single largest source of correctness friction in the library. | A small one-off engineering cost (~3 days) to write and validate the backend. |
| Removes `pypose` from the dep tree (one fewer transitive constraint). | We become responsible for SE3/SO3 numerical stability. Mitigation: every formula is from a textbook (Murray/Li/Sastry, Barfoot, Chirikjian), tested at boundary angles, gradcheck-locked. |
| Makes Adam/LBFGS/future solvers free to call `loss.backward()` like normal users expect. | Some users may have built workflows around PyPose tangent-space gradients; mitigation: `BR_LIE_BACKEND=pypose` for one release, and a release note. |
| Smaller import surface — `lie/` no longer depends on `pypose`, which has a heavy import chain. | None significant. |

## Acceptance criteria

- `lie/_torch_backend.py` exists; passes
  `tests/lie/test_torch_backend_gradcheck.py` at fp64.
- Forward-pass parity with PyPose backend within `1e-6` (fp32) /
  `1e-12` (fp64) on randomised batches at sizes 1, 32, 1024.
- The current 297-test suite passes with `BR_LIE_BACKEND=torch` set.
- `tests/bench/bench_lie.py` shows the torch backend is no slower
  than 1.5× PyPose on CPU and ≤ 1.2× on CUDA for `compose`, `log`,
  `exp`, `inverse`.
- After Phase C, `JacobianStrategy.AUTODIFF` is the default in
  `residual_jacobian`; tests for every analytic residual still pass
  the analytic-vs-autodiff-equivalence check from
  [16_TESTING.md §4.4](../../conventions/16_TESTING.md).
- After Phase E, `import pypose` is no longer present anywhere in the
  source tree; `pyproject.toml` no longer lists `pypose` as a
  dependency.

## Cross-references

- [Proposal 02](02_backend_abstraction.md) — the backend dispatch
  this proposal swaps under.
- [PYPOSE_ISSUES.md](../../status/PYPOSE_ISSUES.md) — the bug catalogue
  this proposal closes.
- [16_TESTING.md §4.4](../../conventions/16_TESTING.md) — the
  analytic-vs-autodiff equivalence test that becomes a one-line
  `torch.func.jacrev` comparison after this lands.
