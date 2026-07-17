# Contracts & Validation

> **Status:** normative. Describes the library's input contracts,
> numerical guarantees, and error-handling policy.

Other docs describe *what* the library does. This one describes the
**rules every public function must honour** and **what the user must
give in return**. Contracts pin the responsibility boundary: with one,
"my IK is wrong" splits cleanly into "you passed me a non-unit
quaternion" and "the library has a bug." Without one, every public
function eventually grows ambiguous defensive checks because nobody is
sure who is responsible for which input shape — and the call site
slowly accrues noise that exists to defend against scenarios that
should not happen.

We picked a small set of explicit rules instead. The target policy validates
inputs **at the boundary** with named exceptions and trusts internal callers.
Where the current code does not yet enforce a rule, this document names the
gap instead of claiming the exception already fires. Cache levels are tracked
with `KinematicsLevel`; the SE(3) layout is a stable convention.

The benefit shows up everywhere downstream. The hot-path lint can
forbid `.item()` because contracts already pin shapes. The solver can
skip "is `q` the right shape?" because the entry point already
checked. Documentation can read like prose because the rules live
here, not scattered through every function.

## 1 · Input contracts

### 1.1 Tensor shapes

All public tensors are shaped `(B..., feature)` where `B...` is zero
or more leading batch axes and `feature` is the semantic last-axis
(e.g. `nq`, `(njoints, 7)`, etc.). See
{doc}`/concepts/batching_and_backends` for the full shape table.

**What we promise:**

- Every return tensor's leading batch shape **equals** the broadcast
  of all input batch shapes.
- We never drop a leading batch axis (`B=1` stays `(1, …)`).
- We never add a leading batch axis the user did not provide.

**What we expect:**

- Inputs are contiguous (`.contiguous()` if not — cost is the caller's).
- `q` has shape `(B..., nq)`; `v` / `a` / `tau` have `(B..., nv)`.
- Broadcast follows PyTorch rules; ambiguous broadcasts raise.

### 1.2 Device & dtype

- `Model.to(device, dtype)` returns a new `Model` with all tensor
  buffers moved.
- `Data` inherits device / dtype from the input `q`. Calling
  `forward_kinematics(model, q)` with `q.device != model.device`
  raises `DeviceMismatchError`.
- fp32 is primary and fp64 is supported. Supported paths preserve the
  working input dtype; see {doc}`engineering` for accumulation, TF32,
  and tolerance policy.
- fp16 and bf16 are outside the supported contract. Cast up before the
  call. FK and dynamics passes reject them with `DtypeMismatchError`.
- A model/query dtype mismatch raises `DtypeMismatchError` before FK or a
  dynamics pass begins; cast the query or move the model explicitly.

### 1.3 Quaternion / SE(3) inputs

- Quaternion format is **`[qx, qy, qz, qw]`** (scalar last). Every
  function accepting an SE(3) pose reads it as `[tx, ty, tz, qx, qy,
  qz, qw]`.
- Quaternions are assumed unit-norm on input; FK does not normalize them.
  `forward_kinematics(..., check_quaternion_norm=True)` enables an opt-in
  boundary check for free-flyer configurations. The check synchronizes
  accelerator tensors and is therefore disabled on the default hot path.
- With that diagnostic enabled, a norm outside `[0.9, 1.1]` raises
  `QuaternionNormError`. Normalize configurations before calling FK.

### 1.4 Joint limits

- `model.lower_pos_limit <= q <= model.upper_pos_limit` is *not*
  automatically enforced. FK accepts any `q`; the solver honours
  limits only when told to (see `IKCostConfig.limit_weight`).
- Continuous revolute joints (kind `revolute_unbounded`) have `±inf`
  limits — wrapping is the user's responsibility if they care.
- Free-flyer: the four quaternion components live on the sphere; the
  library treats them as a free manifold (no rectangular bounds). To
  constrain orientation, use an `OrientationResidual` with a target
  pose and a large weight, not a limit.

### 1.5 Model well-formedness

After `build_model`:

- `parents[0] == -1` (joint 0 is the universe).
- `parents[i] < i` for `i > 0` (topologically sorted).
- `sum(nqs) == nq`, `sum(nvs) == nv`.
- `idx_qs[i] + nqs[i] == idx_qs[i+1]` (contiguous slicing).

Violations raise `ModelInconsistencyError` at build time — never at
query time.

## 2 · Error taxonomy

The library raises a small set of typed exceptions. Each has a single
responsible layer and a documented remediation.

| Exception | Raised in | Meaning | Remediation |
|-----------|-----------|---------|-------------|
| `ModelInconsistencyError` | `io.build_model` | Parsed IR violates topology invariants | Fix the URDF / MJCF or the programmatic builder |
| `DeviceMismatchError` | `kinematics`, `dynamics`, `optim` | `q.device != model.device` | Call `model.to(q.device)` or vice versa |
| `DtypeMismatchError` | as above | `q.dtype` incompatible with `model.dtype` | Cast one side; see §1.2 |
| `QuaternionNormError` | `kinematics` opt-in debug check | Free-flyer quaternion norm outside `[0.9, 1.1]` | Normalise before passing or enable the check only while debugging |
| `ShapeError` | every public entry | Wrong trailing-axis size | Match the published shape |
| `StaleCacheError` | `kinematics`, `dynamics` | `Data._kinematics_level` below required level | Call `forward_kinematics(model, data)` first; or `data.invalidate(NONE)` then re-run |
| `ConvergenceError` | Caller policy (optional) | Solver did not converge within `max_iter` | Inspect the returned `SolverState` |
| `UnsupportedJointError` | `io.build_model` | URDF / MJCF joint kind without a built-in `JointModel` | Add a custom joint via {doc}`extension` |
| `SingularityWarning` *(warning, not error)* | `kinematics`, `optim` | Jacobian condition number > 1e12 | Change initial configuration or relax weights |

**Rule of thumb:**

- Library-internal invariants should never raise — they should
  `assert` and fail loudly. Use `assert` for those.
- User-facing input violations raise one of the exceptions above, with
  a message naming the offending tensor and what was expected.

## 3 · Numerical guarantees

### 3.1 Determinism

- `forward_kinematics`, `compute_joint_jacobians`, `rnea`, `aba`,
  `crba` are deterministic bit-for-bit on a pinned PyTorch / CUDA
  version for a given `(device, dtype, input)` triple.
- `solve_ik` is deterministic on a pinned seed and pinned solver
  config, modulo floating-point non-associativity across CUDA
  streams. Tests use `torch.use_deterministic_algorithms(True)` on
  the reference path.

### 3.2 Accuracy

| Routine | Guaranteed accuracy |
|---------|---------------------|
| `se3.exp(log(T))` | `‖Δ‖ < 1e-6` (fp32), `< 1e-12` (fp64) |
| Analytic FK Jacobian vs. `jacrev` | `‖ΔJ‖_F < 1e-4` (fp32), `< 1e-10` (fp64) |
| `solve_ik` pose residual at reported `converged=True` | `‖r‖ < tol` where `tol=1e-4` by default |
| Long-chain FK (30 joints) | 1 ulp of a well-conditioned product of SE(3)s |

Numerical accuracy tests live in `tests/kinematics/` and
`tests/regression/`.

### 3.3 Singularities

- Near SO(3) singularities (rotation angle ≥ `π - 1e-6`), `log` falls
  back to a Taylor expansion for stable gradients.
- Rank-deficient spatial Jacobians are a user concern; the analytic
  inversion in LM handles them via Levenberg damping, not by
  pseudo-inversion. If the user opts into Gauss-Newton and `JᵀJ` is
  singular, the linear solver raises `torch._C._LinAlgError` and the
  outer loop catches it into `ConvergenceError`.

### 3.4 Batched broadcasting

Where two inputs have different leading batch shapes, broadcast rules
are PyTorch's. A broadcast that would create an implicit copy larger
than its input is allowed but logged under `BR_WARN_BROADCAST=1`.
Silent in default runs.

## 4 · Mutability rules

| Object | Mutable? | Notes |
|--------|----------|-------|
| `Model` | Shallowly frozen | `@dataclass(frozen=True)` prevents field reassignment, but contained tensors/dicts remain mutable. Treat them as read-only; deep immutability remains an engineering gap. |
| `Data` | Yes | Mutated by kinematics / dynamics. Thread-local — do not share across threads without copying. |
| `IKResult` | No | Dataclass; `.q` is a view into the solver's tensor but treated as read-only. |
| `Trajectory` | Limited | `slice`, `resample` return new instances. Direct tensor access is read-only unless you know what you are doing. |
| `CostStack` | Controlled | `.add(...)`, `.set_weight(...)`, `.set_active(...)` are the *only* mutation points. |
| `LeastSquaresProblem` | No | Re-build if you want different bounds or initial guesses. |

Rule: if it is shared across threads or workers, it is frozen. If it
is per-query, it may be mutable.

## 5 · Autograd rules

- Autograd guarantees are path- and input-specific. Eager Torch Lie/FK and
  named dynamics paths have gradient tests; that does not make every public
  function differentiable. The complete matrix is in {doc}`engineering`.
- The current `solve_ik` detaches its initial iterate and has no
  differentiable-solve guarantee.
- `residual_jacobian(..., strategy=ANALYTIC)` uses the residual's
  `.jacobian()` method. `strategy=AUTO` prefers analytic and falls back to
  unbatched central finite differences at a cost of `2·nv + 1` residual
  evaluations. Real `torch.func` strategies are scheduled for M2.
- **Forbidden**: in-place mutation of a tensor currently on the
  autograd tape. The library uses functional-style ops throughout;
  contributions must too.

## 6 · Threading & concurrency

- `Model` may be shared only while every contained tensor and dictionary
  is treated as read-only. The current frozen dataclass is not deeply
  immutable.
- `Data` is mutable ⇒ one `Data` per thread. Use `data.clone()` for
  fork points.
- `CostStack` is mutable; one per optimisation problem. Parallelising
  over problems requires a fresh stack per thread.
- The library does not call `torch.set_num_threads` internally; it
  inherits the user's setting.

CUDA-stream and multiprocessing rules live in {doc}`engineering`.

## 7 · Backwards compatibility policy

### 7.1 SemVer scope

Before 1.0, minor releases may change the public surface. Contract tests pin
a required core and reject duplicate or unresolvable exports without freezing
the current symbol count. Once 1.0 is released, a **major bump** is required
to change:

- A stable public symbol named in this section.
- The SE(3) quaternion layout (`[tx, ty, tz, qx, qy, qz, qw]`).
- The `Model` / `Data` dataclass fields (additive is allowed in
  minor; rename is major unless part of a documented migration
  window).
- The DAG (a new edge in {doc}`/concepts/architecture`).

The complete release / deprecation discipline lives in {doc}`packaging`.
This file pins the contract; that file pins the operational mechanism
(extras, version source, release process, CI gating).

A **minor bump** may:

- Add a new public symbol, joint kind, residual, solver.
- Rename a storage field with one release of deprecation shim (see
  {doc}`naming`).
- Tighten a numerical tolerance (never loosen without a major bump).

A **patch bump** is bug fixes; performance improvements that do not
change numerical output beyond tolerance.

### 7.2 Deprecation mechanism

```python
import warnings
warnings.warn(
    "The 'old_name' API is deprecated; use 'new_name'. "
    "Will be removed in vX.Y.",
    DeprecationWarning,
    stacklevel=2,
)
```

Deprecation warnings are on by default under `pytest` and silent in
production. `BR_STRICT=1` promotes them to errors (used in CI).

### 7.3 Stability tier per symbol

| Tier | Meaning | Examples |
|------|---------|----------|
| Stable | SemVer-bound; major bump to remove or rename | `Model`, `Data`, `forward_kinematics`, `solve_ik`, `SE3`, `ModelBuilder`, `LeastSquaresProblem`, `Trajectory` |
| Stable (Protocol) | Extending the protocol (adding methods) is a major bump; using existing methods is stable | `JointModel`, `Residual`, `Optimizer`, `LinearSolver`, `RobustKernel`, `DampingStrategy`, `TrajectoryParameterization`, `AssetResolver` |
| Experimental | May change in minor releases with a deprecation warning | `solve_trajopt`, `compute_centroidal_map`, `BSplineTrajectory`, `MultiStageOptimizer` |

| Module | Stability |
|--------|-----------|
| `lie/`, `spatial/` | Stable from v1. Changes require major bump. |
| `data_model/` | Stable from v1. Field renames follow §7.1 deprecation. |
| `kinematics/`, `dynamics/` | Stable from v1. |
| `residuals/`, `costs/`, `optim/` | Stable from v1 — Protocol signatures are frozen. |
| `tasks/` | Stable from v1 for IK; `solve_trajopt` is experimental. `TrajectoryParameterization` Protocol is stable. |
| `collision/` | Experimental. |
| `viewer/` | Experimental. The `RendererBackend` protocol refers only to scene rendering and is stable; concrete modes may iterate. |
| `io/` (URDF / MJCF) | Stable. Parser edge cases may iterate in patch releases. `AssetResolver` Protocol stable. |

Experimental means: no SemVer guarantee, but the signatures will not
wander without a release note.

## 8 · Logging

`better_robot.logger` is a `logging.Logger` named `"better_robot"`.

| Level | When |
|-------|------|
| DEBUG | First-call compile events, registry registrations |
| INFO | One-time `Model` load summary (`nq`, `nv`, `njoints`) |
| WARNING | `SingularityWarning`, `BroadcastWarning`, deprecated names |
| ERROR | Recoverable failures (convergence, fallback paths engaged) |

No `print` calls in the library. Ever.

## 9 · Assumptions summary (the one-page contract)

Obey **all** of these and the library obeys its numerical guarantees:

1. `q.shape == (B..., nq)`; `v / a / tau.shape == (B..., nv)`.
2. Quaternions scalar-last, unit-norm on entry (tolerance 10%).
3. `q.device == model.device`, `q.dtype in {fp32, fp64}`.
4. `Model` is built once; do not mutate.
5. Self-limits (`q ∈ [lo, hi]`) are the user's responsibility unless
   `limit_weight > 0`.
6. `Data` is per-thread.
7. `CostStack`, `LeastSquaresProblem` are per-optimisation.

Break any of these and the library will do something, but we make no
promise about what.
